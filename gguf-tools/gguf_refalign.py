#!/usr/bin/env python3
"""Realign a GGUF's trailing disk-only tables onto a reference file and write it
as an APFS clone of that reference, so one copy of the tables serves both packs.

The runtime reads the 95.37 GiB BF16 Qwen n-gram table (and the DeepSeek V4.1
engram tables) with plain preads and never maps it, so every sibling pack pays
for its own copy on disk.  Siblings built from the same checkpoint carry
byte-identical tables, but each pack puts them right after its own main
weights, so the copies sit at different absolute offsets.

Reflink deduplication (refdelta, cp -c, this tool) shares bytes between the
same *absolute* file offsets only -- APFS has no range clone -- and it works on
pages, so two payloads share exactly when they are identical and start on the
same page.  Re-padding the tensor data section with general.alignment cannot fix
that: the tables sit behind the main model, so they move with it.  What has to
match is the tables' absolute offset, which is what this tool pins:

    OUT = clone(REFERENCE)              # extents shared, no data copied
    write header + main model           # new space, only where bytes differ
    punch the gap before the tables     # free
    leave the table region untouched    # 95.37 GiB shared with REFERENCE

The output is a complete, independently readable GGUF; the table rel_offsets
and general.alignment are the only header changes.  Deleting the original input
afterwards reclaims everything but the main model.

  python3 gguf-tools/gguf_refalign.py \
      gguf/Qwen3.8-Flash-Next-Q4.gguf \
      gguf/Qwen3.8-Flash-Next-Uncensored-Q4.gguf \
      --verify --report gguf/uncensored.realign.json --install

The reference is never modified; keep it as the family anchor so the next
sibling shares the same tables.  Requires macOS/APFS.
"""

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str, tnbytes

TABLE = 'per_layer_token_embd.weight'
PAGE = os.sysconf('SC_PAGESIZE')
CHUNK = 4 << 20
F_PUNCHHOLE = 99
F_LOG2PHYS_EXT = 65

_libc = ctypes.CDLL(None, use_errno=True)
_libc.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
_libc.clonefile.restype = ctypes.c_int


def clonefile(src, dst):
    """Whole-file reflink: dst shares every extent with src for free."""
    if _libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f'clonefile {src} -> {dst} failed (same APFS volume?)')


def read_exact(fp, off, n):
    fp.seek(off)
    buf = bytearray()
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            raise ValueError(f'truncated GGUF payload at {off}')
        buf += chunk
    return bytes(buf)


def write_exact(fp, off, data):
    """pwrite the whole buffer at off. A short pwrite must resume at
    off + written -- restarting the remainder at off would silently smear
    the payload and read back fine through the page cache until reload."""
    view = memoryview(data)
    done = 0
    while done < len(view):
        n = os.pwrite(fp.fileno(), view[done:], off + done)
        if n <= 0:
            raise OSError(f'pwrite stalled at {off + done}')
        done += n


def punch(fp, off, size):
    """Deallocate a page-aligned range; a clone simply drops the shared blocks."""
    try:
        fcntl.fcntl(fp, F_PUNCHHOLE, struct.pack('=IIqq', 0, 0, off, size))
        return True
    except OSError:
        return False


def payload(reader):
    """Tensors as (name, kind, dims, abs_offset, bytes) in payload order."""
    items = []
    for name, (kind, dims, off) in reader.tensors.items():
        items.append((name, kind, dims, reader.data_start + off,
                      tnbytes(kind, math.prod(dims))))
    items.sort(key=lambda it: it[3])
    if not items:
        raise ValueError(f'{reader.f.name}: no tensors')
    end = items[0][3]
    for _, _, _, off, size in items:
        if off < end:
            raise ValueError(f'{reader.f.name}: tensors overlap at {off}')
        end = max(end, off + size)
    if end > os.fstat(reader.f.fileno()).st_size:
        raise ValueError(f'{reader.f.name}: tensor payload exceeds file size')
    return items


def trailing_tables(items, size, names, aligned):
    """The disk-only tables must be the last payload, in order, clear of the main
    model.  The anchor must also start on a page: that is what lets the runtime
    unmap them and read rows straight from the file, and what lets a clone share
    them page by page.  Returns (start, bytes, [(name, kind, dims, rel, size)])."""
    wanted = set(names)
    tables = sorted((it for it in items if it[0] in wanted), key=lambda it: it[3])
    if len(tables) != len(wanted):
        raise ValueError(f'missing one of the disk-only tensors {",".join(names)}')
    start, end = tables[0][3], tables[-1][3] + tables[-1][4]
    if end != size:
        raise ValueError(f'{",".join(names)} must trail every other tensor')
    if [it for it in items if it[3] >= start] != tables:
        raise ValueError('the disk-only tensors must be the last payload, in order')
    if aligned and start % PAGE:
        raise ValueError(f'reference table must start on a {PAGE}-byte page boundary')
    if max((it[3] + it[4] for it in items if it[0] not in wanted), default=0) > start:
        raise ValueError('the disk-only tensors must not overlap the main model')
    return start, end - start, [(it[0], it[1], it[2], it[3] - start, it[4])
                                for it in tables]


def header(kv, kv_types, plan):
    out = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(kv)))
    for key, kind in kv_types.items():
        out += kv_bytes(key, kind, kv[key])
    for name, kind, dims, off in plan:
        out += w_str(name) + struct.pack('<I', len(dims))
        out += struct.pack('<' + 'Q' * len(dims), *dims)
        out += struct.pack('<IQ', kind, off)
    return bytes(out)


def fill(out, start, end, stats):
    """Clear a range the clone inherited from the reference. Punching keeps the
    gap free; short or unaligned runs can only be written as zeros."""
    if end <= start:
        return
    if end - start >= CHUNK and start % PAGE == 0 and end % PAGE == 0 and \
            punch(out, start, end - start):
        stats['punched'] += end - start
    else:
        write_exact(out, start, bytes(end - start))


def copy(out, src, dst, srcoff, size, stats, zeros):
    """Copy one tensor, leaving chunks the clone already holds untouched."""
    done = 0
    while done < size:
        n = min(CHUNK, size - done)
        data = read_exact(src, srcoff + done, n)
        if data == zeros[:n] and (n == CHUNK or n % PAGE == 0) and \
                (dst + done) % PAGE == 0 and punch(out, dst + done, n):
            stats['punched'] += n
        elif read_exact(out, dst + done, n) == data:
            stats['shared'] += n
        else:
            write_exact(out, dst + done, data)
            stats['written'] += n
        stats['copied'] += n
        done += n
        if stats['copied'] // (8 << 30) > (stats['copied'] - n) // (8 << 30):
            print(f'  {stats["copied"] / 2**30:.1f} GiB walked', flush=True)


def refalign(reference, model, output, tables=(TABLE,), alignment=PAGE,
             verify=True, probe=True, install=False, report_path=None, dry=False):
    if alignment & (alignment - 1) or alignment < 32:
        raise ValueError('alignment must be a power of two of at least 32')
    if Path(output).exists() or Path(output) == Path(model):
        raise ValueError(f'{output} exists or is the input; choose a new path')
    names = tuple(tables)
    ref, src = Reader(str(reference)), Reader(str(model))
    try:
        ref_items, src_items = payload(ref), payload(src)
        anchor, region, ref_tables = trailing_tables(
            ref_items, os.fstat(ref.f.fileno()).st_size, names, True)
        src_anchor, _, src_tables = trailing_tables(
            src_items, os.fstat(src.f.fileno()).st_size, names, False)
        if [(t[0], t[1], t[2], t[4]) for t in ref_tables] != \
                [(t[0], t[1], t[2], t[4]) for t in src_tables]:
            raise ValueError('the disk-only tables differ between the two packs')
        if anchor % alignment:
            raise ValueError(f'reference table offset {anchor} is not {alignment}-aligned')
        # Each table keeps the absolute slot it has in the reference, where its
        # bytes already match: (name, output offset, input offset, bytes).
        pins = [(t[0], anchor + t[3], src_anchor + s[3], t[4])
                for t, s in zip(ref_tables, src_tables)]

        # The main model keeps its own page-aligned slots; only the tables are
        # pinned to the reference, and the gap they leave behind is punched free.
        src.kv['general.alignment'], src.kv_types['general.alignment'] = alignment, 4
        rel, slots = 0, []
        for name, kind, dims, off, size in src_items:
            if name in names:
                continue
            slots.append((name, kind, dims, rel, size, off))
            rel = (rel + size + alignment - 1) // alignment * alignment
        data_start = 0
        for _ in range(2):  # the header length is fixed, one redo settles it
            plan = [(n, k, d, o) for n, k, d, o, _, _ in slots]
            plan += [(n, t[1], t[2], off - data_start)
                     for (n, off, _, _), t in zip(pins, ref_tables)]
            blob = header(src.kv, src.kv_types, plan)
            data_start = (len(blob) + alignment - 1) // alignment * alignment
        main_end = data_start + rel
        if main_end > anchor:
            raise ValueError(
                f'main model needs {(main_end - anchor) / 2**20:.1f} MiB more than the '
                f'gap to the reference table; anchor on a pack with more slack')
        total = anchor + region
        print(f'{reference}: {len(ref_tables)} disk-only table tensor(s) at {anchor} '
              f'(page {anchor // PAGE}, {region / 2**30:.2f} GiB shared)\n'
              f'{model}: tables move {src_anchor} -> {anchor} '
              f'({(anchor - src_anchor) / 2**20:+.1f} MiB)\n'
              f'  header {data_start} ({data_start - len(blob)} spare), main '
              f'{rel / 2**30:.2f} GiB at alignment {alignment}, '
              f'{(anchor - main_end) / 2**20:.1f} MiB gap punched\n'
              f'  {output}: {total / 2**30:.2f} GiB logical, at most '
              f'{rel / 2**30:.2f} GiB new space', flush=True)
        result = dict(reference=str(reference), model=str(model),
                      tables=list(names), table_offset=anchor, table_bytes=region,
                      alignment=alignment, bytes=total, main_bytes=rel,
                      gap=anchor - main_end, copied=0, shared=0, written=0,
                      punched=0, verified=False, dry=dry,
                      tensors=[dict(name=n, offset=data_start + o, bytes=s)
                               for n, _, _, o, s, _ in slots] +
                              [dict(name=n, offset=o, bytes=s) for n, o, _, s in pins])
        stats = result          # the copy counters below land in the report
        if dry:
            return result

        clonefile(reference, output)
        try:
            out = open(output, 'r+b')
            try:
                if os.fstat(out.fileno()).st_size != total:
                    os.ftruncate(out.fileno(), total)
                write_exact(out, 0, blob)
                write_exact(out, len(blob), bytes(data_start - len(blob)))
                # Walk the payload in output order so both streams stay sequential.
                # A chunk the clone already holds byte-identically is left alone
                # and keeps sharing the reference's extents.
                segments = [(data_start + off, src_off, size)
                            for _, _, _, off, size, src_off in slots]
                segments += [(dst, srcoff, size) for _, dst, srcoff, size in pins]
                segments.sort()
                zeros, prev = bytes(CHUNK), data_start
                for dst, srcoff, size in segments:
                    fill(out, prev, dst, stats)
                    copy(out, src.f, dst, srcoff, size, stats, zeros)
                    prev = dst + size
                out.flush()
                os.fsync(out.fileno())
            finally:
                out.close()
            print(f'wrote {output}: {stats["written"] / 2**30:.2f} GiB new, '
                  f'{stats["shared"] / 2**30:.2f} GiB already shared, '
                  f'{stats["punched"] / 2**30:.2f} GiB punched', flush=True)
            if verify:
                verify_model(model, output, data_start, alignment,
                             {n: o for n, o, _, _ in pins})
                result['verified'] = True
        except Exception:
            os.unlink(output)   # a partial or unverified output is never runnable
            raise
        if probe:
            result['sharing'] = probe_sharing(
                output, reference, [(o, s) for _, o, _, s in pins])
            if result['sharing']:
                print(f'{output}: {result["sharing"][0]}/{result["sharing"][1]} '
                      f'sampled table pages resolve to the reference blocks', flush=True)
        result['table_fingerprint'] = sample_sha(
            model, [(s, b) for _, _, s, b in pins])
        if report_path:
            Path(report_path).write_text(json.dumps(result, indent=2) + '\n')
        if install:
            old = Path(str(model) + '.pre-refalign')
            if old.exists():
                raise ValueError(f'{old} exists; remove it first')
            os.rename(model, old)
            os.rename(output, model)
            print(f'installed {model}; rm {old} to reclaim the original pack', flush=True)
        return result
    finally:
        ref.f.close()
        src.f.close()


def verify_model(model, output, data_start, alignment, pins):
    """Compare every payload tensor, then the header. The output is not a byte
    copy of the input: the tables moved and the gap behind them became a hole."""
    src, out = Reader(str(model)), Reader(str(output))
    try:
        if out.data_start != data_start:
            raise ValueError('output data section does not start where the plan said')
        # general.alignment is the only metadata this tool adds or rewrites.
        if set(src.kv_types) | {'general.alignment'} != set(out.kv_types):
            raise ValueError('output metadata differs from the model')
        src_items = {it[0]: it for it in payload(src)}
        out_items = {it[0]: it for it in payload(out)}
        if sorted(src_items) != sorted(out_items):
            raise ValueError('output tensor set differs from the model')
        walked = 0
        t0 = time.time()
        for name, item in src_items.items():
            other = out_items[name]
            if item[1:3] != other[1:3] or item[4] != other[4]:
                raise ValueError(f'{name}: layout changed')
            if name in pins and other[3] != pins[name]:
                raise ValueError(f'{name}: not on the reference offset')
            for off in range(0, item[4], CHUNK):
                n = min(CHUNK, item[4] - off)
                if read_exact(src.f, item[3] + off, n) != \
                        read_exact(out.f, other[3] + off, n):
                    raise ValueError(f'{name}: payload mismatch at {off}')
                walked += n
                if walked // (8 << 30) > (walked - n) // (8 << 30):
                    print(f'  verified {walked / 2**30:.1f} GiB '
                          f'in {time.time() - t0:.0f}s', flush=True)
        for key in src.kv_types:
            value = getattr(src.kv[key], 'values', src.kv[key])
            wrote = getattr(out.kv[key], 'values', out.kv[key])
            if key == 'general.alignment':
                # The one key this tool rewrites; the output must carry the
                # new value, not merely pass a vacuous comparison.
                if wrote != alignment:
                    raise ValueError('general.alignment was not rewritten')
                continue
            if value != wrote:
                raise ValueError(f'metadata {key} changed')
        print(f'verified all {walked / 2**30:.2f} GiB of {output} '
              f'in {time.time() - t0:.0f}s', flush=True)
    finally:
        src.f.close()
        out.f.close()


def probe_sharing(output, reference, ranges):
    """F_LOG2PHYS_EXT proves the tables really share extents with the reference."""
    hits = total = 0
    with open(output, 'rb') as a, open(reference, 'rb') as b:
        for off, size in ranges:
            step = max(PAGE, size // 64) // PAGE * PAGE
            for pos in range(0, size - PAGE, step):
                try:
                    dev = [struct.unpack('=Iqq', fcntl.fcntl(
                                fp, F_LOG2PHYS_EXT,
                                struct.pack('=Iqq', 0, PAGE, off + pos)))[2]
                           for fp in (a, b)]
                except OSError:
                    return None         # filesystem without the mapping call
                total += 1
                hits += dev[0] == dev[1]
    return (hits, total) if total else None


def sample_sha(path, ranges):
    """Cheap identity of the shared tables: one page every 16 MiB."""
    digest = hashlib.sha256()
    with open(path, 'rb') as fp:
        for off, size in ranges:
            for step in range(0, size, 1 << 24):
                digest.update(read_exact(fp, off + step, min(PAGE, size - step)))
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('reference', help='pack that already owns the shared tables (unchanged)')
    ap.add_argument('model', help='sibling pack to realign')
    ap.add_argument('--output', help='result file (default: <model>.realign)')
    ap.add_argument('--table', default=[TABLE], action='append',
                    help='disk-only tensor trailing the main model; repeat or '
                         'comma-separate for several tables')
    ap.add_argument('--alignment', type=int, default=PAGE,
                    help='tensor data alignment; the page size keeps every '
                         'tensor and gap page-aligned')
    ap.add_argument('--verify', action='store_true', help='compare every payload tensor')
    ap.add_argument('--no-probe', action='store_true', help='skip the extent probe')
    ap.add_argument('--install', action='store_true',
                    help='rename the model aside and move the output into its place')
    ap.add_argument('--report', help='write a JSON layout report here')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    names = [n for tables in args.table for n in tables.split(',')]
    try:
        refalign(args.reference, args.model,
                 args.output or args.model + '.realign', names, args.alignment,
                 args.verify or args.install, not args.no_probe, args.install,
                 args.report, args.dry_run)
    except (OSError, ValueError) as error:
        ap.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
