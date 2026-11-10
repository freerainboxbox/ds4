"""Reflink realignment: the shared disk-only tables land on the anchor's offsets,
so only the main model's differing bytes take new space."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gguf_refalign as align
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

PAGE, MB = align.PAGE, 1 << 20
TABLE = 'per_layer_token_embd.weight'
DIMS = [2, 4096 * 4]                  # BF16 -> a 64 KiB table spanning pages


def write_gguf(path, meta, parts, tables, alignment=32):
    """parts: [(name, F32 payload)] padded like llama.cpp, tables
    [(name, dims, kind, payload)] trailing them.  Returns the first table offset."""
    rel, plan, data = 0, [], b''
    for name, raw in parts:
        plan.append((name, [len(raw) // 4], 0, rel))
        data += raw + bytes(-len(raw) % alignment)
        rel = len(data)
    trail = []
    for name, dims, kind, raw in tables:
        trail.append((name, dims, kind, rel))
        data += raw
        rel = len(data)

    def blob():
        out = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(plan) + len(trail), len(meta)))
        out += b''.join(kv_bytes(k, t, v) for k, (t, v) in meta.items())
        for name, dims, kind, off in plan + trail:
            out += w_str(name) + struct.pack('<I', len(dims))
            out += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, off)
        return bytes(out)

    head = blob()
    start = (len(head) + alignment - 1) & -alignment
    path.write_bytes(head + bytes(start - len(head)) + data)
    return start + rel - sum(len(t[3]) for t in tables)


class Refalign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.anchor = root/'anchor.gguf'
        self.model = root/'sibling.gguf'
        self.out = root/'sibling.realign'
        self.table = bytes(range(256)) * 256
        self.main = [bytes([i + 1]) * MB for i in range(10)]
        # The anchor packs ten main tensors, so its table sits on a page.
        self.anchor_table = write_gguf(
            self.anchor, {'general.architecture': (8, 'qwen4exp'),
                          'general.alignment': (4, PAGE),
                          'qwen4exp.ple.row_count': (10, DIMS[1])},
            [(f'blk.{i}.w', raw) for i, raw in enumerate(self.main)],
            [(TABLE, DIMS, 30, self.table)], PAGE)
        # The sibling is a fine-tune: fewer, different main tensors and its own
        # metadata, so its copy of the same table sits at another offset.
        self.model_table = write_gguf(
            self.model, {'general.architecture': (8, 'qwen4exp'),
                         'general.name': (8, 'sibling'),
                         'qwen4exp.ple.row_count': (10, DIMS[1])},
            [('blk.0.w', self.main[0]), ('blk.1.w', bytes([9]) * MB)],
            [(TABLE, DIMS, 30, self.table)])
        self.assertNotEqual(self.model_table, self.anchor_table)
        self.gap = self.anchor_table - PAGE - 2 * MB

    def realign(self, **kw):
        kw.setdefault('probe', False)
        kw.setdefault('verify', True)
        return align.refalign(self.anchor, self.model, self.out, **kw)

    def test_table_moves_to_the_anchor_and_only_diffs_are_written(self):
        report = self.realign()
        self.assertEqual(report['table_offset'], self.anchor_table)
        self.assertEqual(report['table_bytes'], len(self.table))
        self.assertEqual(report['bytes'], self.anchor_table + len(self.table))
        self.assertEqual(report['main_bytes'], 2 * MB)
        self.assertEqual(report['gap'], self.gap)
        # The table plus the one main tensor the sibling shares with the anchor.
        self.assertEqual(report['shared'], len(self.table) + MB)
        self.assertEqual(report['written'], MB)
        self.assertEqual(report['punched'], self.gap)
        self.assertTrue(report['verified'])

        raw = self.out.read_bytes()
        self.assertEqual(len(raw), report['bytes'])
        self.assertEqual(raw[PAGE + 2 * MB:self.anchor_table], bytes(self.gap))
        self.assertEqual(raw[self.anchor_table:], self.table)
        model, wrote = Reader(str(self.model)), Reader(str(self.out))
        self.addCleanup(model.f.close)
        self.addCleanup(wrote.f.close)
        self.assertEqual((wrote.data_start, wrote.kv['general.alignment']), (PAGE, PAGE))
        self.assertEqual(wrote.kv['general.name'], 'sibling')
        for name, (kind, dims, off) in wrote.tensors.items():
            if name == TABLE:
                self.assertEqual((kind, dims, off),
                                 (30, DIMS, self.anchor_table - PAGE))
                continue
            src = model.data_start + model.tensors[name][2]
            blob = self.model.read_bytes()
            self.assertEqual(raw[wrote.data_start + off:wrote.data_start + off + MB],
                             blob[src:src + MB])
        # The punched gap costs no blocks, so the pack stays sparse on disk.
        self.assertLessEqual(self.out.stat().st_blocks * 512,
                             report['bytes'] - self.gap)

    def test_extents_are_shared_with_the_anchor(self):
        self.realign(probe=True, verify=False)
        probe = align.probe_sharing(self.out, self.anchor,
                                    [(self.anchor_table, len(self.table))])
        if probe is None:
            self.skipTest('F_LOG2PHYS_EXT unsupported here')
        self.assertGreater(probe[1], 1)
        self.assertEqual(probe[0], probe[1])

    def test_several_trailing_tables_are_pinned_together(self):
        # DeepSeek V4.1 style: two disk-only tables behind the main weights.
        extra = bytes(range(256)) * 16               # 4 KiB of engram rows
        write_gguf(self.anchor, {'general.architecture': (8, 'deepseek41'),
                                 'general.alignment': (4, PAGE)},
                   [(f'blk.{i}.w', raw) for i, raw in enumerate(self.main)],
                   [('blk.1.engram_embd.weight', [512], 27, extra),
                    (TABLE, DIMS, 30, self.table)], PAGE)
        write_gguf(self.model, {'general.architecture': (8, 'deepseek41')},
                   [('blk.0.w', self.main[0]), ('blk.1.w', bytes([9]) * MB)],
                   [('blk.1.engram_embd.weight', [512], 27, extra),
                    (TABLE, DIMS, 30, self.table)])
        report = self.realign(tables=['blk.1.engram_embd.weight', TABLE])
        self.assertEqual(report['table_bytes'], len(extra) + len(self.table))
        self.assertEqual(report['shared'], len(extra) + len(self.table) + MB)
        wrote = Reader(str(self.out))
        self.addCleanup(wrote.f.close)
        self.assertEqual(wrote.tensors['blk.1.engram_embd.weight'][2],
                         report['table_offset'] - wrote.data_start)
        self.assertEqual(wrote.tensors[TABLE][2],
                         report['table_offset'] + len(extra) - wrote.data_start)

    def test_install_swaps_the_pack_aside(self):
        report = self.realign(install=True)
        self.assertEqual(report['written'], MB)
        self.assertTrue(Path(str(self.model) + '.pre-refalign').exists())
        self.assertFalse(self.out.exists())
        wrote = Reader(str(self.model))
        self.addCleanup(wrote.f.close)
        self.assertEqual(wrote.tensors[TABLE], (30, DIMS, self.anchor_table - PAGE))

    def test_refuses_a_main_model_that_does_not_fit(self):
        write_gguf(self.model, {'general.architecture': (8, 'qwen4exp')},
                   [(f'blk.{i}.w', bytes([i]) * MB) for i in range(12)],
                   [(TABLE, DIMS, 30, self.table)])
        with self.assertRaisesRegex(ValueError, 'more than the gap'):
            self.realign(verify=False)
        self.assertFalse(self.out.exists())

    def test_refuses_an_anchor_without_a_trailing_table(self):
        self.anchor.write_bytes(self.anchor.read_bytes() + b'junk')
        with self.assertRaisesRegex(ValueError, 'must trail every other tensor'):
            self.realign(verify=False)

    def test_a_differing_table_page_is_written_not_shared(self):
        blob = bytearray(self.model.read_bytes())
        blob[-1] = 7                       # the sibling carries another table
        self.model.write_bytes(bytes(blob))
        report = self.realign()
        self.assertEqual(report['written'], MB + len(self.table))
        self.assertEqual(report['shared'], MB)

    def test_zero_payload_is_punched(self):
        write_gguf(self.model, {'general.architecture': (8, 'qwen4exp')},
                   [('blk.0.w', self.main[0]), ('blk.1.w', bytes(MB))],
                   [(TABLE, DIMS, 30, self.table)])
        report = self.realign()
        self.assertEqual(report['punched'], self.gap + MB)
        self.assertEqual(report['written'], 0)
        raw = self.out.read_bytes()
        self.assertEqual(raw[PAGE + MB:PAGE + 2 * MB], bytes(MB))

    def test_dry_run_plans_without_writing(self):
        report = self.realign(dry=True, verify=False)
        self.assertEqual(report['table_offset'], self.anchor_table)
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
