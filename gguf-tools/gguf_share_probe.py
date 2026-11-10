#!/usr/bin/env python3
"""Dense physical-block probe: for every 1 MiB window of the realigned
artifact, compare the physical block address of its pages with the base
file's pages at the SAME file offset (F_LOG2PHYS_EXT). Physical match at the
same offset means the pair shares that block: it costs disk once."""
import fcntl, os, struct, sys

F_LOG2PHYS_EXT = 65
PAGE = os.sysconf('SC_PAGESIZE')

a_path, b_path = sys.argv[1], sys.argv[2]
a, b = open(a_path, 'rb'), open(b_path, 'rb')
size_a = os.fstat(a.fileno()).st_size
size_b = os.fstat(b.fileno()).st_size

def phys(fp, off):
    return struct.unpack('=Iqq', fcntl.fcntl(
        fp.fileno(), F_LOG2PHYS_EXT, struct.pack('=Iqq', 0, PAGE, off)))[2]

shared = unique = beyond = 0
pos = 0
STRIDE = 64  # probe one page per 1 MiB
while pos + PAGE <= size_a:
    pa = phys(a, pos)
    if pos + PAGE <= size_b:
        pb = phys(b, pos)
        if pa == pb and pa != 0:
            shared += 1
        else:
            unique += 1
    else:
        beyond += 1
    pos += STRIDE * PAGE

scale = STRIDE * PAGE
print(f'{a_path}: {size_a/2**30:.2f} GiB logical vs {b_path}')
print(f'  pages shared with base at same offset: {shared*scale/2**30:.2f} GiB')
print(f'  pages unique to the artifact:          {unique*scale/2**30:.2f} GiB')
print(f'  beyond base size (tail):               {beyond*scale/2**30:.2f} GiB')
print(f'pair physical footprint ~= base ({size_b/2**30:.2f} GiB) + unique '
      f'({unique*scale/2**30:.2f} GiB) = {(size_b + unique*scale)/2**30:.2f} GiB')
print(f'vs two standalone packs {(size_a + size_b)/2**30:.2f} GiB -> '
      f'{(size_a - unique*scale)/2**30:.2f} GiB saved by CoW')
