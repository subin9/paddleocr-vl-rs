#!/usr/bin/env bash
# Cross-stack (§2.6): fetch the FIRST-PARTY GGUF (not a community requant, whose quality would be a
# confound rather than a finding).
#
# The repo ships exactly one LM gguf + one mmproj + a chat template -- there is NO quant ladder, so
# the precision is whatever PaddlePaddle shipped. We do not assume it: the script prints the header's
# general.file_type at the end, and that printed value is what BENCHMARKS.md must quote. A quantized
# run compared against our bf16 port without saying so would be a dishonest comparison.
set -euo pipefail
WS="${WS:-$(cd "$(dirname "$0")/../../.." && pwd)}"   # see llamacpp_build.sh
BUILD="${BUILD:-$WS/llamacpp-build}"
REPO="${REPO:-PaddlePaddle/PaddleOCR-VL-1.5-GGUF}"
HF="${HF:-$WS/.venv/bin/hf}"
PY_BIN="${PY_BIN:-$WS/.venv/bin/python}"

"$HF" download "$REPO" --local-dir "$BUILD/gguf"
ls -la "$BUILD/gguf"

# Read the precision off the header rather than trusting the filename or the model card.
"$PY_BIN" - "$BUILD/gguf" <<'PY'
import pathlib, struct, sys
# Minimal GGUF header reader: magic, version, n_tensors, n_kv, then the kv block. We only need
# general.file_type (u32), so we stop as soon as we have it.
GGUF_TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
FT = {0:'ALL_F32', 1:'MOSTLY_F16', 2:'MOSTLY_Q4_0', 7:'MOSTLY_Q8_0', 32:'MOSTLY_BF16'}

def rd(f, n): return f.read(n)
def u32(f): return struct.unpack('<I', rd(f,4))[0]
def u64(f): return struct.unpack('<Q', rd(f,8))[0]
def s(f):
    return rd(f, u64(f)).decode('utf-8', 'replace')

def val(f, t):
    if t == 8: return s(f)
    if t == 9:
        et, n = u32(f), u64(f)
        return [val(f, et) for _ in range(n)]
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
    raw = rd(f, sz)
    return struct.unpack({0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}[t], raw)[0]

for g in sorted(pathlib.Path(sys.argv[1]).glob('*.gguf')):
    with open(g, 'rb') as f:
        assert rd(f,4) == b'GGUF', g
        ver, n_tensors, n_kv = u32(f), u64(f), u64(f)
        kv = {}
        for _ in range(n_kv):
            k = s(f); t = u32(f); kv[k] = val(f, t)
        ft = kv.get('general.file_type')
        print(f"{g.name}: v{ver} tensors={n_tensors} arch={kv.get('general.architecture')} "
              f"file_type={ft} ({FT.get(ft, '?')})")
PY
