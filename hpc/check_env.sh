#!/bin/bash
# Verify the environment before committing to a 38 GB download and days of queue time.
#
#   bash hpc/check_env.sh          # login node: everything except CUDA
#   srun --qos=bbgpu --gres=gpu:a100:1 --time=10:0 --pty bash hpc/check_env.sh   # incl. GPU
#
# Checks imports, the two version pins that silently break checkpoint loading, the imageio
# FreeImage plugin (needed by preprocessing, and undownloadable from a compute node), the split
# lists, and -- if a GPU is visible -- the spikingjelly CuPy backend.

set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
source hpc/env.sh

fail=0
note() { printf '  %-14s %s\n' "$1" "$2"; }

echo "python: $(python -V 2>&1)  ($(command -v python))"
echo

echo "imports:"
python - <<'PY'
import importlib, sys
mods = ["numpy","pandas","yaml","h5py","hdf5plugin","imageio","tqdm","numba","tables",
        "torch","torchvision","timm","einops","mlflow","spikingjelly","cupy"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod,'__version__','ok')}")
    except Exception as e:
        print(f"  {m:<14} MISSING ({type(e).__name__})")
        bad.append(m)
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && fail=1

echo
echo "version pins that matter:"
python - <<'PY'
import sys
ok = True
try:
    import torch
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 6):
        print(f"  torch {torch.__version__}: TOO NEW -- torch.load defaults to weights_only=True")
        print("    from 2.6, which breaks utils.load_model (it unpickles a whole nn.Module).")
        print("    Pin torch<2.6, or pass weights_only=False at both torch.load sites.")
        ok = False
    else:
        print(f"  torch {torch.__version__}: ok (<2.6)")
except Exception as e:
    print(f"  torch: could not check ({e})"); ok = False
try:
    import mlflow
    if int(mlflow.__version__.split(".")[0]) >= 2:
        print(f"  mlflow {mlflow.__version__}: check artifact layout -- utils.py hardcodes the")
        print("    1.x path model/data/model.pth. Pin mlflow<2.0 if load_model cannot find it.")
    else:
        print(f"  mlflow {mlflow.__version__}: ok (<2.0)")
except Exception as e:
    print(f"  mlflow: could not check ({e})"); ok = False
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && fail=1

echo
echo "imageio FreeImage plugin (ground-truth PNG decoding):"
python - <<'PY'
import sys, imageio
try:
    imageio.plugins.freeimage.download()
    print("  available")
except Exception as e:
    print(f"  UNAVAILABLE ({type(e).__name__}: {e})")
    print("  Run this ON THE LOGIN NODE: python -c \"import imageio; imageio.plugins.freeimage.download()\"")
    print("  Compute nodes have no outbound network, so preprocessing would fail at _create_flow_maps.")
    sys.exit(1)
PY
[ $? -ne 0 ] && fail=1

echo
echo "split lists:"
D=data/Datasets/DSEC/saved_flow_data/sequence_lists
for f in train_split_doubleseq:6000 valid_split_doubleseq:2152 train_split_seq:6000 valid_split_seq:2152; do
    name="${f%%:*}"; want="${f##*:}"
    if [ -f "${D}/${name}.csv" ]; then
        got=$(wc -l < "${D}/${name}.csv")
        if [ "${got}" -eq "${want}" ]; then note "${name}" "${got} rows"
        else note "${name}" "${got} rows -- EXPECTED ${want}"; fail=1; fi
    else
        note "${name}" "MISSING -- run: bash hpc/setup_splits.sh"; fail=1
    fi
done

echo
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "spikingjelly CuPy backend:"
    python - <<'PY'
import sys, torch
try:
    from spikingjelly.activation_based import functional, neuron
    n = neuron.LIFNode(step_mode='m').cuda()
    functional.set_backend(n, 'cupy', neuron.LIFNode)
    n(torch.rand(4, 2, 8, device='cuda'))
    print("  compiles and runs")
except Exception as e:
    print(f"  FAILED ({type(e).__name__}: {e})")
    print("  The SNN cannot train without this. Usually a cupy/CUDA version mismatch.")
    sys.exit(1)
PY
    [ $? -ne 0 ] && fail=1
else
    echo "GPU: none visible (expected on a login node)."
    echo "     Re-run under srun to verify CUDA and the spikingjelly CuPy backend."
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "OK -- environment looks ready."
else
    echo "PROBLEMS FOUND (see above). Fix these before downloading or submitting jobs."
    exit 1
fi
