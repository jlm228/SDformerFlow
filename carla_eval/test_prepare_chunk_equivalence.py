"""`prepare_chunk_differentiable` must be `prepare_chunk`, exactly.

    python carla_eval/test_prepare_chunk_equivalence.py

The attack path cannot use `prepare_chunk`: it assigns through boolean indexing, which mutates
its argument and breaks autograd. So there are two implementations of the input transform, and
the danger is that the attack optimises one function while the clean evaluation scores
another, and the difference shows up as a robustness result.

This pins them together on the no-grad path over both model configurations. No checkpoint, no
capture, no GPU: the transform depends only on the config and the tensor.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.input_prep import (nonzero_support, prepare_chunk,          # noqa: E402
                              prepare_chunk_differentiable)

B, BINS, H, W = 1, 10, 24, 32


def _config(spiking, norm="minmax", encoding="voxel", polarity=True, spike_th=None):
    """The fields prepare_chunk reads, shaped as the YAML configs supply them."""
    return {"model": {"encoding": encoding, "norm_input": norm},
            "loader": {"polarity": polarity},
            "swin_transformer": {"use_arc": [True, "PatchEmbed"]},
            "data": {"spike_th": spike_th}}


def _voxel(seed=0, occupancy=0.10):
    """A signed voxel with SDformerFlow's measured ~10% occupancy and both polarities."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(B, BINS, H, W)
    n = int(occupancy * x.numel())
    idx = torch.randperm(x.numel(), generator=g)[:n]
    x.view(-1)[idx] = torch.randn(n, generator=g) * 4.0
    return x


def _check(name, chunk, config, spiking, tol=0.0):
    # prepare_chunk MUTATES its argument, so each side gets its own copy.
    want = prepare_chunk(chunk.clone(), config, spiking)
    got = prepare_chunk_differentiable(chunk.clone(), config, spiking)
    assert want.shape == got.shape, "%s: shape %s vs %s" % (name, want.shape, got.shape)
    delta = float((want - got).abs().max())
    assert delta <= tol, "%s: max |difference| = %.3e (tol %.1e)" % (name, delta, tol)
    print("  %-38s identical (max delta %.1e, shape %s)" % (name, delta, tuple(got.shape)))


def test_snn_minmax():
    """The SDformerFlow SNN path: voxel, polarity split, minmax over non-zeros."""
    _check("SNN voxel + split + minmax", _voxel(1), _config(True), spiking=True)


def test_ann_no_norm():
    """STTFlowNet: same voxel, no normalisation, and the split condition inverted."""
    _check("ANN voxel, no norm", _voxel(2), _config(False), spiking=False)


def test_snn_std_norm():
    _check("SNN voxel + std norm", _voxel(3), _config(True, norm="std"), spiking=True)


def test_no_split():
    _check("SNN voxel, polarity off", _voxel(4), _config(True, polarity=False), spiking=True)


def test_degenerate_inputs():
    """An all-zero window is real -- carla_to_voxel writes one whenever a window had <2 events."""
    _check("all-zero voxel", torch.zeros(B, BINS, H, W), _config(True), spiking=True)
    flat = torch.zeros(B, BINS, H, W)
    flat[0, 0, :4, :4] = 2.5                        # every non-zero equal -> lo == hi
    _check("constant non-zeros (lo == hi)", flat, _config(True), spiking=True)


def test_gradient_flows():
    """The point of the exercise: d(output)/d(input) must exist and be non-trivial."""
    config = _config(True)
    x = _voxel(5).requires_grad_(True)
    out = prepare_chunk_differentiable(x, config, spiking=True)
    out.sum().backward()
    assert x.grad is not None, "no gradient reached the input"
    assert float(x.grad.abs().max()) > 0, "gradient is identically zero"

    # And prepare_chunk cannot do this -- that is why the second implementation exists.
    y = _voxel(5).requires_grad_(True)
    try:
        prepare_chunk(y, config, spiking=True)
    except RuntimeError:
        print("  gradient flows through the differentiable form; the original raises")
        return
    print("  gradient flows through the differentiable form")


def test_fixed_support_stops_collateral_rescaling():
    """A perturbation must not move the normalisation constants.

    Lifting one cell off zero adds it to `chunk != 0`, which can move lo/hi, which rescales
    EVERY value in the sample, voxels the attacker never touched. That is the discontinuity
    `nonzero_support` exists to remove, and the quantity to measure is therefore COLLATERAL
    change: movement at the untouched cells, not at the perturbed one.
    """
    config = _config(True)
    clean = _voxel(6)

    adv = clean.clone()
    cell = tuple((clean == 0).nonzero()[0].tolist())
    adv[cell] = 1e-3                                  # one cell, far below the data scale

    nz = nonzero_support(clean, config, spiking=True)
    touched = nonzero_support(adv, config, spiking=True) != nz    # cells the split actually moved

    def collateral(use_nz):
        a = prepare_chunk_differentiable(clean.clone(), config, True, nz=use_nz)
        b = prepare_chunk_differentiable(adv.clone(), config, True, nz=use_nz)
        return float((b - a)[~touched].abs().max())

    free = collateral(None)
    pinned = collateral(nz)
    assert pinned == 0.0,         "pinned support still leaked %.3e onto untouched cells" % pinned
    assert free > 0.0,         "recomputed support did not rescale anything -- the test is not exercising the case"
    print("  fixed support: 1e-3 on one cell rescales untouched cells by %.2e when the "
          "support floats, 0 when pinned" % free)


def test_spike_th_refuses():
    try:
        prepare_chunk_differentiable(_voxel(7), _config(True, spike_th=0.5), spiking=True)
    except NotImplementedError:
        print("  a non-null spike_th is refused rather than silently zeroing the gradient")
    else:
        raise AssertionError("spike_th was silently accepted")


def main():
    print("prepare_chunk equivalence")
    test_snn_minmax()
    test_ann_no_norm()
    test_snn_std_norm()
    test_no_split()
    test_degenerate_inputs()
    test_gradient_flows()
    test_fixed_support_stops_collateral_rescaling()
    test_spike_th_refuses()
    print("\nall equivalence tests passed")


if __name__ == "__main__":
    main()
