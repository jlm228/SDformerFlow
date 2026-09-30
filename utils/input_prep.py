"""The transform between what DSECDatasetLite yields and what the models consume.

Extracted from eval_DSEC_flow_SNN.py's inner loop so every evaluation applies the same steps.
Two behaviours look like typos and are not:

  * `norm_input` applies to the SNN only; the ANN trains and evaluates on raw voxel values.
  * The polarity split is conditioned on `loader.polarity` in opposite senses -- the SNN splits
    when it is set, the ANN when it is not. That is what the trained weights expect on each
    side.
"""
import torch


def prepare_chunk(chunk, config, spiking):
    """Dataset chunk -> model input tensor.

    spiking: True for the SNN (config["model"]["spiking_neuron"] is not None), False for the
    ANN.
    """
    encoding = config["model"]["encoding"]
    polarity = config["loader"]["polarity"]

    if encoding == "cnt":
        if config["swin_transformer"]["use_arc"][1] == "PatchEmbed3D":  # B,D,P,H,W -> B,P,D,H,W
            chunk = torch.transpose(chunk, 1, 2)
        elif polarity:
            chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))  # [B,40,H,W]

    elif encoding == "voxel":  # B, C, H, W -> B, C, P, H, W when split
        split = polarity if spiking else not polarity
        if split:
            neg = torch.nn.functional.relu(-chunk)
            pos = torch.nn.functional.relu(chunk)
            chunk = torch.cat((torch.unsqueeze(pos, dim=2), torch.unsqueeze(neg, dim=2)), dim=2)

    else:
        raise AttributeError("Config error: event encoding not supported: %r" % encoding)

    if spiking:
        norm = config["model"]["norm_input"]
        # An entirely empty window is not hypothetical: carla_to_voxel.py writes a zero voxel
        # whenever a window carried fewer than two events, and torch.min raises on an empty
        # reduction. Nothing to normalise, so leave it alone.
        mask = chunk != 0
        if mask.any():
            # Per sample, so batching cannot change any one sample's scaling.
            if norm == "minmax":
                lo, hi, any_ = _per_sample_bounds(chunk, mask)
                scale = torch.where((hi > lo) & any_, hi - lo, torch.ones_like(hi))
                shift = torch.where((hi > lo) & any_, lo, torch.zeros_like(lo))
                chunk = torch.where(mask, (chunk - shift) / scale, chunk)
            elif norm == "std":
                mean, stddev, n = _per_sample_moments(chunk, mask)
                ok = (stddev > 0) & (n > 1)
                chunk = torch.where(
                    mask & ok, (chunk - mean) / torch.where(ok, stddev, torch.ones_like(stddev)),
                    chunk)

    spike_th = config["data"]["spike_th"]
    if spike_th is not None:
        # Values exactly equal to the threshold are left untouched, as in the original.
        chunk[chunk > spike_th] = 1
        chunk[chunk < spike_th] = 0

    return chunk


def _per_sample_bounds(chunk, mask):
    """(lo, hi, any) per sample: min and max over the masked cells, keeping dims for broadcast.

    Reduced over every dim but 0, so a batch of k candidates cannot share one lo/hi.
    """
    dims = tuple(range(1, chunk.dim()))
    big = torch.finfo(chunk.dtype).max
    lo = torch.where(mask, chunk, torch.full_like(chunk, big)).amin(dim=dims, keepdim=True)
    hi = torch.where(mask, chunk, torch.full_like(chunk, -big)).amax(dim=dims, keepdim=True)
    return lo, hi, mask.any(dim=dims, keepdim=True)


def _per_sample_moments(chunk, mask):
    """(mean, std, n) per sample over the masked cells, std with Bessel's correction."""
    dims = tuple(range(1, chunk.dim()))
    kept = torch.where(mask, chunk, torch.zeros_like(chunk))
    n = mask.sum(dim=dims, keepdim=True).to(chunk.dtype)
    mean = kept.sum(dim=dims, keepdim=True) / n.clamp(min=1)
    var = torch.where(mask, (chunk - mean) ** 2, torch.zeros_like(chunk)).sum(
        dim=dims, keepdim=True) / (n - 1).clamp(min=1)
    return mean, var.sqrt(), n


def _reshape(chunk, config, spiking):
    """The encoding / polarity-split half of prepare_chunk. Already out of place."""
    encoding = config["model"]["encoding"]
    polarity = config["loader"]["polarity"]

    if encoding == "cnt":
        if config["swin_transformer"]["use_arc"][1] == "PatchEmbed3D":
            return torch.transpose(chunk, 1, 2)
        if polarity:
            return chunk.reshape([chunk.shape[0], -1] + list(chunk.shape[3:]))
        return chunk

    if encoding == "voxel":
        split = polarity if spiking else not polarity
        if split:
            neg = torch.nn.functional.relu(-chunk)
            pos = torch.nn.functional.relu(chunk)
            return torch.cat((torch.unsqueeze(pos, dim=2),
                              torch.unsqueeze(neg, dim=2)), dim=2)
        return chunk

    raise AttributeError("Config error: event encoding not supported: %r" % encoding)


def nonzero_support(clean_chunk, config, spiking):
    """The set minmax normalises over, taken from the CLEAN tensor.

    Returned in POST-RESHAPE space, because that is where the original computes it: the
    polarity split runs before the normalisation, so a mask built on the pre-split tensor has
    the wrong shape and the wrong cells.

    Why fix it at all: `prepare_chunk` recomputes `chunk != 0` from whatever it is given, so
    under a perturbation the set MOVES -- a cell the attack lifts off zero joins it and shifts
    lo/hi for the whole sample. That is a discontinuous, non-differentiable dependence, and it
    lets an arbitrarily small perturbation rescale the entire input. Holding the set at the
    clean sample's is the honest reading: the attacker perturbs the physical voxel, and which
    cells the sample's normalisation was computed over is a property of the recording.
    """
    return _reshape(clean_chunk, config, spiking) != 0


def prepare_chunk_differentiable(chunk, config, spiking, nz=None):
    """`prepare_chunk` with no in-place writes, so autograd can reach the input.

    The original assigns through boolean indexing (`chunk[chunk != 0] = ...`), which mutates
    its argument and breaks the graph -- on a leaf tensor it raises outright. Every step here
    is the same arithmetic expressed out of place, and
    `carla_eval/test_prepare_chunk_equivalence.py` pins the two together on the no-grad path.
    That test is what stops an attack silently optimising a different transform than the one
    the clean evaluation scored.

    `nz` is the normalisation set from `nonzero_support`, in post-reshape space. Defaults to
    this tensor's own non-zero cells, which reproduces the original exactly when nothing has
    been perturbed.
    """
    chunk = _reshape(chunk, config, spiking)

    if spiking:
        norm = config["model"]["norm_input"]
        mask = (chunk != 0) if nz is None else nz
        if mask.shape != chunk.shape:
            raise ValueError(
                "nz has shape %s but the reshaped chunk is %s -- build it with "
                "nonzero_support(clean_chunk, config, spiking), which reshapes first"
                % (tuple(mask.shape), tuple(chunk.shape)))
        if mask.any():
            # Per sample, so batching cannot change any one sample's scaling.
            if norm == "minmax":
                lo, hi, any_ = _per_sample_bounds(chunk, mask)
                scale = torch.where((hi > lo) & any_, hi - lo, torch.ones_like(hi))
                shift = torch.where((hi > lo) & any_, lo, torch.zeros_like(lo))
                chunk = torch.where(mask, (chunk - shift) / scale, chunk)
            elif norm == "std":
                mean, stddev, n = _per_sample_moments(chunk, mask)
                ok = (stddev > 0) & (n > 1)
                chunk = torch.where(
                    mask & ok, (chunk - mean) / torch.where(ok, stddev, torch.ones_like(stddev)),
                    chunk)

    spike_th = config["data"]["spike_th"]
    if spike_th is not None:
        # A hard threshold has zero gradient almost everywhere, so an attack through it would
        # report a converging loss while the input gradient carried no information. Every
        # config in this repo sets spike_th: Null; refuse rather than mislead if that changes.
        raise NotImplementedError(
            "spike_th=%r binarises the input, which is not differentiable. Attacking through "
            "it needs a straight-through estimator, which is deliberately not implemented "
            "here." % (spike_th,))

    return chunk


def forward_model(model, chunk, spiking):
    """Call the model with the arguments its family expects and return the final flow.

    SpikingformerFlowNet.forward(x); STTFlowNet.forward(event_voxel, event_cnt).
    """
    pred_list = model(chunk, None) if not spiking else model(chunk)
    return pred_list["flow"][-1]
