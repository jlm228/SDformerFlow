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
        if norm == "minmax":
            # Over non-zero entries only -- the voxel is mostly empty.
            lo, hi = torch.min(chunk[chunk != 0]), torch.max(chunk[chunk != 0])
            if not lo == hi:
                chunk[chunk != 0] = (chunk[chunk != 0] - lo) / (hi - lo)
        elif norm == "std":
            mean, stddev = chunk[chunk != 0].mean(), chunk[chunk != 0].std()
            if stddev > 0:
                chunk[chunk != 0] = (chunk[chunk != 0] - mean) / stddev

    spike_th = config["data"]["spike_th"]
    if spike_th is not None:
        # Values exactly equal to the threshold are left untouched, as in the original.
        chunk[chunk > spike_th] = 1
        chunk[chunk < spike_th] = 0

    return chunk


def forward_model(model, chunk, spiking):
    """Call the model with the arguments its family expects and return the final flow.

    SpikingformerFlowNet.forward(x); STTFlowNet.forward(event_voxel, event_cnt).
    """
    pred_list = model(chunk, None) if not spiking else model(chunk)
    return pred_list["flow"][-1]
