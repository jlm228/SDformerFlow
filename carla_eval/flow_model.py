"""SDformerFlow and STTFlowNet behind a common flow-model interface.

    window_ms: float
    input_from_events(ev, t0, t1) -> Tensor
    forward(x) -> Tensor   # (B, 2, H, W), pixels per window
    reset_state() -> None

One adapter serves both models. They differ only in configuration, and those differences are
handled by utils/input_prep.py. These models are stateful, so reset_state must be called
between samples or results depend on evaluation order.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from spikingjelly.activation_based import functional, neuron  # noqa: E402

from models.STSwinNet.STSwinNet import STTFlowNet, STTFlowNet_4en  # noqa: E402
from models.STSwinNet_SNN.Spiking_STSwinNet import (  # noqa: E402
    MS_SpikingformerFlowNet, MS_SpikingformerFlowNet_en4, SpikingformerFlowNet)
from models.STSwinNet_SNN.Spiking_submodules import *  # noqa: E402,F401,F403
from utils.input_prep import (forward_model, nonzero_support,  # noqa: E402
                              prepare_chunk, prepare_chunk_differentiable)
from utils.utils import load_model, print_parameters  # noqa: E402

from carla_eval.carla_to_voxel import events_to_voxel  # noqa: E402

MODELS = {
    "STTFlowNet": STTFlowNet,
    "STTFlowNet_4en": STTFlowNet_4en,
    "SpikingformerFlowNet": SpikingformerFlowNet,
    "MS_SpikingformerFlowNet": MS_SpikingformerFlowNet,
    "MS_SpikingformerFlowNet_en4": MS_SpikingformerFlowNet_en4,
}

NEURON_TYPES = {
    "if": getattr(neuron, "IFNode"),
    "lif": getattr(neuron, "LIFNode"),
    "plif": getattr(neuron, "ParametricLIFNode"),
    "glif": GatedLIFNode,  # noqa: F405
    "psn": PSN,  # noqa: F405
    "SLTTlif": SLTTLIFNode,  # noqa: F405
}


class SwinFlowAdapter:
    window_ms: float = 100.0

    def __init__(self, config, runid, device, verbose=True):
        self.config = config
        self.device = device
        self.spiking = config["model"].get("spiking_neuron") is not None
        self.num_bins = config["data"]["num_frames"]
        self.num_chunks = config["data"]["num_chunks"]

        name = config["model"]["name"]
        if name not in MODELS:
            raise SystemExit("unknown model.name %r (known: %s)"
                             % (name, ", ".join(sorted(MODELS))))

        # Must be set before construction: the swin position-bias tables are sized from it.
        crop = config["loader"].get("crop")
        config["swin_transformer"]["input_size"] = list(crop) if crop else \
            list(config["loader"]["resolution"])

        if config["swin_transformer"]["use_arc"][0]:
            model = MODELS[name](config["model"].copy(), config["swin_transformer"].copy())
        else:
            model = MODELS[name](config["model"].copy())

        model.to(device)
        model.init_weights()

        remap = config["loader"].get("remap")
        # test=True strips the "module." prefix a DataParallel checkpoint carries.
        model = load_model(runid, model, device, remap=remap, test=True)

        if self.spiking:
            functional.reset_net(model)
            functional.set_step_mode(model, config["data"]["step_mode"])
            ntype = config["model"]["spiking_neuron"]["neuron_type"]
            if ntype not in NEURON_TYPES:
                raise NotImplementedError("neuron type not implemented: %s" % ntype)
            self.neuron_type = NEURON_TYPES[ntype]
            if device.type != "cpu":
                functional.set_backend(model, "cupy", self.neuron_type)

        # BatchNorm convention from eval_DSEC_flow_SNN.py: the SNN's spike_norm layers are
        # trained at batch_size 1 and validated in train() mode, normalising each sample by its
        # own statistics rather than by running estimates.
        if self.spiking and config["loader"]["batch_size"] == 1:
            model.train()
        else:
            model.eval()

        self.net = model
        if verbose:
            print_parameters(model)

    def input_from_events(self, ev, t0: int, t1: int) -> torch.Tensor:
        """Raw (x, y, t, pol) events spanning [t0, t1) us -> (1, C, H, W) model input.

        t1 - t0 must cover num_chunks windows; for num_chunks 2 the first is the preceding
        window. Used for perturbing events directly; the batch evaluation reads the tensors
        carla_to_voxel.py writes instead.
        """
        h, w = self.config["loader"]["resolution"]
        window_us = int(round(self.window_ms * 1000))
        expected = self.num_chunks * window_us
        if t1 - t0 != expected:
            raise ValueError("this model needs %d us of context (num_chunks=%d), got %d"
                             % (expected, self.num_chunks, t1 - t0))

        chunks = []
        for c in range(self.num_chunks):
            a, b = t0 + c * window_us, t0 + (c + 1) * window_us
            sel = (ev["t"] >= a) & (ev["t"] < b)
            chunks.append(events_to_voxel(ev["x"][sel], ev["y"][sel], ev["t"][sel],
                                          ev["pol"][sel], self.num_bins, h, w))
        return torch.cat(chunks, dim=0).unsqueeze(0)

    def prepare(self, chunk: torch.Tensor) -> torch.Tensor:
        return prepare_chunk(chunk.to(device=self.device, dtype=torch.float32),
                             self.config, self.spiking)

    def forward(self, x: torch.Tensor, prepared: bool = False) -> torch.Tensor:
        """x -> predicted flow (B, 2, H, W) in pixels over one window."""
        if not prepared:
            x = self.prepare(x)
        with torch.no_grad():
            return forward_model(self.net, x.to(self.device), self.spiking)

    def support(self, clean_chunk: torch.Tensor):
        """The minmax normalisation set, taken from the CLEAN chunk. See nonzero_support.

        None for the ANN, which does not normalise, so the caller can pass it through
        unconditionally.
        """
        if not self.spiking:
            return None
        return nonzero_support(clean_chunk.to(device=self.device, dtype=torch.float32),
                               self.config, self.spiking)

    def prepare_grad(self, chunk: torch.Tensor, nz=None) -> torch.Tensor:
        """`prepare` without in-place writes, so autograd reaches the input."""
        return prepare_chunk_differentiable(
            chunk.to(device=self.device, dtype=torch.float32), self.config, self.spiking, nz=nz)

    def forward_grad(self, x: torch.Tensor, nz=None, prepared: bool = False) -> torch.Tensor:
        """The same forward with gradients enabled, for white-box attacks.

        Three differences from `forward`, all deliberate:

        * no `torch.no_grad()`, so d(flow)/d(input) exists;
        * the input transform is the out-of-place `prepare_chunk_differentiable`, pinned
          against `prepare_chunk` by test_prepare_chunk_equivalence.py;
        * state is reset HERE. `forward` leaves that to the evaluation loop, but an attack
          calls this tens of times on one window and every call must start from the same
          state, or the gradient is taken through a network that has been drifting since
          iteration zero.

        `nz` should come from `support(clean_chunk)` and stay fixed for the whole
        optimisation, or the perturbation can rescale the sample through the normalisation.

        BatchNorm mode is NOT touched: the SNN is left in train() at batch size 1, normalising
        each sample by its own statistics, exactly as the clean evaluation does. Attacking a
        differently-configured network than the one being scored would invalidate the result.
        """
        self.reset_state()
        if not prepared:
            x = self.prepare_grad(x, nz=nz)
        return forward_model(self.net, x.to(self.device), self.spiking)

    def reset_state(self) -> None:
        functional.reset_net(self.net)


def numpy_events_to_dict(events):
    """Structured event array -> the dict input_from_events expects."""
    return {k: np.asarray(events[k]) for k in ("x", "y", "t", "pol")}
