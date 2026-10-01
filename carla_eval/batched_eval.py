"""Evaluate a batch of candidate inputs in one forward pass, matching one-at-a-time results.

`per_sample_norms` makes every batch-coupling normalisation layer run one sample at a time.
`BatchedEval` checks, once per batch size, that a batched pass reproduces separate single-sample
passes, and raises if it does not.

The SNN presents its BatchNorms with the batch in two different places, so each wrapper works
out its own layout from the module it wraps rather than sharing one:

  multi-step spikingjelly layer   (T, N, ...)   e.g. SpikingNormLayer's layer.BatchNorm2d
  anything else                   (T*N, ...)    e.g. nn.BatchNorm2d after x.flatten(0, 1) of a
                                                (T, N, ...) tensor, so sample n is the rows
                                                n, n+N, n+2N, ...; with T = 1 this is plain (N, ...)
"""
import torch
import torch.nn as nn

# Tolerance for the check. Reduction-order noise sits far below this; a coupling error sits far
# above.
VERIFY_ATOL = 1e-4


def _is_batch_coupling(module):
    """True for a BatchNorm that normalises by batch statistics rather than running ones."""
    if not isinstance(module, nn.modules.batchnorm._BatchNorm):
        return False
    return bool(module.training or not module.track_running_stats)


def _is_multi_step(module):
    """True for a spikingjelly layer in multi-step mode, which takes (T, N, ...)."""
    return getattr(module, "step_mode", "s") == "m"


class _PerSampleNorm(nn.Module):
    """Runs `inner` once per sample of a batch of `batch`, then reassembles."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.batch = 1

    def forward(self, x):
        k = self.batch
        if k <= 1:
            return self.inner(x)
        # Read at call time: set_step_mode may run after the wrappers are installed.
        if _is_multi_step(self.inner):
            if x.dim() < 2 or x.shape[1] != k:
                raise RuntimeError("multi-step norm expected (T, %d, ...), got %s"
                                   % (k, tuple(x.shape)))
            return torch.cat([self.inner(p) for p in x.split(1, dim=1)], dim=1)
        if x.shape[0] % k:
            raise RuntimeError("norm expected (T*%d, ...) rows, got %s" % (k, tuple(x.shape)))
        # Time-major fold: sample n is every k-th row from n. Stacking on dim 1 and flattening
        # puts row t*k + n back where it came from.
        parts = [self.inner(x[n::k]) for n in range(k)]
        return torch.stack(parts, dim=1).flatten(0, 1)


class NormHandle:
    """The wrappers `per_sample_norms` installed, and `restore` to undo them."""

    def __init__(self, net, replaced, wrappers):
        self.net = net
        self.replaced = replaced          # [(parent, attr, original)]
        self.wrappers = wrappers

    @property
    def n_wrapped(self):
        return len(self.replaced)

    def configure(self, batch):
        for w in self.wrappers:
            w.batch = int(batch)

    def restore(self):
        for parent, attr, original in self.replaced:
            if attr.isdigit():
                parent[int(attr)] = original
            else:
                setattr(parent, attr, original)
        n = len(self.replaced)
        self.replaced, self.wrappers = [], []
        return n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        return False


def per_sample_norms(net):
    """Replace every batch-coupling normalisation in `net` with a per-sample wrapper."""
    replaced, wrappers = [], []
    for name, module in list(net.named_modules()):
        if not _is_batch_coupling(module):
            continue
        parent = net
        parts = name.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        attr = parts[-1]
        wrapper = _PerSampleNorm(module)
        if attr.isdigit():
            parent[int(attr)] = wrapper
        else:
            setattr(parent, attr, wrapper)
        replaced.append((parent, attr, module))
        wrappers.append(wrapper)
    return NormHandle(net, replaced, wrappers)


class BatchedEval:
    """`batch -> stacked output`, checked against `single` called per sample.

    `single` takes one sample, `batched` a stacked batch, `handle` the NormHandle to tell the
    batch size. Each batch size is checked once on first use: SDA's last chunk of a round is
    usually smaller than the rest, and a size-dependent bug would otherwise go unseen.
    """

    def __init__(self, single, batched, handle, atol=VERIFY_ATOL):
        self.single = single
        self.batched = batched
        self.handle = handle
        self.atol = atol
        self.max_seen = 0
        self.checked = set()

    def _run_batched(self, batch):
        # Every other caller of the network (forward_eval, forward_grad) passes one sample, so
        # the wrappers are told the batch size only for the length of this pass.
        self.handle.configure(batch.shape[0])
        try:
            return self.batched(batch)
        finally:
            self.handle.configure(1)

    def _check(self, batch):
        k = batch.shape[0]
        with torch.no_grad():
            want = torch.stack([self.single(batch[i]).detach() for i in range(k)])
            gap = float((self._run_batched(batch).detach() - want).abs().max())
        if gap > self.atol:
            raise RuntimeError(
                "a batch of %d does not match %d separate single-sample passes: max "
                "difference %.3e, tolerance %.1e. %d norm layer(s) are wrapped, so something "
                "else couples the samples or reaches a norm in an unexpected layout.\n"
                "Run with --sda-fd-batch 1 to fall back to the correct, slower path."
                % (k, k, gap, self.atol, self.handle.n_wrapped))
        print("batched candidates: batch %d matches single passes (max diff %.2e)" % (k, gap))
        self.checked.add(k)

    def __call__(self, batch):
        k = batch.shape[0]
        self.max_seen = max(self.max_seen, k)
        if k > 1 and k not in self.checked:
            self._check(batch)
        return self._run_batched(batch)
