"""Evaluate a batch of candidate inputs in one forward pass, matching one-at-a-time results.

`per_sample_norms` makes every batch-coupling normalisation layer run one sample at a time.
`BatchedEval` wraps a batched and a single-sample callable, and on its first batched call checks
the two agree.
"""
import torch
import torch.nn as nn

# Tolerance for the first-call check. Reduction-order noise sits far below this; a coupling
# error sits far above.
VERIFY_ATOL = 1e-4


def _is_batch_coupling(module):
    """True for a BatchNorm that normalises by batch statistics rather than running ones."""
    if not isinstance(module, nn.modules.batchnorm._BatchNorm):
        return False
    return bool(module.training or not module.track_running_stats)


class _PerSampleNorm(nn.Module):
    """Runs `inner` on one sample at a time, then reassembles the batch."""

    def __init__(self, inner, batch_dim):
        super().__init__()
        self.inner = inner
        self.batch_dim = batch_dim

    def forward(self, x):
        if x.dim() <= self.batch_dim or x.shape[self.batch_dim] == 1:
            return self.inner(x)
        parts = [self.inner(s) for s in x.split(1, dim=self.batch_dim)]
        return torch.cat(parts, dim=self.batch_dim)


class NormHandle:
    """The wrappers `per_sample_norms` installed, and `restore` to undo them."""

    def __init__(self, net, replaced):
        self.net = net
        self.replaced = replaced          # [(parent, attr, original)]

    @property
    def n_wrapped(self):
        return len(self.replaced)

    def restore(self):
        for parent, attr, original in self.replaced:
            setattr(parent, attr, original)
        n, self.replaced = len(self.replaced), []
        return n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        return False


def per_sample_norms(net):
    """Replace every batch-coupling normalisation in `net` with a per-sample wrapper.

    The batch axis is 1 for a multi-step layer, which takes (T, N, ...), and 0 otherwise.
    """
    replaced = []
    for name, module in list(net.named_modules()):
        if not _is_batch_coupling(module):
            continue
        parent = net
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        attr = parts[-1]
        batch_dim = 1 if getattr(module, "step_mode", "s") == "m" else 0
        wrapper = _PerSampleNorm(module, batch_dim)
        if attr.isdigit():
            parent[int(attr)] = wrapper
        else:
            setattr(parent, attr, wrapper)
        replaced.append((parent, attr, module))
    return NormHandle(net, replaced)


class BatchedEval:
    """`batch -> stacked output`, checked once against `single` called per sample.

    `single` takes one sample, `batched` a stacked batch. The first call with more than one
    sample runs both and compares; later calls run only `batched`.
    """

    def __init__(self, single, batched, atol=VERIFY_ATOL, verify=True):
        self.single = single
        self.batched = batched
        self.atol = atol
        self.verify = verify
        self.verified = False
        self.max_seen = 0

    def __call__(self, batch):
        k = batch.shape[0]
        self.max_seen = max(self.max_seen, k)
        if k == 1:
            return self.batched(batch)

        out = self.batched(batch)
        if self.verify and not self.verified:
            # The check is a measurement, so it needs no graph of its own.
            with torch.no_grad():
                want = torch.stack([self.single(batch[i]).detach() for i in range(k)])
                gap = float((out.detach() - want).abs().max())
            if gap > self.atol:
                raise RuntimeError(
                    "a batch of %d does not match %d separate single-sample passes: max "
                    "difference %.3e, tolerance %.1e.\n"
                    "Something in this model still couples the samples in a batch. Check that "
                    "per_sample_norms wrapped every BatchNorm (it reported how many), and that "
                    "its batch_dim matches how this model presents them -- spikingjelly's "
                    "multi-step layers take (T, N, ...), single-step ones (N, ...).\n"
                    "Run with --sda-fd-batch 1 to fall back to the correct, slower path."
                    % (k, k, gap, self.atol))
            self.verified = True
        return out
