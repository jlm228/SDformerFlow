"""BatchedEval over train-mode BatchNorm in both layouts the SNN uses. CPU, no checkpoint.

Run: python -m carla_eval.test_batched_eval
"""
import torch
import torch.nn as nn

from carla_eval.batched_eval import BatchedEval, per_sample_norms

T, C, H, W = 3, 4, 6, 5


class MultiStepBN(nn.BatchNorm2d):
    """Stands in for spikingjelly's layer.BatchNorm2d in step_mode 'm': (T, N, C, H, W)."""

    step_mode = "m"

    def forward(self, x):
        return super().forward(x.flatten(0, 1)).reshape(x.shape)


class ToyNet(nn.Module):
    """(N, T*C, H, W) -> (N, 2, H, W), with a BN of each layout as SDformerFlow has."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, padding=1)
        self.norm_ms = MultiStepBN(C)            # SpikingNormLayer
        self.conv2 = nn.Conv2d(C, C, 3, padding=1)
        self.norm_flat = nn.BatchNorm2d(C)       # SpikingPatchEmbed: BN after flatten(0, 1)
        self.bntt = nn.ModuleList([nn.BatchNorm2d(C) for _ in range(T)])   # BNTT: (N, ...)
        self.head = nn.Conv2d(T * C, 2, 1)

    def forward(self, x):
        n = x.shape[0]
        x = x.reshape(n, T, C, H, W).transpose(0, 1)                        # (T, N, C, H, W)
        x = self.conv1(x.flatten(0, 1)).reshape(T, n, C, H, W)
        x = torch.relu(self.norm_ms(x))
        x = self.norm_flat(self.conv2(x.flatten(0, 1))).reshape(T, n, C, H, W)
        x = torch.stack([self.bntt[t](x[t]) for t in range(T)])
        return self.head(x.transpose(0, 1).reshape(n, T * C, H, W))


def _setup():
    torch.manual_seed(0)
    net = ToyNet().train()           # batch statistics, as the SNN is evaluated
    handle = per_sample_norms(net)
    single = lambda x: net(x.unsqueeze(0))[0]
    return net, handle, BatchedEval(single=single, batched=net, handle=handle)


def test_matches_single_passes():
    net, handle, be = _setup()
    assert handle.n_wrapped == 2 + T
    batch = torch.randn(4, T * C, H, W) * torch.tensor([1., 3., .2, 7.]).view(4, 1, 1, 1)
    out = be(batch)
    want = torch.stack([net(batch[i:i + 1])[0] for i in range(4)])
    assert (out - want).abs().max() < 1e-5
    # A smaller final chunk is checked again, not trusted.
    be(batch[:3])
    assert be.checked == {4, 3}


def test_single_passes_unaffected_after_batch():
    """forward_eval / forward_grad share the network and must still see batch 1."""
    net, handle, be = _setup()
    x = torch.randn(1, T * C, H, W)
    before = net(x)
    be(torch.randn(4, T * C, H, W))
    assert all(w.batch == 1 for w in handle.wrappers)
    assert torch.equal(net(x), before)


def test_unwrapped_batch_is_caught():
    """Without the wrappers the samples couple, and the check says so."""
    torch.manual_seed(0)
    net = ToyNet().train()
    handle = per_sample_norms(net)
    handle.restore()
    be = BatchedEval(single=lambda x: net(x.unsqueeze(0))[0], batched=net, handle=handle)
    try:
        be(torch.randn(4, T * C, H, W) * torch.tensor([1., 3., .2, 7.]).view(4, 1, 1, 1))
    except RuntimeError as e:
        assert "does not match" in str(e)
    else:
        raise AssertionError("coupled batch passed the check")


def test_restore():
    net, handle, _ = _setup()
    assert handle.restore() == 2 + T
    assert isinstance(net.norm_flat, nn.BatchNorm2d) and isinstance(net.bntt[0], nn.BatchNorm2d)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
