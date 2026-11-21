"""Small mathematical and integration checks for this independently written code."""

import json
from pathlib import Path

import torch
from torch.distributions import Normal

from model import PaperSFA
from train import make_pinwheel


def main():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    results = []
    for x_dim, z_dim, activation in ((2, 1, "tanh"), (784, 64, "softplus")):
        model = PaperSFA(x_dim, z_dim, width=16, depth=2, activation=activation)
        x = torch.randn(8, x_dim)
        t = torch.linspace(0, 1, 8)
        q = model.posterior(x, t)
        assert isinstance(q, Normal)
        assert q.loc.shape == (8, z_dim) and q.scale.shape == (8, z_dim)
        assert torch.isfinite(q.loc).all() and torch.isfinite(q.scale).all()
        assert (q.scale > 0).all()
        assert model.velocity(t, x, q.rsample()).shape == x.shape
        assert model.posterior(x, 0.5).loc.shape == q.loc.shape
        assert model.posterior(x, t[:, None]).loc.shape == q.loc.shape
        loss = model.loss(x)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        for name, network in (("posterior", model.posterior_net), ("field", model.field_net)):
            gradients = [parameter.grad for parameter in network.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            norm = sum(float(g.square().sum()) for g in gradients) ** 0.5
            assert norm > 0
            results.append({"test": "finite_nonzero_gradients", "x_dim": x_dim,
                            "component": name, "gradient_norm": norm})
        torch.manual_seed(91)
        actual = model.loss(x)
        torch.manual_seed(91)
        manual_t = torch.rand(len(x))
        x0 = torch.randn_like(x)
        xt = (1 - manual_t[:, None]) * x0 + manual_t[:, None] * x
        latent = model.posterior(xt, manual_t).rsample()
        expected = (model.velocity(manual_t, xt, latent) - (x - x0)).square().sum(-1).mean()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        results.append({"test": "single_draw_objective_exact_agreement", "x_dim": x_dim,
                        "loss": float(actual.detach())})
        sample = model.sample_given_latent(q.loc.detach(), steps=3)
        assert sample.shape == x.shape and torch.isfinite(sample).all()
        results.append({"test": "shape_positive_scale_finite_sampling", "x_dim": x_dim})

    class ConstantVelocity(PaperSFA):
        def velocity(self, t, x, z):
            return torch.ones_like(x) * 2

    constant_model = ConstantVelocity(2, 1, width=4, depth=1)
    initial = torch.randn(3, 2)
    original = initial.clone()
    generated = constant_model.sample_given_latent(torch.zeros(3, 1), steps=5, x0=initial)
    torch.testing.assert_close(generated, original + 2)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    results.append({"test": "forward_time_ode_and_input_preservation", "passed": True})
    x_a, y_a = make_pinwheel(100, 123)
    x_b, y_b = make_pinwheel(100, 123)
    assert torch.equal(x_a, x_b) and torch.equal(y_a, y_b)
    assert torch.bincount(y_a).tolist() == [20] * 5
    assert torch.isfinite(x_a).all()
    results.append({"test": "seeded_five_balanced_arms", "passed": True})
    report = {"status": "passed", "torch_version": torch.__version__, "results": results}
    Path(__file__).with_name("smoke_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
