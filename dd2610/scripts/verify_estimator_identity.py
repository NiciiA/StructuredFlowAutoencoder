#!/usr/bin/env python3
"""Check proposal SCFM value/gradient identities; NumPy only, no data downloads.

Condition on the *same* x_t, t and target u. The toy reparameterized posterior is
z = mu + exp(rho) * eps, eps ~ N(0,I); v = a*z+b. Thus both vector-field and
posterior parameters are checked. This verifies algebra, not neural-model quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analytic(params, target, k, reduction):
    a, b, mu, rho = params
    sigma2 = np.exp(2 * rho)
    error = a * mu + b - target
    inv_k = 0.0 if k is None else 1.0 / k
    scale = 1.0 / len(a) if reduction == "mean" else 1.0
    value = scale * np.sum(error**2 + inv_k * a**2 * sigma2)
    gradient = scale * np.stack([
        2 * error * mu + 2 * inv_k * a * sigma2,
        2 * error,
        2 * error * a,
        2 * inv_k * a**2 * sigma2,
    ])
    return float(value), gradient


def samples(params, target, eps, kind, reduction):
    a, b, mu, rho = params
    sigma = np.exp(rho)
    z = mu + sigma * eps
    v = a * z + b
    scale = 1.0 / len(a) if reduction == "mean" else 1.0
    if kind == "cross":
        z1, z2 = z[:, 0], z[:, 1]
        e1, e2 = v[:, 0] - target, v[:, 1] - target
        values = np.sum(e1 * e2, axis=-1)
        grads = np.stack([
            e2 * z1 + e1 * z2,
            e1 + e2,
            a * (e1 + e2),
            a * sigma * (e2 * eps[:, 0] + e1 * eps[:, 1]),
        ], axis=1)
    else:
        zbar, epsbar = z.mean(axis=1), eps.mean(axis=1)
        error = v.mean(axis=1) - target
        values = np.sum(error**2, axis=-1)
        grads = np.stack([
            2 * error * zbar, 2 * error,
            2 * error * a, 2 * error * a * sigma * epsbar,
        ], axis=1)
    return scale * values, scale * grads


def moment_report(values, expected):
    n = len(values)
    mean = np.mean(values, axis=0)
    se = np.std(values, axis=0, ddof=1) / np.sqrt(n)
    error = np.abs(mean - expected)
    # 6-sigma limits are a numerical diagnostic, not a statistical test of SFA.
    tolerance = 6 * se + 1e-10
    return {
        "mean": np.asarray(mean).tolist(),
        "analytic": np.asarray(expected).tolist(),
        "standard_error": np.asarray(se).tolist(),
        "max_abs_error": float(np.max(error)),
        "within_6_mc_standard_errors": bool(np.all(error <= tolerance)),
    }


def run(trials=200_000, seed=20261113, reduction="mean"):
    if trials < 10_000:
        raise ValueError("Use at least 10,000 Monte Carlo trials for this check.")
    params = np.array([
        [0.8, -0.6, 0.5],   # a, vector-field scale
        [-0.1, 0.2, 0.05], # b, vector-field intercept
        [0.3, -0.2, 0.1],  # mu, posterior mean
        [-0.1, 0.15, 0.0], # rho, posterior log-standard-deviation
    ], dtype=np.float64)
    target = params[0] * params[2] + params[1] + np.array([0.08, -0.05, 0.04])
    rng = np.random.default_rng(seed)
    result = {"seed": seed, "trials": trials, "reduction": reduction,
              "parameter_order": ["a", "b", "mu", "rho"], "estimators": {}}
    # Separate conditionally iid epsilon values inside every trial are essential.
    for name, k in (("k1", 1), ("k4", 4), ("cross", None)):
        eps = rng.normal(size=(trials, 2 if k is None else k, 3))
        values, gradients = samples(params, target, eps, name, reduction)
        truth, grad_truth = analytic(params, target, k, reduction)
        # Finite differences verify the analytic derivative independently.
        grad_fd = np.zeros_like(params)
        delta = 1e-5
        for index in np.ndindex(params.shape):
            plus, minus = params.copy(), params.copy()
            plus[index] += delta
            minus[index] -= delta
            grad_fd[index] = (analytic(plus, target, k, reduction)[0]
                              - analytic(minus, target, k, reduction)[0]) / (2 * delta)
        np.testing.assert_allclose(grad_truth, grad_fd, atol=1e-8, rtol=1e-6)
        value_report = moment_report(values, truth)
        gradient_report = moment_report(gradients, grad_truth)
        assert value_report["within_6_mc_standard_errors"], name + " value identity failed"
        assert gradient_report["within_6_mc_standard_errors"], name + " gradient identity failed"
        result["estimators"][name] = {
            "value": value_report, "gradient": gradient_report,
            "gradient_variance_trace": float(np.var(gradients.reshape(trials, -1), axis=0, ddof=1).sum()),
            "negative_fraction": float(np.mean(values < 0)),
        }
    # Guard against two tempting but incorrect implementations.
    eps4 = rng.normal(size=(trials, 4, 3))
    a, b, mu, rho = params
    errors = a * (mu + np.exp(rho) * eps4) + b - target
    factor = 1 / 3 if reduction == "mean" else 1
    wrong_average_of_squares = factor * np.sum(errors**2, axis=-1).mean(axis=1)
    wrong_report = moment_report(wrong_average_of_squares, analytic(params, target, 1, reduction)[0])
    assert wrong_report["within_6_mc_standard_errors"]
    duplicate = np.repeat(eps4[:, :1], 2, axis=1)
    duplicate_cross, _ = samples(params, target, duplicate, "cross", reduction)
    k1, _ = samples(params, target, eps4[:, :1], "k1", reduction)
    np.testing.assert_allclose(duplicate_cross, k1, atol=0, rtol=0)
    result["wrong_average_of_four_squared_losses_matches_k1"] = wrong_report
    result["reusing_same_draw_in_cross_exactly_equals_k1"] = True
    result["passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.trials, args.seed, args.reduction)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {"passed": result["passed"], "trials": args.trials, "reduction": args.reduction}
    for name, record in result["estimators"].items():
        compact[name] = {"expected": record["value"]["analytic"],
                         "observed": record["value"]["mean"],
                         "negative_fraction": record["negative_fraction"],
                         "gradient_variance_trace": record["gradient_variance_trace"]}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
