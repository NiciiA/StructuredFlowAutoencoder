"""Standalone paper-only SFA training for a synthetic Pinwheel or local MNIST.

Example:
    python paper_only/train.py --dataset pinwheel --epochs 100

MNIST is optional, read only from an already prepared local torchvision dataset:
    python paper_only/train.py --dataset mnist --data-root /path/to/mnist

No network access or automatic downloads are performed. Dataset construction,
optimization defaults, and output diagnostics are documented in PROVENANCE.md.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .model import PaperSFA
else:
    from model import PaperSFA


ROOT = Path(__file__).resolve().parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pinwheel(count, seed):
    """Independent five-arm construction; exact source dataset is unspecified.

    Balanced arm labels are shuffled. Radius is uniform on [0.35, 1.8],
    with Gaussian radial noise of sd 0.025. Each arm bends by 0.7*radians
    per radial unit, with angular noise of sd 0.055 radians. No source
    generator, external implementation, or dataset labels enter training.
    """
    rng = np.random.default_rng(seed)
    labels = np.arange(count, dtype=np.int64) % 5
    rng.shuffle(labels)
    radius = rng.uniform(0.35, 1.8, size=count)
    angle = 2 * math.pi * labels / 5 + 0.7 * radius
    angle += rng.normal(0, 0.055, size=count)
    radius += rng.normal(0, 0.025, size=count)
    xy = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return torch.from_numpy(xy.astype(np.float32)), torch.from_numpy(labels)


def load_splits(args):
    if args.dataset == "pinwheel":
        counts = (args.train_size or 10000, args.val_size or 2000, args.test_size or 2000)
        splits = [make_pinwheel(n, args.seed + 11 * (i + 1)) for i, n in enumerate(counts)]
        description = {"name": "independently_constructed_five_arm_pinwheel",
                       "split_seeds": [args.seed + 11 * (i + 1) for i in range(3)],
                       "counts": dict(zip(("train", "validation", "test"), counts)),
                       "normalization": "none"}
        return splits, description

    # Import only when requested; installed package source is not inspected.
    try:
        from torchvision.datasets import MNIST
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("MNIST requires a working optional torchvision installation") from exc
    train_set = MNIST(str(args.data_root), train=True, download=False)
    test_set = MNIST(str(args.data_root), train=False, download=False)
    n_train = args.train_size if args.train_size is not None else 55000
    n_val = args.val_size if args.val_size is not None else 5000
    n_test = args.test_size if args.test_size is not None else 10000
    if n_train + n_val > len(train_set) or n_test > len(test_set):
        raise ValueError("Requested MNIST split sizes exceed the official dataset sizes")
    generator = torch.Generator().manual_seed(args.seed + 11)
    order = torch.randperm(len(train_set), generator=generator)
    test_order = torch.randperm(len(test_set), generator=generator)[:n_test]

    def take(dataset, indices):
        x = dataset.data[indices].reshape(len(indices), -1).float() / 255.0
        return x, dataset.targets[indices].long()

    splits = [take(train_set, order[:n_train]),
              take(train_set, order[n_train:n_train + n_val]),
              take(test_set, test_order)]
    description = {"name": "MNIST", "split_seed": args.seed + 11,
                   "counts": {"train": n_train, "validation": n_val, "test": n_test},
                   "normalization": "uint8 / 255, flattened to 784; no dequantization",
                   "official_test_used_for_training_or_selection": False}
    return splits, description


@torch.no_grad()
def evaluate(model, x, batch_size, draws, seed, device):
    """Fixed random draws make validation comparisons less noisy across epochs."""
    was_training = model.training
    model.eval()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    total = 0.0
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        for _ in range(draws):
            for batch in x.split(batch_size):
                value = model.loss(batch.to(device))
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite evaluation loss")
                total += float(value) * len(batch)
    model.train(was_training)
    return total / (len(x) * draws)


def save_checkpoint(path, model, optimizer, epoch, score, config, shuffle_generator):
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": config["model"],
                "run_config": config,
                "epoch": epoch,
                "validation_loss": score,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "shuffle_rng_state": shuffle_generator.get_state()}, path)


@torch.no_grad()
def save_diagnostics(model, splits, output, count, steps, seed, device):
    """Posterior predictives and samples using the training aggregated posterior."""
    if count == 0:
        return
    model.eval()
    seed_everything(seed)
    train_x = splits[0][0]
    test_x, test_y = splits[2]
    count = min(count, len(train_x), len(test_x))
    observations = test_x[:count].to(device)
    posterior = model.posterior(observations, 1.0)
    posterior_z = posterior.rsample()
    predictive = model.sample_given_latent(posterior_z, steps=steps)
    anchors = train_x[torch.randperm(len(train_x))[:count]].to(device)
    aggregate_z = model.posterior(anchors, 1.0).rsample()
    aggregate_samples = model.sample_given_latent(aggregate_z, steps=steps)
    tensors = {"observations": observations, "labels": test_y[:count],
               "posterior_mean": posterior.loc, "posterior_std": posterior.scale,
               "posterior_z": posterior_z, "posterior_predictive": predictive,
               "aggregated_training_z": aggregate_z,
               "aggregated_training_samples": aggregate_samples}
    if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise FloatingPointError("Nonfinite diagnostic samples")
    np.savez_compressed(output / "samples.npz",
                        **{name: tensor.cpu().numpy() for name, tensor in tensors.items()})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pinwheel", "mnist"), default="pinwheel")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-draws", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--z-dim", type=int)
    parser.add_argument("--activation", choices=("tanh", "softplus"))
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--ode-steps", type=int, default=100)
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "eval_draws", "width", "depth", "threads", "ode_steps"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("train_size", "val_size", "test_size", "z_dim"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.sample_count < 0:
        parser.error("--sample-count must be nonnegative")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    splits, dataset_info = load_splits(args)
    model_config = {"x_dim": splits[0][0].shape[1],
                    "z_dim": args.z_dim or (1 if args.dataset == "pinwheel" else 64),
                    "width": args.width, "depth": args.depth,
                    "activation": args.activation or ("tanh" if args.dataset == "pinwheel" else "softplus")}
    model = PaperSFA(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output = args.output or ROOT / "runs" / f"{args.dataset}-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    config = {"arguments": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
              "model": model_config, "dataset": dataset_info,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "objective": "single-draw SCFM; sum coordinates, mean examples",
              "time_direction": "0=noise, 1=data"}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shuffle_generator = torch.Generator().manual_seed(args.seed + 101)
    loader = DataLoader(TensorDataset(splits[0][0]), batch_size=args.batch_size,
                        shuffle=True, generator=shuffle_generator)
    validation_seed = args.seed + 1001
    initial = evaluate(model, splits[1][0], args.batch_size, args.eval_draws,
                       validation_seed, device)
    best, best_epoch = initial, 0
    save_checkpoint(output / "checkpoint_best.pt", model, optimizer, 0, initial, config, shuffle_generator)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "validation_loss", "seconds"))
        writer.writeheader()
        writer.writerow({"epoch": 0, "train_loss": "", "validation_loss": initial, "seconds": 0.0})
        handle.flush()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            train_total = 0.0
            for (batch,) in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(batch.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                    raise FloatingPointError("Nonfinite model gradient")
                optimizer.step()
                train_total += float(loss.detach()) * len(batch)
            score = evaluate(model, splits[1][0], args.batch_size, args.eval_draws,
                             validation_seed, device)
            row = {"epoch": epoch, "train_loss": train_total / len(splits[0][0]),
                   "validation_loss": score, "seconds": time.perf_counter() - started}
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if score < best:
                best, best_epoch = score, epoch
                save_checkpoint(output / "checkpoint_best.pt", model, optimizer, epoch, score,
                                config, shuffle_generator)
            save_checkpoint(output / "checkpoint_last.pt", model, optimizer, epoch, score,
                            config, shuffle_generator)
    checkpoint = torch.load(output / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss = evaluate(model, splits[2][0], args.batch_size, args.eval_draws,
                         args.seed + 2001, device)
    save_diagnostics(model, splits, output, args.sample_count, args.ode_steps,
                     args.seed + 3001, device)
    result = {"initial_validation_loss": initial, "best_validation_loss": best,
              "best_epoch": best_epoch, "test_loss_at_best_validation": test_loss,
              "epochs": args.epochs, "output": str(output.resolve()),
              "claim": "Training diagnostic only; not a reproduction of published metrics"}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
