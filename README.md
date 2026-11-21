# SFA: Stochastic Flow Analysis

[Open the DD2610 notebook in Google Colab](https://colab.research.google.com/github/NiciiA/StructuredFlowAutoencoder/blob/dd2610-colab-english/DD2610_Colab.ipynb) · [View the notebook](DD2610_Colab.ipynb) · [Experiment guide](dd2610/README.md)

Flow matching framework for generative modeling with mixture latent variables across multiple data modalities.

## Project Structure

```
SFA/
├── src_sfa/                # Main source code
│   ├── main.py             # Entry point (training & evaluation)
│   ├── sfa_llk.py          # Unified vector field (LLK) — supports mlp, cnn, cnn_film, resnet, unet
│   ├── sfa.py              # Continuous priors (GaussianPrior, CNF, FlowMatchingLoss)
│   ├── sfa_discrete.py     # Mixture components (CatNF_fixed, GaussianMixtureComponent, FlowMatchingLossMixture)
│   ├── sfa_lds.py          # Sequence models (LatentDynamicalSystem, fullCNF, FlowMatchingLossSeq)
│   ├── runner_continuous.py    # Runners: PinWheel, MNIST, HVG (continuous latent)
│   ├── runner_mixture.py       # Runners: PinWheel, MNIST (mixture latent)
│   ├── runner_pendulum.py      # Runner: Pendulum (sequential latent)
│   ├── callbacks.py        # Lightning callbacks for plotting
│   ├── Scheduler.py        # Learning rate schedulers (warmup)
│   └── utils.py            # Shared utilities (PCA, t-SNE, metrics)
├── configs/                # YAML config files per experiment
├── models/                 # Neural network architectures (MLP, UNet)
└── dataloader/             # Dataset loaders (pinwheel, MNIST, HVG, pendulum, etc.)
```

## Runners

| Runner | Config | Data | Latent |
|--------|--------|------|--------|
| `pinwheel` | `pinwheel.yml` | 2D synthetic pinwheel | Continuous z |
| `mnist` | `mnist.yml` | MNIST images | Continuous z |
| `hvg` | `hvg.yml` | Highly variable genes | Continuous z |
| `pinwheel_mixture` | `pinwheel_discrete.yml` | 2D synthetic pinwheel | Mixture (pi, z) |
| `mnist_mixture` | `mnist.yml` | MNIST images | Mixture (pi, z) |
| `pendulum` | `pendulum.yml` | Pendulum image sequences | Sequential z(t) |

## Training

```bash
# MNIST with mixture latent
python src_sfa/main.py --config mnist.yml --runner mnist_mixture --ni

# MNIST with continuous latent
python src_sfa/main.py --config mnist.yml --runner mnist --ni

# Pinwheel with mixture latent
python src_sfa/main.py --config pinwheel_discrete.yml --runner pinwheel_mixture --ni

# Pinwheel with continuous latent
python src_sfa/main.py --config pinwheel.yml --runner pinwheel --ni

# Pendulum sequences
python src_sfa/main.py --config pendulum.yml --runner pendulum --ni
```

`--ni` suppresses interactive prompts (auto-overwrites existing logs). Logs and checkpoints are saved to `exp/logs/<runner_name>/`.

### Resume training

```bash
python src_sfa/main.py --config mnist.yml --runner mnist_mixture --resume_training --ni
```

### Custom log folder

```bash
python src_sfa/main.py --config mnist.yml --runner mnist_mixture --doc my_experiment --ni
```

## Evaluation

```bash
# Test / evaluation
python src_sfa/main.py --config mnist.yml --runner mnist_mixture --test --ni

# Generate figures
python src_sfa/main.py --config mnist.yml --runner mnist_mixture --figure --ni
```

## Key Config Options

In `configs/*.yml`:

- `model.type`: LLK architecture — `"cnn"` or `"mlp"`
- `flow.z_dim`: Latent dimension
- `flow.pi_dim`: Number of mixture components (mixture runners)
- `flow.beta`: Softmax temperature for mixture assignment
- `training.n_epochs`: Number of training epochs
- `training.alpha`: Regularization weight
- `optim.lr`: Learning rate

## Setup

```bash
conda env create -f environment.yml
conda activate my_env
```

## Device compatibility

Runners select CUDA, then Apple Silicon MPS, then CPU. CUDA uses
`DDPStrategy(find_unused_parameters=True)`; MPS uses one device, `strategy="auto"`
and `precision="32-true"`. Model computations use float32 on MPS and retain
float64 in CPU/CUDA experiments. Imports do not change PyTorch's default dtype.
Floating batches are converted before device transfer; labels, indices and masks
retain their integer/boolean dtypes. CPU conversion for plotting is explicit.

ODE integration retains the existing solvers and tolerances. MPS uses float32
solver time calculations. Differentiable MPS ODE solves use direct autograd to
avoid a native PyTorch 2.2 MPS assertion in the flattened adjoint state. This
requires more memory than adjoint differentiation; CPU/CUDA retain the adjoint
method. CPU fallback is not enabled.

Run the focused regressions with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m unittest discover -s tests -v
```

The suite checks CPU float32/float64 and, when available, real MPS float32,
including distributions, batch transfer, one Lightning optimizer step, sequence
models, ODE likelihoods and analytic ODE gradients. MPS tests skip when unavailable.
The MNIST mixture posterior now sizes its input from `flow.pi_dim`; checkpoints
from the previous incorrectly sized posterior layer will need regeneration.

## Baseline experiments

The baselines use the existing Pinwheel generator, MNIST train/test datasets,
stratified MNIST samplers, preprocessing, Lightning training and device selection.
Their YAML files copy the dataset counts, batch sizes and optimizer settings from
`pinwheel_discrete.yml` and `mnist.yml`. Use the same `--seed` for SFA and each
baseline. The MLPs are intentionally smaller than SFA; parameter counts are
reported rather than claiming capacity equivalence.

```bash
python src_sfa/main.py --config pinwheel_vae.yml --runner pinwheel_vae --seed 123 --ni
python src_sfa/main.py --config mnist_vae.yml --runner mnist_vae --seed 123 --ni
python src_sfa/main.py --config pinwheel_latentfm.yml --runner pinwheel_latentfm --seed 123 --ni
python src_sfa/main.py --config mnist_latentfm.yml --runner mnist_latentfm --seed 123 --ni
python src_sfa/main.py --config pinwheel_fm.yml --runner pinwheel_fm --seed 123 --ni

python src_sfa/main.py --config hvg_vae.yml --runner hvg_vae --seed 123 --ni
python src_sfa/main.py --config hvg_latentfm.yml --runner hvg_latentfm --seed 123 --ni
python src_sfa/main.py --config hvg_fm.yml --runner hvg_fm --seed 123 --ni
python src_sfa/main.py --config pendulum_vae.yml --runner pendulum_vae --seed 123 --ni
python src_sfa/main.py --config pendulum_latentfm.yml --runner pendulum_latentfm --seed 123 --ni
python src_sfa/main.py --config pendulum_fm.yml --runner pendulum_fm --seed 123 --ni
python src_sfa/main.py --config mnist_fm.yml --runner mnist_fm --seed 123 --ni
```
