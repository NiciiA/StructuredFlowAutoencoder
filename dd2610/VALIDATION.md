# Validation record and scope of the original delivery

These checks were completed on 14 November 2026 (Europe/Stockholm). This record describes what had been tested when the original package was prepared. Subsequent student experiments have their own run records.

## Completed checks

| Check | Result | Scope |
|---|---|---|
| Estimator identities and gradients | Passed | K1, K4, and cross, with 200,000 Monte Carlo trials for each of the sum and mean reductions |
| Protocol unit tests | 5 passed | Known answers for W1, Vendi, SSIM, and softNMI; split handling; order of averaging in K4; gradients through both cross branches |
| Backpropagation through author models | Passed | Continuous and mixture Pinwheel, continuous and mixture MNIST, 5,000-dimensional HVG input, and pendulum; the CNF posterior was also checked |
| Short training and evaluation with author models | Passed | Pinwheel, real MNIST, mixture MNIST, real Kang RNA, and pendulum |
| Short benchmark training | 11 runs passed | Seven Pinwheel methods and four methods on real MNIST (SFA, paper-only, VAE, and LatentFM), using small samples and one epoch |
| Full MNIST split | Passed | 54,000 training, 6,000 validation, and 10,000 test images; training and validation index sets were disjoint |
| Kang download and preprocessing | Passed | Read 13,576 cells by 5,000 genes and found 8 classes; prepared both the source `split_B` and a random cell holdout |
| Two-stage checkpoint recovery | Passed | After an intentional interruption in the second stage of LatentFM, all final parameters matched an uninterrupted run bit for bit on the local CPU |
| Comparison across participant IDs with the same seed | Passed | Two local short runs with IDs `dwei` and `nico` and the same settings produced zero differences on 17 paired metrics. This exercised the comparison script. |
| Gradient diagnostic at a shared checkpoint | Passed | Calculated gradient variance for K1, K4, and cross at the same parameter values |
| Original notebook | Structure and syntax checks passed | 27 cells, including 15 Python cells; the original delivery checks did not execute the notebook in a signed-in Colab browser |
| Paper-only freeze | Hashes of 6 files matched | The generation record and original prompt are in `paper_only/PROVENANCE.md`; the model source remained frozen during comparisons |

The local environment used an Apple Silicon Mac (CPU), Python 3.9, PyTorch 2.8.0, and torchvision 0.23.0. The full local package list is in `verification/local_environment.txt`. Colab retains its installed torch, torchvision, and CUDA combination, installs the extra requirements, and records the versions used for each run.

## Issues found and fixed during validation

1. The author LLK and mixture posterior had input-size mismatches. The wrapper now supplies the correct dimensions while retaining the original upstream files.
2. The released data roles, repeated pendulum trajectories, and reshape order required changes. The project uses fixed disjoint splits and an explicit dimension permutation.
3. Resuming the second LatentFM stage needed two fixes. The optimizer now restores the correct parameter group, and stale gradients on the frozen AE no longer affect clipping. The interruption test compared every final parameter with the uninterrupted run.
4. The data loader and bridge sampler now have separate random-number streams, and their states are both checkpointed. Changing K can no longer change subsequent batch order or bridge noise through a shared random stream.
5. A local Python cache directory was not writable during the first syntax check. Compilation succeeded using a writable temporary cache.

One small RNA run produced a scikit-learn floating-point warning during matrix multiplication. A separate check of the saved latents found a normal input range, finite model coefficients, and identical float32 and float64 predictions. Recalculation did not reproduce the warning. The small-batch metrics checked the data path only.

## Open items at the time of delivery

- Full Colab GPU training, long-run convergence, formal statistics across seeds, and measured peak GPU memory and time budgets.
- The EMNIST download and complete OOD experiment. The evaluation entry point handles separate digit and letter label ranges and corrects image orientation.
- Reproduction of all published values, exact hyperparameters, every paper baseline, and the likelihood tables. The implemented LatentFM freezes the AE during flow training.
- A strict donor holdout and training-only HVG selection from all genes. The downloaded file has no verified donor IDs and contains genes already selected upstream.
- Real MNIST and GPU runs with the standalone paper-only runner. A short real-MNIST run *was* performed with the frozen model inside the shared benchmark runner.
- Cross-hardware bitwise reproducibility. The local CPU comparison held the environment fixed. Keep the environment metadata and per-run results when comparing different GPUs, drivers, or dependencies.

## Evidence

See the files in `verification/`, including `estimator_identity_mean.json`, `estimator_identity_sum.json`, `benchmark_smoke.json`, `official_smoke.json`, `resume_check.json`, `member_comparison_check.json`, and `mnist_full_split_check.json`.

The original local working directory retained checkpoints, logs, sample arrays, and plots from the short runs. The distribution contains code, notes, and compact verification records. Datasets, full experiment results, and the local virtual environment are excluded. Start formal runs from the version being reported. Earlier smoke runs can have different source hashes after subsequent fixes and should be identified separately in comparisons.
