# DD2610 · Structured Flow Autoencoders

[Open the team notebook in Colab](https://colab.research.google.com/github/NiciiA/StructuredFlowAutoencoder/blob/dd2610-colab-english/dd2610/notebooks/DD2610_Colab.ipynb)

This directory contains the group's experimental code, the Colab notebook, and local validation records. It follows the course proposal and feedback and uses the [authors' original implementation at commit 05e8907](https://github.com/edenx/StructuredFlowAutoencoder/tree/05e89071f6bae502eedfd3464fad4e8f75f506c1). Running this project fetches that revision into `upstream/StructuredFlowAutoencoder` inside the project directory. The surrounding team repository has its own runners and settings.

## Start in Colab

1. Open the notebook above after the branch is published. For an offline copy of the *project files*, upload `notebooks/DD2610_Colab.ipynb` to Colab and select the ZIP source in section 1. Upload the **project** archive (`DD2610_code.zip`), which contains the scripts and requirements. Both routes need internet access to install dependencies and fetch the authors' code.
2. Select a GPU runtime, execute section 1, and set `PARTICIPANT` to your own identifier (`nico` or `dwei`). Mount your own Drive for persistent checkpoints. Work from the same package version or team commit and record it with your results.
3. Run the checks in section 2 and the short Pinwheel pilot in section 3. Measure an epoch on the assigned hardware before committing to full runs. GPU availability and session duration vary. [Colab runtime FAQ](https://research.google.com/colaboratory/faq.html)
4. Keep `split_seed=2026` fixed. Start with training seed 0; add matched seeds 1 and 2 as the budget allows. Checkpoints are saved after each epoch. After a disconnect, reinitialize the project, mount the same Drive, and use the same run arguments with `--resume`.
5. Use the [project plan](PROJECT_PLAN.md) to choose experiments and the [reproduction audit](REPRODUCTION_AUDIT.md) to decide which numbers can be compared with published results.

## Experimental sequence

| Stage | Run | What to retain |
|---|---|---|
| 0 | Pin the source and data splits, record the environment, run mathematical and interface checks. | Verification output and one pilot record per member. |
| 1 | Run the authors' models on Pinwheel, MNIST including mixture, real Kang RNA, and pendulum. | Training and validation records, evaluation files, and documented failures. |
| 2 | Compare FM, VAE, and frozen-AE LatentFM on the main tasks where applicable. | Matched data and evaluation settings, parameter counts, and compute budgets. |
| 3 | Compare the frozen paper-guided implementation with the author-derived continuous SFA. | Shared evaluation plus architecture, posterior, loss scale, preprocessing, and solver differences. |
| 4 | Compare K1, K4, and cross on Pinwheel; extend to MNIST if feasible. | Matched seeds, training curves, stability, latent-use diagnostics, and time. |
| 5 | Analyse latent shuffling, ODE steps, latent dimension, and EMNIST OOD. | Quality, diversity, clustering, probe results, and cost under each protocol. |
| 6 | Freeze choices, evaluate on the held-out test set, and compare member runs. | Per-seed results, source audit, failures, and individual contributions. Check Canvas for deadlines. |

One proposed budget is five author configurations at one seed, seven Pinwheel methods at three seeds, and four MNIST methods at three seeds: 38 training runs. LatentFM has two training stages. Independent full repeats by both members would roughly double the work. The proposal's 25–75 accelerator hours are an estimate. Prioritize coverage of the four requested author experiments when allocating repeats.

## Entry points

| File | Purpose |
|---|---|
| [Colab notebook](notebooks/DD2610_Colab.ipynb) | Setup, one-off runs, RNA preparation, campaign preview, final tests, OOD, and member comparisons. |
| [bootstrap.py](scripts/bootstrap.py) | Fetch and verify the pinned author's source. |
| [run_official.py](scripts/run_official.py) | Adapted runner for the authors' architecture and losses, including mixture and pendulum. |
| [run_experiment.py](scripts/run_experiment.py) | Common training and evaluation for Gaussian SFA K1/K4/cross, FM, VAE, frozen-AE LatentFM, and paper-only. |
| [benchmark_models.py](scripts/benchmark_models.py) | Baseline models and the three posterior-sampling loss estimators. |
| [verify_estimator_identity.py](scripts/verify_estimator_identity.py) | Analytic expectations and reparameterization gradients, Monte Carlo checks, and incorrect-estimator counterexamples. |
| [diagnose_estimators.py](scripts/diagnose_estimators.py) | Compare estimator gradient variance on the same input batch at one checkpoint. |
| [metrics.py](scripts/metrics.py) | Raw 2D Euclidean W1, paired SSIM, cosine Vendi, NMI/ARI, linear probes, and softNMI variants. |
| [download_kang.py](scripts/download_kang.py), [prepare_official_data.py](scripts/prepare_official_data.py) | Obtain the real data and record preprocessing and split identities. |
| [evaluate_ood.py](scripts/evaluate_ood.py) | Evaluate a frozen MNIST encoder on EMNIST digits and letters A–J. |
| [run_suite.py](scripts/run_suite.py) | Preview or execute selected jobs sequentially and save logs. |
| [compare_runs.py](scripts/compare_runs.py) | Report per-run results, summarize independent seeds, and pair compatible member runs. |
| [paper_only/](paper_only/) | Frozen implementation, standalone training script, exact prompts, design choices, and pre-freeze checks. |
| [test_protocol.py](tests/test_protocol.py) | Known-answer metric and loss checks, data split checks, and author-model backward passes. |

## Command line

Run these commands **from this directory**, which contains `scripts/` and `requirements-colab.txt`. For a local install, first install the PyTorch and torchvision versions for your machine. Colab keeps its preinstalled PyTorch, torchvision, and CUDA combination. Actual versions are recorded per run.

```bash
python -m pip install -r requirements-colab.txt
python -m scripts.bootstrap
python -m scripts.verify_freeze
python -m unittest discover -s tests -v
python -m scripts.verify_estimator_identity --output results/identity.json
```

A Pinwheel experiment using the full proposal split:

```bash
python -m scripts.run_experiment --dataset pinwheel --method sfa_k1 --profile full --participant nico --seed 0 --epochs 100 --eval-samples 500 --ode-steps 100 --output results/benchmarks --tag main_v1
```

This example uses 500 validation points to match the notebook's section 6 default. Use 1,000 for a separately identified evaluation if desired; compare like sample counts. Change `--method` to `sfa_k4`, `sfa_cross`, `fm`, `vae`, `latent_fm`, or `paper_only`. Select MNIST with `--dataset mnist`. `--z-dim` changes latent dimension. Epochs are **per stage**; LatentFM trains two stages. A starting budget of 100 Pinwheel or 50 MNIST epochs belongs to this comparison protocol.

Reuse the original training command with `--resume` to continue an interrupted run. Use `--eval-only --evaluate-test` after validation choices are fixed; it reloads the validation-selected `best.pt`. Keep training arguments unchanged when resuming and choose a new tag for a new training budget or architecture.

```bash
python -m scripts.run_official --dataset mnist_mixture --profile pilot --participant nico --seed 0 --epochs 3 --output-root results/official --run-id nico_mixture_pilot_v1
python -m scripts.run_suite --participant nico --phase core --seeds 0,1,2 --output results
python -m scripts.diagnose_estimators --run-dir results/benchmarks/nico/pinwheel_sfa_k1_full_s0_main_v1 --repeats 32
python -m scripts.compare_runs results --output comparison
```

The suite previews jobs by default. Select names printed in the preview with `--only` and add `--execute` when ready. Resume existing jobs separately from new jobs.

## Data and protocol

**Pinwheel and MNIST.** Full Pinwheel uses 15,000/2,000/2,000 points and retains the authors' factor-of-ten coordinate scale. MNIST uses a fixed 54,000/6,000 split of the official training pool and reserves the official 10,000 test images. A fixed shuffled test order supports a 1,000-image comparison. Labels are reserved for post-training diagnostics. The data manifest records actual counts and hashes, including pilot subsets.

**Kang RNA.** The [CPA Kang tutorial](https://cpa-tools.readthedocs.io/en/latest/tutorials/Kang.html) supplies a roughly 545 MB file with 13,576 cells, 5,000 upstream-selected genes, and eight observed cell types. The paper reports seven. The verified input SHA-256 is `f04d4d31cc5ff56588e34cb305cdf52b3eddaad273192857a95dd7f5bea1c437`. The two `orig.ident` values identify STIM/CTRL conditions. A donor-level leakage check needs verified patient identifiers, and reselecting genes from this file cannot recover train-only selection from all genes.

```bash
python -m pip install -r requirements-rna.txt
python -m scripts.download_kang --output data/kang_normalized_hvg.h5ad
python -m scripts.prepare_official_data rna --input data/kang_normalized_hvg.h5ad --output data/kang_cellholdout.npz --input-state lognorm --keep-features
python -m scripts.run_official --dataset hvg --rna-file data/kang_cellholdout.npz --profile pilot --epochs 3 --participant nico
```

This main protocol holds out random cells (80/10/10). A separate preparation with `--split-key split_B` reproduces the released partition (11,576 train, 1,226 valid, 774 OOD). The OOD set captures a B-cell condition shift; report it separately from ordinary multiclass clustering. The input matrix has already been normalized and log transformed.

**Pendulum.** Independent simulation seeds separate complete trajectories before windowing. Evaluate posterior reconstruction of observed sequences using pixel/frame RMSE, raw latent coordinates, and separately a mapping to physical states fitted on training trajectories.

## Implementation and evaluation differences

The bootstrap preserves the author checkout. The runner records shape, split, device, gradient clipping, optimizer, integration, and checkpoint adaptations in `adaptations.json`. Profile `full` selects the project data/model preset. Exact agreement with a paper table also requires matching the training and scoring procedures. Use `--config configs/official_pinwheel_gaussian.json` for the appendix-style Gaussian setting. Explicit scaled JSON presets for MNIST, RNA, and pendulum give starting points after the pilot.

The paper-only implementation was generated in a separate context with the extracted paper as its research source and instructions prohibiting browsing and reading author code. Its prompt also prescribed interfaces and some modeling choices. The [provenance record](paper_only/PROVENANCE.md) describes those choices, failures, and input restrictions. Six hashes verify stability after freezing. These records do not establish tool-level isolation or rule out pretraining exposure. The standalone defaults (independent Pinwheel construction, MNIST 55k/5k/10k and unit-scaled pixels) differ from the shared comparison protocol. The shared runner retains the frozen model's coordinate-sum loss and uses its Softplus activation for MNIST; the author-derived comparison averages over coordinates.

The included baselines cover FM, VAE, and frozen-AE LatentFM. The original LatentFM joint second stage, VampVAE, GMVAE/Mixture-SVAE, GLD-SVAE, and full likelihood tables are outside the implemented scope. Treat training losses as optimization diagnostics.

- The SFA estimators share architecture, training budget, data, and `alpha`. K4 averages velocity predictions **before** squaring. Validation checkpoint selection uses a fixed-noise K16 estimate, which retains a variance term scaled by 1/16.
- The independent two-draw cross estimator can have negative minibatch losses. Keep instability and cost records. Use the common-checkpoint diagnostic to isolate posterior-sampling gradient variance.
- Reuse observation noise when swapping latents. Pair SSIM with structural probes. Report the number of samples and the cosine PSD kernel for pixel- or vector-space Vendi scores.
- The implemented Pinwheel W1 uses optimal matching with raw Euclidean costs. The released author's batch-dependent scaling and the team's separate sliced-Wasserstein implementation use different metrics.
- SFA marginal samples draw latents from the aggregate training posterior. Set the official runner's `marginal_latent` to `normal` to investigate a standard-normal assumption explicitly.
- The code separates train-fitted KMeans from clustering fitted on the held-out latents. The logistic probe and scaler are fitted on training representations with fixed `C=1`. Three softNMI formulas have separate names. Most latent diagnostics use posterior means; evaluation records a fixed posterior sample for a CNF without an analytic mean.

## Comparing both members' runs

Start with the same seed, data, code, training configuration, evaluation count, and solver. Exchange the configuration, environment, data manifest, and evaluation JSON before interpreting differences. One possible division is continuous Pinwheel/MNIST and estimator analysis for Dwei, mixture/RNA/pendulum for Nico, followed by at least one mutual repeat.

`compare_runs.py` writes `per_run.csv`, `seed_summary.csv`, and `member_comparison.csv`. Only matching protocols are paired. A same-seed repeat by a second member does not increase the independent seed count. A single seed has no confidence interval; intervals based on three seeds are descriptive. Keep all training curves, hardware, parameter counts, and failed runs.

For delivery, retain the source repository, actual contributions, presentation, and peer review. Check Canvas for the current format and deadlines. The proposal filename uses Group_007 while its text and feedback use Group 5; verify the submission identifier and all members' contributions.

The [validation record](VALIDATION.md) lists verified and unverified behavior. The full project ZIP includes the input paper extraction. The team branch distributes the code, notebook, and provenance; `scripts.verify_freeze` also checks the paper hash when the extraction is present. A separately distributed frozen archive, `DD2610_paper_only_frozen.zip`, contains the isolated implementation and its original input text.
