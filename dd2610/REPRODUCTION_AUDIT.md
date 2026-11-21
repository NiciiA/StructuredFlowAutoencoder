# Reproduction audit of the paper, released code, and course project

The initial audit on 13 November 2026 used the supplied 22-page paper and a local checkout of [the authors' pinned revision, 05e89071f6bae502eedfd3464fad4e8f75f506c1](https://github.com/edenx/StructuredFlowAutoencoder/tree/05e89071f6bae502eedfd3464fad4e8f75f506c1). Documents were research sources. Section 8 records subsequent implementation checks.

This pre-experiment audit describes observable repository behavior and its possible impact. The authors' original run logs were unavailable, so the released code alone does not establish which settings produced each published result.

## 1. Naming the three types of result

| Result | Evidence | Use in this project |
|---|---|---|
| Run of the released implementation | Pin the revision and retain the configuration and failure logs. | Keep the upstream source and evidence from execution alongside the README commands. |
| Controlled or scaled reproduction with author models | Use author models and losses directly. Document each change to compatibility, data splits, or storage. | Main Colab track. Identify changes to training budget, precision, architecture, or ODE integration. |
| Paper-only LLM implementation | Use a fresh context with the paper as its only research source. Disable search and browsing, exclude the author repository, this audit, and code from other tracks, and freeze the result before comparison. | Develop it in a separate session and directory. Code generated in the shared project conversation does not meet that input boundary. See Section 8 and the generation record for the limits of the implementation supplied here. |

The paper does not uniquely determine all training details. Several appendix passages specify training until convergence without complete optimizer settings, seeds, fixed splits, training steps, or metric protocols. Running the released configuration therefore establishes a repository result; matching it to a particular published number requires additional evidence.

## 2. Continuous-latent SCFM conventions

Equation 6 on p. 5 averages over the latent distribution before squaring:

\[
L_\infty=\mathbb E_{x,t,x_t}\left\|\mathbb E_{z\sim q_\phi(z\mid x_t,t)}v_\theta(x_t,t,z)-u_t\right\|_2^2.
\]

Section 3.1 on p. 6 suggests using one posterior draw in practice. At fixed \(x_t,t,u_t\), draw \(z_1,\ldots,z_K\) independently from the same posterior and let \(v_k=v_\theta(x_t,t,z_k)\). Conditionally,

\[
\mathbb E\left\|\frac{1}{K}\sum_{k=1}^{K}v_k-u_t\right\|_2^2
=\|\mathbb E v-u_t\|_2^2+\frac{\operatorname{tr}\operatorname{Cov}(v)}{K}.
\]

To see this, write \(v_k=m+\delta_k\) with \(m=\mathbb E v\). Conditional independence and zero means remove the cross terms, so \(\mathbb E\|\operatorname{mean}(\delta)\|^2=\operatorname{tr}\operatorname{Cov}(v)/K\). Then average over the outer data, time, and noise draws. The elementwise MSE in the author code divides both terms on the right by the observation dimension \(d\).

For two independent draws, the residual cross product satisfies

\[
L_\times=(v_1-u_t)^\top(v_2-u_t),\qquad \mathbb E L_\times=\|\mathbb E v-u_t\|_2^2.
\]

When gradients and expectation can be interchanged, reparameterized gradients have the corresponding expectation as well. The identity also applies to a learned posterior. Gradients must reach the posterior parameters through both draws as well as the vector field.

Implementation details:

- In K4, average the four velocities across the sample axis and then square their error. Averaging four squared errors would retain the expected K1 objective.
- Cross uses the same observation, time, bridge point, and target for both terms, with independent posterior randomness. Reusing the first latent gives K1. Redrawing the bridge point or target changes the quantity estimated.
- A cross loss may be negative. Taking its absolute value, clipping it at zero, or stopping gradients through one branch changes the objective. Record nonfinite losses, gradient norms, termination conditions, and the fraction of negative losses.
- At fixed parameters and a fixed outer minibatch, redraw only posterior noise to measure gradient variance attributable to posterior sampling. Use a separate name for variance that also includes new observations, times, or base noise.
- Compare methods at fixed epochs or training steps and, if possible, provide a second comparison at equal wall-clock time. K4 evaluates the vector field four times per bridge point.
- The identity establishes a conditional velocity-variance penalty. Its magnitude can change because the posterior narrows or because the field becomes less sensitive to the latent. Latent-use measurements are needed to diagnose posterior collapse or determine whether K4 or cross improves results.

Run `python scripts/verify_estimator_identity.py --output results/identity.json` for an independent NumPy check of the identities. It compares analytic values, analytic and finite-difference gradients, and Monte Carlo gradients, and includes a counterexample to the incorrect order of averaging and squaring. The sum reduction (`--reduction sum`) uses the paper's squared norm. The default mean reduction matches the elementwise MSE in the code. This is an algebraic test independent of model training.

### Time convention

| Quantity | Paper | Released continuous loss |
|---|---|---|
| Time 0 | Base noise | Data endpoint |
| Time 1 | Data endpoint | Base noise |
| Interpolation | \(x_t=(1-t)\,\mathrm{noise}+t\,\mathrm{data}\) in the paper's example | \(x_r=(1-r)\,\mathrm{data}+[\sigma_{\min}+(1-\sigma_{\min})r]\,\mathrm{noise}\) |
| Regression target | \(\mathrm{data}-\mathrm{noise}\) in the zero-endpoint-noise example | \((1-\sigma_{\min})\,\mathrm{noise}-\mathrm{data}\) |
| Posterior at an observation | \(q(t=1,z\mid x)\) | GaussianPrior at \(x,r=0\) |
| Conditional generation | Integrate from 0 to 1 | LLK.decode integrates from 1 to 0 while holding the specified latent fixed. |

The change of variable \(r=1-t\) accounts for orientation. A positive minimum noise also changes the data endpoint, so reversing the sign alone is insufficient. The released FlowMatchingLoss defaults to `sig_min=1e-4`. The continuous runner does not forward the YAML `flow.sig_min`, so the configured value can differ from the effective value.

Page 6 allows the latent marginal to be estimated after training from the aggregated posterior. A Gaussian conditional posterior alone imposes no standard-normal marginal. Distinguish the following protocols:

1. Encode a held-out observation with \(q(z\mid x)\) and generate a new observation from \(p(x_{\mathrm{new}}\mid z)\). This posterior predictive check supports the main reconstruction measures.
2. Sample a latent from the training-set aggregate posterior and then generate an observation. This marginal generator has an empirical prior; record that training observations supplied the latent reference pool.
3. Sample \(z\sim N(0,I)\) directly. This assumes a different prior and requires a separately labeled result.

## 3. Discrepancies in the released implementation

Paths in this table are relative to the pinned author repository.

| Source | Behavior observed | Project treatment |
|---|---|---|
| src_sfa/sfa.py, FlowMatchingLoss | Adds \(\alpha\,\mathrm{mean}(z^2)\) to MSE. | Equation 6 omits this term, and p. 16 discusses the absence of explicit regularization. Record alpha and keep its definition fixed across estimators. Use zero in the primary analysis and consider the released value in a sensitivity run. |
| configs/pinwheel.yml and PinWheelRunner in runner_continuous.py | flow.fix_z=false selects a CNF posterior. Appendix B.1 describes a Gaussian. The released freqs=0 also removes the time features needed here for a time-dependent Gaussian. | Report posterior family and time embedding explicitly when comparing the released preset with the paper's Gaussian variant. |
| configs/mnist.yml | Defaults to z_dim=32 and model.type=mlp. The appendix visualization text on pp. 19/21 mentions 64 dimensions. | Record the specification difference. The available material does not determine which setting produced the entire Table 2. |
| dataloader/dataloader_pinwheel.py | num_per_class counts points per arm, with coordinates multiplied by 10. | With five arms, samplesize=10000 can produce 50,000 observations. Check and record total split sizes in the new scripts. |
| dataloader/dataloader_mnist.py and MNISTRunner.train | Training uses stratified sampling and validation uses the official test partition. | Split the official training pool into 54,000/6,000 and reserve the official test set for final evaluation. Document the protocol change. |
| MNISTRunner.sample(in_sample=True) | Draws from MNIST with train=True. | Record the split and sample indices. The released in-sample option does not identify held-out, in-distribution test examples. |
| HVGRunner.sample | Selects the validation loader even though a test loader is returned. | Use a designated test set after choices are fixed. |
| dataloader_hvg.py | Expects kang_normalized_hvg.h5ad, cell_type, and split_B. The dataset is not included, and a missing file triggers dummy-object creation. | Fail explicitly when real data are unavailable. Label synthetic interface checks separately from an RNA experiment. |
| dataloader_pendulum.py, PENDULUM.sample | Recreating the simulator with seed=42 for separate train and test calls can yield identical trajectory prefixes. | Split whole trajectories from one pool or use distinct, recorded simulator seeds. Keep every window from a trajectory in the same split. |
| runner_continuous.py, training_step | Clips gradients after opt.step(). | Moving clipping before the update changes optimization. Record this fix, particularly when studying unstable cross losses. |
| main.py --ni | Can delete a preexisting log directory. | Give each participant, run, and seed a unique path and preserve prior output when resuming. |
| main.py configuration loading and the upstream README | The README instructs the reader to enter src_sfa. Code then looks for configs relative to the working directory, while the directory is at repository root. | Use an explicit working directory or absolute configuration path. |
| sfa_discrete.py, FlowMatchingLossMixture | Adds a negative marginal class-entropy term scaled using a detached loss ratio. | Keep and report the released mixture objective. Switching the posterior alone does not implement this model. |
| sfa_lds.py, FlowMatchingLossSeq | Sums residuals over time before squaring, adds a second-difference penalty for the latent trajectory, and has an upperbound branch. | Preserve the distinction from framewise mean MSE. The first continuous-estimator ablation excludes the sequence objective. |

Some released configurations, model interfaces, and runner branches may be incompatible. Report failed attempts and label runs using compatibility fixes accordingly. Count the actual parameters for each model rather than assuming the roughly 2.4 million parameters mentioned in the paper.

## 4. Choice of comparison models

| Dataset or question | Priority comparisons | Rationale and scope |
|---|---|---|
| Pinwheel marginal fit | Standard FM and VAE, supplemented by LatentFM | FM is the strongest direct Table 1 baseline, with W1 of 0.025 versus SFA's 0.024. A VAE-only comparison omits it. |
| MNIST continuous latent | LatentFM with a deterministic AE, and VAE | Table 2 reports SSIM/NMI of 0.980/0.488 for LatentFM and 0.716/0.490 for SFA. Differences in diversity are central to the comparison. |
| MNIST mixture | Mixture-SVAE/GMVAE as a specialized extension | It controls for discrete structure. An ordinary VAE addresses a different question. |
| RNA | Favor 32-dimensional LatentFM, followed by VAE | LatentFM and SFA have similar NMI/ARI in Table 4a. Compare clustering together with diversity. |
| Pendulum | GLD-SVAE where feasible, or a documented sequential VAE adaptation | A framewise AE does not implement the paper's GLD-SVAE. |

Page 8 describes a two-stage LatentFM procedure, training the AE first and then jointly training the AE and latent flow. The project baseline freezes the AE during flow training and should be labeled **frozen-AE LatentFM**. Compare models using shared splits, preprocessing, evaluation precision for the ODE, and seeds. Match the training budget or explain how it differs. Report actual parameter counts and time because architectural differences make exact parameter matching impractical.

## 5. Metrics and comparability

### Pinwheel W1

Compute 2D W1 by multivariate optimal transport with uniform weights and Euclidean ground cost, for example with `POT.emd2(a, b, euclidean_cost)`. Flattening coordinates and applying the univariate `scipy.stats.wasserstein_distance` measures a different distance. Sliced W1 is a less expensive, separately labeled comparison.

The authors' test_step divides the cost matrix by `M.max()*0.1`, so the result depends on the largest distance in that batch. Distinguish raw empirical W1 in fixed data coordinates (`w1_euclidean_raw`) from the author-style batch-normalized measure used for auditing (`w1_author_batch_normalized`). Match evaluation sample counts, generation protocol, data scale, and cost normalization before comparing results numerically to Table 1. A real-sample versus independent real-sample W1 gives a finite-sample reference.

### MNIST SSIM and Vendi

Compute paired SSIM between each held-out input and an observation generated from its posterior. Base noise can be sampled independently across observations. When comparing correctly paired and shuffled latents, reuse the same base noise in both conditions. Fix the inverse transformation, pixel range [0,1], SSIM implementation and window, and `data_range=1`. Standard unconditional FM provides no paired reconstruction for this measure.

Vendi depends on the features, kernel, normalization, and sample count. The released implementation calls `vendi.score_dual(..., normalize=True)`, with some branches evaluating latents and others generated observations. Label an image cosine-kernel score `Vendi_x_cosine` and specify pixel versus logit space and the actual generated sample count. A latent score cannot substitute for image diversity in Table 2.

For a trace-normalized positive semidefinite sample kernel, the Vendi effective rank cannot exceed the number of samples. Table 4a specifies 500 held-out examples yet reports SFA Vendi(x)=737.7. Possible explanations include multiple generated observations per held-out input or a different evaluation sample count; the published details do not resolve this. Report the number actually generated and retain the specified kernel.

### Clustering and linear probes

Generative training does not read class labels. Giving the clustering algorithm the number of classes follows the paper's protocol with known \(k\). Two useful evaluations are:

1. `Test-transductive KMeans.` Fit unsupervised KMeans on a fixed set of test representations, then use labels to score NMI/ARI. This is close to p. 8 and the released evaluation. Repeatedly consulting test labels when choosing settings introduces selection leakage.
2. `Train-fit KMeans.` Fit using training representations and use predict on held-out representations. This stricter inductive protocol differs from the Table 2 evaluation.

Fit the linear probe and scaler using training representations; train labels are used only for the probe. If tuning the regularization strength, select it on validation data, then reserve the final test set for one evaluation. A posterior mean and a posterior sample define different representations; the paper emphasizes samples. Label both when reporting both. Use t-SNE for visualization and the original latent coordinates for the main clustering metrics.

The EMNIST OOD task uses digits 0–9 and letters A–J. Letter labels are numbered 1–26, so the first ten are 1–10. Fix the test partition, subset indices, image orientation, class counts, and number of clusters in advance. The released sampler's `range(1,num_classes)` can omit J when selecting the first ten letters.

### softNMI

Equation 16 on p. 19 gives `MI/(H(p)+H(q))`, without the factor of two in arithmetic NMI. With true marginal and joint entropies, its upper bound is 0.5, although the text gives 1. The released `utils.soft_nmi` computes entropy over the sample axis per class and averages, which also differs from forming the two marginal entropies.

Keep the released function's result as `soft_nmi_legacy` for auditing. From the joint distribution \(P_{ab}=\operatorname{mean}_i[\mathbf{1}(y_i=a)q_i(b)]\), also calculate \(2\,MI/(H_y+H_q)\), labeled `soft_nmi_arithmetic_corrected`. Use hard-cluster NMI/ARI as clearly defined primary comparisons.

### RNA and pendulum

For RNA, select the 5,000 HVGs from training data where the original gene matrix allows it. Save gene IDs and normalization/log1p settings. If the input is already log-normalized, avoid applying the transform twice. A donor-grouped split is an additional leakage check in the proposal; if donor IDs are missing, record the limitation. Keep results from a random cell split separate from donor-isolated results.

For pendulum, report per-pixel RMSE, raw latent RMSE, and optionally latent RMSE after a linear mapping fitted on training trajectories to address rotation and scale ambiguity. Reserve test ground truth for scoring. The released code needs checks for direct reshaping of time/batch axes, inverse transformation of only the generated values in some branches, and the dimension summed before taking the square root. Identify these differences when interpreting Table 4b.

## 6. Information to retain per run

Record participant, run ID, code revision and changes or source hashes, environment versions, GPU, dtype, data and split hashes, actual counts, preprocessing, resolved hyperparameters, posterior family, alpha and sigma, parameter count, and seed. Save epoch-level training and validation curves, checkpoint selection rule, training and evaluation times, peak GPU memory, ODE method/tolerance/steps/function evaluations, metric definitions, and reasons for failure.

When two people repeat the same seed, use the pair to evaluate reproducibility. Compare environments, inputs, and settings first. Treat distinct seeds as the units for estimating run-to-run variation. Intervals based on roughly three seeds can be unstable, so include each individual result.

## 7. Sources

- *Structured Flow Autoencoders*, supplied as `12981_Structured_Flow_Autoenco.pdf`, especially pp. 2, 5–10, 16, and 18–22.
- The course project proposal, supplied as `DD2610_Deep_Learning_Advanced__Group_007.pdf`, pp. 1–2.
- Proposal feedback, supplied as `group_5_proposal_feedback.pdf`, p. 1.
- [Pinned author revision](https://github.com/edenx/StructuredFlowAutoencoder/tree/05e89071f6bae502eedfd3464fad4e8f75f506c1). The table in Section 3 identifies source locations. The authors' unpublished training records and original data files were unavailable.

## 8. Checks after implementation

- The LLK MLP has an identity embedding for the input observation, so the runner explicitly sets `hidden_dim` to match the flattened observation width. For mixture models, setting the corresponding width to the number of class logits fixes the mismatch between the identity class embedding and the hypernetwork.
- The file downloaded through the [CPA Kang tutorial](https://cpa-tools.readthedocs.io/en/latest/tutorials/Kang.html) contains 13,576 cells, 5,000 genes, and 8 cell types. The download manifest records the source hash and fields. The paper describes 7 classes. Retain the actual labels; `orig.ident` distinguishes STIM and CTRL and provides no verified donor identifier.
- The supplied `split_B` has 11,576 train, 1,226 valid, and 774 OOD cells. The upstream loader treats OOD as test, while the sampling runner selected validation. Multiclass clustering and the B-cell OOD test answer different questions. The project provides a primary random cell holdout and a separate audit using the original split_B.
- The generation record for `paper_only/` describes a separate context that used the extracted paper as its research source and did not browse or read the author code. Its task prompt also supplied interface and implementation guidance. The shared runner incorporates the model only after the freeze, retaining its coordinate-sum loss. Report its standalone and shared-runner protocols separately. See [the original prompt and provenance](paper_only/PROVENANCE.md) for the evidence and its limitations.
- The three shared-runner SFA estimators have the same architecture and separate random streams for data order and bridge draws. The common K16 validation score retains a variance penalty and is biased relative to the exact squared-mean objective. A separate gradient diagnostic compares all three estimators at one fixed checkpoint.
- Three explicitly labeled softNMI variants are implemented: the printed equation, the arithmetic correction, and the released legacy formula. Likelihood tables and some published baselines remain outside the implemented coverage.
- Local short runs, real data loading, and recovery tests checked the execution pipeline. Complete paper-value replication and Colab GPU performance had not been verified at the time of the original delivery. See [Validation](VALIDATION.md).
