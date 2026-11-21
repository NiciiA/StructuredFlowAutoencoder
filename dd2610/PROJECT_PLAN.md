# DD2610 final project: experiment plan

Plan prepared on 13 November 2026. Course requirements and teacher feedback, commitments in the group proposal, and suggested implementation details have different status. Label the scale and scope of each completed experiment explicitly.

## 1. Project question and scope

**Does SFA learn both a good observation distribution and useful latent variables? Does the practical single-sample SCFM estimator introduce a penalty on the variation in velocity induced by the latent?** The project should produce evidence that another researcher can inspect, an analysis of discrepancies, and a record of failures. The proposal does not require a higher score than the paper.

| Source | Implication | Evidence to deliver |
|---|---|---|
| Course instructions, p. 2 | Create a public or KTH-private code repository, submit a presentation with its link by the deadline, and complete the assigned peer review. | Repository, presentation, individual contributions and peer review. Check the relevant Canvas pages for format and dates. |
| Proposal, pp. 1–2 | Reproduce the work using the authors' code and an isolated paper-only LLM implementation. Pinwheel and MNIST are the primary datasets. Add a theoretical and experimental extension. | Provenance for both implementations, frozen versions, run records, a discrepancy table, and the estimator derivation and experiments. |
| Feedback, p. 1 | For the Excellent/A target, run the authors' code on **all four experiment families**, including a mixture model and pendulum. | Results or explicit evidence of failure for Pinwheel, MNIST (including mixture), RNA-seq and pendulum. |
| Feedback, p. 1 | The official repository lacks baselines. The feedback suggests implementing or adapting the most competitive comparisons. | Standard FM on Pinwheel, LatentFM on MNIST, and a VAE as a reference for latent behavior. |
| Feedback, p. 1 | Disable search and browsing for the paper-only LLM track. | Fresh context, paper as the only research source, complete prompts and iteration records. Disclose that pretraining exposure to the public repository cannot be ruled out. |

The proposal originally described RNA-seq and pendulum as breadth checks to attempt. Feedback makes four-family coverage an explicit part of the grade target. If compute is tight, first reduce the number of seeds, training scale or plotting cost. Disclose omitted or failed work. If coverage is infeasible, keep the failure records and discuss an alternative scope with the instructor. No grade is guaranteed.

The proposal filename contains `Group_007`, while the proposal text and feedback say Group 5. The proposal names three members. The two-person running plan below concerns dwei and Nico. Confirm the actual group number in Canvas and record the third member's actual contributions. The proposal still marks individual learning goals as TODO. Each member should fill theirs in. The supplied course PDF links to the final-submission and peer-review pages but does not state those deadlines.

## 2. Three tracks and two meanings of independence

1. **Author-code reproduction.** Pin an official commit and add Colab support, documented data splits, logging, evaluation and any necessary repairs. Record the changes and why each was needed. Report the configuration actually run.
2. **Paper-only LLM implementation.** The proposed protocol calls for a fresh conversation and separate repository. Provide the paper, appendix and stated task constraints. Disable search and browsing, and keep official code, implementation tutorials, signatures copied from the authors, project-generated implementation code and patches out of the context. Generate, test and freeze the implementation and prompts before comparison. The main development conversation had already seen the authors' repository. Only a separately documented generation process can supply this comparison. Section 13 describes the bundled artifact and its limitations.
3. **Extension and controlled baselines.** Compare K=1, K=4 and the independent two-draw cross-product estimator on a fixed code base. Match data splits, network scale and evaluation. The shared comparison harness and discrepancy audit can be used with the paper-only implementation once it has been frozen.

Two people independently running the same notebook test execution reproducibility. Two executions at the same seed are a replication pair, not two independent training seeds. The independent LLM track tests whether the paper alone supplies enough implementation detail.

## 3. Experiment matrix and priorities

| ID | Experiment | Methods | Outputs | Priority |
|---|---|---|---|---|
| A1 | Continuous-latent Pinwheel | Author SFA, FM, VAE, LatentFM | 2D distribution, W1, latent structure, cost | Required. Start here. |
| A2 | Continuous-latent MNIST | Author SFA, LatentFM, VAE | Posterior predictive samples, SSIM, Vendi, NMI/ARI, probe | Required. |
| A3 | Mixture MNIST | Author Mixture-SFA; add GMVAE/Mixture-SVAE if feasible | Mixture probabilities, clustering, images and explicit softNMI definition | Required mixture coverage. |
| A4 | Kang PBMC RNA-seq | Author SFA; prioritize adding LatentFM | Record of 5,000-HVG processing, cell-type clustering, Vendi | Required four-family coverage. |
| A5 | Pendulum | Author LDS-SFA; adapt GLD-SVAE if feasible | Video frames, trajectories, observation and latent RMSE | Required by feedback. |
| L1/L2 | Independent implementation from the paper | Paper-only SFA on Pinwheel and scaled MNIST | Architecture, objective, hyperparameter and result differences against author implementation | Core proposal commitment. |
| X1 | Estimator theory and synthetic check | K=1, K=4, cross | Analytic identity, Monte Carlo convergence, gradient variance | First extension experiment. |
| X2 | Pinwheel estimator ablation | Same SFA architecture with different estimators | Quality, latent use, stability, training cost | Primary extension experiment. |
| X3 | MNIST estimator ablation | K=1, K=4, cross | Transfer of the X2 findings to images | Subject to budget. |
| S1 | Sensitivity | Latent dimension, ODE steps, scale | Changes in quality and cost | After core work. |
| A6 | Mixture Pinwheel | Author Mixture-SFA | Additional comparison with Table 1 and Figure 1 | Recommended if the pilot cost permits. |

FM has no encoder, so latent probe accuracy and NMI are inapplicable. Standard FM is a competitive Pinwheel baseline because its published W1 is close to SFA's. Deterministic LatentFM achieved high MNIST SSIM and provides a stronger reconstruction comparison than VAE alone. Label any simplified architecture or modified two-stage training procedure as a controlled baseline and describe the adaptation.

## 4. Execution order

### Step 0. Freeze the experimental protocol

- Confirm the Canvas group number, members, presentation format, rubric and deadline. The course handout links to some of this information without reproducing the full pages.
- Preserve checksums for the paper, proposal and feedback. Record the official commit, patches and configuration.
- For each experiment, specify its track, data split, network, objective, sampling procedure, evaluation definitions and stopping rule.
- dwei and Nico should use distinct participant identifiers and output directories, with the same split seed. Keep transferable run manifests and metrics, and separate each member's checkpoints.

### Step 1. Set up Colab and run a short pilot

Allow approximately 10–20 minutes as an initial planning target, then measure the actual runtime. Use a small sample to exercise the full path from downloading or loading data, through forward/backward passes, saving and resuming, to evaluation. A pilot checks feasibility. Select model settings on validation data.

Record GPU model and available memory, Python/PyTorch/CUDA and package versions, time per epoch, peak GPU memory, training time and sampling time. Colab GPU and session resources vary. Appendix B reports an M2 Pro with 16 GB; it gives a hardware reference but no guaranteed Colab training time.

Keep training data on runtime-local storage where practical and back up checkpoints, settings, logs and final samples to each participant's Drive. Preserve optimizer and RNG states. After a disconnect, verify the same split and configuration before resuming. Start with the notebook's default precision and validate any mixed-precision change separately.

### Step 2. Run author Pinwheel and a standard FM baseline

Use this inexpensive task to check the entire workflow. Inspect prior/posterior definitions, latent dimension, activation, sampling ODE, objective and regularization. Then run the planned train/validation/test protocol. Plot real and generated points and color the points by their latent value. Check both distribution coverage and encoded structure.

A falling loss alone cannot substantiate a reproduction claim. Save the model, fixed evaluation observations, sampling noise, metric definitions, results and plots. When a published result and a new result differ, inspect the protocols before interpreting the difference.

### Step 3. Generate the isolated paper-only implementation

One member should open a fresh context and provide the paper and agreed constraints. Request a continuous SFA with a Gaussian posterior conditioned on time and observation, a latent-conditioned velocity field, an explicit SCFM estimator, and runnable Pinwheel and scaled MNIST experiments. Disable search and browsing. Keep code from this repository outside the generation context. Save the initial prompt, each repair prompt, errors, final source, freeze date and hashes.

After freezing, check interfaces and mathematics before applying the common comparison protocol. Record details the paper leaves open, such as widths, epochs, optimizer settings, Gaussian scale constraints and ODE solver. Attribute each such choice to the implementer.

### Step 4. Cover continuous and mixture MNIST, then the remaining datasets

Run reduced pilots, then the chosen formal settings. Continuous SFA with post hoc k-means and Mixture-SFA answer different questions; label each result with the actual trained model.

For RNA-seq, check access to genuine observations, metadata and preprocessing. When starting from full raw counts, select highly variable genes (HVGs) and fit scaling parameters on training cells, then apply them to validation and test cells. If reproducing the paper's preprocessing uses information from all data, describe that protocol separately from a leakage-controlled experiment. Only actual Kang results belong in the Kang table. Section 13 explains the processed file's limitations.

For pendulum, split complete trajectories before extracting windows. Keep adjacent frames from the same trajectory together. Record dynamics, damping, duration, time interval, initial-condition distribution, image size and noise, and check them against the official generator.

### Step 5. Run the estimator ablation and latent diagnostics

Check the analytic formula and finite-sample Monte Carlo estimates before training K=1/K=4/cross on the same network. Record nonfinite values, gradient norms, training/validation curves, wall-clock time, peak memory, negative-loss frequency and failures.

First compare a fixed number of training updates. A supplementary fixed-wall-clock comparison can show how much of a gain depends on K=4's extra computation. Select each checkpoint using a shared nonnegative validation criterion with fixed evaluation randomness. A particularly negative cross-product training loss is not a checkpoint selection criterion.

### Step 6. Add matched seeds and run the final test after freezing choices

Get one seed working for each required experiment, then increase the primary comparisons to three seeds if the measured budget allows. Preserve the four author-code experiment families, the two core implementations and the three-estimator Pinwheel comparison first.

Use validation data for configuration choice. Freeze the configuration and checkpoint rule before running the untouched test set. Report all planned seeds, including failures, rather than selecting the seed with the best test performance.

### Step 7. Exchange results and prepare the presentation

Exchange manifests and metrics first. Compare source and data hashes, configuration, split IDs, hardware, training seed and resume history. When a matched run differs, check data, initialization, numerical settings, checkpoint, and sampling/evaluation randomness in that order.

In the presentation, report which paper findings reproduced, which depended on implementation or metric choices, how the estimator affected latent use and cost, and what failed. Include reproducible commands, individual contributions, LLM usage and known limitations.

## 5. Data and evaluation protocols

### Pinwheel

The proposal specifies 15,000/2,000/2,000 training/validation/test points, five arms and separate fixed generator seeds. Arm labels are excluded from training. Record the generator parameters and coordinate scale. Compare W1 at matched held-out and generated sample sizes.

Use empirical optimal transport with Euclidean ground cost in two dimensions. Averaging independent one-dimensional Wasserstein distances over coordinates gives a different metric. If using sliced Wasserstein or a fixed subset to save time, report the metric, number of projections and subset sizes explicitly. Put numerical comparisons on a common definition.

Evaluate arm structure with NMI/ARI and angle information with a suitable angular diagnostic. Since angle wraps around, predict `(sin(angle), cos(angle))` or use circular correlation. Simple linear correlation with the raw discontinuous `atan2` angle can mislead.

### MNIST and EMNIST

Use fixed indices to divide the 60,000 MNIST training images into 54,000 training and 6,000 validation images. Reserve the official 10,000 test images and use a fixed subset of 1,000 for comparisons with Tables 2 and 3. Share subset indices, preprocessing, pixel range and resolution across methods. Specify the training count and resolution for every scaled experiment.

- **SSIM.** For an observation x, draw z from q(z|x) and x' from p(x|z). Compute SSIM for the paired posterior predictive image and its source. Fix the window, data range and preprocessing. Randomly pairing unconditional samples with random test images does not measure reconstruction.
- **Vendi.** State sample size, feature space, kernel and normalization. Jointly interpret diversity, SSIM, sample grids and clustering; noise can have high diversity.
- **Clustering.** Fix k-means settings and seed, and state whether representations are posterior means or draws. Using k=10 for evaluation conveys knowledge of the number of classes; the generative model can still train without class labels.
- **Linear probe.** Freeze the encoder after unsupervised training. Fit the probe on training representations and labels. If tuning probe hyperparameters, use validation labels. Reserve test labels for final evaluation. The bundled basic probe uses fixed C=1.
- **OOD.** Prepare EMNIST digits and the first ten letter classes as in the proposal/paper. Record the split, label remapping, orientation and per-class counts. Assign nonoverlapping class IDs to the two sources. If OOD evaluation is incomplete, mark that result as unavailable.

### RNA-seq

The paper (p. 9 and Appendix B.2) describes Kang PBMC data from eight patients, seven cell types, library-size normalization, a log1p transform, 5,000 HVGs and a 32-dimensional continuous latent. Table 4a describes 500 held-out observations. Record data provenance and prior preprocessing to avoid applying normalization or log1p twice.

Add a donor-grouped assessment if reliable patient IDs are available. Fit the generative model on expression values and use cell-type labels for clustering evaluation. Record author-style preprocessing and a stricter leakage-controlled protocol separately. Reducing the number of genes or cells constitutes a scaled breadth check.

### Pendulum

The paper (p. 10 and Appendix B.4) uses angle and angular velocity as latent states and observes image sequences. The LDS posterior uses sequence information. Define RMSE's reduction over pixels, frames or trajectories, any normalization, and the number of held-out sequences.

Unsupervised coordinates may be unidentified up to a rotation, sign or change of scale. To report calibrated RMSE against physical coordinates, fit the mapping on training trajectories, select its settings on validation trajectories and apply it unchanged to test trajectories. Label its relation to the paper's raw RMSE. Fitting the alignment to test ground truth would use the target being evaluated.

## 6. Original extension: mathematics and implementation checks

For fixed `(x_t, t, u_t)`, let `V_k = v_theta(x_t, t, Z_k)`, where the Z draws are independent and identically distributed from the current posterior. With `m = E[V]`,

\[
E\left\|\frac{1}{K}\sum_{k=1}^{K} V_k-u_t\right\|^2
= \|m-u_t\|^2 + \frac{1}{K}\operatorname{tr}\operatorname{Cov}(V).
\]

The K=1 objective therefore penalizes variation in the latent-conditioned velocity **in expectation**. For K=4 the coefficient of that term falls to one quarter. The identity alone establishes neither latent collapse nor an improvement in learned representations.

For two independent posterior draws, the cross estimator satisfies

\[
L_{\mathrm{cross}} = (V_1-u_t)^\top(V_2-u_t),
\qquad E[L_{\mathrm{cross}}] = \|m-u_t\|^2.
\]

Both branches share the outer observation, time and reference velocity, use independent posterior noise and retain their reparameterization gradients. Do not detach either branch. For K draws, average the velocities before squaring. Averaging K single-draw squared losses leaves the K=1 expectation unchanged. Use a consistent sum or mean over coordinates in all three methods and theoretical checks, and record the scale.

One small synthetic check uses `V = A Z + b` and `Z ~ N(mu, Sigma)`. Compare Monte Carlo means to the analytic mean and `tr(A Sigma A^T)`. Use multiple independent minibatches to estimate loss/gradient variance. Check that the excess term falls as 1/K and that the cross estimator approaches the squared-mean target despite occasional negative minibatch values.

Record exploding gradients, nonfinite values and unusual downward drift. Treat changes to clipping, learning rate or warmup as explicitly named variants so the tuning budget is visible. Audit any author-code regularizer or optimization target outside Eq. (6). Either keep it identical for all estimators or remove it consistently to align with the theoretical objective.

## 7. Measuring latent use

For a held-out x, infer z, fix the base noise epsilon and ODE solver, then compare `g(epsilon,z)` and `g(epsilon,shuffle(z))`. Measure changes in generated observations and paired reconstruction. A complementary control holds z fixed and varies epsilon.

1. **Does z influence the output?** Measure the effect of changing z at fixed epsilon.
2. **Does z contain usable information?** Inspect arm/class probes and clustering.
3. **Does using the correct z benefit the task?** Compare input-paired scores before and after shuffling.
4. **What are the quality, diversity and cost trade-offs?** Interpret these measures together.

Shuffling keeps the batch distribution of z values and breaks each sample's pairing with its inferred z. The marginal generation distribution can remain similar. Large output changes may reflect arbitrary sensitivity, so also inspect probes and structure recovery.

## 8. Published metrics to audit

- Table 4a describes 500 held-out samples and reports SFA Vendi=737.7. The effective rank of a standard normalized positive-semidefinite sample kernel is at most its sample count. Check the actual generated batch size, kernel and sampling scheme. The discrepancy warrants an audit; the available description does not establish an error.
- Equation (16) on p. 19 prints `MI/(H(p)+H(q))` and states range [0,1]. The usual symmetric NMI includes a factor of two. Check the official implementation and keep printed and conventional variants separate when discussing Table 3.
- Some network, training and sampling details are underspecified or differ between the paper and code. Report parameter counts, latent dimensions, activations, posterior family, regularization, ODE solver, preprocessing and checkpoint policy.
- Likelihood requires density evaluation and explicit ODE trace handling. Mark unimplemented or unvalidated values as `not evaluated`. Training loss, MSE and unnormalized scores do not estimate `log p(x|z)` interchangeably.

## 9. Two-person running plan

Freeze the protocol together, run core comparisons independently and divide responsibility for breadth experiments before cross-running them.

| Phase | dwei | Nico | What to exchange |
|---|---|---|---|
| Pilot | Check environment and author Pinwheel | Run the same configuration and seed independently | Environment, data/config hashes and checkpoint metrics |
| Coverage | Lead continuous Pinwheel and MNIST, plus estimators | Lead mixture MNIST, RNA and pendulum | A notebook runnable from scratch and success/failure records per task |
| Two core tracks | Run author and frozen paper-only code on Pinwheel and scaled MNIST | Run both independently, or begin with the largest observed discrepancy | Participant-labeled results, distinguishing replicate runs from new seeds |
| Cross-check | Rerun at least one of Nico's breadth experiments | Rerun at least one of dwei's extension experiments or baselines | Differences between matched runs and findings |
| If budget allows | Complete the required matrix | Complete the required matrix | Two independent manifests and a common analysis |

Decide who will generate the paper-only implementation before that work begins. Keep author implementation details out of the generation context until freeze. Assign the third member's duties (data audit, baselines, review, presentation or other work) by actual agreement. This suggested plan cannot substitute for all three individual contribution statements.

## 10. Budget, stopping rules and deliverables

The proposal estimates 25–40 runs, 25–75 accelerator-hours and about 15 GB of storage. Repeating all experiments per participant adds cost. Two runs of the same configuration and seed form one cross-member replication pair.

Prioritize one seed for each required experiment, competitive baselines, both core implementations and all three Pinwheel estimators. Use pilot measurements to decide whether to expand primary comparisons to three matched seeds. Estimate the budget using measured seconds per epoch, planned epochs, ODE evaluations, repetitions and time for failures/disconnects. Running every breadth and sensitivity check at three seeds is not a prerequisite.

Suggested stopping rules include saving diagnostics and stopping a run on nonfinite losses or gradients, and resolving reproducible resume failures before continuing. Set validation patience, maximum epochs or steps and wall-clock limits before the formal runs when those rules are used. Log failures and label unrun experiments explicitly.

Final materials should include:

- A README and notebook runnable from a new Colab runtime, with versions and installation instructions.
- The official commit, adaptations and configuration differences; paper-only prompts, freeze records and failures.
- Split indices, preprocessing, full per-run configurations, environment, seeds, checkpoints and training/evaluation logs.
- Raw metrics per seed, aggregate mean/std or explicitly constructed intervals, and replication pairs across participants. Intervals from three seeds carry considerable uncertainty.
- Coverage of the four experiment families, baseline and estimator comparisons, latent diagnostic figures, quality versus time, and a table of failures and limitations.
- A presentation, repository link, individual contributions and disclosure of LLM use. Follow the current Canvas instructions for submission and peer review.

## 11. Code entry points and the scope of this release

Read the verified commands and settings in [README.md](README.md).

| File | Purpose |
|---|---|
| `notebooks/DD2610_Colab.ipynb` | Colab setup, experiment selection, training, persistence and summaries |
| `scripts/run_official.py` | Load the pinned author implementation for all four families and the mixture model. Record compatibility repairs in the audit. |
| `scripts/prepare_official_data.py` | Prepare and check the data, splits and preprocessing |
| `scripts/run_experiment.py` | Author-derived continuous SFA K=1/K=4/cross comparisons, FM/VAE/LatentFM baselines and the frozen paper-only model |
| `scripts/metrics.py` | Shared evaluation metrics with explicit definitions |
| `scripts/compare_runs.py` | Aggregate by experiment, seed and participant, and compare matched runs |

The harness supports controlled reproductions. Formal results, convergence, training cost, stability conclusions and comparisons with the paper require documented runs on the relevant data. Small smoke tests and runs with fewer epochs, smaller networks or lower resolution check the pipeline or inform budgeting. Label their actual scale.

Use the proposal splits and frozen configuration for planned full runs. Give subsets distinct identifiers and document their size. Tables should identify scope (`full / scaled / smoke`) and implementation (`author / controlled / paper-only`). The paper-only code is a separate artifact. Its generation record and input restrictions establish what can be said about its provenance.

## 12. Source materials

The following readings inform the project. Course and feedback documents specify student deliverables. They do not authorize anyone to publish a repository, send messages or submit work on the group's behalf.

- *Structured Flow Autoencoders* (original paper, `12981_Structured_Flow_Autoenco.pdf`): Eq. (6), p. 5; continuous Gaussian posterior, p. 6; Table 1/Figure 1, p. 2; Table 2, p. 8; Table 3, p. 9; Table 4, p. 10; Appendix B, pp. 18–22. The project ZIP includes the extracted input in `references/12981_Structured_Flow_Autoenco.txt`.
- Group proposal (`DD2610_Deep_Learning_Advanced__Group_007.pdf`): two reproduction tracks and estimator extension, p. 1; splits, latent diagnostics, budget and fallback plan, p. 2.
- Proposal feedback (`group_5_proposal_feedback.pdf`): four-family coverage, baseline suggestions and paper-only isolation, p. 1.
- Course *Final Project* handout: Important Dates reference, p. 1; repository, presentation, peer review and late-submission policy, p. 2. Consult Canvas for current details.
- [Authors' repository](https://github.com/edenx/StructuredFlowAutoencoder). Use the revision identified in the release manifest and reproduction audit.

## 13. Implementation and data verification notes

The package supplies entry points for all author models, the three estimators, FM/VAE/frozen-AE LatentFM, the separately generated frozen paper-only model, and a common evaluation harness. See `paper_only/PROVENANCE.md` for the exact prompt, generation restrictions, freeze information and limitations. The model was generated in a separate context. Resource restrictions were communicated in the task prompt rather than enforced by filesystem or browser controls. The exact prompt also supplied interface and implementation guidance. The provenance record describes what the LLM actually received; possible pretraining exposure cannot be excluded.

Inspection of the processed Kang file from the CPA tutorial found 13,576 cells × 5,000 HVGs and eight cell types, whereas the paper describes seven. `orig.ident` distinguishes stimulated/control groups and provides no reliable donor ID. The default controlled RNA experiment splits cells randomly 80/10/10. A separate author-loader audit preserves the published `split_B` train/valid/ood partition, whose OOD set represents a specific B-cell condition shift. Report these protocols separately. The upstream file already has selected HVGs, so the full gene-selection process cannot be certified leakage-free.

Controlled estimator training uses independent RNG streams for data order and bridge noise so different posterior sample counts do not change later minibatches or bridge draws. Checkpoints preserve these states. The K=16 validation score is a common selection criterion with a remaining 1/16 variance term. The basic probe uses fixed C=1.

The current implementation trains for fixed epochs and saves the best validation checkpoint. Patience or wall-clock early stopping would need to be implemented and frozen before using either criterion. Likelihood, all structured baselines, full GPU convergence, OOD runs and formal statistical conclusions were outstanding when the package was prepared. [VALIDATION.md](VALIDATION.md) documents which local checks were performed.
