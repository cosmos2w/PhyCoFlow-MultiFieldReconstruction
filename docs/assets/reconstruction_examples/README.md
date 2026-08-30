# Reconstruction examples

This tracked gallery stores selected, review-ready reconstruction and evaluation figures used by the project documentation. Routine generated figures remain under ignored case run directories.

## Current examples

- `senseiver_test_snapshot_0000_last.png`: turbulent-combustion Senseiver reconstruction from `tc_senseiver_5000ep/20260828T190145Z_fea0fc25`, using `last.pt` and test snapshot index `0`.
- `senseiver_base_5000ep_loss_history.png`: total training and fixed-validation loss history from the same completed 5,000-epoch Senseiver base run.
- `senseiver_test_best_relative_l2_violin.png`: field-wise relative-L2 distributions for 200 evenly spaced test snapshots from the same Senseiver run, using `best.pt`.
- `senseiver_test_best_global_distribution_marginal.png`: marginal field-distribution discrepancy statistics for the same 200 test snapshots and `best.pt`.
- `senseiver_test_best_cross_spectrum_same_frequency.png`: pooled same-frequency spectral-coherence scores for the same 200 test snapshots and `best.pt`.
- `gl_rbf_A_test_last_global_distribution_marginal_base.png`: source `best.pt` marginal field-distribution discrepancies for the matched 200-snapshot test comparison generated from `coherence_fix_A_global_distribution/20260828T131104Z_942a5f40`.
- `gl_rbf_A_test_last_global_distribution_marginal_posttraining.png`: post-training `last.pt` counterpart from the same matched comparison and shared logarithmic axis.
- `ab_test_last_global_distribution_joint_pdf_CO-T_base.png`: source `best.pt` CO–T joint-density comparison for 200 matched test snapshots generated from `coherence_fix_AB_balanced/20260829T235221Z_b3b586c4`.
- `ab_test_last_global_distribution_joint_pdf_CO-T_posttraining.png`: AB post-training `last.pt` counterpart using the same samples, histogram bins, axes, and density normalization.
- `ab_train_last_cross_spectrum_cross_frequency_base.png`: source `last.pt` cross-frequency coherence scores over 12 deterministic training-aligned ensembles of 16 snapshots generated from `coherence_fix_AB_balanced/20260829T235221Z_b3b586c4`.
- `ab_train_last_cross_spectrum_cross_frequency_posttraining.png`: AB post-training `last.pt` counterpart using the same 12 ensembles, graph, field pairs, and bounded score axis.
- `ab_train_last_global_distribution_pairwise_base.png`: source `last.pt` pairwise field-distribution discrepancies for the matched 200-snapshot training comparison generated from `coherence_fix_AB_balanced/20260829T235221Z_b3b586c4`.
- `ab_train_last_global_distribution_pairwise_posttraining.png`: AB post-training `last.pt` counterpart using the same training snapshots and logarithmic vertical limits.

Add future examples with descriptive, stable filenames and record their source run, checkpoint, split, and snapshot selection here.
