# Phase 8 integration audit

This release verifies reproducible wiring, not converged model quality.

- Scope: one Brusselator validation trajectory, 128 sparse `u` sensors, complete
  192×192 query grid, and one optimizer update per ablation.
- Included stages: plain coordinate MLP, global-distribution post-training,
  periodic PDE post-training, and direct PINN.
- Excluded: mass-transport demo (benchmark-ineligible), tuned generative sweeps,
  confidence intervals, and performance rankings.
- Lineage: post-training rows are immutable children; source hashes before and
  after refinement agree in their run manifests.
- Data: canonical stored 80/10/10 trajectory split; full payload SHA-256 is in
  `suite.yaml`; the portable sensor manifest is checked against every row.
- Claims: metric values demonstrate executable contracts only. They do not show
  superiority or physical validity after one update.
- Code: the repository commit and a relevant-file hash snapshot are stored in
  `results.yaml`. The release was prepared in a dirty worktree, so file hashes,
  not the commit alone, identify the exact implemented paths.
- Licenses/provenance: `UPSTREAM.md` records NeuralOperator (MIT), Senseiver
  (BSD-3-Clause reference), DiffusionPDE (CC BY-NC-SA reference), DeepXDE
  (LGPL-2.1 reference), Meta Flow Matching (CC BY-NC reference), and the
  repository-local Demo50 extraction. No additional upstream code was added in
  Phases 7–8.
