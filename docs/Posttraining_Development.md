# Extending post-training methods

This note is for collaborators changing shared post-training code. Start with [Posttraining strategies](Posttraining_Strategies.md) for supported user-facing routes and [Posttraining results](Posttraining_Results.md) for the measured Run 00/06/11/11f comparison; preserve the evidence limits there when writing new claims.

## Code ownership

| Concern | Maintained implementation |
| --- | --- |
| Config composition and selector validation | `src/phycoflow_reconstruction/config/validate.py` |
| Shared coherence trainer, SOURCE loading, optimizer and recovery flow | `src/phycoflow_reconstruction/training/post_training.py` |
| Family scalarization and external task aggregation | `src/phycoflow_reconstruction/training/multitask.py` |
| SOURCE-relative risk math and Lagrangian / rectify / barrier controllers | `src/phycoflow_reconstruction/training/fidelity_budget.py` |
| Fixed-panel checkpoint eligibility and aggregate MSE tracking | `src/phycoflow_reconstruction/training/checkpointing.py` |
| Research profile | `cases/turbulent_combustion/configs/posttrain/fidelity_budget/lagrangian_config_warmup.yaml` |

Keep model behavior behind the existing differentiable-reconstruction contract. A new policy should compose with the shared trainer and its model, data, observation, optimizer, EMA and checkpoint boundaries; do not add a second trainer, model-specific adapter, teacher copy or implicit launch manager.

## Preserve the update contract

The external-policy path computes one observation-consistent differentiable reconstruction for coherence and paired-target endpoint risks. It does not replay `model.training_loss()` or construct a teacher model. Scalar policies build one objective direction; internal A/B/T ConFIG obtains three ordered family-gradient rows and combines the selected external-policy contribution. Family gradients and the constraint direction are distinct from the reported scientific A+B+T sum.

One ordinary update performs its configured gradient traversals, checks the resulting direction, clips the combined model gradient once, and takes one existing AdamW step. A stateful external controller updates only after the successful model step using the detached pre-update violations. Preserve this ordering and fail on invalid/nonfinite directions without silently switching to weighted-sum behavior.

SOURCE-relative budgets use pooled masked endpoint MSE per normalized output field. Calibration is anchored to the original SOURCE on TRAIN batches; fixed validation SOURCE metrics remain an independent checkpoint-selection reference. The denominator floor scales violations and never enlarges the acceptance bound. Keep TRAIN feedback, validation eligibility, physical relative-L2, and held-out claims distinct.

The global model LR warm-up reads the restored successful-update cursor, so resume must continue the schedule without resetting it. Ordinary resume must keep the resolved config contract; changing the epoch horizon is a separate, explicit `RunStore.extend_epochs` operation with its preserved-snapshot safeguards.

## Add or change a policy

Write the mathematical objective and state transition first, including which predictions, reference quantities, update cursor and model parameters it uses. Identify whether it is a scalar family policy, a family-gradient aggregator, an external fidelity policy, or a composition of those layers; use the narrowest existing selector that expresses that change.

Add only selected method parameters beneath that method's own configuration key. Update validation in `config/validate.py` so invalid, inactive or incompatible maps fail before model execution. The validator intentionally rejects unknown nested keys: changing the primary Lagrangian policy to another method requires replacing the whole `external_policy` map, removing its `lagrangian` settings instead of overriding only `method`.

Preserve the public composition rules: an external policy requires a null/omitted `outer_aggregation`; internal `coherence_aggregation` supports TorchJD ConFIG, requires `family_scalarization.method: none`, requires the external policy, and requires all three A/B/T families. Scalar external policies can use an allowed family scalarization without internal ConFIG. Legacy `optimization.multitask` omission keeps its prior route; component-constrained and topology-regularized modes remain separate.

If a policy carries state, bind and serialize every value needed for exact continuation: calibrated SOURCE references, controller multipliers or schedules, completed update cursor and optimizer state where applicable. Verify strict resume compatibility and interrupted-history recovery. A stateless route should not create checkpoint state it does not need.

## Verification entry points

Start with the focused existing contracts that match the edited boundary; do not rely only on configuration composition if the change affects real gradient or recovery behavior.

- Policy equations and validation: `tests/test_fidelity_budget.py`, `tests/test_fidelity_budget_config.py`, `tests/test_fidelity_budget_config_direction.py`, and `tests/test_fidelity_budget_formal_config.py`.
- Shared trainer composition and model execution: `tests/test_fidelity_budget_integration.py` and `tests/test_fidelity_budget_composition_integration.py`.
- Existing scalarization, joint aggregation and backward compatibility: `tests/test_multitask.py`, `tests/test_multitask_config.py`, `tests/test_multitask_joint.py`, and `tests/test_multitask_integration.py`.
- State, cadence and optimizer-schedule boundaries: `tests/test_checkpointing.py`, `tests/test_epoch_extension.py`, `tests/test_fidelity_budget_sparse_penalty.py`, and `tests/test_posttraining_lr_warmup.py`.

Keep configuration-only validation separate from measured model execution. When a required real-model smoke is run, record the actual SOURCE identity, checkpoint, resolved profile, update count, stop reason, selected/final checkpoint names, and any physical device used; an accepted config or successful process launch is not completion evidence.

Do not use TEST during policy development or branch validation. For results, report every field and the total MSE guard, raw A/B/T values, selected and final checkpoint identities, physical relative-L2 when measured, runtime scope and memory scope. A passed intervention or panel does not establish causal physical influence, population fidelity, method superiority or complete-runtime speedup.
