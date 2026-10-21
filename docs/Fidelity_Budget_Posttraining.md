# SOURCE-relative fidelity policies and internal ConFIG

This reference records the mathematical and configuration contracts for SOURCE-relative endpoint-fidelity policies. Use [Posttraining strategies](Posttraining_Strategies.md) for the maintained portable recipe and commands, [Posttraining results](Posttraining_Results.md) for the completed comparison, and [Posttraining development](Posttraining_Development.md) when changing shared implementation. Older operational notes and generated run evidence are optional historical archive material and may not be present in a fresh clone.

## Endpoint-risk definition

For an already model-normalized prediction and target, the per-field risk pools squared error over valid sample/query entries rather than averaging samplewise ratios:

\[
E_f=\frac{\sum_{bq}m_{bq}(\hat u_{bqf}-u_{bqf})^2}{\sum_{bq}m_{bq}},\qquad
\tau_f=(1+\epsilon)E_f^0+\delta,\qquad
a_f=\max(E_f^0,10^{-8}),\qquad
h_f=(E_f-\tau_f)/a_f.
\]

Here, $E_f^0$ is measured from the original SOURCE on fixed TRAIN calibration batches, $m$ masks valid entries, and $h_f$ is the normalized constraint violation. Defaults are a 5% relative increase and zero absolute allowance for each output field. The denominator floor scales violations; it never raises the accepted bound. A zero SOURCE risk with zero absolute allowance therefore has a zero threshold. TRAIN feedback, validation SOURCE measurements, and held-out assessments remain distinct.

## Scalar external-policy route

The reported scientific coherence sum is $C_{\mathrm{scientific}}=\sum_k s_k w_k L_k$, with the configured family preferences and SOURCE calibration scales. A scalar family policy produces $\Phi$; the external objective uses $\bar C=\Phi/C_0$, where $C_0=\max(C^0_{\mathrm{scientific}},10^{-8})$. With `family_scalarization.method: none`, ordered calibrated family values remain separate until the external policy's explicit `vector_reduction: sum`; this has the same value and gradient as `fixed_sum` for that scalar route.

The Lagrangian policy minimizes $J=\bar C+\sum_f\lambda_f h_f$ and projects each multiplier after a successful model update using the detached pre-update violations. Its optional penalty cadence changes only whether the primal constraint-gradient term is evaluated on that update; risk measurement and dual updates still occur each batch. The `rectify` policy uses the coherence direction when every field is feasible and otherwise uses the direction for the most violated field. The extended log barrier minimizes $J=\bar C+\sum_f b_t(h_f)$, with a continuous value and first derivative across its feasible and infeasible branches.

Multipliers initialize at zero and use $\lambda_f\leftarrow\max(0,\lambda_f+\eta_\lambda\operatorname{stopgrad}(h_f))$, with default dual LR $\eta_\lambda=0.01$. The detached violations are measured before the model update. Rectification chooses the first field in declared field order when maximum violations tie. The barrier is

\[
b_t(h)=\begin{cases}-\log(-h)/t,&h\le-1/t^2,\\th+(2\log t+1)/t,&h>-1/t^2.\end{cases}
\]

Its continuation uses $t=\min(t_{\max},t_0\gamma^{\lfloor k/I\rfloor})$ at completed-update cursor $k$; defaults are $t_0=1$, $\gamma=1.1$, $I=100$, $t_{\max}=100$. The controller restores that cursor on resume. These settings, the source-reference protocol and the budget remain explicit research choices.

These scalar policies use one ordinary model backward and the existing AdamW step. Invalid or nonfinite calculations stop the update; they do not fall back to a different aggregation method. Optional method settings belong only under the selected policy map, and changing methods requires replacing the full map so inactive settings are removed.

## Internal A/B/T TorchJD ConFIG route

The internal route keeps the three calibrated coherence-family gradients separate. In family order $k\in\{A,B,T\}$, let $g_k=\nabla_\theta(s_k w_k L_k)$, and stack rows as $G_{ABT}=[g_A^\top;g_B^\top;g_T^\top]$. TorchJD ConFIG maps this matrix to a parameter direction $d_C=\operatorname{ConFIG}(G_{ABT})$, which is normalized by $C_0=\max(C^0_{\mathrm{scientific}},10^{-8})$. The reported $C_{\mathrm{scientific}}=A+B+T$ remains a scientific diagnostic; it is not a scalar objective whose gradient equals the ConFIG direction.

For Lagrangian and barrier updates, the optimizer direction composes $d_C/C_0$ with the active policy-specific constraint-gradient contribution. In a feasible `rectify` update it uses $d_C/C_0$; in an infeasible `rectify` update it uses the selected worst-field constraint direction. The implementation obtains all three family gradients from the shared reconstruction graph, passes the full-width gradient matrix to TorchJD in FP64 (the Gram matrix is reused only for diagnostics), and fails explicitly on zero or nonfinite family or composed directions.

The internal route requires all three A/B/T families, `family_scalarization: {method: none}`, `coherence_aggregation: {method: torchjd_config}`, one `external_policy`, and `outer_aggregation: null` or omission. Scalar external policies can instead use an allowed family subset and do not set `coherence_aggregation`. Both routes require `vector_reduction: sum` and `objective_scale: source_mean`; the validator rejects contradictory selectors and inactive nested settings.

```yaml
optimization:
  multitask:
    family_scalarization: {method: none}
    outer_aggregation: null
    coherence_aggregation: {method: torchjd_config}
    external_policy:
      method: lagrangian
      vector_reduction: sum
      objective_scale: source_mean
      lagrangian: {dual_lr: 0.01, penalty_interval_updates: 1}
    fidelity_budget:
      metric: endpoint_mse_model_units
      relative_increase: 0.05
      absolute_allowance: 0.0
      denominator_floor: 1.0e-8
```

The route does not compose with native reconstruction retention, endpoint/source-anchor objectives, component constraints, non-full-model trainable scope, legacy gradient balancing, coherence warm-up or interval rescaling, recurring family-gradient diagnostics, or a non-null outer aggregator. See [configuration composition rules](configuration.md#source-relative-external-fidelity-policies) for the complete boundary.

## Checkpoints and interpretation

TRAIN controller risks and fixed validation SOURCE measurements serve different purposes. A successful training update does not certify validation or unseen samples. `last.pt` is the terminal recovery state; `best.pt` follows the configured validation selection rule and can remain a SOURCE fallback when no child checkpoint is eligible; `best_fidelity.pt` tracks aggregate fixed-panel MSE and does not certify that every field meets its guard. Inspect checkpoint selection metadata and its committed update cursor before reporting a checkpoint.

Per-field normalized MSE changes are not physical relative-L2 errors. Report each field and the total endpoint MSE alongside physical relative-L2 when evaluated, and keep raw A/B/T values separate from any scalar diagnostic sum. A passing fixed panel does not prove universal fidelity, method superiority, physical causality, or end-to-end speedup. Keep TEST locked until a separate selection protocol authorizes its use.

## Implementation and verification

The reusable code contracts are in [fidelity_budget.py](../src/phycoflow_reconstruction/training/fidelity_budget.py), [post_training.py](../src/phycoflow_reconstruction/training/post_training.py), and [validate.py](../src/phycoflow_reconstruction/config/validate.py). [Posttraining development](Posttraining_Development.md) maps the shared code boundaries and recovery requirements.

Focused contract coverage is maintained in [test_fidelity_budget.py](../tests/test_fidelity_budget.py), [test_fidelity_budget_config.py](../tests/test_fidelity_budget_config.py), [test_fidelity_budget_config_direction.py](../tests/test_fidelity_budget_config_direction.py), and [test_fidelity_budget_integration.py](../tests/test_fidelity_budget_integration.py). These suites cover pooled masked risk and bounds, policy behavior, selector validation, recovery, and shared trainer execution; they are automated checks, not evidence of long-horizon efficacy.
