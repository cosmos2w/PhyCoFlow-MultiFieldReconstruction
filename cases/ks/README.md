# KS Quasi-Super-Resolution Case

Uniform sparse samples in both time and space condition reconstruction of the
complete `[401,256,1]` trajectory. Temporal and spatial ratios are independent;
the default is `(4,4)`. This is not future forecasting.

Named sensor configs cover `(time,space)` ratios `(2,4)`, `(4,4)`, and `(4,8)`.
Point models flatten the state to `(t,x)` tokens; grid adapters retain the full
reversible `[time,x]` logical shape.

The Phase-7 gate reconstructed the exact 401×256 query state for both `(2,4)`
and `(4,8)` protocols through `evaluate-run`. Reports include errors aggregated
independently over time and space plus temporal- and spatial-derivative MSE.
`configs/posttrain/global_distribution_reference.yaml` enables only the valid
single-field marginal component.
