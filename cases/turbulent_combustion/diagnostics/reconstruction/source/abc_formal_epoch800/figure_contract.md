# Training reconstruction preview

- **Claim:** qualitative sparse-reconstruction quality of the configured evaluation weights.
- **Training epoch:** `800.000`
- **Sample:** `trajectory_000000:8000` from the configured preview split.
- **Panels:** physical target, reconstruction, and absolute error; each error panel reports its field-wise relative L2 error, and white circles mark conditioned sensors.
- **Metrics:** `latest_metrics.json`; reusable arrays: `latest_reconstruction.npz`.
- **Caveat:** this fixed-sample diagnostic is not an aggregate benchmark.
