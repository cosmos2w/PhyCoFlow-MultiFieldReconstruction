# Dataset Catalog

Only Markdown explanations are synchronized here. HDF5/PT payloads and links
are local-only and ignored by Git. A collaborator may create the documented
relative link or place an equivalent validated file at the same path.

| Case | Local payload | Benchmark split |
|---|---|---|
| Turbulent combustion | `turbulent_combustion/Merged_*.h5` | ordered frames 80/10/10 |
| Brusselator | `brusselator/brusselator.h5` | stored trajectories 80/10/10 |
| Kolmogorov flow | `kolmogorov/kolmogorov.h5` | stored trajectories 80/10/10 |
| KS | `ks/ks.h5` | stored trajectories 80/10/10 |
| Mass transport-fluid | `mass_transport_fluid/mass_transport_fluid_demo.h5` | future scaffold; current file is integration-only |

Validate links and schemas from the project root:

```bash
python scripts/validate_dataset.py --all
```

See `SCHEMA.md` for the accepted HDF5/PT contract.
