# Turbulent Combustion Case

Sparse measurable combustion fields condition reconstruction of all five
fields. The single long trajectory uses chronological frames 0–7999 for train,
8000–8999 for validation, and 9000–9999 for test.

The legacy clock resets at frame 4000; sample identity and splitting therefore
use saved frame indices while retaining the raw clock as provenance.

New PointCloudFFM runs use `GL_rbf_ENH/topk_rbf`; Demo 50 uses the isolated
legacy compatibility path with explicit `CO,T,U_0,U_1,p` field mapping.
Run `python import_demo50.py` here to perform the strict non-destructive import
and write its local compatibility manifest.

`configs/posttrain/demo50_global_distribution.yaml` maps the historical flat
direct-coherence settings into the structured Phase-5 schema. It preserves the
paired-supervised self/mutual/cross estimators, two-step clean rollout,
endpoint-smooth observation consistency, data-retention loss, and optional
ConFIG update while writing only to a new child run. The compatibility dataset
uses the checkpoint's actual positional fields `CO,T,U_0,U_1,p` and explicitly
retains the stored constant third coordinate required by Demo50's encoders.

The canonical loader verifies 403 unique x positions, 100 unique y positions,
one constant z coordinate, and all 40,300 unique `(x,y)` pairs. It reorders the
stored permutation to ascending y then x, giving logical shape `(100,403)`.
New base and training-reference post-training templates live in `configs/base/`
and `configs/posttrain/`; the compatibility configuration remains separate.
