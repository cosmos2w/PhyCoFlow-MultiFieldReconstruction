# MIMONet source and adaptation

- Paper: Kobayashi et al., *Nature Communications* (2026), DOI
  [10.1038/s41467-026-77463-7](https://doi.org/10.1038/s41467-026-77463-7).
- Official release: [Zenodo v2](https://zenodo.org/records/21986357), DOI
  [10.5281/zenodo.21986357](https://doi.org/10.5281/zenodo.21986357).
- The release metadata points to
  [kkazuma19/MIMONet](https://github.com/kkazuma19/MIMONet). On 2026-09-26
  that URL returned HTTP 404, so the locally available
  `0_demo_TurbulentCombustion/src/mimonet_upstream/` copy extracted from the
  Zenodo v2 archive was the implementation reference. Its own `UPSTREAM.md`
  records the extraction and package-import change. The archive's large data
  files were not added here.
- The Zenodo metadata grants MIT for code and distinguishes the released data's
  CC BY-NC 4.0 license. The upstream code notice is retained in
  `MIMONET_UPSTREAM_LICENSE.txt` and included in the Python package.

`mimonet.py` adapts the released FCN branch/trunk computation to
`ObservationBatch`: measured values and normalized sensor locations go through
separate ReLU FCNs with validity flags, their 256-dimensional outputs multiply,
and a coordinate FCN produces a field-wise basis contracted with the shared
branch vector. The combustion profile uses the canonical repository's 2-D
coordinates rather than the demo wrapper's stored 3-D coordinates with a
constant third component. Sensor inputs use fixed field-specific slots, and
the shared trainer/evaluator supplies normalization, masked MSE, checkpointing,
and post-training reconstruction.
