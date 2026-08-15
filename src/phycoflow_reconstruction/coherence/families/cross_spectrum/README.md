# Reserved Cross-Spectrum Coherence Family

This package reserves the peer family name `cross_spectrum`; it does not define
or return a placeholder loss. A future contribution must specify its component
taxonomy, grid/spacing requirements, periodicity, detrending and windowing,
wavenumber bands, invariances, units, differentiability, cost, failure cases,
and degenerate examples before registering an implementation.

Cross-spectrum components must remain beneath `cross_spectrum.*` and must not
be represented as `global_distribution.self`, `mutual`, or `cross` terms.
