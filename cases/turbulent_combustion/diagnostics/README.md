# Turbulent-combustion diagnostics

The current, complete [A+B+C ExampleVisual gallery](ExampleVisual/README.md)
contains training history, physical-coordinate reconstruction, matched source
and A+B+C set evaluations, and examples for all three configured coherence
families. It uses a frozen epoch-1440 copy of the formal run checkpoint and
does not write to or restart that run.

The gallery keeps the standard post-processing figures and metrics together
with concise explanatory views. Its manifests state the checkpoint hash,
validation sample selection, ensemble size, topology raster, and limits of
interpretation. Generated run outputs remain under `runs/`; the gallery is
the versioned demonstration suite.
