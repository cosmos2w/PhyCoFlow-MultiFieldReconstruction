# Case Workspaces

Each case owns physical meaning, dataset/config selection, optional physics,
visualization, and ignored run artifacts. `run.py` remains a thin delegate to
the shared package; model and training implementations do not live here.

The initial formal cases are turbulent combustion, Brusselator, Kolmogorov
flow, and KS. `mass_transport_fluid` is retained as a future case scaffold; its
current three-frame payload is useful only for integration checks.
