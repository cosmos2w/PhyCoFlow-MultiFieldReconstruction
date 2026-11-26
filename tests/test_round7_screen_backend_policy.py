"""The screen sets the production seed/strict policy before reading backend state."""
from scripts.qualification import car_round7_executor as screen


def test_production_backend_policy_is_set_before_inspection(monkeypatch):
    calls=[]
    def seed(seed, deterministic, *, strict_algorithms):
        calls.append((seed,deterministic,strict_algorithms))
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    def deterministic_enabled():
        assert calls == [(47,True,True)]
        return True
    monkeypatch.setattr(screen,'seed_everything',seed)
    monkeypatch.setattr(screen.torch,'are_deterministic_algorithms_enabled',deterministic_enabled)
    cfg={'runtime':{'seed':47,'deterministic':True},
         'coherence':{'scalable_anchored_coherence':{'common_prefit':{'sha256':'fixture'}}}}
    policy=screen.configure_backend(cfg)
    assert policy['strict_algorithms'] is True
    assert policy['deterministic_algorithms_enabled'] is True
    assert policy['cublas_workspace_config']==':4096:8'
