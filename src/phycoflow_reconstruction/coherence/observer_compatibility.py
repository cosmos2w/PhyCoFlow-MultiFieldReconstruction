"""Exact additive observer compatibility, without refitting.

The pinned current revisions add empirical reduction and preserve the historical
fair calculations; the pinned query sampler adds condition-group assembly only.
Focused AST and value/gradient regression tests cover those paths. A saved
artifact may contain any combination of these pinned old and new revisions.
Any further source or numerical-backend change requires a new qualification.
"""

_PAIRS = {'pooled': {'family.py': ('d4789b0fca32e648d913e01a5c98f8e70d2985db5e1e88f3838b66f13f72c5ce', '4ba684a380f2dbb9d1878a1d248be5ed775ef77566ecd48f32fb6cf7f1a3bbd8'), 'shape_compute.py': ('4df122f90cafdde1ee5963c81f034d40924adc719332ae7684fd90c30675e945', '8cd0ec1b9f5577af082a91b59c9188fd60cf7e4100bd151955be7c43d3c43531'), 'query_sampling.py': ('40eecb8c47bce77f995fc2f2304cef7b2c3c29f93f75c9b0dfef0f8ed0182f3c', 'cd578538f403ffe1b08419c78aceee4f637268f4102af831538d0ee81e490d20')}, 'scale': {'family.py': ('d231a54729de23b90fd11c94440f872fc772158d3fe5d6a414da25a437db9e0c', '3f48225a1cdaa4484563fadeb17f407a902e44fa7776e30962599c3910ea94c9'), 'losses.py': ('cd5f15327aeec9e0c9fe353994c844ce155aec68ae702d025e3940a4df7e9186', '9178cafacdba82c2e5d4ebed3f83cb730245d91651d1dd99f1d450da52dd1f1f')}}

def _compatible(saved, current, key, kind):
    if saved == current:
        return True
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    previous = dict(current)
    now = current.get(key)
    before = saved.get(key)
    if not isinstance(now, dict) or not isinstance(before, dict):
        return False
    altered = dict(now)
    for name, (old_hash, new_hash) in _PAIRS[kind].items():
        if before.get(name) not in (old_hash, new_hash) or now.get(name) != new_hash:
            return False
        altered[name] = before[name]
    previous[key] = altered
    return saved == previous

def compatible_pooled_source(saved, current):
    return _compatible(saved, current, "source_sha256", "pooled")

def compatible_scale_source(saved, current):
    return _compatible(saved, current, "sources", "scale")
