"""A fixed SSS support plus rotating native-field complement for PI feedback."""
from __future__ import annotations

import torch

from ..coherence.physical_risk import physical_ensemble_risks


def complement_indices(selected, native_count, count, *, seed, absolute_step):
    """Uniform without-replacement complement draws, independent of training RNG."""
    selected = torch.as_tensor(selected, dtype=torch.long, device='cpu')
    if selected.ndim != 2 or not 0 < count <= native_count-selected.shape[1]:
        raise ValueError('invalid selected support or complement count')
    generator = torch.Generator(device='cpu').manual_seed(int(seed) + int(absolute_step))
    universe = torch.arange(native_count)
    rows = []
    for sites in selected:
        if (sites < 0).any() or (sites >= native_count).any() or torch.unique(sites).numel() != sites.numel():
            raise ValueError('SSS support must contain unique native indices')
        mask = torch.ones(native_count, dtype=torch.bool)
        mask[sites] = False
        available = universe[mask]
        rows.append(available[torch.randperm(available.numel(), generator=generator)[:count]])
    return torch.stack(rows)


def stratified_mean_risks(selected_members, selected_truth, complement_members, complement_truth,
                          selected_indices, complement, native_masses):
    """J-member signed risk on exact S and uniform complement sample C."""
    native = torch.as_tensor(native_masses, device=selected_truth.device, dtype=selected_truth.dtype)
    selected_indices = selected_indices.to(device=native.device, dtype=torch.long)
    complement = complement.to(device=native.device, dtype=torch.long)
    if native.ndim != 1 or selected_indices.ndim != 2 or complement.ndim != 2:
        raise ValueError('native mass and batched support shapes required')
    if selected_indices.shape[0] != complement.shape[0]: raise ValueError('support batch mismatch')
    s, m, n = selected_indices.shape[1], complement.shape[1], native.numel()
    if not 0 < m <= n-s: raise ValueError('invalid complement inclusion probability')
    indices = torch.cat([selected_indices, complement], dim=1)
    if any(torch.unique(row).numel() != s+m for row in indices): raise ValueError('overlapping or duplicate quadrature indices')
    if not torch.all((indices >= 0) & (indices < n)): raise ValueError('quadrature index out of range')
    masses = native[indices]
    probabilities = torch.cat([torch.ones_like(selected_indices, dtype=native.dtype),
        torch.full_like(complement, m/(n-s), dtype=native.dtype)], dim=1)
    members = torch.cat([selected_members, complement_members], dim=-2)
    truth = torch.cat([selected_truth, complement_truth], dim=-2)
    return physical_ensemble_risks(members, truth, masses, total_mass=native.sum(),
        sampling='without_replacement', probabilities=probabilities)[1]


def rotating_field_risks(model, endpoint, dataset, config, *, absolute_step, rollout_seed, device):
    """Replay only C with the original observation batch and paired RFF member draws."""
    from dataclasses import replace
    from ..data.sensor_protocols import build_observation_batch
    from .common import sensor_protocol_from_config
    from .post_training import _conditional_draw_seed, _without_target
    from .rollout import differentiable_reconstruction

    selected = endpoint['coupling_selected_batch']
    sites = selected.metadata.get('query_indices')
    if sites is None or sites.shape[1] != 4096 or not bool(selected.query_valid_mask.all()):
        raise ValueError('rotating complement requires the fixed valid Q4096 SSS support')
    if len(endpoint['members']) != 2: raise ValueError('rotating complement requires two members')
    native = dataset.physical_weights
    settings = config['optimization']['multitask']['fidelity_budget']['rotating_complement']
    complement = complement_indices(sites, native.numel(), settings['points'], seed=settings['seed'], absolute_step=absolute_step)
    # Resolve the exact existing training conditions without consuming the batch stream.
    if not hasattr(dataset, '_fidelity_condition_lookup'):
        dataset._fidelity_condition_lookup = {f'{dataset.trajectory_ids[t]}:{frame}':i for i,(t,frame) in enumerate(dataset._items)}
    samples = [dataset[dataset._fidelity_condition_lookup[sample_id]] for sample_id in selected.sample_ids]
    observations = {}
    query = {}
    for i, sample_id in enumerate(selected.sample_ids):
        valid = selected.obs_valid_mask[i].detach().cpu()
        obs = selected.obs_indices[i].detach().cpu()[valid].tolist()
        fields = selected.obs_field_ids[i].detach().cpu()[valid].tolist()
        observations[sample_id] = list(zip(obs, fields, strict=True))
        query[sample_id] = complement[i].tolist()
    rebuilt = build_observation_batch(samples, sensor_protocol_from_config(config), query_indices=query, manifest_indices=observations).to(device)
    if rebuilt.sample_ids != selected.sample_ids or not torch.equal(rebuilt.metadata['query_indices'].cpu(), complement):
        raise ValueError('complement sample or global query indices differ')
    # The compact producer fills masked padding with real points, whereas the
    # manifest builder pads with -1/zeros and may use a shorter padded width.
    # Validate every valid observation exactly and keep the original full
    # tensors, including its padding and mask, for the actual rollout.
    for i in range(len(selected.sample_ids)):
        original_mask, rebuilt_mask = selected.obs_valid_mask[i], rebuilt.obs_valid_mask[i]
        if int(original_mask.sum()) != int(rebuilt_mask.sum()):
            raise ValueError('complement observation count differs')
        for name in ('obs_indices','obs_field_ids','obs_coords','obs_values'):
            if not torch.equal(getattr(selected,name)[i][original_mask], getattr(rebuilt,name)[i][rebuilt_mask]):
                raise ValueError('complement valid observation differs: '+name)
    metadata = dict(rebuilt.metadata)
    metadata['protocol'] = selected.metadata['protocol']
    batch = replace(selected, query_coords=rebuilt.query_coords,
        query_valid_mask=rebuilt.query_valid_mask, target_fields=rebuilt.target_fields,
        metadata=metadata)
    batch.validate()
    members = []
    for member in range(2):
        generator = torch.Generator(device=device)
        if member == 0: generator.set_state(endpoint['rollout_generator_state'])
        else: generator.manual_seed(_conditional_draw_seed(rollout_seed, member))
        members.append(differentiable_reconstruction(model, _without_target(batch),
            steps=int(config['rollout']['steps']), solver=config['rollout']['solver'],
            generator=generator, observation_config=dict(config['observation_consistency'])))
    risks = stratified_mean_risks(torch.stack(endpoint['members'],dim=1), endpoint['target'],
        torch.stack(members,dim=1), batch.target_fields, sites, complement, native)
    return risks, complement
