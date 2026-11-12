#!/usr/bin/env python3
"""Bounded CPU-only R8 query scaling, independent of trained models or jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
import time
import tracemalloc
from pathlib import Path
import torch
import gudhi
from phycoflow_reconstruction.coherence.round8_queries import SharedQuerySampler, pair_count, VERSION, SHAPE_SAMPLING_VERSION
from phycoflow_reconstruction.coherence.round8_scale import sampled_scale
from phycoflow_reconstruction.coherence.round8_shape import LazyPooledShape


def _size(tensor): return tensor.numel() * tensor.element_size()


def importance_experiment():
    import statistics
    sampler = SharedQuerySampler([str(i) for i in range(8)], seed=104, capacity=8)
    # Populate a bounded, unequal priority snapshot using only exposed IDs.
    for _ in range(8):
        q = sampler.draw()
        sampler.finish(pair_difficulties={d.index:(d.index + 1., d.index + 1.) for d in q.pair_queries})
    sampler.refresh(10)
    n = pair_count(8)
    values = [i / 4. - 4. for i in range(n)]
    gradients = [1. + i * i / 100. for i in range(n)]
    distribution = sampler.distribution['pair']
    probabilities = [sampler.epsilon / n + (1-sampler.epsilon)*distribution.get(i, 0.) for i in range(n)]
    exact = sum(values) / n
    exact_grad = sum(gradients) / n
    theory_var = sum(p * (v / (n*p) - exact)**2 for p,v in zip(probabilities,values)) / sampler.mutual_draws
    theory_grad_var = sum(p * (v / (n*p) - exact_grad)**2 for p,v in zip(probabilities,gradients)) / sampler.mutual_draws
    observed, observed_grad = [], []
    for _ in range(12000):
        q = sampler.draw()
        observed.append(sum(d.weight*values[d.index] for d in q.pair_queries))
        observed_grad.append(sum(d.weight*gradients[d.index] for d in q.pair_queries))
        sampler.finish()
    return {'population':n,'draws_per_estimate':sampler.mutual_draws,'trials':len(observed),
            'cache_entries':len(distribution),'negative_query_values':sum(v < 0 for v in values),
            'exact_loss_mean':exact,'sample_loss_mean':statistics.mean(observed),
            'exact_loss_variance':theory_var,'sample_loss_variance':statistics.variance(observed),
            'exact_gradient_mean':exact_grad,'sample_gradient_mean':statistics.mean(observed_grad),
            'exact_gradient_variance':theory_grad_var,'sample_gradient_variance':statistics.variance(observed_grad)}


def run():
    torch.manual_seed(803); random.seed(803); torch.set_num_threads(1)
    rows=[]
    for c in (1, 2, 5, 20, 100):
        fields=tuple(str(i) for i in range(c))
        sampler=SharedQuerySampler(fields,seed=91)
        draw=sampler.draw(); sampler.finish()
        a=torch.randn(1,c,4,4); b=torch.randn_like(a); target=torch.randn_like(a)
        shape=LazyPooledShape(torch.zeros(c),torch.ones(c),dimensions=(0,),projections=4)
        shape.accumulate(a,target,draw)
        artifact=shape.freeze(provenance={'split':'train','sample_ids':['synthetic:0'],'source_checkpoint_sha256':'a'*64})
        coef=torch.randn(2,1,8,c); coef_target=torch.randn(1,8,c)
        filters=torch.ones(2,8); scales=torch.ones(2,c); mass=torch.ones(8); weights=torch.ones(2)/2
        desc=torch.randn(2,1,2*c); target_desc=torch.randn(1,2*c)
        means=torch.randn(2,1,1,c); target_means=torch.randn(1,1,c)
        kwargs={'draw_piece_means':means,'target_piece_means':target_means,'piece_weights':torch.ones(1),'mean_scales':torch.ones(1,c)}
        def scale_call():
            return sampled_scale(coef,coef_target,draw,filters=filters,scales=scales,mass=mass,power_floor=scales,band_weights=weights,self_descriptors=desc,target_self_descriptor=target_desc,mean_kwargs=kwargs)
        def shape_call(): return shape.score(a,b,target,draw)
        scale_call(); shape_call()
        def measure(fn, repeats):
            start=time.perf_counter()
            for _ in range(repeats): fn()
            elapsed=(time.perf_counter()-start)*1000/repeats
            tracemalloc.start(); fn(); peak=tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
            return {'mean_ms':elapsed,'python_tracemalloc_peak_bytes':peak,'repeats':repeats}
        scale_times=measure(scale_call,20); shape_times=measure(shape_call,5)
        t0=time.perf_counter()
        for _ in range(1000): sampler.draw(); sampler.finish()
        sampler_us=(time.perf_counter()-t0)*1e6/1000
        tensor_inputs=sum(_size(x) for x in (a,b,target,coef,coef_target,filters,scales,mass,weights,desc,target_desc,means,target_means,*kwargs.values()))
        rows.append({'fields':c,'pair_population':pair_count(c),'self_unique':len(draw.self_queries),'pair_unique':len(draw.pair_queries),
                     'sampler_max_priority_entries_per_kind':sampler.capacity,'sampler_state_json_bytes':len(json.dumps(sampler.state_dict())),
                     'shape_calibration_categories':len(artifact['scales']),'shape_selected_filtrations':len(tuple(shape._sampled_queries(draw))),'shape_artifact_json_bytes':len(json.dumps(artifact)),
                     'declared_tensor_input_bytes_upper_bound':tensor_inputs,'sampler_draw_finish_mean_us':sampler_us,
                     'scale':scale_times,'shape':shape_times})
    files=['src/phycoflow_reconstruction/coherence/round8_queries.py','src/phycoflow_reconstruction/coherence/round8_scale.py',
           'src/phycoflow_reconstruction/coherence/round8_shape.py','src/phycoflow_reconstruction/coherence/round8_state.py',
           'tests/test_round8_queries.py','tests/test_round8_policy_state_calibration.py',
           'tests/test_round8_parameter_diagnostics.py', 'scripts/round8_numerical_qualification.py']
    return {'schema':'round8_cpu_numerical_qualification_v1','seed':803,'device':'cpu','torch_num_threads':1,
            'versions':{'python':platform.python_version(),'torch':torch.__version__,'gudhi':gudhi.__version__},
            'query_protocol':VERSION,'shape_sampling':SHAPE_SAMPLING_VERSION,
            'calibration_semantics':'all pooled sign/line categories fitted; training samples one inner query per unique outer field ID',
            'setup':{'batch':1,'members':2,'spatial_shape':[4,4],'homology_dimensions':[0],'sliced_projections':4,
                     'bands':2,'spectral_coefficients':8,'mutual_draws':3,'scale_repeats':20,'shape_repeats':5},
            'measure_boundary':'Synthetic forward score kernels and sampler only; excludes model forward, backward, optimizer, calibration, GPU, native/tensor allocator peaks, and concurrent-load control. tracemalloc measures Python allocations only. Tensor input bound may double-count shared references.',
            'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},
            'numerical_checks':['real Scale/Shape finite-bank value and gradient expectations and Monte Carlo variance under nonuniform importance sampling',
              'fitted retained Scale anchors and gradients', 'distinct weighted endpoint State value and gradient parity',
              'State chain-rule decomposition', 'sampler and calibration checkpoint replay', 'physical risk and output-blend quadratic'],
            'checks_command':'KEOPS_CACHE_FOLDER=/tmp/keops_round8_queries python -m pytest -q tests/test_round8_queries.py tests/test_round8_policy_state_calibration.py tests/test_round8_parameter_diagnostics.py::test_control_state_blocks_match_full_bank_gradient tests/test_round8_parameter_diagnostics.py::test_control_shape_and_scale_live_blocks_reproduce_family_gradients tests/test_round8_parameter_diagnostics.py::test_candidate_diagnostic_uses_frozen_block_coefficients_and_family_scales',
            'importance_experiment':importance_experiment(), 'results':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('cases/turbulent_combustion/runs/coherence_compute_round8/artifacts/numerical_qualification.json'))
    args=parser.parse_args()
    data=run(); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'artifact':str(args.output),'results':data['results']},indent=2))
if __name__=='__main__': main()
