#!/usr/bin/env python3
"""Fit frozen Round 8 artifacts from the matched SOURCE TRAIN cache.

Loads frozen R7 common observers and the verified per-condition cache; no model
inference or optimizer updates are performed. Output must not already exist.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
from phycoflow_reconstruction.training.round8_artifacts import prepare_from_panel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True,help='resolved frozen Round 7 T-style config')
    parser.add_argument('--manifest',type=Path,required=True,help='frozen Round 8 population manifest')
    parser.add_argument('--source-panel',type=Path,required=True,help='completed SOURCE TRAIN panel.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shape-samples',type=int,default=16)
    parser.add_argument('--shape-queries',type=int,default=4)
    parser.add_argument('--probe-seed',type=int,default=800047)
    parser.add_argument('--score-floor',type=float,default=1e-8)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=prepare_from_panel(args.config,args.manifest,args.source_panel,shape_samples=args.shape_samples,shape_queries=args.shape_queries,probe_seed=args.probe_seed,score_floor=args.score_floor)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream: torch.save(result,stream)
    print(f"Saved TRAIN artifacts to {args.output} ({len(result['provenance']['sample_ids'])} conditions).")

if __name__=='__main__': main()
