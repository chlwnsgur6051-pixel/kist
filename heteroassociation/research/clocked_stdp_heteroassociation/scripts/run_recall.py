"""Teacher-free Clocked LIF readout from an archived or newly trained checkpoint."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.datasets import load
from src.training import config_lif
from src.recall import recall_batch

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('dataset',choices=['original','modified'])
 parser.add_argument('--k',type=int,default=10)
 parser.add_argument('--checkpoint',type=Path)
 args=parser.parse_args()
 x,y=load(args.dataset,args.k)
 config=json.loads((ROOT/'configs/validated_model.json').read_text())
 path=args.checkpoint or ROOT/'results/checkpoints'/f'{args.dataset}_K{args.k}.npz'
 with np.load(path,allow_pickle=False) as checkpoint:
  w=checkpoint['final_weight'] if 'final_weight' in checkpoint else checkpoint['weight_by_epoch'][-1]
 pred,voltage=recall_batch(x,w,config_lif(config),config['timing_ms'])
 print(json.dumps(dict(dataset=args.dataset,K=args.k,checkpoint=str(path),
     exact=float(np.mean(np.all(pred==y,axis=1))),bit=float(np.mean(pred==y)),
     association_correct=np.all(pred==y,axis=1).astype(int).tolist(),
     prediction=pred.tolist(),clock_voltage=voltage.tolist())))
