"""Replay actual spike-event learning and compare every archived epoch weight."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.datasets import load
from src.training import train,config_lif
from src.model import simulate_trial
from src.recall import recall_batch
from src.stdp import window,event_pair_update

def verify(full=False):
 config=json.loads((ROOT/'configs/validated_model.json').read_text())
 timing=config['timing_ms'];params=config['window'];cfg=config_lif(config)
 assert abs(window(2,params)+1/15)<1e-14
 assert abs(window(6,params)-1/15)<1e-14
 # Four cases are generated as distinct spike event sources, with no error mask.
 x=np.zeros(15,dtype=np.int8);x[0]=1
 for target,prediction in ((0,0),(0,1),(1,0),(1,1)):
  w=np.zeros((15,15));w[0,0]=1.0 if prediction else 0.0
  y=np.zeros(15,dtype=np.int8);y[0]=target
  trial=simulate_trial(x,w,cfg,timing,teacher=y)
  assert trial['prediction'][0]==prediction
  assert trial['teacher_emitted'][0]==target
  delta,_=event_pair_update(x,trial,timing,params)
  assert abs(delta[0,0]-(target-prediction)/15)<1e-14
  assert not np.any(delta[1:])
  if target==0:assert not np.any(trial['teacher_emitted'])
 cases=[10,15,20] if full else [10]
 rows=[]
 for size in cases:
  for name in ('original','modified'):
   x,y=load(name,size)
   archive=np.load(ROOT/'results/checkpoints'/f'{name}_K{size}.npz')
   result=train(x,y,config)
   err=float(np.max(np.abs(result['weight_by_epoch']-archive['weight_by_epoch'])))
   assert err<1e-12,(name,size,err)
   if 'final_clocked_prediction' in archive or 'final_prediction' in archive:
    archived_pred=archive['final_clocked_prediction'] if 'final_clocked_prediction' in archive else archive['final_prediction']
    assert np.array_equal(result['final_prediction'],archived_pred)
   assert np.allclose(result['exact_by_epoch'],archive['exact_by_epoch']) if 'exact_by_epoch' in archive else True
   pred,_=recall_batch(x,result['weight_by_epoch'][-1],cfg,timing)
   assert np.array_equal(pred,result['final_prediction'])
   rows.append(dict(dataset=name,K=size,max_epoch_weight_error=err,
       first_100_epoch=result['first_100_epoch'],final_exact=float(result['exact_by_epoch'][-1]),
       final_bit=float(result['bit_by_epoch'][-1]),teacher_reliability=result['teacher_emitted']/result['teacher_due'],
       status='pass'))
 return rows

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--full',action='store_true',help='also replay K15 v2 and K20, 500 epochs each')
 parser.add_argument('--output',type=Path)
 args=parser.parse_args()
 rows=verify(args.full)
 print(json.dumps(rows,ensure_ascii=False,indent=2))
 if args.output:
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
