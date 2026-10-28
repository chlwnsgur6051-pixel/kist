"""Independent training entry point using the archived model and datasets."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.datasets import load
from src.training import train

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('dataset',choices=['original','modified'])
 parser.add_argument('--k',type=int,default=10)
 parser.add_argument('--epochs',type=int,default=500)
 parser.add_argument('--shuffle-seed',type=int)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 x,y=load(args.dataset,args.k)
 config=json.loads((ROOT/'configs/validated_model.json').read_text())
 order=np.random.default_rng(args.shuffle_seed).permutation(args.k).tolist() if args.shuffle_seed is not None else list(range(args.k))
 result=train(x,y,config,epochs=args.epochs,order=order)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(args.output,**{key:value for key,value in result.items() if isinstance(value,np.ndarray)},
     dataset=args.dataset,K=args.k,presentation_order=np.array(order),
     best_epoch=result['best_epoch'],first_100_epoch=-1 if result['first_100_epoch'] is None else result['first_100_epoch'])
 print(json.dumps(dict(dataset=args.dataset,K=args.k,epochs=args.epochs,order=order,
     best_exact=float(np.max(result['exact_by_epoch'])),best_epoch=result['best_epoch'],
     final_exact=float(result['exact_by_epoch'][-1]),final_bit=float(result['bit_by_epoch'][-1]),
     first_100_epoch=result['first_100_epoch'],teacher_reliability=result['teacher_emitted']/result['teacher_due'])))
