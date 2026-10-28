import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.datasets import load
from src.training import config_lif
from src.model import simulate_trial
from src.stdp import window,event_pair_update
from src.recall import recall_batch

class MechanismTest(unittest.TestCase):
 def setUp(self):
  self.config=json.loads((ROOT/'configs/validated_model.json').read_text())
  self.cfg=config_lif(self.config)

 def test_four_event_cases_and_inactive_synapses(self):
  x=np.zeros(15,dtype=np.int8);x[0]=1
  for target,prediction in ((0,0),(0,1),(1,0),(1,1)):
   w=np.zeros((15,15));w[0,0]=int(prediction)
   y=np.zeros(15,dtype=np.int8);y[0]=target
   trial=simulate_trial(x,w,self.cfg,self.config['timing_ms'],teacher=y)
   self.assertEqual(trial['prediction'][0],prediction)
   self.assertEqual(trial['teacher_emitted'][0],target)
   delta,_=event_pair_update(x,trial,self.config['timing_ms'],self.config['window'])
   self.assertAlmostEqual(delta[0,0],(target-prediction)/15,places=13)
   self.assertFalse(np.any(delta[1:]))

 def test_teacher_free_recall(self):
  for name in ('original','modified'):
   x,y=load(name)
   with np.load(ROOT/'results/checkpoints'/f'{name}_K10.npz') as d:w=d['final_weight']
   pred,_=recall_batch(x,w,self.cfg,self.config['timing_ms'])
   self.assertTrue(np.array_equal(pred,y))

 def test_dataset_lineage_and_final_targets(self):
  expected_pt=np.array([
   [1,1,1,1,0,1,1,1,1,1,0,0,1,0,0],
   [1,1,1,1,0,1,1,0,1,1,1,1,0,0,1],
   [1,1,1,1,0,1,1,1,1,1,1,0,1,0,1],
   [1,1,1,1,0,0,1,1,1,0,0,1,1,1,1],
   [1,1,1,0,1,0,0,1,0,0,1,0,0,1,0]],dtype=np.int8)
  for name in ('original','modified'):
   x10,y10=load(name,10);x15,y15=load(name,15);x20,y20=load(name,20)
   self.assertTrue(np.array_equal(x10,x15[:10]))
   self.assertTrue(np.array_equal(y10,y15[:10]))
   self.assertTrue(np.array_equal(x15,x20[:15]))
   self.assertTrue(np.array_equal(y15,y20[:15]))
   self.assertTrue(np.array_equal(y20[15:20],expected_pt))
   self.assertFalse(np.array_equal(x20[17],x20[19]))
   self.assertEqual(len(set(map(tuple,x20))),20)

if __name__=='__main__':unittest.main()
