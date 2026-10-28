import csv
import hashlib
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class ArtifactTest(unittest.TestCase):
 def test_selected_file_hashes(self):
  with (ROOT/'artifact_manifest.csv').open() as stream:
   rows=list(csv.DictReader(stream))
  self.assertEqual(len(rows),38)
  for row in rows:
   self.assertNotEqual(row['status'],'MISSING')
   path=ROOT/row['archive_file']
   self.assertTrue(path.is_file(),path)
   self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['archive_sha256'])

 def test_reported_k20_results(self):
  with (ROOT/'results/csv/k20_capacity_accuracy_summary.csv').open() as stream:
   records=list(csv.DictReader(stream))
  with (ROOT/'results/csv/k20_optimal_readout_results.csv').open() as stream:
   optimum=list(csv.DictReader(stream))
  for dataset,final,best,maximum in (('Original',55,60,18),('Modified',70,75,19)):
   row=next(r for r in records if r['dataset']==dataset and r['K']=='20' and r['order_name']=='ascending')
   opt=next(r for r in optimum if r['dataset']==dataset and r['K']=='20')
   self.assertAlmostEqual(float(row['stdp_final_exact_percent']),final)
   self.assertAlmostEqual(float(row['stdp_best_exact_percent']),best)
   self.assertEqual(int(opt['maximum_correct']),maximum)
   self.assertEqual(opt['certified'],'True')

if __name__=='__main__':unittest.main()
