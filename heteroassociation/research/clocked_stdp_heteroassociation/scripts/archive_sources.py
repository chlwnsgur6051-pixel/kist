"""Copy the exact selected immutable artifacts from their original result packages.

Run only in the original research checkout. The resulting archive is standalone.
"""
from pathlib import Path
import csv
import hashlib
import shutil

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[1]
K10=Path('results/inverted_mexican_hat_refined_explainer/20261027_122711')
K15=Path('results/association_capacity_experiment/20261028_141753')
K20=Path('results/custom_K20_capacity_experiment/20261028_160342')
NOISE=Path('results/input_noise_robustness/20261027_171141')
JITTER=Path('results/spike_timing_jitter_robustness/20261027_181527')

FILES={
 'configs/validated_model.json':K10/'configs/config.json',
 'datasets/original_K10.npz':Path('my_research/reference/datasets/Original.npz'),
 'datasets/modified_K10.npz':Path('my_research/reference/datasets/B.npz'),
 'datasets/original_K15_v2.npz':K15/'custom/original_custom_K15_v2.npz',
 'datasets/modified_K15_v2.npz':K15/'custom/modified_custom_K15_v2.npz',
 'datasets/original_K20.npz':K20/'datasets/original_custom_K20.npz',
 'datasets/modified_K20.npz':K20/'datasets/modified_custom_K20.npz',
 'results/csv/k10_run_summary.csv':K10/'csv/run_summary.csv',
 'results/csv/k15_capacity_accuracy_summary.csv':K15/'csv/capacity_accuracy_summary.csv',
 'results/csv/k20_capacity_accuracy_summary.csv':K20/'csv/capacity_accuracy_summary.csv',
 'results/csv/k20_optimal_readout_results.csv':K20/'csv/optimal_readout_results.csv',
 'results/csv/k20_rank_and_feasibility.csv':K20/'csv/rank_and_feasibility.csv',
 'results/csv/k20_presentation_order_results.csv':K20/'csv/presentation_order_results.csv',
 'results/csv/k20_capacity_per_association.csv':K20/'csv/capacity_per_association.csv',
 'results/csv/input_noise_exact_accuracy.csv':NOISE/'csv/exact_accuracy_summary.csv',
 'results/csv/input_noise_bit_accuracy.csv':NOISE/'csv/bit_accuracy_summary.csv',
 'results/csv/jitter_accuracy_summary.csv':JITTER/'csv/jitter_accuracy_summary.csv',
 'results/csv/jitter_cancellation_residual.csv':JITTER/'csv/cancellation_residual_statistics.csv',
 'experiments/input_noise/methods.md':NOISE/'summary/methods.md',
 'experiments/input_noise/source_summary.md':NOISE/'summary/summary.md',
 'experiments/spike_jitter/methods.md':JITTER/'summary/methods.md',
 'experiments/spike_jitter/source_summary.md':JITTER/'summary/summary.md',
 'results/figures/validated_stdp_window.png':K10/'figures/final_window_clean.png',
 'results/figures/four_case_spike_events.png':K10/'figures/four_cases_raster_summary_refined.png',
 'results/figures/k15_capacity.png':K15/'figures/custom_capacity_comparison.png',
 'results/figures/k20_capacity.png':K20/'figures/custom_K20_capacity_comparison.png',
 'results/figures/k20_rank_capacity.png':K20/'figures/custom_K20_rank_and_capacity.png',
 'results/figures/original_k20_learning.png':K20/'figures/original_K20_accuracy_by_epoch.png',
 'results/figures/modified_k20_learning.png':K20/'figures/modified_K20_accuracy_by_epoch.png',
 'results/figures/original_noise.png':NOISE/'figures/original_exact_accuracy_vs_input_noise.png',
 'results/figures/modified_noise.png':NOISE/'figures/modified_exact_accuracy_vs_input_noise.png',
 'results/figures/jitter_accuracy.png':JITTER/'figures/exact_recall_accuracy_vs_jitter.png',
}
for name in ('original','modified'):
 FILES[f'results/checkpoints/{name}_K10.npz']=K10/f'npz/{name}_training.npz'
 FILES[f'results/checkpoints/{name}_K15.npz']=K15/f'npz/{name}_custom_K15_ascending.npz'
 FILES[f'results/checkpoints/{name}_K20.npz']=K20/f'npz/{name}_K20_ascending.npz'

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 rows=[]
 for destination,source in FILES.items():
  src=REPO/source
  if not src.is_file():
   rows.append(dict(archive_file=destination,source_path=str(source),sha256='',bytes='',status='MISSING'))
   continue
  dst=HERE/destination
  dst.parent.mkdir(parents=True,exist_ok=True)
  filtered=(destination=='results/csv/k15_capacity_accuracy_summary.csv')
  is_csv=dst.suffix=='.csv'
  if dst.exists() and not is_csv and digest(dst)!=digest(src):
   raise RuntimeError(f'Existing archive file differs: {destination}')
  shutil.copyfile(src,dst)
  if is_csv:
   with dst.open(newline='') as stream:
    source_rows=list(csv.DictReader(stream));fields=list(source_rows[0])
   selected=[row for row in source_rows if row['dataset_group']=='Custom' and 10<=int(row['K'])<=15] if filtered else source_rows
   with dst.open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(selected)
  rows.append(dict(archive_file=destination,source_path=str(source),sha256=digest(src),bytes=src.stat().st_size,status='filtered_custom_only' if filtered else ('newline_normalized' if is_csv else 'copied'),archive_sha256=digest(dst)))
 with (HERE/'artifact_manifest.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
  writer.writeheader();writer.writerows(rows)
 missing=[r for r in rows if r['status']=='MISSING']
 print(f'Copied {len(rows)-len(missing)} artifacts; missing {len(missing)}')
 for row in missing:print('MISSING',row['source_path'])

if __name__=='__main__':main()
