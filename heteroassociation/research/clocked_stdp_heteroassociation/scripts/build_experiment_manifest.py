"""Link frozen datasets, config, code, results and figures in a compact CSV."""
from pathlib import Path
import csv
import json

ROOT=Path(__file__).resolve().parents[1]
ART={r['archive_file']:r for r in csv.DictReader((ROOT/'artifact_manifest.csv').open())}
REG={(r['dataset'],r['K']):r for r in json.loads((ROOT/'results/regression_verification.json').read_text())}
CAP={(r['dataset'].lower(),int(r['K'])):r for r in csv.DictReader((ROOT/'results/csv/k20_capacity_accuracy_summary.csv').open()) if r['order_name']=='ascending'}
rows=[]
for size,experiment in ((10,'baseline_K10'),(15,'capacity_K15_v2'),(20,'capacity_K20_final')):
 for dataset in ('original','modified'):
  d=f'datasets/{dataset}_K{size}{"_v2" if size==15 else ""}.npz'
  c=f'results/checkpoints/{dataset}_K{size}.npz'
  reg=REG[(dataset,size)]
  figs=['results/figures/validated_stdp_window.png'] if size==10 else [f'results/figures/k{size}_capacity.png']
  rows.append(dict(experiment_name=experiment,dataset=dataset,K=size,
   dataset_file=d,dataset_sha256=ART[d]['archive_sha256'],
   source_code='src/model.py;src/stdp.py;src/training.py;src/recall.py',
   config_file='configs/validated_model.json',training_epochs=500,random_seed='none; ascending',
   final_exact_accuracy=reg['final_exact'],best_exact_accuracy=float(CAP[(dataset,size)]['stdp_best_exact_percent'])/100,
   result_file=c,figure_files=';'.join(figs),original_source_path=ART[c]['source_path'],
   validation_status=f"epoch_weights_match; max_error={reg['max_epoch_weight_error']}"))
for experiment,csvfile,figure in (
 ('input_noise_K10','results/csv/input_noise_exact_accuracy.csv','results/figures/original_noise.png;results/figures/modified_noise.png'),
 ('spike_timing_jitter_K10','results/csv/jitter_accuracy_summary.csv','results/figures/jitter_accuracy.png')):
 for dataset in ('original','modified'):
  d=f'datasets/{dataset}_K10.npz'
  rows.append(dict(experiment_name=experiment,dataset=dataset,K=10,
   dataset_file=d,dataset_sha256=ART[d]['archive_sha256'],
   source_code='see original source path in artifact_manifest.csv',
   config_file='configs/validated_model.json',training_epochs='500 for jitter; clean K10 checkpoint for noise',
   random_seed='see original methods and CSV',final_exact_accuracy='',best_exact_accuracy='',
   result_file=csvfile,figure_files=figure,original_source_path=ART[csvfile]['source_path'],
   validation_status='archived result; not rerun in cleanup'))
with (ROOT/'experiment_manifest.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
 writer.writeheader();writer.writerows(rows)
print('Wrote',len(rows),'experiment rows')

if __name__=='__main__':pass
