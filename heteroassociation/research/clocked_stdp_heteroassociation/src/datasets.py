"""Load frozen binary associations copied into this archive."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(name: str, k: int = 10):
    name=name.lower()
    if name not in ('original','modified') or not 1 <= k <= 20:
        raise ValueError('name must be original/modified and 1 <= K <= 20')
    size=10 if k<=10 else (15 if k<=15 else 20)
    path=ROOT/'datasets'/f'{name}_K{size}{"_v2" if size==15 else ""}.npz'
    with np.load(path,allow_pickle=False) as d:
        x=d['input_binary'] if 'input_binary' in d else d['X_binary']
        y=d['target_binary'] if 'target_binary' in d else d['Y_binary']
        x=np.asarray(x[:k],dtype=np.int8).copy()
        y=np.asarray(y[:k],dtype=np.int8).copy()
    assert x.shape==y.shape==(k,15)
    assert set(np.unique(x))<=set((0,1)) and set(np.unique(y))<=set((0,1))
    return x,y
