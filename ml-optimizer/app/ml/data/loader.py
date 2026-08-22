import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd

def get_config(dataset_name):
    if dataset_name == 'nsl-kdd':
        import app.ml.data.nsl_kdd_config as config
    elif dataset_name == 'cicids2017':
        import app.ml.data.cicids2017_config as config
    elif dataset_name in ('unsw-nb15', 'unsw_nb15'):
        import app.ml.data.unsw_nb15_config as config
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return config

def load_tabular_data(dataset_name, split='train', samples=1000):
    config = get_config(dataset_name)
    
    # Handle naming differences
    safe_name = dataset_name.replace('-', '_')
    if dataset_name == 'unsw-nb15':
        safe_name = 'unsw_nb15'
        
    parquet_path = f"./data/{safe_name}_{split}.parquet"
    csv_path = f"./data/{dataset_name}-{split}.csv"
    
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        y_data = torch.tensor(df['label'].values, dtype=torch.long)
        df = df.drop(columns=['label'])
        x_data = torch.tensor(df.values, dtype=torch.float32)
        print(f"Label distribution for {parquet_path}:", torch.bincount(y_data))
        return TensorDataset(x_data, y_data)
    elif os.path.exists(csv_path):
        import numpy as np
        data = np.loadtxt(csv_path, delimiter=',')
        x_data = torch.tensor(data[:, :-1], dtype=torch.float32)
        y_data = torch.tensor(data[:, -1], dtype=torch.long)
        print(f"Label distribution for {csv_path}:", torch.bincount(y_data))
        return TensorDataset(x_data, y_data)

    print(f"WARNING: No data found for {dataset_name} ({split}). Falling back to synthetic.")
    x_data = torch.rand(samples, config.FEATURE_DIM)

    for group in config.CATEGORICAL_GROUPS:
        indices = torch.randint(0, len(group), (samples,))
        one_hot = torch.nn.functional.one_hot(indices, num_classes=len(group)).float()
        x_data[:, group] = one_hot

    y_data = torch.randint(0, 2, (samples,))
    print(f"Label distribution for synthetic {dataset_name}:", torch.bincount(y_data))

    return TensorDataset(x_data, y_data)

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

class StreamingParquetDataset(IterableDataset):
    def __init__(self, parquet_path, batch_size):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        
        # Read metadata to get total rows
        pf = pq.ParquetFile(parquet_path)
        self.num_rows = pf.metadata.num_rows
        
    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        for batch in pf.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()
            if 'label' in df.columns:
                y = torch.tensor(df['label'].values, dtype=torch.long)
                x = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
            else:
                y = torch.zeros(len(df), dtype=torch.long)
                x = torch.tensor(df.values, dtype=torch.float32)
            
            yield x, y

    def __len__(self):
        return self.num_rows

def get_train_loader(dataset_name='nsl-kdd', batch_size=32768):
    safe_name = dataset_name.replace('-', '_')
    parquet_path = f"./data/{safe_name}_train.parquet"
    if os.path.exists(parquet_path):
        dataset = StreamingParquetDataset(parquet_path, batch_size)
        return DataLoader(dataset, batch_size=None)
    else:
        dataset = load_tabular_data(dataset_name, split='train', samples=5000)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_test_loader(dataset_name='nsl-kdd', batch_size=1000):
    safe_name = dataset_name.replace('-', '_')
    parquet_path = f"./data/{safe_name}_test.parquet"
    if os.path.exists(parquet_path):
        dataset = StreamingParquetDataset(parquet_path, batch_size)
        return DataLoader(dataset, batch_size=None)
    else:
        dataset = load_tabular_data(dataset_name, split='test', samples=1000)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
