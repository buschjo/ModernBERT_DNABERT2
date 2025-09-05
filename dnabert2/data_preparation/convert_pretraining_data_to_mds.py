# Convert DNABERT2 Pretraining Data to Streaming Format (MDS)
# adapted from
# * https://github.com/mosaicml/streaming?tab=readme-ov-file#1-prepare-your-data (accessed: 13.03.2025)
# * https://docs.mosaicml.com/projects/streaming/en/latest/getting_started/main_concepts.html (accessed: 13.03.2025)
# * https://docs.mosaicml.com/projects/streaming/en/latest/preparing_datasets/basic_dataset_conversion.html#configuring-dataset-writing (accessed: 13.03.2025)

import argparse
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from streaming import MDSWriter

def pars_ini():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretraining_data_path', type=str, help='Path and file name to load pretraining data')
    parser.add_argument('--target_path', type=str, help='Target folder to store the converted MDS dataset')
    return parser

def create_shards(path, df, column_name):
    columns = {
        column_name: 'str'
    }

    # Save the samples as shards using MDSWriter
    with MDSWriter(out=path, columns=columns) as out:
        for i in range(len(df)):
            out.write(df.iloc[i])

    print(f'Dataset written to {path}...')

def convert_pretraining_data(args):
	column_name = 'text'
	df = pd.read_csv(args.pretraining_data_path, sep=" ", names=[column_name])
	create_shards(args.target_path, df, column_name)

if __name__ == "__main__":
    print('parsing arguments...')
    parser = pars_ini()
    args = parser.parse_args()
    convert_pretraining_data(args)
