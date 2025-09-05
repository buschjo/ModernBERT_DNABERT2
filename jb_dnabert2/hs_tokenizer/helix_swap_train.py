import argparse
import pandas as pd
from helix_swap import HelixSwapRust as hsb

def pars_ini():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_training_data', type=str, help='Data for tokenizer training')
    parser.add_argument('--tokenizer_save_path', type=str, help='Name of final tokenizer')
    return parser

def create_train_tokenizer(args):
    print('initializing helix swap')
    tokenizer=hsb.HelixSwapBpe(30,4,4096)
    print(f'training on data from {args.tokenizer_training_data}')
    tokenizer.train_from_file(args.tokenizer_training_data)
    print(f'storing tokenizer in {args.tokenizer_save_path}')
    tokenizer.save(args.tokenizer_save_path)

if __name__ == "__main__":
    print('parsing arguments...')
    parser = pars_ini()
    args = parser.parse_args()
    create_train_tokenizer(args)
