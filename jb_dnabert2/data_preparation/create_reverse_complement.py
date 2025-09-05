import argparse
import pandas as pd

def pars_ini():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, help='Path and file name to load dataset')
    parser.add_argument('--target_path', type=str, help='Target folder to store the new datasets')
    parser.add_argument('--num_sequences', type=str, help='Number of sequences in dataset')
    return parser

def revert_string(line):
	return line [::-1]

def complement_string(line):
	uc_line = line.upper()
	tmp_complement = uc_line.replace('A', 'Z')
	tmp_complement = tmp_complement.replace('T', 'A')
	tmp_complement = tmp_complement.replace('Z', 'T')
	tmp_complement = tmp_complement.replace('C', 'Y')
	tmp_complement = tmp_complement.replace('G', 'C')
	tmp_complement = tmp_complement.replace('Y', 'G')
	return tmp_complement

def reverse_complement_row(line):
	return complement_string(revert_string(line))

def create_rc_dataset(args):
	num_sequences_to_load = int(int(args.num_sequences)/2)
	# load half of the data
	df = pd.read_csv(args.source_path, sep=" ", header=None, nrows=num_sequences_to_load)
	# save the half dataset
	print(f'Saving half of the dataset to {args.target_path}')
	df.to_csv(f'{args.target_path}/split_{num_sequences_to_load}.txt', header=False, index=False)

	# add a column with the reverse complement of the sequences
	df['reverse_complement']=df.apply(lambda x: reverse_complement_row(x[0]), axis=1)
	# concat the rc and original sequences to one df
	rc_split = pd.concat([df[0], df['reverse_complement']])
	# save the new rc dataset
	rc_split_path = f'{args.target_path}/rc_{args.num_sequences}.txt'
	print(f'Saving new combined (original + reverse complement) dataset to {rc_split_path}')
	rc_split.to_csv(rc_split_path, header=False, index=False)

if __name__ == "__main__":
    print('parsing arguments...')
    parser = pars_ini()
    args = parser.parse_args()
    create_rc_dataset(args)
