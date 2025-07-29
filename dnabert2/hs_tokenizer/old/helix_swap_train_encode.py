import argparse
import pandas as pd
import helix_swap_bpe as hsb

def pars_ini():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_training_data', type=str, help='Data for tokenizer training')
    parser.add_argument('--tokenizer_save_path', type=str, help='Name of final tokenizer')
    parser.add_argument('--pretraining_data', type=str, help='Path to data that should be tokenized')
    parser.add_argument('--output_tokenized', type=str, help='Name of tokenized data file')
    return parser

def create_tokenizer(input_file_name, output_file_name):
    #x= hsb.TestClass(3,4)
    #print(x.a)
    print('initializing helix swap')
    tokenizer=hsb.HelixSwapBpe(30,4,4096)
    print(f'training on data from {input_file_name}')
    tokenizer.train_from_file(input_file_name)
    print(f'storing tokenizer in {output_file_name}')
    tokenizer.save(output_file_name)
    return tokenizer

def tokenize(args):
    #x=hsb.HelixSwapBpe(30,4,4096)
    #x.train_from_file("train_sub3m.txt")

    # Load tokenizer
    # tokenizer = hsb.helixswap_loader(args.tokenizer)
    tokenizer = create_tokenizer(args.tokenizer_training_data, args.tokenizer_save_path)

    # column_name = 'input_ids'
    # sequences = pd.read_csv(args.input_file, sep=" ", names=[column_name])
    # input_sequences = sequences.input_ids.to_list()
    input_sequences = ["AATTATATATATATATATATGGCCCACACggacaaaaaa","AGGAGTATAVGGACCAGATTGCAC"]
    print('Examples of input sequences:')
    print(input_sequences[0])
    print(input_sequences[1])

    tokenized = tokenizer.encode(input_sequences)
    
    print('Examples of decoded sequences')
    print(tokenizer.decode(tokenized[0]))
    print(tokenizer.decode(tokenized[1]))

    # print(f'Save tokenized sequences to {args.output_file}')
    # tokenized_sequences = pd.DataFrame({'tokenized_sequence': tokenized})
    # tokenized_sequences.to_csv(args.output_file, index=False)

if __name__ == "__main__":
	print('parsing arguments...')
	parser = pars_ini()
	args = parser.parse_args()
	tokenize(args)
