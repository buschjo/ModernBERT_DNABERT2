# adapted from https://huggingface.co/learn/nlp-course/chapter6/2?fw=pt and https://huggingface.co/learn/nlp-course/chapter6/8 (accessed: 16.02.2025)
import argparse
from datasets import load_dataset
from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

def parsIni():
    parser = argparse.ArgumentParser(description='HF Tokenizer Training')
    parser.add_argument('--data_path', type=str,
                    help='Storage path to training data')
    parser.add_argument('--save_path', type=str,
                    help='Storage path for finished tokenizer')
    parser.add_argument('--vocab_size', type=int,
                    help='Size of final vocabulary')
    parser.add_argument('--tokenizer_name', type=str,
                    help='')
    parser.add_argument('--pre_tokenizer', type=str,
                    help='Choose between ByteLevel, Metaspace, Whitespace, WhitspaceSplit')
    return parser

# adapted from https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html (accessed: 16.02.2025)
def get_dataset(dataset_path):
    train_data = load_dataset('text', data_files={'train': f'{dataset_path}'})
    return train_data["train"]

def create_tokenizer(args):
    def get_training_corpus():
        for i in range(0, len(train_data), 1000):
            yield train_data[i : i + 1000]["text"]

    print('Load data...')
    train_data = get_dataset(args.data_path)

    print('Setting BPE model...')
    tokenizer = Tokenizer(models.BPE())

    print(f'Setting Pre-Tokenizer to {args.pre_tokenizer}')
    if args.pre_tokenizer == 'ByteLevel':
        print('This pre-tokenizer takes care of replacing all bytes of the given string ')
        print('with a corresponding representation, as well as splitting into words.')
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if args.pre_tokenizer == 'Metaspace':
        print('This pre-tokenizer replaces any whitespace by the provided replacement ')
        print('character. It then tries to split on these spaces.')
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    if args.pre_tokenizer == 'Whitespace':
        print('This pre-tokenizer simply splits using the following regex: \w+|[^\w\s]+')
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if args.pre_tokenizer == 'WhitspaceSplit':
        print('This pre-tokenizer simply splits on the whitespace. Works like .split()')
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    initial_vocabulary = ['A', 'C', 'T', 'G']
    print(f'Training the tokenizer with vocabulary size {args.vocab_size}...')
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size,
                                   special_tokens=special_tokens,
                                   initial_alphabet = initial_vocabulary)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    print('Wrap tokenizer in PreTrainedTokenizerFast for saving...')
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token= special_tokens[0],
        sep_token= special_tokens[1],
        pad_token= special_tokens[2],
        cls_token= special_tokens[3],
        mask_token= special_tokens[4]
        )

    save_path = '/scratch/jbusch/ma/tokenizer/'
    tokenizer_name = 'hf_bpe_tokenizer'
    print(f'Save tokenizer to {save_path}{tokenizer_name}_{args.pre_tokenizer}')
    wrapped_tokenizer.save_pretrained(f"{save_path}{tokenizer_name}")

if __name__ == "__main__":
    parser = parsIni()
    args = parser.parse_args()
    create_tokenizer(args)
