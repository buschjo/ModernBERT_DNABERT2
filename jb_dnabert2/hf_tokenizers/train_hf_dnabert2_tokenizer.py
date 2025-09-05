# adapted from https://github.com/MAGICS-LAB/DNABERT_2/issues/74 (accessed: 05.09.2025)
import argparse
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

print('run python script...')

def parsIni():
    print('Parse arguments...')
    parser = argparse.ArgumentParser(description='HF Tokenizer Training')
    parser.add_argument('--data_path', type=str,
                    help='Storage path to training data')
    parser.add_argument('--save_path', type=str,
                    help='Storage path for finished tokenizer')
    parser.add_argument('--vocab_size', type=int,
                    help='Size of final vocabulary')
    parser.add_argument('--tokenizer_name', type=str,
                    help='')
    return parser

def main(args):
    print('Load prerequisites...')
    tokenizer_name = f'hf_dnabert_{args.tokenizer_name}'
    print('Load data...')
    train_data = load_dataset('text', data_files={'train': f'{args.data_path}'})

    dataset = train_data["train"]
    def get_training_corpus():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    print('Initialize tokenizer training...')
    vocab_size = args.vocab_size
    special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size = vocab_size, min_frequency=2)
    tokenizer.pre_tokenizer = Whitespace()
    print(f'Initialized tokenizer trainer with vocab size: {trainer.vocab_size} and special tokens: {trainer.special_tokens}')

    print('Start training...')
    # tokenizer.train([args.data_path], trainer)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer, length=len(dataset))
    print('Adding template post processor...')
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    print("train finished")
    print('Save tokenizer...')

    tokenizer.save(os.path.join(args.save_path,
                                 f"tokenizer_{tokenizer_name}.json"))

    # generate and save tokenizer config

    print(f'Save pretrained tokenizer ... to {args.save_path}{tokenizer_name}')
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token= special_tokens[0],
        sep_token= special_tokens[1],
        pad_token= special_tokens[2],
        cls_token= special_tokens[3],
        mask_token= special_tokens[4]
        )

    wrapped_tokenizer.save_pretrained(f"{args.save_path}{tokenizer_name}")

    print(f"tokenizer saved to {args.save_path}{tokenizer_name}")

if __name__ == "__main__":
    print('Entering...')
    parser = parsIni()
    args = parser.parse_args()
    main(args)
