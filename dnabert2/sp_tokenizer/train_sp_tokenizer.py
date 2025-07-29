# https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb (accessed: 16.03.2025)
import sentencepiece as spm

def create_tokenizer():
    control_symbols = ['A', 'T', 'G', 'C']
    user_defined_symbols = '[MASK]'
    save_path = '/scratch/jbusch/ma/'
    spm.SentencePieceTrainer.train(
        input=f'{save_path}data/dnabert_2_pretrain/train.txt',
        model_prefix=f'{save_path}tokenizer/sentencePiece/sentencePiece_tokenizer_train',
        vocab_size=4096,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[CLS]',
        eos_piece='[SEP]',
        # control_symbols = control_symbols,
        user_defined_symbols = user_defined_symbols,
        )
    


if __name__ == "__main__":
    create_tokenizer()