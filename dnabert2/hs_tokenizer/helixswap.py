import helix_swap_bpe as hsb
from transformers import PreTrainedTokenizer


# Beispiel:
# from helixswap import HelixSwap

# if tokenizer_name == "HelixSwap5m":
# 	tokenizer = HelixSwap("pretrained_5m_seqs.json")
# elif tokenizer_name

class HelixSwap(PreTrainedTokenizer):
	def __init__(self,trained_tokenizer_file):
		
		self.helix_swap_pretrained = hsb.helixswap_loader(trained_tokenizer_file)
		super().__init__()

	def get_vocab(self):
		return(self.helix_swap_pretrained.vocab())
	
	def _tokenize(self,x):
		return self.helix_swap_pretrained.tokenize(x)

	def _convert_id_to_token(self,ids):
		return self.helix_swap_pretrained.decode([ids])

	def vocab_size(self):
		return len(self.helix_swap_pretrained.get_vocab())
		# requires new wheel of helixswapbpe
		#self.helix_swap_pretrainedvocab_size()

	def _convert_token_to_id(self,token):
		return self.helix_swap_pretrained.encode(token)
