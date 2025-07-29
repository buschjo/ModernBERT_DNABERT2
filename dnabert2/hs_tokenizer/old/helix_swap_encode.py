import helix_swap_bpe as hsb

#x=hsb.HelixSwapBpe(30,4,4096)
#x.train_from_file("train_sub3m.txt")

# Load encoding
trained_encoding = hsb.helixswap_loader("helixswap_trained.json")
sequences = ["AATTATATATATATATATATGGCCCACACggacaaaaaa","AGGAGTATAVGGACCAGATTGCAC"]

tokenized = trained_encoding.encode(sequences)
print(tokenized)

print(trained_encoding.decode(tokenized[0]))
print(trained_encoding.decode(tokenized[1]))

