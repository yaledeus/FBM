import torch
from torch import nn
import numpy as np
import esm

import sys
sys.path.append('..')
from utils.bio_utils import *


esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_batch_converter = esm_alphabet.get_batch_converter()
esm_model.eval()  # disables dropout for deterministic results


### borrowed from https://github.com/facebookresearch/esm
def esm_embedding(batch_seq):
    data = [(f"protein{i}", batch_seq[i]) for i in range(len(batch_seq))]
    batch_labels, batch_strs, batch_tokens = esm_batch_converter(data)
    batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]  # (B, 1 + seq_len + 1, 1280)
    atom_representations = []
    for i, tokens_len in enumerate(batch_lens):
        atom_representations.append(token_representations[i, 1: tokens_len - 1])

    return atom_representations
