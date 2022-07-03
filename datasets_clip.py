

import torch
from torch.utils.data import Dataset
import sys
from typing import Tuple


class ClipDataset(Dataset):
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.captions_tokens[item],
            self.masks[item],
            self.prefixes[item],
        )

    def __init__(self, data_path: str, prefix_length: int):
        """
        Args:
            data_path: path to train.pkl, result of parse_viecap.py
            prefix_length:
        """
        self.prefix_length = prefix_length
        self.max_seq_len = 128
        dt = torch.load(data_path)
        sys.stdout.flush()
        self.captions_tokens = dt["target"]
        self.captions_tokens[self.captions_tokens.eq(1)] = 0
        self.prefixes = dt["clip_embedding"].float()
        self.masks = []
        for tokens in self.captions_tokens:
            # 5 is token <pad> in tokenizer
            mask = (tokens.greater(0)).long()
            mask = torch.cat((torch.ones(prefix_length), mask))
            self.masks.append(mask)

