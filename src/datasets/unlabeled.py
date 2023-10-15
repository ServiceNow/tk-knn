import torch
from collections import defaultdict

class Unlabeled:
    def __init__(self, label_map):
        self.label_map = label_map
        self.texts = []
        self.input_ids = []
        self.input_masks = []
        self.label_ids = []
        self.idxs = []

    def add_example(self, text, input_ids, label, idx):
        self.texts.append(text)
        self.input_ids.append(input_ids)
        self.label_ids.append(label)
        self.idxs.append(idx)

    def convert_to_tensor(self):
        self.input_ids = torch.tensor(self.input_ids)
        self.label_ids = torch.tensor(self.label_ids)

        for sent in self.input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            self.input_masks.append(att_mask)

        self.input_masks = torch.tensor(self.input_masks)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text):

        encoded_sent = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=64,
                padding="max_length",
                truncation=True,
            )

        return encoded_sent

    def __getitem__(self, idx):

        batch_dict = {
            "idx": idx,
            "texts": self.texts[idx],
            "input_ids": self.input_ids[idx],
            "input_masks": self.input_masks[idx],
            "label_ids": self.label_ids[idx],
        }
        return batch_dict

    def __len__(self):
        return len(self.input_ids)