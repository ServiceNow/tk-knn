from torch.utils.data import Dataset
import pickle


"""
This dataset serves as a generic wrapper for the format of datasets to be used.
Useful when merging datasets together. Or to return after succesfully preprocessing a dataset of choice.
"""
class Intent(Dataset):
    def __init__(self, input_ids, input_masks, label_ids, label_masks, label_map):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.label_ids = label_ids
        self.label_masks = label_masks
        self.label_map = label_map

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
            "input_ids": self.input_ids[idx],
            "input_masks": self.input_masks[idx],
            "label_ids": self.label_ids[idx],
            "label_masks": self.label_masks[idx],
        }
        return batch_dict

    def __len__(self):
        return len(self.input_ids)