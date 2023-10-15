from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
from . import unlabeled, intent, label_map
from itertools import combinations
import numpy as np
import math, torch, datetime
import pandas as pd

import math
import datetime
import torch.nn as nn

from torch.utils.data import Dataset


def get_label_map(label_list):
    LABEL_MAP = {}
    for (i, label) in enumerate(label_list):
        LABEL_MAP[label] = i
    return LABEL_MAP

def convert_to_dataset_hwu(file_path, label_map, sep='\t'):
    dataframe = pd.read_csv(file_path, sep=sep, lineterminator='\n')

    dataset = []
    for idx, row in dataframe.iterrows():
        example = {}

        example["text"] = row[0]
        example["label"] = label_map[row[1].strip()]

        dataset.append(example)

    return dataset

def load_banking_dataset(split):
    if split == "train":
        dataset = load_dataset('csv', data_files=f"./data/banking77/banking77_train.csv")["train"]
    elif split == "val":
        dataset = load_dataset('csv', data_files=f"./data/banking77/banking77_val.csv")["train"]
    else:
        dataset = load_dataset("banking77", split=split)
    return dataset


class SemiLoader(Dataset):
    def __init__(self, split, exp_dict):

        dataset_name = exp_dict["dataset"]

        #handle loading the different datasets
        if dataset_name == "banking77":
            self.label_map = get_label_map(label_map.BANKING77_LABELS)
            #banking77 loading
            dataset = load_banking_dataset(split)
        elif dataset_name == "clinc":
            self.label_map = get_label_map(label_map.CLINC_LABELS)
            #clinc loading
            if split == "val":
                split = "validation"
            dataset = load_dataset("clinc_oos", exp_dict["subset"], split=split)
        elif dataset_name == "hwu64":
            self.label_map = get_label_map(label_map.HWU_LABELS)
            #hwu64 loading
            dataset = convert_to_dataset_hwu(f"data/hwu64/{split}.csv", self.label_map, sep=",")

        #everything below does the setup for the semi-supervised learning
        self.exp_dict = exp_dict

        if exp_dict["model"] in ["ganbert"]:
            label_type = bool
        else:
            label_type = float

        input_ids = []
        input_mask_array = []
        label_id_array = []

        self.unlabeled_dataset = unlabeled.Unlabeled(self.label_map)
        max_seq_length = 64
        if split == "train" and "percent_labeled" in exp_dict:
            label_masks = split_percent_labeled(dataset, exp_dict["percent_labeled"], label_type)
        else:
            label_masks = np.ones(len(dataset), dtype=label_type)

        model_name = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        for idx, example in enumerate(tqdm(dataset, desc=f"Tokenizing {dataset_name}  dataset")):
            text = example["text"]
            if "label" in example:
                label = example["label"]
            elif "intent" in example:
                label = example["intent"]

            encoded_sent = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
            )

            #When running GAN-BERT always include the unlabeled examples in the training
            if (split == "train" and label_masks[idx]) or exp_dict["model"] in ["ganbert"]:
                input_ids.append(encoded_sent)
                label_id_array.append(label)

            elif split == "val" or split =="validation" or split == "test":
                input_ids.append(encoded_sent)
                label_id_array.append(label)
                
            else: # we have an unlabeled example and are not using GAN-BERT
                #add these instances to the unlabeled clinc dataset
                self.unlabeled_dataset.add_example(text, encoded_sent, label, idx)


        # Attention to token (to ignore padded input wordpieces)
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            input_mask_array.append(att_mask)
        

        self.input_ids = torch.tensor(input_ids)
        self.input_masks = torch.tensor(input_mask_array)
        self.label_ids = torch.tensor(label_id_array, dtype=torch.long)
        self.label_masks = torch.tensor(label_masks)

        if split == "train":
            self.unlabeled_dataset.convert_to_tensor()
            self.unlabeled_dataset.set_tokenizer(self.tokenizer)


    def get_unlabeled_dataset(self):
        if len(self.unlabeled_dataset) > 0:
            return self.unlabeled_dataset
        else:
            return None

    def get_intent_dataset(self):

        dataset = intent.Intent(self.input_ids, self.input_masks, self.label_ids, self.label_masks, self.label_map)
        dataset.set_tokenizer(self.tokenizer)

        return dataset


def split_percent_labeled(dataset, percent_labeled, label_type): 
    """
    Sample the labels for each intent based on the percentage of labeled data used.
    Get the list of all indexes for each intent. Then, randomly sample based on the size to 
    get a list of all the indexes that we will have a label for. Update the label mask accordingly
    to reflect the change.

    Label Mask is set to True if have a label for it. False Otherwise
    """
    mask_value = 1.0
    if label_type == bool:
        mask_value = True

    intent_to_idx = defaultdict(list)
    label_mask = np.zeros(len(dataset), dtype=label_type)

    #sample from each list of idx
    #each of those idx's gets a label mask of True
    for idx, example in enumerate(dataset):
        if "label" in example:
            label = example["label"]
        elif "intent" in example:
            label = example["intent"]
        intent_to_idx[label].append(idx)

    for intent, examples in intent_to_idx.items():

        sample_size = math.ceil(len(examples) * percent_labeled) #get the number of samples that should be labeled.
        sampled_idxs = np.random.choice(examples, sample_size, replace=False)
        # sampled_idxs = random.sample(examples, sample_size)
        for idx in sampled_idxs:
            label_mask[idx] = mask_value

    return label_mask