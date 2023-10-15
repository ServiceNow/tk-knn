from . import intent, semi_loader
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from collections import defaultdict


def get_dataset(split, exp_dict):
    # Get dataset that has a 'label_map', text/label combo 'Xy', and mask 'm'

    semi_dataset = semi_loader.SemiLoader(split, exp_dict)
    dataset = semi_dataset.get_intent_dataset()
    unlabeled_dataset = semi_dataset.get_unlabeled_dataset()

    # Shuffle and Balance for train only
    if split == "train":
        if unlabeled_dataset:
            tot_len = len(dataset) + len(unlabeled_dataset)
            sampler = RandomSampler(dataset, replacement=True, num_samples=tot_len)
        else:
            sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    # Building the DataLoader
    loader = DataLoader(
        dataset,  # The training samples.
        sampler=sampler,
        batch_size=exp_dict["batch_size"],
        num_workers=8,
    )

    unlabeled_loader = None
    if (exp_dict["dataset"] == "clinc" or exp_dict["dataset"] == "banking77" or exp_dict["dataset"] == "hwu64") and unlabeled_dataset:
        #Create the unlabeled Dataloader
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            sampler=RandomSampler(unlabeled_dataset),
            batch_size=exp_dict["batch_size"],
            num_workers=0,
        )

    # Get data loader
    print("=" * 10)
    print(f"# {split}: {len(loader.dataset)}")
    print("=" * 10)

    if "contrastive" in exp_dict and exp_dict["contrastive"] and split == "train":
        contrastive_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=exp_dict["contrastive_batch_size"],
            num_workers=0,
        )

        return loader, unlabeled_loader, contrastive_loader

   

    #Return the unlabeled loader as well for the train split
    if split == "train":
        return loader, unlabeled_loader
    else:
        return loader



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def combine_data(exp_dict, train_loader, n_total, input_ids, input_masks, pseudo_labels):

    #We didn't create any new pseudo labels because of the threshold
    if len(pseudo_labels) == 0:
        return train_loader

    label_masks = torch.zeros(len(pseudo_labels))
    #Add in the original training data to the pseudo labeled data
    input_ids = torch.cat((input_ids, train_loader.dataset.input_ids))
    input_masks = torch.cat((input_masks, train_loader.dataset.input_masks))
    pseudo_labels = torch.cat((pseudo_labels, train_loader.dataset.label_ids))

    label_masks = torch.cat((label_masks, torch.ones(len(train_loader.dataset))))
    label_map = train_loader.dataset.label_map

    #create dataloader
    data_loader = generate_data_loader(exp_dict, n_total, input_ids, input_masks, pseudo_labels, label_masks, label_map, train_loader)

    return data_loader


def generate_data_loader(exp_dict, n_total, input_ids, input_masks, label_ids, label_masks, label_map, train_loader):
    """
    Generate a Dataloader given the input that we have.
    This function is used to generate a new Dataloader once we have pseudo labels
    """

    # Building the TensorDataset
    dataset = intent.Intent(input_ids, input_masks, label_ids, label_masks, label_map)

    dataset.set_tokenizer(train_loader.dataset.tokenizer)

    # If we are including augmented data into the training process.
    if "add_augmented" in exp_dict and exp_dict["add_augmented"] and len(input_ids) > n_total:
        n_total = len(input_ids)

    #Only use this function for the training data, so always random sampler
    sampler = RandomSampler(dataset, replacement=True,  num_samples=n_total)

    # Building the DataLoader
    return DataLoader(
        dataset,  # The training samples.
        sampler=sampler,
        batch_size=exp_dict["batch_size"],
    )  # Trains with this batch size.

