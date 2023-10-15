import torch
import io, os, tqdm
import torch.nn.functional as F
import random, exp_configs
import numpy as np
import copy
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from haven import haven_wizard as hw
from haven import haven_utils as hu
import pandas as pd

from src import datasets, models

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # Create data loader and model
    set_seed(43, exp_dict)

    # train_loader, unlabeled_set = datasets.get_dataset(split="train", exp_dict=exp_dict)
    train_loader, unlabeled_dataloader = datasets.get_dataset(split="train", exp_dict=exp_dict)
    # n_total is used to repeatedly sample when the data size is small
    if exp_dict["model"] in ["basic"] and "pseudo_labels" in exp_dict and exp_dict["pseudo_labels"]:
        n_total = len(train_loader.dataset) + len(unlabeled_dataloader.dataset)
    val_loader  = datasets.get_dataset(split="val", exp_dict=exp_dict)
    test_loader = datasets.get_dataset(split="test", exp_dict=exp_dict)
    model = models.get_model(train_set=train_loader.dataset, exp_dict=exp_dict)

    # Resume or initialize checkpoint
    cm = hw.CheckpointManager(savedir)
 
    #store the old train loader as we use it to rebuild the dataset each time
    old_train_loader = copy.deepcopy(train_loader)

    # Train and Validate
    for cycle in range(0, 31):
        score_list_cycle = []
        patience = 0
        score_dict = {
            "cycle": cycle,
        }
        for epoch in range(cm.get_epoch(), (2 * (cycle+1))):
            # Get Metrics
            score_dict = {
                "epoch": epoch,
                #add length of the dataset
                "n_train": len(train_loader.dataset),
                "cycle": cycle,
            }

            train_dict = model.train_on_loader(train_loader)

            val_dict = model.val_on_loader(val_loader)
            val_dict["val_loss"] = val_dict.pop("test_loss")
            val_dict["val_acc"] = val_dict.pop("test_acc")
            score_list_cycle.append(val_dict)

            # Best val acc - save the model and evaluate on the test set
            df = pd.DataFrame(score_list_cycle)
            if len(score_list_cycle) > 0 and val_dict["val_loss"] <= df["val_loss"].min():
                savedir_best = f"{savedir}/model_best.pth"
                torch.save(model, savedir_best)
                test_acc = model.val_on_loader(test_loader)['test_acc']
                patience = 0
            else:
                test_acc = -1
                patience += 1
            score_dict['test_acc'] = test_acc
            
            score_dict.update(train_dict)
            score_dict.update(val_dict)

            cm.log_metrics(score_dict)
            if patience > 5:
                print(f"Stopping criteria met.")
                break

        #Pseudo Labeling performed after each cycle
        if unlabeled_dataloader and exp_dict["pseudo_labels"]:

            if "reuse_train_loader" in exp_dict and exp_dict["reuse_train_loader"]:
                input_ids, input_masks, pseudo_labels, input_idxs, pseudo_label_dict = model.get_pseudo_labels(train_loader, unlabeled_dataloader, exp_dict)
            else:
                input_ids, input_masks, pseudo_labels, input_idxs, pseudo_label_dict = model.get_pseudo_labels(old_train_loader, unlabeled_dataloader, exp_dict)

            #Prevents an error that can occur when the score dict is not initalized somehow
            if "pseudo_label_dict" not in locals():
                pseudo_label_dict = {}

            #check if score_dict is initalized
            if "score_dict" not in locals():
                score_dict = {}
                
            score_dict.update(pseudo_label_dict)

            if "top-k" in exp_dict:
                if cycle == 0:
                    k_value = exp_dict["top-k"]

                exp_dict["top-k"] = exp_dict["top-k"] + k_value #Expand the top-k parameter for the next cycle

            cm.log_metrics(score_dict)
            #modify the train_loader now
            if cycle == 0:
                # train_loader = datasets.combine_data(exp_dict, train_loader, unlabeled_dataloader, pseudo_labels)
                train_loader = datasets.combine_data(exp_dict, train_loader, n_total, input_ids, input_masks, pseudo_labels)
            else:
                train_loader = datasets.combine_data(exp_dict, old_train_loader, n_total, input_ids, input_masks, pseudo_labels)

            if "adjustable_threshold" in exp_dict and exp_dict["adjustable_threshold"] == "flexmatch":
                per_class_threshold = model.per_class_threshold
                initalized = model.initalized

            #reset the model for the next iteration of self-training
            model = models.get_model(train_set=train_loader.dataset, exp_dict=exp_dict)

            if "adjustable_threshold" in exp_dict and exp_dict["adjustable_threshold"] == "flexmatch":
                model.per_class_threshold = per_class_threshold
                model.initalized = initalized
                model.per_class_sum = dict.fromkeys(list(per_class_threshold.keys()), 0)


    print("Experiment done\n")


def set_seed(seed, exp_dict):

    seed = seed + exp_dict.get("runs", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group_list",
        nargs="+",
        default="resnet",
        help="name of an experiment in exp_configs.py",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="/mnt/home/haven_output",
        help="folder where logs will be saved",
    )
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-d", "--datadir", type=str, default="./data")
    parser.add_argument("-md", "--modeldir", type=str, default=None)
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Overwrite previous results"
    )
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument(
        "-j",
        "--job_scheduler",
        type=str,
        default=None,
        help="If 1, runs in toolkit in parallel",
    )
    parser.add_argument("-v", default="results.ipynb", help="orkestrator")
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )
    parser.add_argument("--disable_wandb", default=1, type=int)

    args, unknown = parser.parse_known_args()

    file_name = os.path.basename(__file__)[:-3]  # remove .py
    hw.run_wizard(
        func=trainval,
        exp_groups=exp_configs.EXP_GROUPS,
        job_config=None,
        python_binary_path=args.python_binary,
        # python_file_path=f"-m runners.{file_name}",
        use_threads=True,
        args=args,
        results_fname="results/ssl_nlp.ipynb",
    )
