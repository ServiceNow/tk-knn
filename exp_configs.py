EXP_GROUPS = {}


# Baseline Experiments to run from the paper

# Example Experiments

# First three experiment shows how to run our method with each of the datasets.
# Note that when running these experiments the batch_size may need to be adjusted down depending on the size of your GPU. We used a value of 256
# for the scores reported in the paper.
EXP_GROUPS["tk_knn_clinc_one_percent"] = [
    {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 128, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": 0.01, "runs": 0, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": "mean", "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
]
EXP_GROUPS["tk_knn_banking_one_percent"] = [
    {"model": "basic", "dataset": "banking77", "batch_size": 128, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": 0.01, "runs": 0, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": "mean", "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
]
EXP_GROUPS["tk_knn_hwu64_one_percent"] = [
    {"model": "basic", "dataset": "hwu64", "batch_size": 128, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": 0.01, "runs": 0, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": "mean", "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
]

# These three experiments are evaluating clinc at 1% of labeled data for three different methods Thresholding, FlexMatch, and GAN-BERT
EXP_GROUPS["threshold_clinc_one_percent"] = [
    {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": 0.01, "runs": 0, "pseudo_labels": True, "confidence_threshold": 0.95},
]

EXP_GROUPS["flexmatch_clinc_one_percent"] = [
    {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": 0.01, "pseudo_labels": True, "runs": 0, "confidence_threshold": 0.95, "adjustable_threshold": "flexmatch"},
]

EXP_GROUPS["ganbert_clinc_one_percent"] = [
    {"model": "ganbert", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": 0.01, "runs": 0, "pseudo_labels": False},
]

# Experiment group to run training of a model with cross entropy on the entire dataset
EXP_GROUPS["supervised_all_data"] =[
    {"model": "basic", "dataset": "banking77", "batch_size": 64, "pseudo_labels": False, "percent_labeled": 1.0, "runs": 0},
    #Hwu64
    {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": False, "percent_labeled": 1.0, "runs": 0},
    #clinc
    {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": 1.0, "runs": 0, "pseudo_labels": False},
]


# Below experiment groups are defined to run the full suite of experiments and ablations that we did in the paper.
# If you go to run these I would recommend to chose a specific experiment configuration to evaluate. If you go to run them all
# you'll need to use a lot of GPU hours as we reported in the paper.
EXP_GROUPS["supervised_baselines"]= []
EXP_GROUPS["pseudo_label_baselines"]= []
EXP_GROUPS["pseudo_label_threshold_baselines"]= []
EXP_GROUPS["flexmatch_baseline"] = []
EXP_GROUPS["top_k_baselines"] = []
EXP_GROUPS["top_k_unbalanced"] = []
EXP_GROUPS["knn_top_k_unbalanced"] = []
EXP_GROUPS["knn_top_k_baselines"] = []
EXP_GROUPS["top_k_best_possible"] = []

#Experiment groups for performing the ablations
EXP_GROUPS["top_k_unbalanced_CE_Contrastive"] = [] 
EXP_GROUPS["top_k_unbalanced_CE_entropy"] = [] 
EXP_GROUPS["top_k_unbalanced_CE_Contrastive_entropy"] = [] 
EXP_GROUPS["top_k_balanced_CE_Contrastive"] = [] 
EXP_GROUPS["top_k_balanced_CE_entropy"] = [] 
EXP_GROUPS["top_k_balanced_CE_Contrastive_entropy"] = [] 

EXP_GROUPS["knn_top_k_unbalanced_CE_Contrastive"] = [] 
EXP_GROUPS["knn_top_k_unbalanced_CE_entropy"] = [] 
EXP_GROUPS["knn_top_k_unbalanced_CE_Contrastive_entropy"] = [] 
EXP_GROUPS["knn_top_k_balanced_CE"] = [] 
EXP_GROUPS["knn_top_k_balanced_CE_Contrastive"] = [] 
EXP_GROUPS["knn_top_k_balanced_CE_entropy"] = [] 
EXP_GROUPS["knn_top_k_balanced_CE_Contrastive_entropy"] = [] #This is our new method.

EXP_GROUPS["contrastive_loss_entropy_beta_experiments"] = []
EXP_GROUPS["beta_ablation_experiments"] = []
EXP_GROUPS["k_value_experiments"] = []

beta_experiment_values = [0.0, 0.25, 0.5, 0.75, 1.0]
k_value_experiments = [4,6,8]
# If you want to run the experiment over 5 seeds use this version of runs
#runs = [0, 10, 20, 30, 40] 

#Single seed experiment
runs = [0]

for beta in beta_experiment_values:
    for run in runs:
        EXP_GROUPS["beta_ablation_experiments"] += [
            {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": 0.01, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": beta, "contrast_mode": "cosine",  "hidden_representation": "mean", "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
        ]

for k in k_value_experiments:
    for run in runs:
        EXP_GROUPS["k_value_experiments"] += [
            {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": 0.01, "runs": run, "top-k": k, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": "mean", "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
        ]


#Change these values to different experiments and ablations
k_values = [6]
beta_values = [0.75]
low_percents = [0.01, 0.02]
percent_labeled = [0.01, 0.02, 0.05, 0.10]
contrastive_batch_size = [64, 128, 256]
percent_labeled_supervised = percent_labeled + [1.0]
hidden_representation = ["mean"]



#Ablation Experiments to test the different aspects of our method(CE loss, CON loss, Entropy Regularization, top-k, knn)
for percent in low_percents:
    for representation in hidden_representation:
        for run in runs:

            #top-k ablations below
            EXP_GROUPS["top_k_unbalanced_CE_Contrastive"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
            ]

            EXP_GROUPS["top_k_unbalanced_CE_entropy"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]

            EXP_GROUPS["top_k_unbalanced_CE_Contrastive_entropy"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]

            EXP_GROUPS["top_k_balanced_CE_Contrastive"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
            ]

            EXP_GROUPS["top_k_balanced_CE_entropy"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]

            EXP_GROUPS["top_k_balanced_CE_Contrastive_entropy"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]

            #knn ablations below
            EXP_GROUPS["knn_top_k_unbalanced_CE_Contrastive"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
            ]

            EXP_GROUPS["knn_top_k_unbalanced_CE_entropy"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]

            EXP_GROUPS["knn_top_k_unbalanced_CE_Contrastive_entropy"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "unbalanced": True, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]

            EXP_GROUPS["knn_top_k_balanced_CE"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},

            ]

            EXP_GROUPS["knn_top_k_balanced_CE_Contrastive"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2},
            ]

            EXP_GROUPS["knn_top_k_balanced_CE_entropy"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": False, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]

for percent in percent_labeled:
    for representation in hidden_representation:
        for beta in beta_values:
            EXP_GROUPS["contrastive_loss_entropy_beta_experiments"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "contrastive_loss": False, "joint_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": 50, "top-k": 6, "knn_labels": "closest", "beta": beta, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "contrastive_loss": False, "joint_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": 50, "top-k": 6, "knn_labels": "closest", "beta": beta, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "contrastive_loss": False, "joint_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": 50, "top-k": 6, "knn_labels": "closest", "beta": beta, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]
        for run in runs:
            EXP_GROUPS["knn_top_k_balanced_CE_Contrastive_entropy"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "banking77", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 256, "cross_entropy_loss": True, "contrastive_loss": True, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": 6, "knn_labels": "closest", "beta": 0.75, "contrast_mode": "cosine",  "hidden_representation": representation, "projection_dropout": 0.2, "entropy_regularization": True, "entropy_regularization_weight": 0.1},
            ]


for percent in percent_labeled_supervised:
    for run in runs:
        EXP_GROUPS["supervised_baselines"] += [
            #Banking77
            {"model": "basic", "dataset": "banking77", "batch_size": 64, "pseudo_labels": False, "percent_labeled": percent, "runs": run},
            {"model": "ganbert", "dataset": "banking77", "batch_size": 64, "pseudo_labels": False, "percent_labeled": percent, "runs": run},
            #Hwu64
            {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": False, "percent_labeled": percent, "runs": run},
            {"model": "ganbert", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": False, "percent_labeled": percent, "runs": run},
            #clinc
            {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": percent, "runs": run, "pseudo_labels": False},
            {"model": "ganbert", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": percent, "runs": run, "pseudo_labels": False},
        ]

for percent in percent_labeled:
    for run in runs:
        EXP_GROUPS["pseudo_label_baselines"] += [
            #clinc
            {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": percent, "runs": run, "pseudo_labels": True},
            #banking77
            {"model": "basic", "dataset": "banking77", "batch_size": 64, "percent_labeled": percent, "runs": run,  "pseudo_labels": True},
            #hwu64
            {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run},
        ]

        EXP_GROUPS["pseudo_label_threshold_baselines"] += [
            #banking77
            {"model": "basic", "dataset": "banking77", "batch_size": 64, "percent_labeled": percent, "runs": run,  "pseudo_labels": True, "confidence_threshold": 0.95},
            #hwu64
            {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": True, "confidence_threshold": 0.95, "percent_labeled": percent, "runs": run},
            #clinc
            {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": percent, "runs": run, "pseudo_labels": True, "confidence_threshold": 0.95},
        ]
        EXP_GROUPS["flexmatch_baseline"] += [
            #banking77
            {"model": "basic", "dataset": "banking77", "batch_size": 64, "percent_labeled": percent, "runs": run,  "pseudo_labels": True, "confidence_threshold": 0.95, "adjustable_threshold": "flexmatch"},
            #hwu64
            {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": True, "confidence_threshold": 0.95, "percent_labeled": percent, "runs": run, "adjustable_threshold": "flexmatch"},
            #clinc
            {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "percent_labeled": percent, "runs": run, "pseudo_labels": True, "confidence_threshold": 0.95, "adjustable_threshold": "flexmatch"},
        ]

for percent in percent_labeled:
    for k in k_values:
        for run in runs:
            EXP_GROUPS["top_k_baselines"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k},
                {"model": "basic", "dataset": "banking77", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k},
            ]

            EXP_GROUPS["top_k_unbalanced"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "unbalanced": True},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "unbalanced": True},
                {"model": "basic", "dataset": "banking77", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "unbalanced": True},
            ]

            EXP_GROUPS["top_k_best_possible"] += [
                {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "use_true_labels": True},
                {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "use_true_labels": True},
                {"model": "basic", "dataset": "banking77", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "use_true_labels": True},
            ]

for percent in percent_labeled:
    for k in k_values:
        for beta in beta_values:
            for run in runs:
                EXP_GROUPS["knn_top_k_baselines"] += [
                    {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "knn_labels": "closest", "beta": beta},
                    {"model": "basic", "dataset": "banking77", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "knn_labels": "closest", "beta": beta},
                    {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "knn_labels": "closest", "beta": beta},
                ]

                EXP_GROUPS["knn_top_k_unbalanced"] += [
                    {"model": "basic", "dataset": "hwu64", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "knn_labels": "closest", "beta": beta, "unbalanced": True},
                    {"model": "basic", "dataset": "banking77", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "knn_labels": "closest", "beta": beta, "unbalanced": True},
                    {"model": "basic", "dataset": "clinc", "subset": "plus", "batch_size": 64, "pseudo_labels": True, "percent_labeled": percent, "runs": run, "top-k": k, "knn_labels": "closest", "beta": beta, "unbalanced": True},
                ]
