
# TK-KNN: A Balanced Distance-Based Pseudo Labeling Approach for Semi-Supervised Intent Classification

This repo provides the source code & data of our paper [TK-KNN: A Balanced Distance-Based Pseudo Labeling Approach for Semi-Supervised Intent Classification](ADD Arxiv Link) (Findings EMNLP 2023). If you use any of our code, processed data or pretrained models, please cite:
```bib
@inproceedings{zhang2021greaselm,
  title={TK-KNN: A Balanced Distance-Based Pseudo Labeling Approach for Semi-Supervised Intent Classification},
  author={Botzer, Nicholas and Vasquez, David and Weninger, Tim and Laradji, Issam},
  booktitle={Findings of EMNLP 2023},
  year={2023}
}
```


### 1. Install Requirements

Please install all the dependencies below to run TK-KNN. This install setup will get the code running, but we also provide a requirements.txt file if you wish
to install the dependencies that way.

```bash
conda create -y -n tk-knn python=3.9.16
conda activate tk-knn
pip install torch==2.1.0
pip install tqdm
pip install numpy==1.26
pip install --upgrade git+https://github.com/haven-ai/haven-ai@37efcf6
pip install transformers==4.34.0
pip install datasets==2.4.0
```

### 2. Train & Validate

We have setup some commands to run a few individual experiments. These experiments will run training and testing
for CLINC150, Banking77, and Hwu64 in the 1% labeled data scenario with our method. 

Run TK-KNN for CLINC at 1% with the following command.

```python
python trainval.py -e tk_knn_clinc_one_percent -sb results -r 1 
```

Argument Descriptions:
```
-e  [Experiment group to run like `tk_knn_clinc_one_percent` which is defined in `exp_configs.py`] 
-sb [Directory where the experiments are saved]
-r  [Flag for whether to reset the experiments]
```

For other baseline comparisons for CLINC at 1% we provide the following experiment groups. Simply pass these
instead for the -e parameter to run them.

```
threshold_clinc_one_percent
flexmatch_clinc_one_percent
ganbert_clinc_one_percent
```

In exp_configs.py experiment groups can be found that we used to run all methods we used for evaluation in the paper. 


### 3. Results

Open `results/ssl_nlp.ipynb` and visualize the results as follows.

<p align="center" width="100%">
<img width="100%" src="https://raw.githubusercontent.com/haven-ai/haven-ai/master/docs/vis.gif">
</p>



