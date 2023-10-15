from transformers import (
    AutoModel,
    AutoConfig,
)

import torch, tqdm
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from . import ganbert
import pandas as pd
from collections import defaultdict
from . import supervised_contrastive_loss

device = "cuda"


class Basic:
    def __init__(self, train_set, exp_dict):
        # Load Transformer and config
        self.exp_dict = exp_dict
        model_name = "bert-base-cased"
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Load Discrimnator (used as classifier)
        num_hidden_layers_d = 1
        hidden_size = int(config.hidden_size)
        hidden_levels_d = [hidden_size for _ in range(0, num_hidden_layers_d)]
        out_dropout_rate = 0.2
        self.num_labels = len(train_set.label_map)

        self.discriminator = ganbert.Discriminator(
            input_size=hidden_size,
            hidden_sizes=hidden_levels_d,
            num_labels=self.num_labels,
            dropout_rate=out_dropout_rate,
        )

        self.transform_dropout = nn.Dropout(exp_dict.get("projection_dropout", 0.5))

        self.contrastive_hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.contrastive_output_layer = nn.Linear(hidden_size, 128)


        # Put everything in the GPU if available
        if torch.cuda.is_available():
            self.transformer.cuda()
            self.contrastive_hidden_layer.cuda()
            self.contrastive_output_layer.cuda()
            self.discriminator.cuda()

        # models parameters
        transformer_vars = [i for i in self.transformer.parameters()]
        contrastive_vars = [i for i in self.contrastive_hidden_layer.parameters()] + [i for i in self.contrastive_output_layer.parameters()]
        d_vars = transformer_vars + contrastive_vars + [v for v in self.discriminator.parameters()]

        # optimizer
        learning_rate_discriminator = 5e-5
        self.epsilon = 1e-8

        self.contrastive_optimizer = torch.optim.AdamW(transformer_vars, lr=learning_rate_discriminator)
        self.dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
        # loss
        self.nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        #contrastive loss
        self.criterion = supervised_contrastive_loss.SupervisedContrastiveLoss(similarity=exp_dict.get("contrast_mode", "dot_product"))

        #Only needed when using flexmatch
        self.initalized = False
        self.per_class_threshold = {}
        self.per_class_sum = {}

        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

    def get_state_dict(self):
        return {
            "transformer": self.transformer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "contrastive_hidden_layer": self.contrastive_hidden_layer.state_dict(),
            "dis_optimizer": self.dis_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.transformer.load_state_dict(state_dict['transformer'])
        self.discriminator.load_state_dict(state_dict['discriminator'])
        self.contrastive_hidden_layer.load_state_dict(state_dict['contrastive_hidden_layer'])
        self.dis_optimizer.load_state_dict(state_dict['dis_optimizer'])
    
    def sentence_forward(self, inputs, mask):
        # Encode real data in the Transformer
        model_outputs = self.transformer(inputs, attention_mask=mask)

        #Lets mean pool the hidden states
        hidden_states = torch.mean(model_outputs.last_hidden_state, dim=1)

        return hidden_states

    def contrastive_forward(self, inputs, mask):

        hidden_states = self.sentence_forward(inputs, mask)

        hidden_states = self.contrastive_hidden_layer(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.contrastive_output_layer(hidden_states)

        #Normalize to the unit sphere
        hidden_states = F.normalize(hidden_states, dim=1)

        return hidden_states

    def train_on_loader(self, train_dataloader):

        # Put the model into training mode.
        self.transformer.train()

        # For each batch of training data...
        loss_list = []
        for batch in tqdm.tqdm(train_dataloader, desc="Training"): 
            # Unpack this training batch from our dataloader.
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["input_masks"].to(device)
            b_labels = batch["label_ids"].to(device)

            # Encode real data in the Transformer
            model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]

            _, logits, probs = self.discriminator(hidden_states)
            filtered_logits = logits[:, 0:-1]

            self.dis_optimizer.zero_grad()
            d_loss = self.nll_loss(filtered_logits, b_labels)

            lbl_masks = batch["label_masks"]
            # At least one piece of data comes from the labeled dataset
            loss = 0.0
            if self.exp_dict.get("cross_entropy_loss", True):
                #Only do cross entropy loss if we are not using contrastive loss
                lbl_loss = 0.0
                ubl_loss = 0.0
                if lbl_masks.sum() > 0:
                    lbl_loss = d_loss[lbl_masks == 1].sum()

                # At least one piece of data is unlabeled
                if lbl_masks.mean() != 1:
                    ubl_loss = d_loss[lbl_masks == 0].sum() * self.exp_dict.get("alpha", 1)

                loss = lbl_loss + ubl_loss
                loss = loss / len(batch["input_ids"])

                loss_list += [{'lbl_loss': float(lbl_loss)/len(batch["input_ids"]), 'ubl_loss': float(ubl_loss)/len(batch["input_ids"])}]

            if self.exp_dict.get("contrastive_loss", False):
                #Run the inputs through the contrastive examples to add to the loss
                b_labels = batch["label_ids"].repeat(1,2).to(device)

                b_labels = b_labels.squeeze(0)

                hidden_states = self.contrastive_forward(b_input_ids, b_input_mask)

                #Peform a transformation on the hidden states with dropout
                transformed_hidden_states = self.transform_dropout(hidden_states)

                all_hidden_states = torch.cat((hidden_states, transformed_hidden_states), dim=0)
                contrastive_loss = self.criterion(all_hidden_states, b_labels)
                loss = loss + contrastive_loss

                loss_list += [{'contrastive_loss': float(contrastive_loss)}]

            if self.exp_dict.get("entropy_regularization", False):
                #Add entropy regularization to the loss


                #Calculate the cosine similarity matrix between all samples in the batch
                similarity_matrix = self.criterion.cosine_sim_matrix(hidden_states, hidden_states)
                #To get cosine distances, subtract 1 from the similarity matrix
                cosine_distances = 1 - similarity_matrix

                #Set the diagonal to 1 because the distance will be 0 for the same sample
                cosine_distances = cosine_distances + torch.eye(cosine_distances.shape[0]).to(device)

                #take the minimum distance for each sample
                min_distances = torch.min(cosine_distances, dim=1)[0]

                #Take the log of the minimum distance
                entropy = -torch.sum(torch.log(min_distances + 1e-8))
    
                # Take the mean over the batch
                mean_entropy = entropy / len(batch["input_ids"])

                entropy_loss = mean_entropy * self.exp_dict.get("entropy_regularization_weight", 0.1)

                loss_list += [{'entropy_loss': float(entropy_loss)}]

                # Add the entropy to the loss
                loss = loss + entropy_loss

            loss.backward()

            # Apply modifications
            self.dis_optimizer.step()

        # Calculate the average loss over all of the batches.

        return pd.DataFrame(loss_list).mean().to_dict()

    @torch.no_grad()
    def get_pseudo_labels(self, labeled_dataloader, unlabeled_dataloader, exp_dict):

        if "confidence_threshold" in exp_dict or "top-k" in exp_dict:
            return self.get_pseudo_labels_threshold(labeled_dataloader, unlabeled_dataloader, exp_dict)
        else:
            return self.get_pseudo_labels_original(unlabeled_dataloader, exp_dict)

    @torch.no_grad()
    def get_pseudo_labels_original(self, unlabeled_dataloader, exp_dict):
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.transformer.eval()  # maybe redundant
        self.discriminator.eval()

        all_preds = []
        all_input_ids = torch.empty(0, dtype=torch.int64)
        all_input_masks = torch.empty(0, dtype=torch.int64)
        all_idxs = []
        all_labels_ids = []

        for batch in tqdm.tqdm(unlabeled_dataloader, desc="Generating Pseudo Labels"):

            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["input_masks"].to(device)
            b_labels = batch["label_ids"].to(device)

            model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, probs = self.discriminator(hidden_states)

            filtered_logits = logits[:, 0:-1]

            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            
            all_input_ids = torch.cat((all_input_ids, batch["input_ids"]))
            all_input_masks = torch.cat((all_input_masks, batch["input_masks"]))
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()
            all_idxs.extend(batch["idx"])


        all_preds = torch.stack(all_preds)

        overall_label_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)

        pseudo_label_dict = {}
        pseudo_label_dict["overall_label_accuracy"] = overall_label_accuracy

        return all_input_ids, all_input_masks, all_preds, all_idxs, pseudo_label_dict

    @torch.no_grad()
    def get_pseudo_labels_threshold(self, labeled_dataloader, unlabeled_dataloader, exp_dict):
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.transformer.eval()  # maybe redundant
        self.discriminator.eval()

        #Needed for the first iteration to initalize flex match thresholds
        if "adjustable_threshold" in exp_dict and exp_dict["adjustable_threshold"] == "flexmatch" and not self.initalized:
            self.initialize_flex_match_thresholds(unlabeled_dataloader, exp_dict)
            self.initalized = True

        all_pseudo_labels = []
        all_preds = []
        all_input_ids = []
        all_input_masks = []
        all_idxs = []
        all_labels_ids = []
        all_method_label_ids = []
        entropy_losses = []

        if "top-k" in exp_dict:
            top_k_per_class = defaultdict(list)
        
        if "unbalanced" in exp_dict:
            unbalanced_predictions = []

        if "knn_labels" in exp_dict:
            labeled_representations = defaultdict(list)
            lbl_input_ids = labeled_dataloader.dataset.input_ids
            lbl_input_masks = labeled_dataloader.dataset.input_masks
            lbl_ids = labeled_dataloader.dataset.label_ids
            for input_ids, input_masks, lbl_id in tqdm.tqdm(zip(lbl_input_ids, lbl_input_masks, lbl_ids), desc="Getting latent represenations"): #Don't iterate over labeled loader! Just use the 
                input_ids = input_ids.to(device)
                input_masks = input_masks.to(device)

                input_ids = torch.unsqueeze(input_ids, dim=0)
                input_masks = torch.unsqueeze(input_masks, dim=0)

                #Old method
                if exp_dict.get("hidden_representation", "pooled") == "pooled":
                    model_outputs = self.transformer(input_ids, attention_mask=input_masks)
                    hidden_states = model_outputs[-1] #Gets the pooled output 
                #New method
                elif exp_dict.get("hidden_representation", "mean") == "mean":
                    hidden_states = self.sentence_forward(input_ids, input_masks)

                labeled_representations[lbl_id.item()].append(hidden_states[0])

        #Want overall pseudo label accuracy based on just the unlabeled data and model
        #Want pseudo label accuracy based on those accepted by the method in use.
        method_label_accuracy = 0
        for batch in tqdm.tqdm(unlabeled_dataloader, desc="Generating Pseudo Labels"):

            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["input_masks"].to(device)
            b_labels = batch["label_ids"].to(device)

            pred_confidences, preds, entropy, hidden_states = self.get_predictions_and_confidence(b_input_ids, b_input_mask)
            if exp_dict.get("hidden_representation", "mean") == "mean":
                hidden_states = self.sentence_forward(b_input_ids, b_input_mask)
            entropy_losses.append(entropy)

            all_labels_ids += b_labels.detach().cpu()
            all_pseudo_labels += preds.detach().cpu()

            batch_iterator = [dict(zip(batch,t)) for t in zip(*batch.values())]
            for b, pred_confidence, pred, hidden_state in zip(batch_iterator, pred_confidences, preds, hidden_states):

                if "top-k" in exp_dict:

                    # This section is used to experiment with top-k when giving the GT label
                    # Used to see how the model learns when it gets everything right
                    #perform top-k selection here
                    if "use_true_labels" in exp_dict and exp_dict["use_true_labels"]:
                        predicted_label = b["label_ids"].item() #This is actually the GT label
                    else:
                        predicted_label = pred.item()

                    #If we are using the knn similarity as part of the ranking decision
                    if "knn_labels" in exp_dict:

                        cos_sim_scores = []
                        #Time to do cosign similarity between labeled and unlabeled examples
                        for latent_representation in labeled_representations[predicted_label]:

                            cos_sim = self.cos(hidden_state, latent_representation)
                            cos_sim_scores.append(cos_sim.item())

                        if exp_dict["knn_labels"] == "closest":
                            sim_score = max(cos_sim_scores)
                        

                        #We are using KNN to help sample so calculate the new confidence based on model likelihood and the knn decision
                        #This is equation 3 in the paper
                        pred_confidence = (pred_confidence *  (1 - exp_dict["beta"])) + (sim_score * exp_dict["beta"])

                    class_list = top_k_per_class[predicted_label]

                    confidence_batch_tuple = (pred_confidence.item(), b, predicted_label)

                    # This section of code is for unbalanced top-k sampling
                    if "unbalanced" in exp_dict and exp_dict["unbalanced"] and len(unbalanced_predictions) < (exp_dict["top-k"] * self.num_labels):
                        #Unbalanced sampling
                        unbalanced_predictions.append(confidence_batch_tuple)
                        unbalanced_predictions = sorted(unbalanced_predictions, key=lambda x: x[0])
                    elif "unbalanced" in exp_dict and exp_dict["unbalanced"] and len(unbalanced_predictions) == (exp_dict["top-k"] * self.num_labels):
                        if confidence_batch_tuple[0] > unbalanced_predictions[0][0]:
                            unbalanced_predictions.pop(0)
                            unbalanced_predictions.insert(0, confidence_batch_tuple)
                            unbalanced_predictions = sorted(unbalanced_predictions, key=lambda x: x[0])

                    # This section of code is for top-k sampling balanced
                    if "unbalanced" not in exp_dict:
                        if len(class_list) < exp_dict["top-k"]:
                            #we don't have predictions for this class yet, so start adding them in
                            class_list.append(confidence_batch_tuple)
                            top_k_per_class[predicted_label] = sorted(class_list, key=lambda x: x[0])
                        else:
                            #compare the probabilites to only hold onto the top ones
                            if confidence_batch_tuple[0] > class_list[0][0]:
                                class_list.pop(0) #remove the first element
                                class_list.insert(0, confidence_batch_tuple)
                                top_k_per_class[predicted_label] = sorted(class_list, key=lambda x: x[0])


                elif "adjustable_threshold" in exp_dict and exp_dict["adjustable_threshold"] == "flexmatch":
                    if pred_confidence.item() > self.per_class_threshold[pred.item()]:
                        self.add_unlabeled_sample(b, all_input_ids, all_input_masks, all_idxs, all_preds, all_method_label_ids, pred, flexmatch=True)

                else:
                    # Check for only confidence threshold for pseudo labels
                    if pred_confidence.item() > exp_dict["confidence_threshold"]:
                        self.add_unlabeled_sample(b, all_input_ids, all_input_masks, all_idxs, all_preds, all_method_label_ids, pred, flexmatch=False)


        
        if "adjustable_threshold" in exp_dict and exp_dict["adjustable_threshold"] == "flexmatch":
            portion_labeled = len(unlabeled_dataloader.dataset) - len(all_preds) #used for determing if we need to warm up the threshold
            self.update_thresholds(exp_dict, portion_labeled)

        #take all the predictions, then get the top-k * self.num_labels
        if "unbalanced" in exp_dict and "top-k" in exp_dict:
            for confidence, b, pred in unbalanced_predictions:
                self.add_unlabeled_sample(b, all_input_ids, all_input_masks, all_idxs, all_preds, all_method_label_ids, torch.tensor(pred), flexmatch=False)

        if "top-k" in exp_dict and "unbalanced" not in exp_dict:
            for label_id, top_tupl in top_k_per_class.items():
                for confidence, b, pred in top_tupl:
                    self.add_unlabeled_sample(b, all_input_ids, all_input_masks, all_idxs, all_preds, all_method_label_ids, torch.tensor(label_id), flexmatch=False)

        if len(all_preds) > 0:
            method_preds = torch.stack(all_preds).numpy()
            all_labels_ids = torch.stack(all_labels_ids).numpy()
            all_pseudo_labels = torch.stack(all_pseudo_labels).numpy()
            all_method_label_ids = torch.stack(all_method_label_ids).numpy()

            overall_label_accuracy = np.sum(all_pseudo_labels == all_labels_ids) / len(all_pseudo_labels)
            method_label_accuracy = np.sum(method_preds == all_method_label_ids) / len(method_preds)
        else:
            overall_label_accuracy = 0.0
            method_label_accuracy = 0.0

        #check if we are using confidence threshold and if any exceed it
        if len(all_preds) > 0:
            all_input_ids = torch.stack(all_input_ids)
            all_input_masks = torch.stack(all_input_masks)
            all_preds = torch.stack(all_preds)

        entropy_loss = torch.cat(entropy_losses).sum() # Sum of H(X) from all steps

        #Variance across class predictions
        if len(all_preds) > 0:
            _, all_counts = np.unique(all_preds, return_counts=True)
            _, method_counts = np.unique(method_preds, return_counts=True)
            all_variance = np.var(all_counts)
            method_variance = np.var(method_counts)
        else:
            all_variance = 0.0
            method_variance = 0.0

        pseudo_label_dict = {}
        pseudo_label_dict["overall_label_accuracy"] = overall_label_accuracy
        pseudo_label_dict["method_label_accuracy"] = method_label_accuracy
        pseudo_label_dict["entropy"] = entropy_loss.item()
        pseudo_label_dict["overall_variance"] = all_variance
        pseudo_label_dict["method_variance"] = method_variance

        return all_input_ids, all_input_masks, all_preds, all_idxs, pseudo_label_dict

    @torch.no_grad()
    def get_predictions_and_confidence(self, ids, masks):

        model_outputs = self.transformer(ids, attention_mask=masks)
        hidden_states = model_outputs[-1]
        _, logits, probs = self.discriminator(hidden_states)

        probs = probs[:, 0:-1]
        logits = logits[:, 0:-1]

        entropy = self.calculate_entropy(logits, probs)

        # Accumulate the predictions and the input labels
        # Original version was using the max of the logits and not the softmax results
        confidences, preds = torch.max(probs, 1)

        #If we utilize a confidence threshold only add pseudo labels that the model is confident in
        confidences = confidences.detach().cpu()
        preds = preds.detach().cpu()

        return confidences, preds, entropy, hidden_states

    @torch.no_grad()
    def val_on_loader(self, test_dataloader):

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.transformer.eval()  # maybe redundant
        self.discriminator.eval()

        # Tracking variables
        total_test_loss = 0


        all_hidden_representations = torch.tensor([]).to(device)
        all_preds = []
        all_labels_ids = []

        # Evaluate data for one epoch
        for batch in tqdm.tqdm(test_dataloader, desc="Validating"):

            # Unpack this training batch from our dataloader.
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["input_masks"].to(device)
            b_labels = batch["label_ids"].to(device)

            model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, _ = self.discriminator(hidden_states)

            filtered_logits = logits[:, 0:-1]
            # Accumulate the test loss.
            total_test_loss += self.nll_loss(filtered_logits, b_labels).mean()

            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

            #Gather the hidden representations so we can determine the rank of the similarity matrix
            if self.exp_dict.get("hidden_representation", "pooled") == "pooled":
                model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1] #Gets the pooled output 
            elif self.exp_dict.get("hidden_representation", "mean") == "mean":
                hidden_states = self.sentence_forward(b_input_ids, b_input_mask)

            all_hidden_representations = torch.cat((all_hidden_representations, hidden_states))


        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_loss = float(avg_test_loss)

        #Calculate the rank of the similarity matrix
        all_hidden_representations = all_hidden_representations.detach().cpu().numpy()
        #convert hidden representations to pytorch tensor
        all_hidden_representations = torch.tensor(all_hidden_representations)
        similarity_matrix = self.criterion.cosine_sim_matrix(all_hidden_representations, all_hidden_representations)
        rank = np.linalg.matrix_rank(similarity_matrix)

        #Calculate the vector norm for each hidden representation
        norms = torch.linalg.vector_norm(all_hidden_representations, dim=1)
        #Calculate the mean, min, and max of the norms
        mean_norm = torch.mean(norms).item()
        min_norm = torch.min(norms).item()
        max_norm = torch.max(norms).item()

        # Record all statistics from this epoch.
        return {"test_loss": avg_test_loss, "test_acc": test_accuracy, "matrix_rank": rank, "mean_norm": mean_norm, "min_norm": min_norm, "max_norm": max_norm}


    def calculate_entropy(self, logits, probs):
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        p_log_p = logits * probs
        return -p_log_p.sum(-1)


    def add_unlabeled_sample(self, b, input_ids, input_masks, idxs, preds, method_label_ids, pred, flexmatch=False):
        if flexmatch:
            self.per_class_sum[pred.item()] += 1

        input_ids.append(b["input_ids"])
        input_masks.append(b["input_masks"])
        idxs.append(b["idx"])
        method_label_ids.append(b["label_ids"])
        preds.append(pred)

    def update_thresholds(self, exp_dict, portion_labeled):

        max_learning_effect = max(list(self.per_class_sum.values()))

        summed_learning_effect = sum(list(self.per_class_sum.values()))

        for label, flex_threshold in self.per_class_threshold.items():
            class_difficulty = self.per_class_sum[label] / max(max_learning_effect, portion_labeled - summed_learning_effect)

            self.per_class_threshold[label] = class_difficulty * exp_dict["confidence_threshold"]


    def initialize_flex_match_thresholds(self, loader, exp_dict):

        labels_list = list(loader.dataset.label_map.values())

        for label in labels_list:
            self.per_class_threshold[label] = exp_dict["confidence_threshold"]
            self.per_class_sum[label] = 0.0
