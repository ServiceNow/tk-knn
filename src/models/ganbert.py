from transformers import (
    AutoModel,
    AutoConfig,
)

import torch, tqdm
import torch.nn.functional as F
import numpy as np
import time
import torch.nn as nn

device = "cuda"


class GanBert:
    def __init__(self, train_set, exp_dict):
        # Get config and tokenizer
        model_name = "bert-base-cased"
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Define parameters for dis and gen
        self.noise_size = 100
        hidden_size = int(config.hidden_size)
        num_hidden_layers_g = 1
        num_hidden_layers_d = 1
        hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
        hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]
        out_dropout_rate = 0.2
        self.num_labels = len(train_set.label_map)

        # Instantiate the Generator and Discriminator
        self.generator = Generator(
            noise_size=self.noise_size,
            output_size=hidden_size,
            hidden_sizes=hidden_levels_g,
            dropout_rate=out_dropout_rate,
        )
        self.discriminator = Discriminator(
            input_size=hidden_size,
            hidden_sizes=hidden_levels_d,
            num_labels=self.num_labels,
            dropout_rate=out_dropout_rate,
        )

        # Put everything in the GPU if available
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.transformer.cuda()

        # models parameters
        transformer_vars = [i for i in self.transformer.parameters()]
        d_vars = transformer_vars + [v for v in self.discriminator.parameters()]
        g_vars = [v for v in self.generator.parameters()]

        # optimizer
        learning_rate_discriminator = 5e-5
        learning_rate_generator = 5e-5
        self.epsilon = 1e-8

        self.dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
        self.gen_optimizer = torch.optim.AdamW(g_vars, lr=learning_rate_generator)

        # loss
        self.nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def train_on_loader(self, train_dataloader):
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        tr_g_loss = 0
        tr_d_loss = 0

        # Put the model into training mode.
        self.transformer.train()
        self.generator.train()
        self.discriminator.train()

        # For each batch of training data...
        for batch in tqdm.tqdm(train_dataloader, desc="Training"):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["input_masks"].to(device)
            b_labels = batch["label_ids"].to(device)
            b_label_mask = batch["label_masks"].to(device)
            real_batch_size = len(b_input_ids)

            # Encode real data in the Transformer
            model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]

            # Generate fake data that should have the same distribution of the ones
            # encoded by the self.transformer.
            # First noisy input are used in input to the Generator
            noise = torch.zeros(
                real_batch_size, self.noise_size, device=device
            ).uniform_(0, 1)
            # Gnerate Fake data
            gen_rep = self.generator(noise)

            # Generate the output of the Discriminator for real and fake data.
            # First, we put together the output of the tranformer and the self.generator
            disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
            # Then, we select the output of the disciminator
            features, logits, probs = self.discriminator(disciminator_input)

            # Finally, we separate the self.discriminator's output for the real and fake
            # data
            features_list = torch.split(features, real_batch_size)
            D_real_features = features_list[0]
            D_fake_features = features_list[1]

            logits_list = torch.split(logits, real_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]

            probs_list = torch.split(probs, real_batch_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]

            # ---------------------------------
            #  LOSS evaluation
            # ---------------------------------
            # Generator's LOSS estimation
            g_loss_d = -1 * torch.mean(
                torch.log(1 - D_fake_probs[:, -1] + self.epsilon)
            )
            g_feat_reg = torch.mean(
                torch.pow(
                    torch.mean(D_real_features, dim=0)
                    - torch.mean(D_fake_features, dim=0),
                    2,
                )
            )
            g_loss = g_loss_d + g_feat_reg

            # Disciminator's LOSS estimation
            logits = D_real_logits[:, 0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            # The self.discriminator provides an output for labeled and unlabeled real data
            # so the loss evaluated for unlabeled data is ignored (masked)
            label2one_hot = torch.nn.functional.one_hot(b_labels, self.num_labels)
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(
                per_example_loss, b_label_mask.to(device)
            )
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            # It may be the case that a batch does not contain labeled examples,
            # so the "supervised loss" in this case is not evaluated
            if labeled_example_count == 0:
                D_L_Supervised = 0
            else:
                D_L_Supervised = torch.div(
                    torch.sum(per_example_loss.to(device)), labeled_example_count
                )

            D_L_unsupervised1U = -1 * torch.mean(
                torch.log(1 - D_real_probs[:, -1] + self.epsilon)
            )
            D_L_unsupervised2U = -1 * torch.mean(
                torch.log(D_fake_probs[:, -1] + self.epsilon)
            )
            d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

            # ---------------------------------
            #  OPTIMIZATION
            # ---------------------------------
            # Avoid gradient accumulation
            self.gen_optimizer.zero_grad()
            self.dis_optimizer.zero_grad()

            # Calculate weigth updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            g_loss.backward(retain_graph=True)
            d_loss.backward()

            # Apply modifications
            self.gen_optimizer.step()
            self.dis_optimizer.step()

            # A detail log of the individual losses
            # print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".
            #      format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U,
            #             g_loss_d, g_feat_reg))

            # Save the losses to print them later
            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()

        # Calculate the average loss over all of the batches.
        avg_train_loss_g = tr_g_loss / len(train_dataloader)
        avg_train_loss_d = tr_d_loss / len(train_dataloader)

        return {"train_loss_g": avg_train_loss_g, "train_loss_d": avg_train_loss_d}

    def get_state_dict(self):
        return {
            "transformer": self.transformer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "generator": self.generator.state_dict(),
            "dis_optimizer": self.dis_optimizer.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.transformer.load_state_dict(state_dict["transformer"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.generator.load_state_dict(state_dict["generator"])
        self.dis_optimizer.load_state_dict(state_dict["dis_optimizer"])
        self.gen_optimizer.load_state_dict(state_dict["gen_optimizer"])

    @torch.no_grad()
    def get_pseudo_labels(self, labeled_dataloader, unlabeled_dataloader):
        # Put the model in evaluation mode
        self.transformer.eval()  # maybe redundant
        self.discriminator.eval()
        self.generator.eval()

        all_preds = []

        # Evaluate data for one epoch
        for batch in tqdm.tqdm(unlabeled_dataloader, desc="Generating Pseudo Labels"):

            # Unpack this training batch from our dataloader.
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["input_masks"].to(device)

            model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, _ = self.discriminator(hidden_states)

            filtered_logits = logits[:, 0:-1]

            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()

        pseudo_label_dict = {}

        return None, None, all_preds, None, pseudo_label_dict


    @torch.no_grad()
    def val_on_loader(self, test_dataloader):
        # Put the model in evaluation mode
        self.transformer.eval()  # maybe redundant
        self.discriminator.eval()
        self.generator.eval()

        # Tracking variables
        total_test_loss = 0

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
            total_test_loss += self.nll_loss(filtered_logits, b_labels)

            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_loss = float(avg_test_loss)

        # Record all statistics from this epoch.
        return {"test_loss": avg_test_loss, "test_acc": test_accuracy}


class Generator(nn.Module):
    def __init__(
        self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1
    ):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


class Discriminator(nn.Module):
    def __init__(
        self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1
    ):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )

        self.layers = nn.Sequential(*layers)  # per il flatten
        self.logit = nn.Linear(
            hidden_sizes[-1], num_labels + 1
        )  # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs
