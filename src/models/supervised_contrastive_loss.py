import torch
import torch.nn as nn


from math import log


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, similarity="dot_product"):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362
        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.similarity = similarity

    def cosine_sim_matrix(self, a, b, eps=1e-8):
        """
        Calculate the cosine similarity between all the examples.
        """
        a_n, b_n = torch.linalg.vector_norm(a, dim=1)[:, None], torch.linalg.vector_norm(b, dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        if self.similarity == "dot_product":
            dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
            # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
            similarity_matrix = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
            )
        elif self.similarity == "cosine":
            #Calcualte the cosine similarity between all the examples
            similarity_matrix = self.cosine_sim_matrix(projections, projections) / self.temperature
            similarity_matrix = (
                torch.exp(similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0]) + 1e-5
            )


        #Create a 2d matrix has True in the matrix if the targets are the same False otherwise
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(similarity_matrix.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(similarity_matrix / (torch.sum(similarity_matrix * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss