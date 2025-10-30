import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/tomastokar/Additive-Margin-Softmax/blob/main/AMSloss.py
# https://arxiv.org/abs/1801.05599

class AMSoftmaxLoss(nn.Module):

    def __init__(self, embedding_dim, no_classes, scale=30.0, margin=0.4, reduction="none"):
        '''
        Additive Margin Softmax Loss


        Attributes
        ----------
        embedding_dim : int
            Dimension of the embedding vector
        no_classes : int
            Number of classes to be embedded
        scale : float
            Global scale factor
        margin : float
            Size of additive margin
        '''
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.no_classes = no_classes
        self.embedding = nn.Embedding(no_classes, embedding_dim, max_norm=1)
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x, onehot):
        '''
        Input shape (N, embedding_dim)
        '''
        n, m = x.shape
        assert n == len(onehot)
        assert m == self.embedding_dim
        # assert torch.min(labels) >= 0
        # assert torch.max(labels) < self.no_classes

        x = F.normalize(x, dim=1)
        w = self.embedding.weight
        cos_theta = torch.matmul(w, x.T).T
        psi = cos_theta - self.margin

        # onehot = F.one_hot(labels, self.no_classes)
        labels = onehot.argmax(dim=1)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)
        err = self.loss(logits, labels)

        # Normalize by scale so that loss magnitude ~1 (roughly BCE)
        # err = err / self.scale

        return err, logits
