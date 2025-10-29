import torch
import torch.nn.functional as F
import torch.nn as nn

class CostSensitiveClassLoss(nn.Module):
    def __init__(self, cost_matrix):
        super(CostSensitiveClassLoss, self).__init__()
        self.cost_matrix = cost_matrix

    def forward(self, predictions, targets):
        mae_loss = torch.mean(torch.abs(predictions - targets))
        cosine_loss = 1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=0).mean()
        return mae_loss + cosine_loss

if __name__ == "__main__":
    target = F.one_hot(torch.arange(0, 8) % 5, num_classes=6).to(torch.float32)
    output = torch.full([8, 6], 1.5)  # A prediction (logit)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(output, target)
    print(target)
    print(output)
    print(loss)
