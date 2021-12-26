from torch import nn


class AuxLoss:
    def __init__(self, margin):
        self.mse = nn.MSELoss()
        self.margin = nn.MarginRankingLoss(margin=margin)

    def __call__(
        self,
        less_toxic_outputs,
        more_toxic_outputs,
        less_toxic_targets,
        more_toxic_targets,
        targets,
    ):
        less_mse_loss = self.mse(more_toxic_outputs, more_toxic_targets)
        more_mse_loss = self.mse(less_toxic_outputs, less_toxic_targets)
        margin_loss = self.margin(more_toxic_outputs, less_toxic_outputs, targets)
        return margin_loss * 0.5 + (less_mse_loss + more_mse_loss) * 0.5
