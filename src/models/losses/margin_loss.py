from torch import nn


class MarginLoss:
    def __init__(self, margin):
        self.margin = nn.MarginRankingLoss(margin=margin)

    def __call__(
        self,
        less_toxic_outputs,
        more_toxic_outputs,
        less_toxic_targets,
        more_toxic_targets,
        targets,
    ):
        margin_loss = self.margin(more_toxic_outputs, less_toxic_outputs, targets)
        return margin_loss
