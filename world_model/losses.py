import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, use_top_k=False, top_k_ratio=1.0):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
        )

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)


class RegressionLoss(nn.Module):
    def __init__(self, norm, channel_dim=-1):
        super().__init__()
        self.norm = norm
        self.channel_dim = channel_dim

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class ProbabilisticLoss(nn.Module):
    def forward(self, present_mu, present_log_sigma, future_mu, future_log_sigma):
        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss


class KLBalancing(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss()

    def forward(self, present_mu, present_log_sigma, future_mu, future_log_sigma):
        prior_loss = self.loss(present_mu, present_log_sigma, future_mu.detach(), future_log_sigma.detach())
        posterior_loss = self.loss(present_mu.detach(), present_log_sigma.detach(), future_mu, future_log_sigma)

        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss
