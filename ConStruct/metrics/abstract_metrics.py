import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError

from ConStruct.utils import PlaceHolder


class TrainAbstractMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred, masked_true, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self, current_epoch, local_rank):
        return None


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMSE(MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
        """Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
        tensors.
            preds: Predicted tensor
            target: Ground truth tensor
        """
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff)
        n_obs = preds.shape[0]
        return sum_squared_error, n_obs


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction="sum")
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
        target: Ground truth values     (bs * n, d) or (bs * n * n, d)."""
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction="sum")
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class ProbabilityMetric(Metric):
    def __init__(self):
        """This metric is used to track the marginal predicted probability of a class during training."""
        super().__init__()
        self.add_state("prob", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        self.prob += preds.sum()
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples


class XKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.X, target.X)


class ChargesKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        if preds.charges is None:
            return
        super().update(preds.charges, target.charges)


class EKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.E, target.E)


class YKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.y, target.y)


class XLogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.X * preds.X.log())


class ChargesLogp(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.charges * preds.charges.log())


class ELogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.E * preds.E.log())


class YLogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.y * preds.y.log())
