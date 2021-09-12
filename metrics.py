import torch


class Accuracy:
    """
    Computes how often predictions equals true labels.
    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of The predicted values. shape = [batch_size, d0, .., dN]
        threshold: Threshold value for binary or multi-label logits. default: `0.5`
        from_logits: If the predictions are logits/probabilites or actual labels. default: `True`
            * `True` for Logits
            * `False` for Actual labels
    Returns:
        Tensor of Accuracy metric
    """

    def __init__(self, threshold: float = 0.5, from_logits: bool = True):
        self.threshold = threshold
        self.from_logits = from_logits

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            y_pred, y_true = self._conversion(y_pred, y_true, self.threshold)
        return torch.mean((y_pred == y_true).float())

    def _conversion(self, y_pred, y_true, threshold):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"y_pred and y_true must have the same shape got y_pred {y_pred.shape} and y_true {y_true.shape}")

        if len(y_pred.shape) == len(y_true.shape) and y_pred.dtype == torch.float:
            y_pred = (y_pred >= threshold).float()

        return y_pred, y_true

def accuracy(y_pred,y_true):
    acc = Accuracy() #

    accuracy = acc(y_pred,y_true)
    return accuracy

if __name__ == '__main__':
    y_pred = torch.Tensor([0.5,0.6,0.1,0.45,0.95])
    y_true = torch.Tensor([1,0,1,0,1])
    print(accuracy(y_pred, y_true))