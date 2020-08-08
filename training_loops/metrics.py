from .regimports import *


class accuracy:
    def __init__(self,reduction=True):
        self.summable=True
        self.reduction = True
    def __call__(self,preds,target):
        if isinstance(preds,torch.Tensor):
            preds = preds.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        if preds.dtype=='float64' or preds.dtype=='float32':
            preds = preds.argmax(axis=-1)
        return accuracy_score(target,preds)