import torch
import warnings


class AveragingModel(torch.nn.Module):
    def __init__(self, models, return_summed_variance=False):
        super().__init__()
        if len(models) == 1:
            warnings.warn("AveragingModel received only one model!")
        self.models = models if isinstance(models, torch.nn.ModuleList) else torch.nn.ModuleList(models)
        self.return_summed_variance = return_summed_variance

    def forward(self, x):
        if not self.return_summed_variance:
            y = self.models[0](x.clone())
            if len(self.models) > 1:
                for m in self.models[1:]:
                    y += m(x.clone())  # always make a clone in case the model changes its input
            y /= len(self.models)
            return y
        else:
            preds = torch.concat([m(x.clone()) for m in self.models])
            variances = torch.sum(torch.var(preds, dim=0),dim=(1,2)).detach()  # this should be a rough estimate of model agreement

            return torch.mean(preds, dim=0, keepdim=True), variances
