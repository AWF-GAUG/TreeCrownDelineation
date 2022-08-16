import torch

class InferenceModel(torch.nn.Module):
    """Just a wrapper to apply the sigmoid activation to mask and outlines during inference.."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if len(output) == 2:
            y, metric = output
            y[:,:2] = torch.sigmoid(y[:,:2])
            return y, metric
        else:
            y = output
            y[:,:2] = torch.sigmoid(y[:,:2])
            return y
