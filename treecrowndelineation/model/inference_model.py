import torch

class InferenceModel(torch.nn.Module):
    """Just a wrapper to apply the sigmoid activation to mask and outlines during inference.."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        y[:,:2] = torch.sigmoid(y[:,:2])
        return y
