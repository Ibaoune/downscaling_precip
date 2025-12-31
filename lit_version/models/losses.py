import torch
import torch.nn as nn
import torch.nn.functional as F

class BernoulliGammaLoss(nn.Module):
    def __init__(self):
        super(BernoulliGammaLoss, self).__init__()
        self.eps = 1e-6
        self.tol = 1e-5
        
    def forward(self, y_pred, y_true):
        ocurrence = y_pred[:, 0, :, :].clamp(self.tol, 1-self.tol)
        shape_parameter = y_pred[:, 1, :, :].clamp(-5, 5).clamp(self.tol, 1e3)
        scale_parameter = y_pred[:, 2, :, :].clamp(-5, 5).clamp(self.tol, 1e3)
        bool_rain = torch.where(y_true > 0, torch.tensor(1.0), torch.tensor(0.0))
        # check if any of these tensors contain NaN values
        if torch.isnan(ocurrence).any():
            print("NaN values found in ocurrence tensor")
        if torch.isnan(shape_parameter).any():
            print("NaN values found in shape_parameter tensor")
        if torch.isnan(scale_parameter).any():
            print("NaN values found in scale_parameter tensor")
        #print(f"Valeur de epsilon : {epsilon}")

        # Calcul de la perte en combinant les différentes parties
        loss = -torch.mean(
                (1 - bool_rain) * torch.log(1 - ocurrence + self.eps) +  # (1 - y) * log(1 - p)
                bool_rain * (torch.log(ocurrence + self.eps) +           # + y * log(p)
                (shape_parameter - 1) * torch.log(y_true + self.eps) -   # + y * (shape - 1) * log(y)
                shape_parameter * torch.log(scale_parameter + self.eps) - # - y * shape * log(scale)
                torch.lgamma(shape_parameter + self.eps) -               # - y * log(Gamma(shape))
                y_true / (scale_parameter + self.eps))                   # - y * (y / scale)
            )
        print(f"Current loss: {loss.item()}")
        return loss
