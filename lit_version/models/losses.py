import torch
import torch.nn as nn
import torch.nn.functional as F

class BernoulliGammaLoss(nn.Module):
    def __init__(self):
        super(BernoulliGammaLoss, self).__init__()
        self.eps = 1e-6
        self.tol = 1e-5
        
    def forward(self, true, pred):
        ocurrence = torch.sigmoid(pred[:, 0, :, :]).clamp(self.tol, 1-self.tol)
        shape_parameter = torch.exp(pred[:, 1, :, :].clamp(-5, 5)).clamp(self.tol, 1e3)
        scale_parameter = torch.exp(pred[:, 2, :, :].clamp(-5, 5)).clamp(self.tol, 1e3)
        bool_rain = torch.where(true > 0, torch.tensor(1.0), torch.tensor(0.0))
        #print(f"Valeur de epsilon : {epsilon}")

        # Calcul de la perte en combinant les différentes parties
        loss = (-torch.mean((1 - bool_rain) * torch.log(1 - ocurrence + self.eps) +  # Partie pour les valeurs où il ne pleut pas
                             bool_rain * (torch.log(ocurrence + self.eps) +           # Partie pour les valeurs où il pleut
                                          (shape_parameter - 1) * torch.log(true + self.eps) -
                                          shape_parameter * torch.log(scale_parameter + self.eps) -
                                          torch.lgamma(shape_parameter + self.eps) -         # Calcul du log gamma pour shape_parameter
                                          true / (scale_parameter + self.eps))))  # Terme de normalisation avec scale_parameter

        return loss
