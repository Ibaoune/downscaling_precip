import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class BernoulliGammaLoss(nn.Module):
    def __init__(self):
        super(BernoulliGammaLoss, self).__init__()

    def forward(self, true, pred,output_shape):
        dim1 = output_shape[1] #160 #70
        dim2 = output_shape[2] #100 #100
        # Calcul de D : nombre de paramètres par exemple (ocurrence, shape, scale)
        D = pred.size(1) // 3
        #print(f"Dimension D : {D}, prédiction a {pred.size(1)} valeurs par exemple")

        # Séparation des paramètres dans la prédiction :
        # La première partie (ocurrence) est extraite et redimensionnée en forme (B, 70, 100)
        ocurrence = pred[:, :D].view(-1, dim1, dim2).clamp(min=1e-5)
        #print(f"Ocurrence (partie prédite) : {ocurrence.shape}")

        # La deuxième partie (shape_parameter) est l'exponentielle de la deuxième moitié
        shape_parameter = torch.exp(pred[:, D:2*D]).view(-1, dim1, dim2).clamp(min=1e-5)
        #print(f"Shape Parameter (exponentielle de la prédiction) : {shape_parameter.shape}")

        # La troisième partie (scale_parameter) est l'exponentielle de la troisième moitié
        scale_parameter = torch.exp(pred[:, 2*D:3*D]).view(-1, dim1, dim2).clamp(min=1e-5)
        #print(f"Scale Parameter (exponentielle de la prédiction) : {scale_parameter.shape}")

        # Calcul de bool_rain qui est un masque binaire (1 si true > 0, sinon 0)
        bool_rain = torch.where(true > 0, torch.tensor(1.0), torch.tensor(0.0))
        #print(f"Bool Rain (masque binaire) : {bool_rain.shape}")

        # Petite constante pour éviter les valeurs infinies dans les logarithmes
        epsilon = 0.000001
        #print(f"Valeur de epsilon : {epsilon}")

        # Calcul de la perte en combinant les différentes parties
        loss = (-torch.mean((1 - bool_rain) * torch.log(1 - ocurrence + epsilon) +  # Partie pour les valeurs où il ne pleut pas
                             bool_rain * (torch.log(ocurrence + epsilon) +           # Partie pour les valeurs où il pleut
                                          (shape_parameter - 1) * torch.log(true + epsilon) -
                                          shape_parameter * torch.log(scale_parameter + epsilon) -
                                          torch.lgamma(shape_parameter + epsilon) -         # Calcul du log gamma pour shape_parameter
                                          true / (scale_parameter + epsilon))))  # Terme de normalisation avec scale_parameter
        #print(f"Perte calculée : {loss.item()}")

        return loss

    
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class CNN(nn.Module):
    def __init__(self, input_shape, last_filter_map=1, output_shape=(1, 160, 170)):
        super(CNN, self).__init__()

        self.output_shape = output_shape

        # Définition des couches de convolution
        # La première couche convolutionnelle : prend l'entrée de forme (C, H, W), applique 50 filtres de taille 3x3 et ajoute un padding de 1
        self.conv1 = nn.Conv2d(input_shape[0], 50, kernel_size=3, padding=1)
        #print(f"Conv1 créée : entrée de {input_shape[0]} canaux, 50 filtres")

        # La deuxième couche convolutionnelle : prend 50 canaux en entrée et applique 25 filtres
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        #print("Conv2 créée : entrée de 50 canaux, 25 filtres")

        # La troisième couche convolutionnelle : prend 25 canaux en entrée et applique last_filter_map filtres
        self.conv3 = nn.Conv2d(25, last_filter_map, kernel_size=3, padding=1)
        #print(f"Conv3 créée : entrée de 25 canaux, {last_filter_map} filtres")

        # Aplatissement de la sortie de la dernière couche convolutionnelle (passage de (C, H, W) à un vecteur)
        self.flatten = nn.Flatten()

        # Calcul de la taille d'entrée de la couche linéaire (fully connected layer)
        # Cela dépend de la taille de l'image après la convolution et du nombre de filtres de sortie
        fc_input_size = last_filter_map * input_shape[1] * input_shape[2]
        #print(f"fc_input_size calculé : {fc_input_size}")

        # Définition des couches linéaires
        # Chaque couche linéaire prend l'entrée aplatie (de forme fc_input_size) et produit une sortie de taille np.prod(output_shape)
        self.fc2 = nn.Linear(fc_input_size, np.prod(output_shape))
        #print(f"fc2 créé : entrée de {fc_input_size}, sortie de {np.prod(output_shape)}")

        self.fc3 = nn.Linear(fc_input_size, np.prod(output_shape))
        #print(f"fc3 créé : entrée de {fc_input_size}, sortie de {np.prod(output_shape)}")

        self.fc4 = nn.Linear(fc_input_size, np.prod(output_shape))
        #print(f"fc4 créé : entrée de {fc_input_size}, sortie de {np.prod(output_shape)}")

    def forward(self, x):
        #print(f"Entrée du modèle, forme de x : {x.shape}")

        # Passage à travers les couches de convolution avec activation ReLU
        x = F.relu(self.conv1(x))
        #print(f"Après conv1, forme de x : {x.shape}")

        x = F.relu(self.conv2(x))
        #print(f"Après conv2, forme de x : {x.shape}")

        x = F.relu(self.conv3(x))
        #print(f"Après conv3, forme de x : {x.shape}")

        # Aplatissement de la sortie
        x = self.flatten(x)
        #print(f"Après flatten, forme de x : {x.shape}")

        # Passage à travers les couches linéaires avec activation sigmoïde pour la première couche
        parameter1 = torch.sigmoid(self.fc2(x))
        #print(f"Après fc2, forme de parameter1 : {parameter1.shape}")

        parameter2 = self.fc3(x)
        #print(f"Après fc3, forme de parameter2 : {parameter2.shape}")

        parameter3 = self.fc4(x)
        #print(f"Après fc4, forme de parameter3 : {parameter3.shape}")

        # Concaténation des résultats des trois couches linéaires
        outputs = torch.cat((parameter1, parameter2, parameter3), dim=1)
        #print(f"Après concaténation, forme de outputs : {outputs.shape}")

        return outputs

