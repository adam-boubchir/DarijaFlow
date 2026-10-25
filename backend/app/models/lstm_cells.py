# backend/app/models/lstm_cells.py

import torch
import torch.nn as nn


class VanillaLSTMCell(nn.Module):
    """
    Cellule Vanilla LSTM personnalisée.

    Équations :
        i_t = sigmoid(W_xi*x_t + W_hi*h_prev + b_i)
        f_t = sigmoid(W_xf*x_t + W_hf*h_prev + b_f)
        o_t = sigmoid(W_xo*x_t + W_ho*h_prev + b_o)
        g_t = tanh(W_xg*x_t + W_hg*h_prev + b_g)

        c_t = f_t*c_prev + i_t*g_t
        h_t = o_t*tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Transformation de l'entrée pour les 4 portes.
        self.input_linear = nn.Linear(input_size, 4 * hidden_size)

        # Transformation de l'état caché précédent.
        self.hidden_linear = nn.Linear(
            hidden_size,
            4 * hidden_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ):
        # Calcul commun des quatre portes.
        gates = self.input_linear(x) + self.hidden_linear(h_prev)

        # Séparation en quatre parties : i, f, o et g.
        i, f, o, g = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Mise à jour de la mémoire.
        c = f * c_prev + i * g

        # Mise à jour de l'état caché.
        h = o * torch.tanh(c)

        return h, c


class WMCLSTMCell(nn.Module):
    """
    Cellule WMC LSTM personnalisée.

    Cette implémentation ajoute des connexions de l'état mémoire
    aux portes d'entrée, d'oubli et de sortie.

    Les projections de mémoire sont transformées par tanh.
    La porte de sortie utilise la nouvelle mémoire c_t.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Porte d'entrée.
        self.x_i = nn.Linear(input_size, hidden_size)
        self.h_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_i = nn.Linear(hidden_size, hidden_size, bias=False)

        # Porte d'oubli.
        self.x_f = nn.Linear(input_size, hidden_size)
        self.h_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_f = nn.Linear(hidden_size, hidden_size, bias=False)

        # Porte de sortie.
        self.x_o = nn.Linear(input_size, hidden_size)
        self.h_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidat de mémoire.
        self.x_g = nn.Linear(input_size, hidden_size)
        self.h_g = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ):
        # Porte d'entrée : utilise l'entrée, l'état caché
        # précédent et la mémoire précédente.
        i = torch.sigmoid(
            self.x_i(x)
            + self.h_i(h_prev)
            + torch.tanh(self.c_i(c_prev))
        )

        # Porte d'oubli.
        f = torch.sigmoid(
            self.x_f(x)
            + self.h_f(h_prev)
            + torch.tanh(self.c_f(c_prev))
        )

        # Candidat de contenu à mémoriser.
        g = torch.tanh(
            self.x_g(x) + self.h_g(h_prev)
        )

        # Nouvelle mémoire.
        c = f * c_prev + i * g

        # Porte de sortie : utilise la nouvelle mémoire.
        o = torch.sigmoid(
            self.x_o(x)
            + self.h_o(h_prev)
            + torch.tanh(self.c_o(c))
        )

        # Nouvel état caché.
        h = o * torch.tanh(c)

        return h, c


class CIFGLSTMCell(nn.Module):
    """
    Cellule CIFG LSTM personnalisée.

    CIFG signifie Coupled Input-Forget Gate.

    Particularité :
        i_t = 1 - f_t

    La porte d'entrée est donc liée à la porte d'oubli :
    on ne calcule pas une porte d'entrée indépendante.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Porte d'oubli.
        self.x_f = nn.Linear(input_size, hidden_size)
        self.h_f = nn.Linear(hidden_size, hidden_size, bias=False)

        # Porte de sortie.
        self.x_o = nn.Linear(input_size, hidden_size)
        self.h_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidat de mémoire.
        self.x_g = nn.Linear(input_size, hidden_size)
        self.h_g = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ):
        # Porte d'oubli.
        f = torch.sigmoid(
            self.x_f(x) + self.h_f(h_prev)
        )

        # Porte d'entrée couplée à la porte d'oubli.
        i = 1.0 - f

        # Candidat de mémoire.
        g = torch.tanh(
            self.x_g(x) + self.h_g(h_prev)
        )

        # Nouvelle mémoire.
        c = f * c_prev + i * g

        # Porte de sortie.
        o = torch.sigmoid(
            self.x_o(x) + self.h_o(h_prev)
        )

        # Nouvel état caché.
        h = o * torch.tanh(c)

        return h, c


def count_parameters(model: nn.Module) -> int:
    """Retourne le nombre total de paramètres entraînables."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )