# backend/app/models/test_lstm_cells.py

import torch
import torch.nn as nn

from lstm_cells import (
    VanillaLSTMCell,
    WMCLSTMCell,
    CIFGLSTMCell,
    count_parameters,
)


def test_cell(cell_class, name, input_size=32, hidden_size=64):
    batch_size = 4

    # Entrée d'un pas de temps.
    x = torch.randn(batch_size, input_size)

    # États précédents initialisés à zéro.
    h_prev = torch.zeros(batch_size, hidden_size)
    c_prev = torch.zeros(batch_size, hidden_size)

    # Création de la cellule.
    cell = cell_class(input_size, hidden_size)

    # Calcul de la sortie.
    h, c = cell(x, h_prev, c_prev)

    # Vérification des dimensions.
    assert h.shape == (batch_size, hidden_size)
    assert c.shape == (batch_size, hidden_size)

    # Vérification des valeurs.
    assert torch.isfinite(h).all()
    assert torch.isfinite(c).all()

    # Vérification : aucune cellule nn.LSTM intégrée.
    assert not any(
        isinstance(module, nn.LSTM)
        for module in cell.modules()
    )

    print(f"\n--- {name} ---")
    print("Entrée :", tuple(x.shape))
    print("État caché :", tuple(h.shape))
    print("Mémoire :", tuple(c.shape))
    print("Paramètres entraînables :", count_parameters(cell))
    print("Test réussi !")


if __name__ == "__main__":
    test_cell(
        VanillaLSTMCell,
        "Vanilla LSTM",
    )

    test_cell(
        WMCLSTMCell,
        "WMC LSTM",
    )

    test_cell(
        CIFGLSTMCell,
        "CIFG LSTM",
    )

    print("\nToutes les cellules fonctionnent correctement.")