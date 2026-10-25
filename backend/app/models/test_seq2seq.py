# backend/app/models/test_seq2seq.py

import json
from pathlib import Path

import torch
import torch.nn as nn

from .seq2seq import Seq2Seq


ROOT_DIR = Path(__file__).resolve().parents[3]


def load_vocab_size(filename):
    path = ROOT_DIR / "artifacts" / filename
    with open(path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    return len(vocab["itos"])


def test_model(cell_type, source_vocab_size, target_vocab_size):
    batch_size = 4
    source_length = 7
    target_length = 6

    # Données fictives pour vérifier le fonctionnement du modèle.
    source = torch.randint(
        4, source_vocab_size, (batch_size, source_length)
    )
    source_lengths = torch.tensor([7, 6, 5, 4])

    # Les cibles commencent par SOS (2) et contiennent EOS (3).
    target = torch.randint(
        4, target_vocab_size, (batch_size, target_length)
    )
    target[:, 0] = 2
    target[:, -1] = 3

    model = Seq2Seq(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=32,
        hidden_size=64,
        cell_type=cell_type,
    )

    logits = model(
        source=source,
        target=target,
        source_lengths=source_lengths,
        teacher_forcing_ratio=1.0,
    )
    expected = target[:, 1:]

    assert logits.shape == (
        batch_size,
        target_length - 1,
        target_vocab_size,
    )
    assert expected.shape == (batch_size, target_length - 1)

    # Vérifie que la fonction de perte peut être calculée.
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(
        logits.reshape(-1, target_vocab_size),
        expected.reshape(-1),
    )
    assert torch.isfinite(loss)

    # Vérifie la génération gloutonne.
    predictions = model.translate(
        source[:2],
        source_lengths[:2],
        max_length=10,
    )
    assert predictions.shape[0] == 2

    # Vérifie l'absence de nn.LSTM intégré.
    assert not any(
        isinstance(module, nn.LSTM)
        for module in model.modules()
    )

    print(f"\n--- {cell_type.upper()} Seq2Seq ---")
    print("Logits :", tuple(logits.shape))
    print("Cibles attendues :", tuple(expected.shape))
    print("Perte de test :", round(loss.item(), 4))
    print("Prédictions :", tuple(predictions.shape))
    print("Paramètres entraînables :", model.count_parameters())
    print("Test réussi !")


if __name__ == "__main__":
    source_size = load_vocab_size("english_vocab.json")
    target_size = load_vocab_size("darija_vocab.json")

    print("Taille vocabulaire anglais :", source_size)
    print("Taille vocabulaire Darija :", target_size)

    for variant in ("vanilla", "wmc", "cifg"):
        test_model(variant, source_size, target_size)

    print("\nTous les modèles Seq2Seq fonctionnent correctement.")