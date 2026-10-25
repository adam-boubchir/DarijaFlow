
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from backend.app.data.translation_dataset import create_dataloader
from backend.app.models.seq2seq import Seq2Seq


# Configuration du test
ROOT_DIR = Path(__file__).resolve().parents[3]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
STEPS = 100
LEARNING_RATE = 0.001

PAD_IDX = 0
SOS_IDX = 2
EOS_IDX = 3


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 60)
    print("TEST DIAGNOSTIQUE : OVERFIT SUR 8 EXEMPLES")
    print("=" * 60)
    print(f"Appareil : {DEVICE}")

    # Charger un petit batch du jeu d'entraînement
    loader, english_tokenizer, darija_tokenizer = create_dataloader(
        "train",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    batch = next(iter(loader))

    source = batch["source"].to(DEVICE)
    source_lengths = batch["source_lengths"]
    target = batch["target"].to(DEVICE)

    print(f"Source : {tuple(source.shape)}")
    print(f"Cible  : {tuple(target.shape)}")

    # Créer un modèle neuf, sans charger ni écraser de checkpoint
    model = Seq2Seq(
        source_vocab_size=english_tokenizer.vocab_size(),
        target_vocab_size=darija_tokenizer.vocab_size(),
        embedding_dim=64,
        hidden_size=128,
        cell_type="vanilla",
        pad_idx=PAD_IDX,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    expected = target[:, 1:]

    def calculate_loss():
        model.eval()
        with torch.no_grad():
            logits = model(
                source=source,
                target=target,
                source_lengths=source_lengths,
                teacher_forcing_ratio=1.0,
            )
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                expected.reshape(-1),
            )
        return loss.item()

    initial_loss = calculate_loss()
    print(f"\nLoss initiale : {initial_loss:.4f}")

    model.train()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            source=source,
            target=target,
            source_lengths=source_lengths,
            teacher_forcing_ratio=1.0,
        )

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            expected.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step % 10 == 0:
            current_loss = calculate_loss()
            print(f"Étape {step:3d}/{STEPS} | loss : {current_loss:.4f}")

    final_loss = calculate_loss()

    print("\n" + "=" * 60)
    print("RÉSULTAT")
    print("=" * 60)
    print(f"Loss initiale : {initial_loss:.4f}")
    print(f"Loss finale   : {final_loss:.4f}")
    print(f"Réduction     : {initial_loss - final_loss:.4f}")

    if final_loss < initial_loss * 0.5:
        print(
            "\nLa loss a fortement diminué : le modèle apprend "
            "sur ce petit batch."
        )
        print(
            "Il faudra ensuite examiner les données, la généralisation "
            "et la génération libre."
        )
    else:
        print(
            "\nLa loss n'a pas suffisamment diminué. "
            "Il faudra approfondir le diagnostic."
        )

    print(
        "\nTest terminé. Aucun checkpoint existant n'a été modifié."
    )


    # Évaluer les traductions en génération libre
    print("\n" + "=" * 60)
    print("TRADUCTIONS EN GÉNÉRATION LIBRE")
    print("=" * 60)

    model.eval()

    with torch.no_grad():
        for i in range(source.size(0)):
            src = source[i:i + 1]
            src_length = source_lengths[i:i + 1]

            generated_ids = model.translate(
                source=src,
                source_lengths=src_length,
                max_length=30,
            )

            # Récupérer la phrase anglaise
            source_ids = source[i].tolist()
            source_ids = [
                token for token in source_ids
                if token != PAD_IDX
            ]
            english_text = english_tokenizer.decode(source_ids)

            # Récupérer la traduction de référence
            target_ids = target[i].tolist()
            target_ids = [
                token for token in target_ids
                if token not in (PAD_IDX, SOS_IDX, EOS_IDX)
            ]
            reference_text = darija_tokenizer.decode(target_ids)

            # Convertir la sortie du modèle en texte
            if isinstance(generated_ids, torch.Tensor):
                generated_ids = generated_ids.detach().cpu()

            if generated_ids.dim() > 1:
                generated_ids = generated_ids[0]

            prediction_ids = generated_ids.tolist()
            prediction_ids = [
                token for token in prediction_ids
                if token not in (PAD_IDX, SOS_IDX, EOS_IDX)
            ]
            prediction_text = darija_tokenizer.decode(prediction_ids)

            print(f"\nExemple {i + 1}")
            print(f"Anglais   : {english_text}")
            print(f"Référence : {reference_text}")
            print(f"Prédiction: {prediction_text}")


if __name__ == "__main__":
    main()
