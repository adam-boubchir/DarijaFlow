
import torch
import torch.nn as nn

from backend.app.data.translation_dataset import create_dataloader
from backend.app.models.seq2seq import Seq2Seq
from backend.app.training.train import DEVICE, run_epoch, evaluate_chrf


def main():
    train_loader, en_tokenizer, dr_tokenizer = create_dataloader(
        "train", batch_size=4, shuffle=True
    )
    validation_loader, _, _ = create_dataloader(
        "validation", batch_size=4, shuffle=False
    )

    # Petit sous-ensemble pour le test technique.
    train_batches = list(train_loader)[:3]
    validation_batches = list(validation_loader)[:2]

    model = Seq2Seq(
        source_vocab_size=en_tokenizer.vocab_size(),
        target_vocab_size=dr_tokenizer.vocab_size(),
        embedding_dim=64,
        hidden_size=128,
        cell_type="vanilla",
        pad_idx=0,
        sos_idx=2,
        eos_idx=3,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Appareil :", DEVICE)
    print("Test sur 3 batches d'entraînement.")

    initial_loss = run_epoch(
        model,
        train_batches,
        criterion,
        optimizer=optimizer,
    )

    validation_loss = run_epoch(
        model,
        validation_batches,
        criterion,
        optimizer=None,
    )

    chrf = evaluate_chrf(
        model,
        validation_batches,
        dr_tokenizer,
    )

    print(f"Loss entraînement : {initial_loss:.4f}")
    print(f"Loss validation : {validation_loss:.4f}")
    print(f"chrF validation : {chrf:.2f}")
    print("Test technique terminé.")


if __name__ == "__main__":
    main()
