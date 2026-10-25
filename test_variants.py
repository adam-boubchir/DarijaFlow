
import torch
import torch.nn as nn

from backend.app.data.translation_dataset import create_dataloader
from backend.app.models.seq2seq import Seq2Seq


def main():
    loader, eng, dar = create_dataloader("train", batch_size=4)
    batch = next(iter(loader))

    for variant in ["vanilla", "wmc", "cifg"]:
        print(f"\nTest de la variante : {variant.upper()}")

        model = Seq2Seq(
            source_vocab_size=len(eng),
            target_vocab_size=len(dar),
            embedding_dim=64,
            hidden_size=128,
            cell_type=variant,
            pad_idx=0,
            sos_idx=2,
            eos_idx=3,
        )

        logits, expected = model(
            batch["source"],
            batch["source_lengths"],
            batch["target"],
        )

        criterion = nn.CrossEntropyLoss(ignore_index=0)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            expected.reshape(-1),
        )

        loss.backward()

        print("Dimensions des prédictions :", tuple(logits.shape))
        print("Loss :", round(loss.item(), 4))
        print("Forward et backward fonctionnels !")


if __name__ == "__main__":
    main()
