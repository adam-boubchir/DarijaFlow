
import torch

from translation_dataset import create_dataloader


def main():
    loader, english_vocab, darija_vocab = create_dataloader(
        split="train",
        batch_size=4,
        shuffle=True,
    )

    batch = next(iter(loader))

    print("Taille du vocabulaire anglais :", len(english_vocab))
    print("Taille du vocabulaire Darija :", len(darija_vocab))

    print("\nDimensions des tenseurs :")
    print("Source :", batch["source"].shape)
    print("Longueurs source :", batch["source_lengths"].shape)
    print("Cible :", batch["target"].shape)
    print("Longueurs cible :", batch["target_lengths"].shape)

    print("\nIndices des quatre phrases anglaises :")
    print(batch["source"])

    print("\nIndices des quatre cibles Darija :")
    print(batch["target"])

    assert batch["source"].shape[0] == 4
    assert batch["target"].shape[0] == 4
    assert batch["source"].dtype == torch.long
    assert batch["target"].dtype == torch.long

    print("\nDataLoader fonctionnel !")


if __name__ == "__main__":
    main()
