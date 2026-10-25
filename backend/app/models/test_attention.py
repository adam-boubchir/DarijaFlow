
import torch
import torch.nn.functional as F

from backend.app.data.translation_dataset import create_dataloader
from backend.app.models.seq2seq import Seq2Seq


def main():
    loader, english_tokenizer, darija_tokenizer = create_dataloader(
        "train",
        batch_size=4,
    )

    batch = next(iter(loader))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device :", device)
    print("Vocabulaire anglais :", english_tokenizer.vocab_size())
    print("Vocabulaire darija :", darija_tokenizer.vocab_size())

    source = batch["source"].to(device)
    target = batch["target"].to(device)
    source_lengths = batch["source_lengths"]

    for cell_type in ["vanilla", "wmc", "cifg"]:
        model = Seq2Seq(
            source_vocab_size=english_tokenizer.vocab_size(),
            target_vocab_size=darija_tokenizer.vocab_size(),
            embedding_dim=64,
            hidden_size=128,
            cell_type=cell_type,
        ).to(device)

        model.train()
        model.zero_grad()

        logits = model(
            source,
            target,
            source_lengths=source_lengths,
            teacher_forcing_ratio=1.0,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target[:, 1:].reshape(-1),
            ignore_index=0,
        )

        loss.backward()

        print(
            f"{cell_type.upper()} | "
            f"logits : {tuple(logits.shape)} | "
            f"loss : {loss.item():.4f} | "
            "backward : OK"
        )


if __name__ == "__main__":
    main()
