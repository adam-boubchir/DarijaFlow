
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data" / "processed"
TOKENIZER_DIR = ROOT_DIR / "artifacts" / "sentencepiece"


class TranslationDataset(Dataset):
    def __init__(self, csv_path, english_tokenizer, darija_tokenizer):
        self.df = pd.read_csv(csv_path)
        self.english_tokenizer = english_tokenizer
        self.darija_tokenizer = darija_tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Encoder la phrase anglaise en unités BPE.
        source_ids = self.english_tokenizer.encode(
            str(row["eng"]), out_type=int
        )

        # La cible commence par SOS et se termine par EOS.
        target_ids = (
            [self.darija_tokenizer.bos_id()]
            + self.darija_tokenizer.encode(str(row["darija"]), out_type=int)
            + [self.darija_tokenizer.eos_id()]
        )

        return {
            "source": torch.tensor(source_ids, dtype=torch.long),
            "target": torch.tensor(target_ids, dtype=torch.long),
        }


def collate_fn(batch):
    source_sequences = [item["source"] for item in batch]
    target_sequences = [item["target"] for item in batch]

    source_lengths = torch.tensor(
        [len(seq) for seq in source_sequences],
        dtype=torch.long,
    )

    target_lengths = torch.tensor(
        [len(seq) for seq in target_sequences],
        dtype=torch.long,
    )

    source_padded = torch.nn.utils.rnn.pad_sequence(
        source_sequences,
        batch_first=True,
        padding_value=0,
    )

    target_padded = torch.nn.utils.rnn.pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=0,
    )

    return {
        "source": source_padded,
        "source_lengths": source_lengths,
        "target": target_padded,
        "target_lengths": target_lengths,
    }


def create_dataloader(split, batch_size=32, shuffle=False):
    if split not in {"train", "validation", "test"}:
        raise ValueError(
            "split doit être 'train', 'validation' ou 'test'."
        )

    english_model = TOKENIZER_DIR / "english.model"
    darija_model = TOKENIZER_DIR / "darija.model"

    if not english_model.exists() or not darija_model.exists():
        raise FileNotFoundError(
            "Tokenizers SentencePiece introuvables. "
            "Exécute d'abord : "
            "python -m backend.app.data.train_tokenizers"
        )

    english_tokenizer = spm.SentencePieceProcessor(
        model_file=str(english_model)
    )
    darija_tokenizer = spm.SentencePieceProcessor(
        model_file=str(darija_model)
    )

    dataset = TranslationDataset(
        DATA_DIR / f"{split}.csv",
        english_tokenizer,
        darija_tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    return loader, english_tokenizer, darija_tokenizer
