
from collections import Counter
from pathlib import Path
import json
import re


SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


def tokenize(text: str) -> list[str]:
    """Découpe une phrase en mots et conserve la ponctuation."""
    text = str(text).strip().lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


class Vocabulary:
    def __init__(self):
        self.itos = list(SPECIAL_TOKENS)
        self.stoi = {
            token: index
            for index, token in enumerate(self.itos)
        }

    def build(self, sentences: list[str], min_freq: int = 1):
        counter = Counter()

        for sentence in sentences:
            counter.update(tokenize(sentence))

        for token, frequency in counter.most_common():
            if frequency >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, sentence: str) -> list[int]:
        tokens = tokenize(sentence)

        return [
            self.stoi.get(token, self.stoi[UNK_TOKEN])
            for token in tokens
        ]

    def decode(self, indices: list[int]) -> list[str]:
        return [
            self.itos[index]
            if 0 <= index < len(self.itos)
            else UNK_TOKEN
            for index in indices
        ]

    def __len__(self):
        return len(self.itos)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {"itos": self.itos},
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        vocab = cls()
        vocab.itos = data["itos"]
        vocab.stoi = {
            token: index
            for index, token in enumerate(vocab.itos)
        }

        return vocab
