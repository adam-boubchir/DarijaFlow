
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT_DIR / "data" / "raw" / "darija_english_raw.csv"

df = pd.read_csv(CSV_PATH)

# Conserver temporairement les lignes avec anglais et Darija présents
valid_mask = (
    df["eng"].notna()
    & df["darija"].notna()
    & df["eng"].astype("string").str.strip().ne("")
    & df["darija"].astype("string").str.strip().ne("")
)

valid = df.loc[valid_mask].copy()

print("=" * 60)
print("BILAN DES PAIRES")
print("=" * 60)
print("Total initial :", len(df))
print("Paires avec anglais et Darija :", len(valid))
print("Paires restantes à examiner :", len(df) - len(valid))
print("Anglais distincts :", valid["eng"].nunique())
print("Traductions Darija distinctes :", valid["darija"].nunique())

print("\n--- 20 paires exploitables ---")
print(
    valid[["eng", "darija"]]
    .head(20)
    .to_string(index=False)
)

print("\n--- 10 paires aléatoires ---")
print(
    valid[["eng", "darija"]]
    .sample(n=min(10, len(valid)), random_state=42)
    .to_string(index=False)
)

print("\n--- Exemples avec anglais manquant ---")
print(
    df.loc[df["eng"].isna(), ["darija", "darija_ar"]]
    .head(10)
    .to_string(index=False)
)

print("\n--- Doublons parmi les paires exploitables ---")
print(
    "Paires dupliquées :",
    int(valid.duplicated(subset=["eng", "darija"]).sum()),
)

print("\nInspection terminée. Aucun fichier n'a été modifié.")
