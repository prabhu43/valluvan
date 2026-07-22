"""Normalize the raw HuggingFace Thirukkural parquet into a clean canonical JSON.

Source dataset: https://huggingface.co/datasets/yuvarajvelmurugan/thirukkural
The raw schema uses ambiguous names (division/chapter/section). This script maps
them to unambiguous canonical fields used everywhere else in the project.

Run:  python ingestion/normalize.py
Input:  data/thirukkural_hf.parquet
Output: data/thirukkural.json
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_DIR / "thirukkural_hf.parquet"
OUT = DATA_DIR / "thirukkural.json"


def normalize() -> list[dict]:
    df = pd.read_parquet(RAW)
    records = []
    for _, r in df.iterrows():
        records.append(
            {
                "kural_no": int(r["couplet_number"]),          # 1..1330
                "section_no": int(r["paal_en"]),               # 1..3
                "section_ta": r["paal"],                        # அறத்துப்பால் ...
                "section_en": r["division"],                    # Virtue / Polity / Love
                "iyal_no": int(r["chapter_number"]),
                "iyal_ta": r["iyal"],
                "iyal_en": r["chapter"],
                "adhigaram_no": int(r["section_number"]),       # 1..133
                "adhigaram_ta": r["athikaaram"],
                "adhigaram_en": r["section"],                   # e.g. "The Praise of God"
                "kural_ta": r["kural"].strip(),                 # Tamil couplet (2 lines)
                "transliteration": r["couplet_transliteration"].strip(),
                "translation_en": r["couplet"].strip(),         # English verse translation
                "explanation_en": r["explanation"].strip(),     # English prose meaning
                "meaning_ta": r["porul"].strip(),               # Tamil short meaning
                "explanation_ta": r["vilakkam"].strip(),        # Tamil detailed explanation
            }
        )
    records.sort(key=lambda x: x["kural_no"])
    return records


def main() -> None:
    records = normalize()
    assert len(records) == 1330, f"expected 1330 kurals, got {len(records)}"
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} kurals -> {OUT}")


if __name__ == "__main__":
    main()
