import os
import argparse
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Optional: move caches if running out of space
# os.environ["HF_HOME"] = "/path/to/big_disk/hf"
# os.environ["TRANSFORMERS_CACHE"] = "/path/to/big_disk/hf/transformers"
# os.environ["HF_DATASETS_CACHE"] = "/path/to/big_disk/hf/datasets"
# os.environ["TMPDIR"] = "/path/to/big_disk/tmp"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"

# Map dataset language hints to FLORES tags used by IndicTrans2
# Assumes: language column values are "hi" for Hindi, "pa" for Punjabi, and "en" for English.
LANG_TO_FLORES = {
    "hi": "hin_Deva",
    "pa": "pan_Guru",
    "en": "eng_Latn",
}

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        # attn_implementation="eager",  # optional; avoid flash_attention_2 unless installed
    ).to(DEVICE)
    model.eval()
    # Workaround past_key_values issue
    model.config.use_cache = False
    ip = IndicProcessor(inference=True)
    return tokenizer, model, ip

def translate_batch(model, tokenizer, ip, sentences, src_lang, tgt_lang="eng_Latn"):
    if not sentences:
        return []
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            use_cache=False,    # critical to avoid NoneType past_key_values path
            num_beams=1,        # can try 5 later after confirming stability
            do_sample=False,
            min_length=0,
            max_length=256,
        )
    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    return translations

def process_csv(input_csv: str, output_csv: str, batch_size: int = 32):
    tokenizer, model, ip = load_model(MODEL_NAME)

    df = pd.read_csv(input_csv)
    if "transcript" not in df.columns:
        raise ValueError("Input CSV must contain a 'transcript' column.")
    if "language" not in df.columns:
        raise ValueError("Input CSV must contain a 'language' column with codes like 'hi', 'pa', 'en'.")

    tran_eng = [""] * len(df)

    # 1) Copy English rows directly
    en_mask = df["language"].astype(str).str.lower().eq("en")
    tran_eng_en = df.loc[en_mask, "transcript"].astype(str).tolist()
    for idx, val in zip(df.index[en_mask], tran_eng_en):
        tran_eng[idx] = val

    # 2) Translate Hindi rows
    for lang_code in ["hi", "pa"]:
        mask = df["language"].astype(str).str.lower().eq(lang_code)
        if not mask.any():
            continue
        src_lang = LANG_TO_FLORES[lang_code]
        texts = df.loc[mask, "transcript"].astype(str).tolist()

        outputs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            outputs.extend(translate_batch(model, tokenizer, ip, batch_texts, src_lang, "eng_Latn"))

        # assign back preserving order
        for idx, val in zip(df.index[mask], outputs):
            tran_eng[idx] = val

    df["tran_eng"] = tran_eng
    df.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Path to input CSV with columns: language, transcript, ...")
    ap.add_argument("--output_csv", required=True, help="Path to output CSV with new 'tran_eng' column.")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    process_csv(args.input_csv, args.output_csv, args.batch_size)
