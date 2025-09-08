import os
# Force transformers to use PyTorch only (no TensorFlow)
os.environ["TRANSFORMERS_NO_TF"] = "1"
# Optional: silence TF warnings if TF is installed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10"  # Punjabi model
# MODEL_ID = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"  # Hindi model
# MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self" # English model


TARGET_SR = 16000  # Wav2Vec2 models expect 16k

LANG_MATCH = {"pa"} 
# LANG_MATCH = {"hi"}  # accepted language labels
# LANG_MATCH = {"en"} 

# -----------------------------
# Load model & processor
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

print(f"[INFO] Loading processor: {MODEL_ID}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

print(f"[INFO] Loading model: {MODEL_ID}")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
model.eval()

# Reusable resampler
_resamplers = {}

def _get_resampler(sr_from: int, sr_to: int):
    key = (sr_from, sr_to)
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(sr_from, sr_to)
    return _resamplers[key]

# -----------------------------
# Core transcription function
# -----------------------------
def transcribe_segment(audio_path: str, start_sec: float, end_sec: float) -> str:
    # Load audio
    waveform, sr = torchaudio.load(audio_path)  # shape: [channels, samples]

    # Convert to mono if stereo/multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16k if needed
    if sr != TARGET_SR:
        resampler = _get_resampler(sr, TARGET_SR)
        waveform = resampler(waveform)
        sr = TARGET_SR

    # Slice segment
    start_sample = int(float(start_sec) * sr)
    end_sample = int(float(end_sec) * sr)
    if end_sample <= start_sample:
        return ""

    seg = waveform[:, start_sample:end_sample].squeeze(0)  # [samples]

    # Guard: empty slice
    if seg.numel() == 0:
        return ""

    # Model expects float32 tensor on device
    inputs = processor(
        seg.numpy(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]  # take first batch element
        transcription = processor.decode(pred_ids, skip_special_tokens=True)

    return transcription.strip()

# -----------------------------
# CSV pipeline
# -----------------------------
def process_csv(input_csv: str, output_csv: str):
    new_df = pd.read_csv(input_csv)
    new_df = new_df.head(100)

    for col in ["source_start_sec", "source_end_sec", "source_file", "language"]:
        if col not in new_df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    transcripts = []
    total = len(new_df)
    print(f"[INFO] Processing {total} rows...")

    for idx, row in new_df.iterrows():
        try:
            lang = str(row["language"]).strip().lower()
            if lang not in LANG_MATCH:
                transcripts.append("")
                continue

            src = str(row["source_file"])
            start = float(row["source_start_sec"])
            end = float(row["source_end_sec"])

            if not os.path.isfile(src):
                print(f"[WARN] File not found (row {idx}): {src}")
                transcripts.append("")
                continue

            text = transcribe_segment(src, start, end)
            transcripts.append(text)

        except Exception as e:
            print(f"[ERROR] row {idx}: {e}")
            transcripts.append("")

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"[INFO] Progress: {idx + 1}/{total}")

    # -----------------------------
    # Merge with existing output
    # -----------------------------
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        df = df.head(100)
       
    else:
        df = new_df.copy()

    if "transcript" not in df.columns:
        df["transcript"] = ""

    # Update only rows where language matches and transcript is empty
    for i in range(len(new_df)):
        if str(new_df.at[i, "language"]).strip().lower() in LANG_MATCH:
            if pd.isna(df.at[i, "transcript"]) or df.at[i, "transcript"] == "":
                df.at[i, "transcript"] = transcripts[i]

    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved output to: {output_csv}")

# -----------------------------
# Entry
# -----------------------------



if __name__ == "__main__":
    process_csv("audio_transcript_test.csv", "test_transcript.csv")
