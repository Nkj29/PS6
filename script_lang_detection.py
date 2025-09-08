#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-pass segment-level language ID (hi/en/pa) and evaluation on CPU.

Pass 1: Whisper 'small' (CPU) for all segments -> predicted_language, probability.
Pass 2: Re-run only low-confidence hi/pa with 'medium' (CPU) to improve Hindi/Punjabi.
Evaluation: Compare 'language' vs 'predicted_language', output per-language accuracy and confusion matrix.

Requirements:
  pip install faster-whisper pandas scikit-learn
  ffmpeg available on PATH

Example:
  python lid_two_pass_eval.py \
      --input segments.csv \
      --output predictions.csv \
      --report report.txt \
      --confusion confusion.csv \
      --model1 small \
      --model2 medium \
      --min_dur 0.5 \
      --threshold 0.70
"""

import os
import csv
import argparse
import subprocess
import tempfile
from typing import Tuple, List

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from faster_whisper import WhisperModel

LANG_MAP = {
    "hi": "hi", "hindi": "hi",
    "en": "en", "english": "en",
    "pa": "pa", "punjabi": "pa", "panjabi": "pa",
}

def ffmpeg_slice(src: str, start: float, end: float, dst: str) -> None:
    dur = max(0.0, float(end) - float(start))
    if dur <= 0:
        raise ValueError("non-positive duration")
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-ss", f"{start:.6f}",
        "-t", f"{dur:.6f}",
        "-i", src,
        "-ac","1","-ar","16000","-vn","-acodec","pcm_s16le",
        dst
    ]
    subprocess.run(cmd, check=True)
    if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
        raise FileNotFoundError(f"slice not created: {dst}")

def detect_lang_cpu(model: WhisperModel, wav_path: str, beam_size: int = 1) -> Tuple[str, float]:
    segments, info = model.transcribe(wav_path, beam_size=beam_size, vad_filter=True)
    # Consume generator to finalize metadata
    _ = list(segments)
    lang = (info.language or "").lower()
    prob = float(info.language_probability or 0.0)
    return LANG_MAP.get(lang, "unk"), prob

def evaluate_predictions(df: pd.DataFrame, out_report: str, out_confusion: str) -> None:
    # Normalize columns
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df["predicted_language"] = df["predicted_language"].astype(str).str.strip().str.lower()

    labels = sorted(pd.unique(pd.concat([df["language"], df["predicted_language"]], ignore_index=True)))
    y_true = df["language"].values
    y_pred = df["predicted_language"].values

    total = len(df)
    correct = int((df["language"] == df["predicted_language"]).sum())
    overall_acc = correct / total if total else 0.0

    grp = df.assign(correct=(df["language"] == df["predicted_language"]).astype(int))
    per_lang = grp.groupby("language")["correct"].agg(["sum", "count"]).rename(columns={"sum":"correct","count":"total"})
    per_lang["accuracy"] = (per_lang["correct"] / per_lang["total"]).fillna(0.0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(out_confusion, index=True)

    cls_report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    with open(out_report, "w", encoding="utf-8") as f:
        f.write(f"Total rows: {total}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f} ({correct}/{total})\n\n")
        f.write("Per-language accuracy (by ground-truth):\n")
        for lang, row in per_lang.iterrows():
            f.write(f"  {lang}: acc={row['accuracy']:.4f} ({int(row['correct'])}/{int(row['total'])})\n")
        f.write("\nClassification report (precision/recall/F1):\n")
        f.write(cls_report)
        f.write("\n")

    # Console summary
    print(f"Total: {total}  Overall acc: {overall_acc:.4f} ({correct}/{total})")
    print("Per-language accuracy:")
    print(per_lang[["correct","total","accuracy"]])
    print("Confusion matrix ->", out_confusion)
    print("Detailed report ->", out_report)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV with columns: source_file, source_start_sec, source_end_sec, language")
    ap.add_argument("--output", required=True, help="Output CSV with predicted_language, language_probability (two-pass improved)")
    ap.add_argument("--report", default="report.txt", help="Evaluation report path")
    ap.add_argument("--confusion", default="confusion.csv", help="Confusion matrix CSV path")
    ap.add_argument("--model1", default="small", help="First-pass Whisper model (e.g., tiny, base, small)")
    ap.add_argument("--model2", default="medium", help="Second-pass Whisper model for hard hi/pa cases")
    ap.add_argument("--min_dur", type=float, default=0.5, help="Minimum duration to attempt LID (seconds)")
    ap.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold for accepting hi/pa")
    ap.add_argument("--tmp_dir", default=None, help="Optional persistent temp dir for slices; if not set, uses TemporaryDirectory")
    args = ap.parse_args()

    # Load models on CPU (no CUDA/cuDNN), int8 for speed on CPU
    print(f"Loading first-pass model: {args.model1} (CPU)")
    model1 = WhisperModel(args.model1, device="cpu", compute_type="int8")  # CPU-only [WhisperModel CPU]
    print(f"Loading second-pass model: {args.model2} (CPU)")
    model2 = WhisperModel(args.model2, device="cpu", compute_type="int8")  # CPU-only [WhisperModel CPU]

    # Prepare temp dir
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)
        tmp_ctx = None
        tmp_dir = args.tmp_dir
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="lid_twopass_")
        tmp_dir = tmp_ctx.name

    # Read input CSV
    df = pd.read_csv(args.input)
    for col in ["source_file", "source_start_sec", "source_end_sec"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prepare output columns
    df_out = df.copy()
    if "predicted_language" not in df_out.columns:
        df_out["predicted_language"] = "unk"
    if "language_probability" not in df_out.columns:
        df_out["language_probability"] = 0.0

    # First pass
    print("Pass 1: detecting language with first-pass model...")
    first_pass_low_conf_indices: List[int] = []

    for i, row in df_out.iterrows():
        src = row["source_file"]
        try:
            start = float(row["source_start_sec"])
            end = float(row["source_end_sec"])
        except Exception:
            df_out.at[i, "predicted_language"] = "unk"
            df_out.at[i, "language_probability"] = 0.0
            continue

        dur = end - start
        if dur < args.min_dur:
            df_out.at[i, "predicted_language"] = "unk"
            df_out.at[i, "language_probability"] = 0.0
            continue

        seg_id = row.get("file_name", None)
        if seg_id is None:
            seg_id = row.get("segment_index", i)
        slice_path = os.path.join(tmp_dir, f"slice_{seg_id}.wav")

        try:
            ffmpeg_slice(src, start, end, slice_path)
            lang1, prob1 = detect_lang_cpu(model1, slice_path)
            df_out.at[i, "predicted_language"] = lang1
            df_out.at[i, "language_probability"] = float(prob1)

            # Identify hard cases for escalation: low-confidence hi/pa
            if lang1 in ("hi", "pa") and prob1 < args.threshold:
                first_pass_low_conf_indices.append(i)

        except Exception:
            df_out.at[i, "predicted_language"] = "unk"
            df_out.at[i, "language_probability"] = 0.0
        finally:
            # Keep slices during debugging; optionally remove here:
            # if os.path.exists(slice_path): os.remove(slice_path)
            pass

    # Second pass on selected rows
    print(f"Pass 2: escalating {len(first_pass_low_conf_indices)} hard hi/pa cases with second-pass model...")
    for i in first_pass_low_conf_indices:
        row = df_out.iloc[i]
        src = row["source_file"]
        start = float(row["source_start_sec"])
        end = float(row["source_end_sec"])
        seg_id = row.get("file_name", None)
        if seg_id is None:
            seg_id = row.get("segment_index", i)
        slice_path = os.path.join(tmp_dir, f"slice_{seg_id}.wav")

        try:
            # Recreate slice to be safe
            ffmpeg_slice(src, start, end, slice_path)
            lang2, prob2 = detect_lang_cpu(model2, slice_path)

            # If second pass is more confident, replace
            if float(prob2) > float(df_out.at[i, "language_probability"]):
                df_out.at[i, "predicted_language"] = lang2
                df_out.at[i, "language_probability"] = float(prob2)
        except Exception:
            # Keep first-pass result on failure
            pass
        finally:
            # Optionally remove slice
            # if os.path.exists(slice_path): os.remove(slice_path)
            pass

    # Write predictions CSV
    df_out.to_csv(args.output, index=False)
    print("Predictions written to:", args.output)

    # If ground truth 'language' present, evaluate
    if "language" in df_out.columns:
        evaluate_predictions(df_out, args.report, args.confusion)
    else:
        print("No 'language' column found; skipping evaluation.")

    # Cleanup temp dir if ephemeral
    if tmp_ctx is not None:
        tmp_ctx.cleanup()

if __name__ == "__main__":
    main()
