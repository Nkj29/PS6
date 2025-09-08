#!/usr/bin/env python3
import argparse
import random
import re
from pathlib import Path
import csv

import numpy as np
import soundfile as sf

# Expect input names like: hi_spk0012_0045.wav, en_spk0007_0003.wav, pa_spk0101_0123.wav
# Extract language (lang), spkID (spkXXXX), uttID (numeric or token), require .wav
NAME_RX = re.compile(r"^(?P<lang>[a-z]{2,3})_(?P<spk>spk[0-9]+)_(?P<utt>[^.]+)\.(?P<ext>wav)$", re.IGNORECASE)

def parse_name(p: Path):
    m = NAME_RX.match(p.name)
    if not m:
        return None
    return {
        "lang": m.group("lang").lower(),
        "spk": m.group("spk"),
        "utt": m.group("utt"),
        "ext": m.group("ext").lower()
    }

def load_wav_48k(path: Path):
    # Read as 2D (frames, channels), then downmix to mono and return float32 1D
    audio, sr = sf.read(str(path), always_2d=True)  # explicit channel axis [web:232]
    if sr != 48000:
        raise ValueError(f"Sampling rate is {sr}, expected 48000 for {path}")  # enforce 48 kHz [web:231]
    # audio shape: (frames, channels); downmix to mono
    mono = audio.mean(axis=1).astype(np.float32)  # mono 1D [web:230]
    return mono, sr

def pick_random_segment(x: np.ndarray, sr: int, dur_sec_min: float, dur_sec_max: float):
    # Ensure 1D ndarray and get integer length in frames
    x = np.asarray(x).reshape(-1)               # 1D samples [web:233]
    total = int(x.shape[0])                     # number of frames [web:238]
    want = int(sr * random.uniform(dur_sec_min, dur_sec_max))  # desired length [web:231]
    want = max(1, min(want, total))  # clamp to available length [web:233]
    if total <= want + 1:
        start = 0
        end = total
    else:
        start = random.randint(0, total - want - 1)
        end = start + want
    seg = x[start:end]
    return seg, start / sr, end / sr  # return segment and source times [web:231]

def build_pool(in_dirs):
    pool = []  # entries: {path, lang, spk, speaker_key, utt}
    for d in in_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in d.rglob("*.wav"):
            info = parse_name(p)
            if not info:
                continue
            pool.append({
                "path": p,
                "lang": info["lang"],
                "spk": info["spk"],
                "speaker_key": f"{info['lang']}_{info['spk']}",  # unique across languages [web:216]
                "utt": info["utt"],
            })
    return pool

def main():
    ap = argparse.ArgumentParser(description="Create multilingual diarization mixtures from 48 kHz corpora and log segments to CSV")
    ap.add_argument("--hindi_dir", default=None, help="Folder with Hindi files named hi_spkXXXX_uttID.wav")
    ap.add_argument("--english_dir", default=None, help="Folder with English files named en_spkXXXX_uttID.wav")
    ap.add_argument("--punjabi_dir", default=None, help="Folder with Punjabi files named pa_spkXXXX_uttID.wav")
    ap.add_argument("--in_dirs", nargs="*", default=None, help="Additional language folders (each with lang_spkXXXX_uttID.wav)")
    ap.add_argument("--out_dir", required=True, help="Output folder to write augmented WAVs")
    ap.add_argument("--manifest_csv", default="mixtures_manifest.csv", help="CSV path inside out_dir (default: mixtures_manifest.csv)")
    ap.add_argument("--aug_per_file", type=int, default=4, help="Augmented files per input seed (min 3–4 recommended)")
    ap.add_argument("--segments_min", type=int, default=6, help="Min segments per mixture")
    ap.add_argument("--segments_max", type=int, default=12, help="Max segments per mixture")
    ap.add_argument("--seg_dur_min", type=float, default=4.0, help="Min segment duration (sec)")
    ap.add_argument("--seg_dur_max", type=float, default=8.0, help="Max segment duration (sec)")
    ap.add_argument("--shuffle_lang", action="store_true", help="Alternate languages to force stronger language mixing")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / args.manifest_csv

    # Collect all input dirs
    in_dirs = []
    for d in [args.hindi_dir, args.english_dir, args.punjabi_dir]:
        if d:
            in_dirs.append(d)
    if args.in_dirs:
        in_dirs.extend(args.in_dirs)

    if not in_dirs:
        raise SystemExit("Provide at least one input directory via --hindi_dir, --english_dir, --punjabi_dir, or --in_dirs")  # validation [web:219]

    # Build file pool
    pool = build_pool(in_dirs)
    if not pool:
        raise SystemExit("No valid input files found matching lang_spkXXXX_uttID.wav pattern")  # validation [web:216]

    # Index by language to encourage alternation if requested
    by_lang = {}
    for rec in pool:
        by_lang.setdefault(rec["lang"], []).append(rec)  # language buckets [web:216]

    # Prepare CSV writer
    fieldnames = [
        "mix_file", "sr", "duration_sec",
        "segment_index", "start_sec", "end_sec",
        "speaker_key", "language", "source_file", "source_start_sec", "source_end_sec"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        sr_target = 48000  # 48 kHz output [web:231]

        # For each pool item, generate at least aug_per_file mixtures
        mix_counter = 0
        for idx_seed, seed_item in enumerate(pool):
            for k in range(max(3, args.aug_per_file)):
                n_segments = random.randint(max(3, args.segments_min), max(args.segments_min, args.segments_max))  # number of segments [web:219]
                segments_meta = []
                segments_audio = []

                # Select candidate records
                if args.shuffle_lang and len(by_lang) >= 2:
                    langs = list(by_lang.keys())
                    random.shuffle(langs)
                    lang_seq = (langs * ((n_segments // len(langs)) + 1))[:n_segments]  # alternation [web:216]
                    candidates = [random.choice(by_lang[L]) for L in lang_seq]  # per-language pick [web:216]
                else:
                    candidates = random.choices(pool, k=n_segments)  # random sampling [web:219]

                # Extract random segments and accumulate
                cur_pos = 0.0
                for seg_idx, cand in enumerate(candidates):
                    try:
                        audio, sr = load_wav_48k(cand["path"])  # robust mono 48k loader [web:232]
                    except Exception:
                        continue
                    seg_audio, src_start, src_end = pick_random_segment(audio, sr, args.seg_dur_min, args.seg_dur_max)  # safe slicing [web:233]
                    seg_dur = len(seg_audio) / sr  # seconds [web:231]
                    if seg_audio.size == 0:
                        continue
                    segments_audio.append(seg_audio)
                    segments_meta.append({
                        "speaker_key": cand["speaker_key"],   # lang_spkXXXX identity [web:216]
                        "language": cand["lang"],
                        "source_file": str(cand["path"]),
                        "source_start_sec": src_start,
                        "source_end_sec": src_end,
                        "mix_start_sec": cur_pos,
                        "mix_end_sec": cur_pos + seg_dur
                    })
                    cur_pos += seg_dur

                if not segments_audio:
                    continue

                # Concatenate segments into the final mixture
                mixture = np.concatenate(segments_audio, axis=0).astype(np.float32)  # 1D waveform [web:233]
                mix_dur = len(mixture) / sr_target  # total duration [web:231]

                # Name the mixture
                mix_name = f"mix_{idx_seed:04d}_{k:02d}_{mix_counter:06d}.wav"  # deterministic naming [web:219]
                mix_path = out_dir / mix_name

                # Write audio
                sf.write(str(mix_path), mixture, sr_target, subtype="PCM_16")  # 16-bit WAV at 48 kHz [web:231]

                # Log segments to CSV
                for i, m in enumerate(segments_meta):
                    writer.writerow({
                        "mix_file": mix_name,
                        "sr": sr_target,
                        "duration_sec": f"{mix_dur:.3f}",
                        "segment_index": i,
                        "start_sec": f"{m['mix_start_sec']:.3f}",
                        "end_sec": f"{m['mix_end_sec']:.3f}",
                        "speaker_key": m["speaker_key"],
                        "language": m["language"],
                        "source_file": m["source_file"],
                        "source_start_sec": f"{m['source_start_sec']:.3f}",
                        "source_end_sec": f"{m['source_end_sec']:.3f}",
                    })

                mix_counter += 1

    print(f"Done. Mixtures written to {out_dir}. CSV: {csv_path}")  # completion notice [web:219]

if __name__ == "__main__":
    main()
