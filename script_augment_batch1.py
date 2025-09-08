#!/usr/bin/env python3
import argparse
import random
import re
from pathlib import Path
import csv
import numpy as np
import soundfile as sf

# Expect input names like: hi_spk0012_0045.wav, en_spk0007_0003.wav, pa_spk0101_0123.wav
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
    audio, sr = sf.read(str(path), always_2d=True)  # explicit channel axis [7]
    if sr != 48000:
        raise ValueError(f"Sampling rate is {sr}, expected 48000 for {path}")  # enforce 48 kHz [7]
    mono = audio.mean(axis=1).astype(np.float32)  # mono 1D [7]
    return mono, sr

def pick_random_segment(x: np.ndarray, sr: int, dur_sec_min: float, dur_sec_max: float):
    # Ensure 1D ndarray
    x = np.asarray(x).reshape(-1)                         # flatten to 1D samples [4]
    total =len(x)                             # number of frames (length) [1]
    want = int(sr * random.uniform(dur_sec_min, dur_sec_max))  # desired length in samples [4]
    want = max(1, min(want, total))                       # clamp to available length [1]
    if total <= want + 1:
        start = 0
        end = total
    else:
        start = random.randint(0, total - want - 1)
        end = start + want
    seg = x[start:end]
    return seg, start / sr, end / sr                      # segment and source times [1]



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
                "speaker_key": f"{info['lang']}_{info['spk']}",  # unique across languages [7]
                "utt": info["utt"],
            })
    return pool

def add_noise_to_target_snr(x: np.ndarray, snr_db: float, eps=1e-12):
    # Compute noise variance for desired SNR and add zero-mean Gaussian noise [1]
    sig_power = np.mean(x**2) + eps  # watts proxy [1]
    noise_power = sig_power / (10**(snr_db/10.0))  # from SNR_dB = 10 log10(Ps/Pn) [1]
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)  # white noise [1]
    y = x + noise  # inject [8]
    return y

def safe_mix_add(dst: np.ndarray, src: np.ndarray, offset: int):
    # Add src into dst starting at offset; expand dst if needed; return possibly extended dst [6]
    end = offset + len(src)
    if end > len(dst):
        pad = end - len(dst)
        dst = np.concatenate([dst, np.zeros(pad, dtype=np.float32)], axis=0)  # extend [9]
    dst[offset:end] += src  # overlap-add [6]
    return dst

def normalize_or_clip_pcm16(x: np.ndarray):
    # Clip to [-1,1] then scale to int16 when writing PCM_16 to avoid clipping artifacts [10][7]
    x = np.clip(x, -1.0, 1.0)  # bound float [-1,1] [10]
    return x

def main():
    ap = argparse.ArgumentParser(description="Create multilingual mixtures from 48 kHz corpora with optional silence, overlap, noise, and CSV logging")
    ap.add_argument("--hindi_dir", default=None, help="Folder with Hindi files named hi_spkXXXX_uttID.wav")
    ap.add_argument("--english_dir", default=None, help="Folder with English files named en_spkXXXX_uttID.wav")
    ap.add_argument("--punjabi_dir", default=None, help="Folder with Punjabi files named pa_spkXXXX_uttID.wav")
    ap.add_argument("--in_dirs", nargs="*", default=None, help="Additional language folders (each with lang_spkXXXX_uttID.wav)")
    ap.add_argument("--out_dir", required=True, help="Output folder to write augmented WAVs")
    ap.add_argument("--manifest_csv", default="audio_augment_batch1.csv", help="CSV path inside out_dir (default: mixtures_manifest.csv)")
    # Reduction controls
    ap.add_argument("--max_mixes", type=int, default=6000, help="Maximum number of mixtures to write (cap total at ~5k)")  # [1]
    # Per-seed augment count (lower to reduce size)
    ap.add_argument("--aug_per_file", type=int, default=2, help="Augmented mixtures per input seed (lower value reduces total)")
    # Segment selection
    ap.add_argument("--segments_min", type=int, default=3, help="Min segments per mixture")
    ap.add_argument("--segments_max", type=int, default=6, help="Max segments per mixture")
    ap.add_argument("--seg_dur_min", type=float, default=3.0, help="Min segment duration (sec)")
    ap.add_argument("--seg_dur_max", type=float, default=6.0, help="Max segment duration (sec)")
    # Arrangement: silence and overlap probabilities and ranges
    ap.add_argument("--silence_prob", type=float, default=0.6, help="Probability to insert silence between segments")
    ap.add_argument("--overlap_prob", type=float, default=0.4, help="Probability to overlap next segment with previous")
    ap.add_argument("--silence_min", type=float, default=0.15, help="Min silence between segments (sec)")
    ap.add_argument("--silence_max", type=float, default=0.6, help="Max silence between segments (sec)")
    ap.add_argument("--overlap_frac_min", type=float, default=0.05, help="Min fraction of next segment duration that overlaps")
    ap.add_argument("--overlap_frac_max", type=float, default=0.35, help="Max fraction of next segment duration that overlaps")
    # Noise injection controls
    ap.add_argument("--add_noise", action="store_true", help="Enable additive white noise at random SNR")
    ap.add_argument("--add_noise_snr_db_min", type=float, default=15.0, help="Min SNR dB for noise injection")
    ap.add_argument("--add_noise_snr_db_max", type=float, default=30.0, help="Max SNR dB for noise injection")
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
        raise SystemExit("Provide at least one input directory via --hindi_dir, --english_dir, --punjabi_dir, or --in_dirs")  # [7]

    # Build file pool
    pool = build_pool(in_dirs)
    if not pool:
        raise SystemExit("No valid input files found matching lang_spkXXXX_uttID.wav pattern")  # [7]

    # Optionally subsample seeds to reduce total size
    total_pool = len(pool)
    num_seed_used = max(1, int(total_pool * args.sample_pool_ratio))
    seed_pool = random.sample(pool, k=num_seed_used) if num_seed_used < total_pool else pool  # [1]

    # Index by language to encourage alternation if requested
    by_lang = {}
    for rec in pool:
        by_lang.setdefault(rec["lang"], []).append(rec)  # [7]

    # Prepare CSV writer
    fieldnames = [
        "mix_file", "sr", "duration_sec",
        "segment_index", "start_sec", "end_sec",
        "speaker_key", "language", "source_file", "source_start_sec", "source_end_sec"
    ]

    sr_target = 48000  # 48 kHz [7]

    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        mix_counter = 0
        for idx_seed, seed_item in enumerate(seed_pool):
            # Stop when reaching cap
            if mix_counter >= args.max_mixes:
                break

            # Per-seed augmentations
            for k in range(max(1, args.aug_per_file)):
                if mix_counter >= args.max_mixes:
                    break

                n_segments = random.randint(max(2, args.segments_min), max(args.segments_min, args.segments_max))  # [7]
                segments_meta = []
                segments_audio = []

                # Select candidate records
                if args.shuffle_lang and len(by_lang) >= 2:
                    langs = list(by_lang.keys())
                    random.shuffle(langs)
                    lang_seq = (langs * ((n_segments // len(langs)) + 1))[:n_segments]  # alternation [7]
                    candidates = [random.choice(by_lang[L]) for L in lang_seq]  # [7]
                else:
                    candidates = random.choices(pool, k=n_segments)  # [7]

                # Extract random segments
                for cand in candidates:
                    try:
                        audio, sr = load_wav_48k(cand["path"])  # [7]
                    except Exception:
                        continue
                    seg_audio, src_start, src_end = pick_random_segment(audio, sr, args.seg_dur_min, args.seg_dur_max)  # [7]
                    if seg_audio.size == 0:
                        continue
                    segments_audio.append((seg_audio, cand, src_start, src_end))

                if not segments_audio:
                    continue

                # Build mixture timeline with optional silence/overlap
                mixture = np.zeros(1, dtype=np.float32)  # dynamic buffer [9]
                cur_pos_samples = 0  # in samples [7]
                segments_out_meta = []

                for seg_idx, (seg_audio, cand, src_start, src_end) in enumerate(segments_audio):
                    seg_len = len(seg_audio)
                    # Decide spacing: silence or overlap relative to previous
                    if seg_idx > 0:
                        do_sil = random.random() < args.silence_prob
                        do_ovl = (not do_sil) and (random.random() < args.overlap_prob)
                        if do_sil:
                            sil_dur = random.uniform(args.silence_min, args.silence_max)
                            sil_samples = int(round(sil_dur * sr_target))
                            cur_pos_samples += sil_samples  # push start forward [9]
                        elif do_ovl:
                            frac = random.uniform(args.overlap_frac_min, args.overlap_frac_max)
                            ovl_samples = int(round(frac * seg_len))
                            cur_pos_samples = max(0, cur_pos_samples - ovl_samples)  # pull back to overlap [6]
                        # else contiguous, no change
                    # Place/add segment
                    mixture = safe_mix_add(mixture, seg_audio.astype(np.float32), cur_pos_samples)  # [6]
                    seg_start_sec = cur_pos_samples / sr_target
                    seg_end_sec = (cur_pos_samples + seg_len) / sr_target
                    segments_out_meta.append({
                        "speaker_key": cand["speaker_key"],
                        "language": cand["lang"],
                        "source_file": str(cand["path"]),
                        "source_start_sec": src_start,
                        "source_end_sec": src_end,
                        "mix_start_sec": seg_start_sec,
                        "mix_end_sec": seg_end_sec
                    })
                    # Advance nominal position to end (for next spacing decision)
                    cur_pos_samples = cur_pos_samples + seg_len

                # Optional additive noise at random SNR
                if args.add_noise and mixture.size > 0:
                    snr = random.uniform(args.add_noise_snr_db_min, args.add_noise_snr_db_max)
                    mixture = add_noise_to_target_snr(mixture, snr)  # [1][8]

                # Safety clip/normalize before PCM16 write
                mixture = normalize_or_clip_pcm16(mixture)  # [10][7]
                mix_dur = len(mixture) / sr_target  # [7]

                # Name the mixture
                mix_name = f"mix_{idx_seed:04d}_{k:02d}_{mix_counter:06d}.wav"  # [7]
                mix_path = out_dir / mix_name

                # Write audio
                sf.write(str(mix_path), mixture, sr_target, subtype="PCM_16")  # PCM16 at 48k [7]

                # Log segments to CSV
                for i, m in enumerate(segments_out_meta):
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

    print(f"Done. Mixtures written to {out_dir}. CSV: {csv_path}")  # [7]

if __name__ == "__main__":
    main()
