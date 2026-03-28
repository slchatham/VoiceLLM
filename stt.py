"""
Phase 3 — Parakeet STT wrapper
nvidia/parakeet-tdt-0.6b-v3 via NeMo / MPS

Usage:
    python stt.py                  # push-to-talk: Enter to start, Enter to stop
    python stt.py --file audio.wav # transcribe a file
    python stt.py --test           # one push-to-talk then exit
"""

import argparse
import logging
import os
import select
import sys
import tempfile
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

import log_utils

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME  = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# NeMo log suppression
# ---------------------------------------------------------------------------

def _silence_nemo_loggers() -> None:
    """Force ERROR level on all NeMo / Lightning loggers (they reset on every call)."""
    _PREFIXES = ("nemo", "lightning", "pytorch_lightning",
                 "torch.distributed", "nv_one_logger", "one_logger")
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and any(
            name.startswith(p) for p in _PREFIXES
        ):
            logger.setLevel(logging.ERROR)
            logger.propagate = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(log):
    import warnings
    warnings.filterwarnings("ignore")
    _silence_nemo_loggers()

    import torch
    import nemo.collections.asr as nemo_asr

    log.info(f"loading {MODEL_NAME} …")
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)

    if torch.backends.mps.is_available():
        try:
            model = model.to("mps")
            log.info("Parakeet on MPS (Apple Silicon)")
        except Exception as e:
            log.warning(f"MPS failed ({e}), falling back to CPU")
    else:
        log.info("Parakeet on CPU")

    model.eval()
    _silence_nemo_loggers()
    log.info("model ready")
    return model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(model, audio: np.ndarray, log) -> tuple[str, float]:
    """Transcribe a float32 mono 16kHz array. Returns (text, elapsed_seconds)."""
    if len(audio) == 0:
        return "", 0.0

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        sf.write(tmp_path, audio, SAMPLE_RATE)
        _silence_nemo_loggers()
        t0     = time.perf_counter()
        result = model.transcribe([tmp_path], verbose=False)
        elapsed = time.perf_counter() - t0

        if isinstance(result, tuple):
            result = result[0]
        text = result[0] if result else ""
        if hasattr(text, "text"):
            text = text.text
        text = str(text).strip()

        dur = len(audio) / SAMPLE_RATE
        log.info(f"STT: {elapsed:.2f}s for {dur:.1f}s audio — RTF: {elapsed/max(dur,0.01):.2f}x")
        log.info(f"transcript: {text!r}")
        return text, elapsed

    except Exception as e:
        log.error(f"transcribe error: {e}")
        return "", 0.0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Push-to-talk recording
# ---------------------------------------------------------------------------

def _await_enter() -> None:
    """Block until Enter is pressed, reliably interruptible by Ctrl+C.

    Uses select.select with a short timeout instead of input() so that
    SIGINT is delivered between polls. The SIGINT handler is set once at
    startup in pipeline.py — no save/restore needed here.

    The 0.4s sleep before flushing covers stray Enter presses that arrive
    while afplay is playing back a long audio clip.
    """
    time.sleep(0.4)
    while select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.readline()
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            sys.stdin.readline()
            return


def record_push_to_talk(log) -> np.ndarray:
    """Record mic audio. Press Enter to start, Enter to stop."""
    print("\033[92m╔══════════════════════════════════════╗\033[0m", flush=True)
    print("\033[92m║        🎙  PRESS ENTER TO TALK        ║\033[0m", flush=True)
    print("\033[92m╚══════════════════════════════════════╝\033[0m", flush=True)
    _await_enter()

    frames     = []
    keep_going = [True]

    def callback(indata, frame_count, time_info, status):
        if keep_going[0]:
            frames.append(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    )
    stream.start()
    print("\033[91m╔══════════════════════════════════════╗\033[0m", flush=True)
    print("\033[91m║        ⏹  PRESS ENTER TO STOP         ║\033[0m", flush=True)
    print("\033[91m╚══════════════════════════════════════╝\033[0m", flush=True)
    t0 = time.perf_counter()
    try:
        _await_enter()
    except KeyboardInterrupt:
        keep_going[0] = False
        stream.stop()
        stream.close()
        raise

    keep_going[0] = False
    stream.stop()
    stream.close()
    print("\033[93m⏳  Processing…\033[0m", flush=True)

    elapsed = time.perf_counter() - t0
    audio   = np.concatenate(frames) if frames else np.array([], dtype=np.float32)
    log.info(f"recorded {elapsed:.1f}s ({len(audio)/SAMPLE_RATE:.1f}s audio)")
    return audio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parakeet STT wrapper")
    parser.add_argument("--file", metavar="WAV", help="Transcribe a WAV file")
    parser.add_argument("--test", action="store_true", help="One push-to-talk then exit")
    args = parser.parse_args()

    log = log_utils.setup("stt")

    model = load_model(log)

    if args.file:
        audio, sr = sf.read(args.file, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        text, _ = transcribe(model, audio, log)
        print(text)
        return

    # push-to-talk loop
    while True:
        audio = record_push_to_talk(log)
        if len(audio) < SAMPLE_RATE * 0.5:
            log.warning("recording too short (<0.5s), skipped")
            continue
        text, _ = transcribe(model, audio, log)
        print(f"\n>>> {text}\n")
        if args.test:
            break


if __name__ == "__main__":
    main()
