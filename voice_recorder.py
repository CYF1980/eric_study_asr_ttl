# filename: voice_recorder.py
import time
import wave
import audioop
import shutil
import subprocess
import collections
from typing import Optional, Tuple, Dict

import pyaudio

# =========================
# Defaults (可在呼叫時覆寫)
# =========================
DEFAULTS = {
    "SAMPLE_RATE": 16000,
    "CHANNELS": 1,
    "SAMPLE_WIDTH": 2,           # 16-bit
    "FRAME_DURATION_MS": 30,     # 30ms frame
    "CALIBRATE_SEC": 0.5,        # 前0.5秒環境噪音校正
    "SILENCE_TIMEOUT_SEC": 2.0,  # 靜音多久判定為一句話結束
    "MAX_UTTERANCE_SEC": 60,     # 單句最長限制
    "THRESHOLD_BOOST": 1.5,      # 閾值 = 噪音均值 * 係數
    "PRESPEECH_MS": 300,         # 起音前緩衝(避免吃掉開頭音)
}

# -------- utils ----------
def _frame_bytes(cfg: Dict) -> int:
    samples = int(cfg["SAMPLE_RATE"] * (cfg["FRAME_DURATION_MS"] / 1000.0))
    return samples * cfg["CHANNELS"] * cfg["SAMPLE_WIDTH"]

def write_wav(path: str, audio_bytes: bytes, cfg: Optional[Dict] = None) -> None:
    cfg = {**DEFAULTS, **(cfg or {})}
    with wave.open(path, "wb") as wf:
        wf.setnchannels(cfg["CHANNELS"])
        wf.setsampwidth(cfg["SAMPLE_WIDTH"])
        wf.setframerate(cfg["SAMPLE_RATE"])
        wf.writeframes(audio_bytes)

def ensure_cmd_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def play_wav_with_aplay(path: str) -> None:
    if not ensure_cmd_exists("aplay"):
        print("Cannot play audio: 'aplay' command not found.")
        return
    try:
        subprocess.run(["aplay", path], check=False)
    except Exception as e:
        print(f"Play exception: {e}")

def _rms(data: bytes, sample_width: int) -> float:
    return audioop.rms(data, sample_width)

# -------- core ----------
def record_until_silence(
    cfg: Optional[Dict] = None,
    input_device_index: Optional[int] = None,
) -> Tuple[bytes, float, float]:
    """
    以簡易 VAD 錄音：說話→偵測靜音→回傳這段音訊
    Returns: (audio_bytes, noise_floor_rms, threshold_rms)
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=cfg["CHANNELS"],
        rate=cfg["SAMPLE_RATE"],
        input=True,
        frames_per_buffer=int(cfg["SAMPLE_RATE"] * cfg["FRAME_DURATION_MS"] / 1000),
        input_device_index=input_device_index,
    )

    fb = _frame_bytes(cfg)
    voiced = bytearray()
    start_time = time.time()
    last_loud_time = None

    # 1) 噪音校正
    noise_samples = []
    need_frames = int(cfg["CALIBRATE_SEC"] * 1000 / cfg["FRAME_DURATION_MS"])
    print(f"\n[rec] Calibrating noise for {cfg['CALIBRATE_SEC']} sec...")
    for _ in range(need_frames):
        chunk = stream.read(int(cfg["SAMPLE_RATE"] * cfg["FRAME_DURATION_MS"] / 1000), exception_on_overflow=False)
        if len(chunk) == fb:
            noise_samples.append(_rms(chunk, cfg["SAMPLE_WIDTH"]))
    noise_floor = (sum(noise_samples) / max(1, len(noise_samples))) if noise_samples else 0.0
    threshold = max(200.0, noise_floor * cfg["THRESHOLD_BOOST"])
    print(f"[rec] Noise floor RMS ≈ {noise_floor:.1f}. Threshold ≈ {threshold:.1f}")
    print("[rec] Start talking (Ctrl+C to Quit)")

    prebuf = collections.deque(maxlen=int(cfg["PRESPEECH_MS"] / cfg["FRAME_DURATION_MS"]))
    speech_started = False

    try:
        while True:
            data = stream.read(int(cfg["SAMPLE_RATE"] * cfg["FRAME_DURATION_MS"] / 1000), exception_on_overflow=False)
            if len(data) != fb:
                continue

            level = _rms(data, cfg["SAMPLE_WIDTH"])
            is_loud = level >= threshold

            if not speech_started:
                prebuf.append(data)
                if is_loud:
                    speech_started = True
                    last_loud_time = time.time()
                    # 把起音前的緩衝也寫進去，避免吃字
                    for b in prebuf:
                        voiced.extend(b)
            else:
                voiced.extend(data)
                if is_loud:
                    last_loud_time = time.time()

                # 偵測靜音結束
                if last_loud_time and (time.time() - last_loud_time) >= cfg["SILENCE_TIMEOUT_SEC"]:
                    break

            # 最長時長保護
            if (time.time() - start_time) > cfg["MAX_UTTERANCE_SEC"]:
                print("[rec] Speak time limit reached.")
                break

    except KeyboardInterrupt:
        print("\n[rec] Recording stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return (bytes(voiced), noise_floor, threshold)
