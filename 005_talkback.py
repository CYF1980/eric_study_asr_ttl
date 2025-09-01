# filename: talkback_simple_vad.py
import os
import time
import wave
import math
import audioop
import shutil
import subprocess
import collections

import pyaudio
from openai import OpenAI
from init_env import init_env, get_env_variable

# =========================
# Config
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2               # 16-bit
FRAME_DURATION_MS = 30         # 30ms frame
CALIBRATE_SEC = 0.5            # Auto calibrate noise for first 0.5 sec
SILENCE_TIMEOUT_SEC = 2.0      # Silence timeout to end the recording
MAX_UTTERANCE_SEC = 60         # Limit max utterance length
THRESHOLD_BOOST = 1.5          # Threshold = noise_floor * THRESHOLD_BOOST

TEMP_WAV = "/tmp/asr_input.wav"
TTS_WAV = "/tmp/tts_output.wav"

ASR_MODEL = "gpt-4o-mini-transcribe"   # "whisper-1"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "nova"
# =========================

def frame_bytes():
    samples = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))
    return samples * CHANNELS * SAMPLE_WIDTH

def write_wav(path, audio_bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)

def ensure_cmd_exists(cmd):
    return shutil.which(cmd) is not None

def play_wav_with_aplay(path):
    if not ensure_cmd_exists("aplay"):
        print("Cannot play audio: 'aplay' command not found.")
        return
    try:
        subprocess.run(["aplay", path], check=False)
    except Exception as e:
        print(f"Play exceptoin: {e}")

def rms(data):
    return audioop.rms(data, SAMPLE_WIDTH)

def record_until_silence():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=CHANNELS,
                     rate=SAMPLE_RATE,
                     input=True,
                     frames_per_buffer=int(SAMPLE_RATE * FRAME_DURATION_MS / 1000),
                     input_device_index=None)

    fb = frame_bytes()
    voiced = bytearray()
    start_time = time.time()
    last_loud_time = None

    # 1) Auto calibrate noise level
    noise_samples = []
    need_frames = int(CALIBRATE_SEC * 1000 / FRAME_DURATION_MS)
    print(f"Calibrating noise for {CALIBRATE_SEC} sec...")
    for _ in range(need_frames):
        chunk = stream.read(int(SAMPLE_RATE * FRAME_DURATION_MS / 1000), exception_on_overflow=False)
        if len(chunk) == fb:
            noise_samples.append(rms(chunk))
    noise_floor = (sum(noise_samples) / max(1, len(noise_samples))) if noise_samples else 0
    threshold = max(200, noise_floor * THRESHOLD_BOOST)
    print(f"Noise floor RMS ≈ {noise_floor:.1f}. Threshold ≈ {threshold:.1f}")

    print("Start talking (Ctrl+C to Quit)")
    prebuf = collections.deque(maxlen= int(300/FRAME_DURATION_MS))  # keep last 300ms
    speech_started = False

    try:
        while True:
            data = stream.read(int(SAMPLE_RATE * FRAME_DURATION_MS / 1000), exception_on_overflow=False)
            if len(data) != fb:
                continue

            level = rms(data)
            is_loud = level >= threshold

            if not speech_started:
                prebuf.append(data)
                if is_loud:
                    speech_started = True
                    last_loud_time = time.time()
                    for b in prebuf:
                        voiced.extend(b)
                # No speech yet, keep waiting
            else:
                voiced.extend(data)
                if is_loud:
                    last_loud_time = time.time()

                # Detect silence timeout
                if last_loud_time and (time.time() - last_loud_time) >= SILENCE_TIMEOUT_SEC:
                    break

            # Detect max utterance length
            if (time.time() - start_time) > MAX_UTTERANCE_SEC:
                print("Speak time limit reached.")
                break

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return bytes(voiced)

def main():
    init_env()
    client = OpenAI(api_key=get_env_variable("OPENAI_API_KEY"))

    audio_bytes = record_until_silence()
    if not audio_bytes:
        print("No audio recorded.")
        return

    write_wav(TEMP_WAV, audio_bytes)
    print(f"Save to {TEMP_WAV}")

    # 3) ASR
    print("Transcribing...")
    with open(TEMP_WAV, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=ASR_MODEL,
            file=f
        )
    text = (transcript.text or "").strip()
    print("Result: ", text if text else "No speech detected.")
    if not text:
        return

    # 4) TTS
    print("Synthesizing speech...")
    try:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="wav",
        ) as response:
            response.stream_to_file(TTS_WAV)
        print(f"Generated {TTS_WAV}")
        play_wav_with_aplay(TTS_WAV)
    except Exception as e:
        print(f"TTS Exception: {e}")

if __name__ == "__main__":
    main()
