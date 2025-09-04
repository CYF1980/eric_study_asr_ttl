# filename: talkback_faster_whisper_vad.py
import os
from faster_whisper import WhisperModel

from voice_recorder import record_until_silence, write_wav

# =========================
# Config
# =========================
TEMP_WAV = "/tmp/asr_input.wav"
WHISPER_MODEL_SIZE = os.getenv("WHISPER_SIZE", "small")   # 可改 "tiny" / "base" / "small" / "medium" / "large-v3"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")      # "cpu" / "cuda" / "auto"
WHISPER_PREC = os.getenv("WHISPER_PREC", "int8")          # "int8" / "int8_float16" / "float16" / "float32"
LANG = os.getenv("ASR_LANG", "zh")                        # 預設中文
INPUT_DEVICE_INDEX = None                                  # 如需指定麥克風，可填整數 index

# faster-whisper 參數（可依你的 Pi 5 / Hailo 搭配做調整，這裡不使用 Hailo，單純 CPU/GPU）
TRANSCRIBE_KW = dict(
    language=LANG,
    beam_size=5,
    vad_filter=False,              # 我們已用簡易 VAD 截斷句子，故關閉內建 VAD
    condition_on_previous_text=False,
    temperature=0.0,
)

def main():
    print(f"[asr] Loading faster-whisper model '{WHISPER_MODEL_SIZE}' (device={WHISPER_DEVICE}, compute_type={WHISPER_PREC}) ...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_PREC)

    print("Talkback (faster-whisper) is running.")
    print("我會無限迴圈：錄音→辨識→印出結果。按 Ctrl+C 離開。")

    try:
        while True:
            # 1) 錄到一段語音（直到偵測到靜音結束）
            audio_bytes, noise_floor, threshold = record_until_silence()

            if not audio_bytes:
                print("[asr] No audio captured. Listening again...")
                continue

            # 2) 存成 wav（16k/單聲道/16bit）
            write_wav(TEMP_WAV, audio_bytes)
            print(f"[asr] Saved input to {TEMP_WAV} (noise≈{noise_floor:.1f}, thr≈{threshold:.1f})")

            # 3) 轉文字
            print("[asr] Transcribing...")
            try:
                segments, info = model.transcribe(TEMP_WAV, **TRANSCRIBE_KW)

                # 輸出逐段與合併結果
                final_text_parts = []
                for s in segments:
                    print(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")
                    final_text_parts.append(s.text)

                final_text = "".join(final_text_parts).strip()
                if final_text:
                    print(f"[asr] >> {final_text}")
                else:
                    print("[asr] No speech detected.")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[asr] Exception: {e}")

            # 4) 回到聆聽
            print("[asr] Listening again... (Ctrl+C to Quit)")

    except KeyboardInterrupt:
        print("\nBye! 👋")

if __name__ == "__main__":
    main()
