# 先用以下命令錄製聲音
# arecord -f S16_LE -r 16000 -c 1 -d 5 /tmp/test.wav
from faster_whisper import WhisperModel

model = WhisperModel("small", device="auto", compute_type="int8")  # 可改 "small"
segments, info = model.transcribe("/tmp/test.wav", language="zh")
for s in segments:
    print(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")

