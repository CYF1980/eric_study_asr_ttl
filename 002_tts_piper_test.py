from piper import PiperVoice, SynthesisConfig
import wave

voice = PiperVoice.load(
    "./voices/zh_CN-huayan-medium.onnx",
    config_path="./voices/zh_CN-huayan-medium.onnx.json",
)

text = "你好,第一次來用餐嗎? 需要吃點什麼呢?"
cfg = SynthesisConfig(length_scale=1.0, noise_scale=0.667)

gen = voice.synthesize(text, cfg)
first = next(gen)  # 先拿第一個 chunk 取得音訊格式

# 從 chunk 讀取格式資訊
sr = first.sample_rate
channels = getattr(first, "sample_channels", 1)
sampwidth = getattr(first, "sample_width", 2)  # bytes per sample，piper 通常是 2 (16-bit)

out_wav = "test_py.wav"
with wave.open(out_wav, "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(sr)
    wf.writeframes(first.audio_int16_bytes)
    for chunk in gen:
        wf.writeframes(chunk.audio_int16_bytes)

print(f"寫出 {out_wav}")
import subprocess
subprocess.run(["aplay", out_wav], check=False)
