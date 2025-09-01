from init_llm_openai import get_openai_model
from init_env import init_env, get_env_variable

# 初始化環境變數
init_env()
model = get_openai_model()

from openai import OpenAI

client = OpenAI(api_key=get_env_variable("OPENAI_API_KEY"))

# 上傳音檔並轉文字
with open("/tmp/test.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",  # 也可以用 whisper-1
        file=f
    )

print("轉錄結果：", transcript.text)
