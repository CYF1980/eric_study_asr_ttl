from openai import OpenAI
from init_env import init_env, get_env_variable

# 初始化環境變數
init_env()
client = OpenAI(api_key=get_env_variable("OPENAI_API_KEY"))

# SAVE TO FILE
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="nova",
    input="你好,我是您的旅遊助理,你有什麼旅遊規劃或是需要幫你介紹我們的商品嗎",
    response_format="wav",
) as response:
    response.stream_to_file("output.wav")

print("已產生 output.wav")
import subprocess
subprocess.run(["aplay", "output.wav"], check=False)