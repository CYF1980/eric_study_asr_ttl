# -*- coding: utf-8 -*-
"""
RAG 語音對答機器人：錄音(VAD) -> ASR -> RAG檢索 -> 生成回覆 -> TTS播放

依賴：
- pyaudio (錄音)
- aplay (播放 wav；或自行改為 afplay/ffplay)
- langchain, langchain-openai, langchain-community
- openai
- pandas (CSV loader 會用到)
- python-dotenv (可選，用於載入 .env)

環境變數：
- OPENAI_API_KEY

資料：
- 一個 CSV，例如 NaturalGarden.csv（含你要檢索的內容）
"""

import os
import time
import wave
import math
import audioop
import shutil
import subprocess
import collections
from typing import List, TypedDict, Optional

import pyaudio
from openai import OpenAI

# ===== 可調參數 =====
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2               # 16-bit
FRAME_DURATION_MS = 30
CALIBRATE_SEC = 0.5
SILENCE_TIMEOUT_SEC = 1.8
MAX_UTTERANCE_SEC = 60
THRESHOLD_BOOST = 1.6

TEMP_WAV = "/tmp/asr_input.wav"
TTS_WAV = "/tmp/tts_output.wav"

ASR_MODEL = "gpt-4o-mini-transcribe"   # 你原本用的
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "nova"

# RAG / LLM 設定
EMBED_MODEL = "text-embedding-3-large"
GEN_MODEL = "gpt-4.1-mini"  # 文字回答的模型，可換 "gpt-4o-mini" 等
CSV_PATH = "NaturalGarden.csv"
TOP_K = 4  # 召回文件數

# ====== LangChain / RAG ======
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# -- RAG 提示模板（避免依賴 hub.pull）--
RAG_PROMPT = """You are a helpful assistant. Use ONLY the context to answer the question.
If the answer is not in the context, say you don't know.
Answer in the same language as the user's question.

Question:
{question}

Context:
{context}
"""

def ensure_cmd_exists(cmd):
    return shutil.which(cmd) is not None

def play_wav_with_aplay(path):
    # 你也可改成 'afplay' (macOS) 或 'ffplay -autoexit -nodisp'
    player = "aplay"
    if not ensure_cmd_exists(player):
        print(f"Cannot play audio: '{player}' command not found.")
        return
    try:
        subprocess.run([player, path], check=False)
    except Exception as e:
        print(f"[Play] exception: {e}")

def frame_bytes():
    samples = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))
    return samples * CHANNELS * SAMPLE_WIDTH

def write_wav(path, audio_bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)

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

    # 1) Auto calibrate
    noise_samples = []
    need_frames = int(CALIBRATE_SEC * 1000 / FRAME_DURATION_MS)
    print(f"\nCalibrating noise for {CALIBRATE_SEC} sec...")
    for _ in range(need_frames):
        chunk = stream.read(int(SAMPLE_RATE * FRAME_DURATION_MS / 1000), exception_on_overflow=False)
        if len(chunk) == fb:
            noise_samples.append(rms(chunk))
    noise_floor = (sum(noise_samples) / max(1, len(noise_samples))) if noise_samples else 0
    threshold = max(200, noise_floor * THRESHOLD_BOOST)
    print(f"Noise floor RMS ≈ {noise_floor:.1f}. Threshold ≈ {threshold:.1f}")
    print("Start talking (Ctrl+C to Quit)")

    prebuf = collections.deque(maxlen=int(300/FRAME_DURATION_MS))  # 300ms pre-roll
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
            else:
                voiced.extend(data)
                if is_loud:
                    last_loud_time = time.time()

                if last_loud_time and (time.time() - last_loud_time) >= SILENCE_TIMEOUT_SEC:
                    break

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

# ===== RAG Pipeline 構建 =====
class RAGBot:
    def __init__(self, csv_path: str, embed_model: str, gen_model: str, top_k: int = 4):
        self.csv_path = csv_path
        self.top_k = top_k

        # Embeddings + 向量庫
        self.embeddings = OpenAIEmbeddings(model=embed_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)

        # 載入 CSV -> 切塊 -> 建索引
        self._load_and_index_csv(csv_path)

        # 文字生成模型（走 LangChain ChatOpenAI，方便後續換成工具調用等）
        self.llm = ChatOpenAI(model=gen_model, temperature=0.2)

    def _load_and_index_csv(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        loader = CSVLoader(path)
        docs = loader.load()

        # 你也可把每列某欄位挑出來組合，這裡先簡化整列文字
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No chunks produced from CSV. Check file content/encoding.")
        _ = self.vector_store.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks from {path}")

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata or {}
            parts.append(f"[{i}] " + d.page_content)
        return "\n\n".join(parts)

    def answer(self, question: str) -> str:
        retrieved = self.vector_store.similarity_search(question, k=self.top_k)
        ctx = self._format_context(retrieved)
        prompt = RAG_PROMPT.format(question=question, context=ctx)
        # LangChain 的 ChatOpenAI 接口：輸入一個 human message 字串即可
        resp = self.llm.invoke(prompt)
        text = resp.content.strip() if hasattr(resp, "content") else str(resp)
        return text

# ===== 主程式：ASR -> RAG -> TTS =====
def main():
    # 載 .env（可選）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Init OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    # 準備 RAG Bot
    print("Building RAG index...")
    rag = RAGBot(csv_path=CSV_PATH, embed_model=EMBED_MODEL, gen_model=GEN_MODEL, top_k=TOP_K)

    print("\nRAG Voice Bot is running.")
    print("Speak after the 'Start talking' line; I'll answer from CSV knowledge and talk back.\n")

    try:
        while True:
            # 1) 錄音（含 VAD）
            audio_bytes = record_until_silence()
            if not audio_bytes:
                print("No audio recorded. Listening again...")
                continue

            write_wav(TEMP_WAV, audio_bytes)
            print(f"Saved input to {TEMP_WAV}")

            # 2) ASR：語音轉文字
            print("Transcribing...")
            try:
                with open(TEMP_WAV, "rb") as f:
                    transcript = client.audio.transcriptions.create(
                        model=ASR_MODEL,
                        file=f
                    )
                text = (transcript.text or "").strip()
                print("You said:", text if text else "(empty)")
            except Exception as e:
                print(f"[ASR] Exception: {e}")
                continue

            if not text:
                print("Nothing recognized. Listening again...")
                continue

            # 3) RAG 產生回答
            try:
                answer = rag.answer(text)
            except Exception as e:
                print(f"[RAG] Exception: {e}")
                answer = "抱歉，我剛剛檢索資料時遇到問題。"

            print("Bot:", answer)

            # 4) TTS：把回答講出來
            print("Synthesizing speech...")
            try:
                with client.audio.speech.with_streaming_response.create(
                    model=TTS_MODEL,
                    voice=TTS_VOICE,
                    input=answer,
                    response_format="wav",
                ) as response:
                    response.stream_to_file(TTS_WAV)
                print(f"Generated {TTS_WAV}")
                play_wav_with_aplay(TTS_WAV)
            except Exception as e:
                print(f"[TTS] Exception: {e}")

            print("Listening again... (Ctrl+C to Quit)")

    except KeyboardInterrupt:
        print("\nBye! 👋")

if __name__ == "__main__":
    main()
