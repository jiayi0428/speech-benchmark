"""Smoke test for Qwen2-Audio pipeline."""
import numpy as np
import soundfile as sf
import os
from src.direct_qwen import QwenAudioPipeline

# Create test audio
t = np.linspace(0, 3, 48000, endpoint=False)
audio = (0.3 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
sf.write("_test.wav", audio, 16000)

print("Loading Qwen2-Audio (downloading ~14GB on first run, ~5-10 min)...")
qwen = QwenAudioPipeline()

print("Running summarization...")
r = qwen.run("_test.wav", "summarization")
print(f"OK! Output: {r['output'][:200]}")
print(f"Latency: {r['latency_seconds']}s")

os.remove("_test.wav")
print("QWEN2-AUDIO SMOKE TEST: PASSED")
