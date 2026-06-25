"""Gradio demo for interactive Cascade vs Direct speech understanding comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import tempfile
from src.cascade import CascadePipeline
from src.data import inject_noise, load_audio, save_audio
from src.config import TASKS, SPEECH_LLM_PROVIDER

# Auto-select Direct pipeline
if SPEECH_LLM_PROVIDER == "gemini":
    from src.direct_gemini import GeminiDirectPipeline
    DirectClass = GeminiDirectPipeline
elif SPEECH_LLM_PROVIDER == "openai":
    from src.direct import DirectPipeline
    DirectClass = DirectPipeline
else:
    from src.direct_qwen import QwenAudioPipeline
    DirectClass = QwenAudioPipeline

# Load pipelines once at startup
print(f"Loading Cascade pipeline (faster-whisper + DeepSeek)...")
cascade = CascadePipeline()
print(f"Loading Direct pipeline ({SPEECH_LLM_PROVIDER}: {DirectClass.__name__})...")
direct = DirectClass()
print("Both pipelines ready!")

NOISE_CHOICES = [
    ("Clean", {}),
    ("Babble 20dB", {"noise_type": "babble", "snr_db": 20}),
    ("Babble 10dB", {"noise_type": "babble", "snr_db": 10}),
    ("Babble 0dB", {"noise_type": "babble", "snr_db": 0}),
    ("White 20dB", {"noise_type": "white", "snr_db": 20}),
    ("White 0dB", {"noise_type": "white", "snr_db": 0}),
    ("Reverb 1.0s", {"noise_type": "reverb", "rt60": 1.0}),
]

NOISE_LABELS = [c[0] for c in NOISE_CHOICES]
NOISE_KWARGS = {c[0]: c[1] for c in NOISE_CHOICES}


def process_audio(audio_file, noise_choice):
    """Main processing function: run both pipelines and return results."""
    if audio_file is None:
        return ("Please upload an audio file.", "", "", "", "",
                "", "", "", "", "")

    # Apply noise if selected
    audio_path = audio_file
    noise_kwargs = NOISE_KWARGS.get(noise_choice, {})

    if noise_kwargs:
        audio, sr = load_audio(audio_path)
        noisy = inject_noise(audio, sr, seed=42, **noise_kwargs)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_audio(noisy, tmp.name, sr)
        audio_path = tmp.name

    # Run cascade on all tasks
    cascade_summary = cascade.run(audio_path, "summarization")
    cascade_sentiment = cascade.run(audio_path, "sentiment")
    cascade_keywords = cascade.run(audio_path, "keywords")
    cascade_intent = cascade.run(audio_path, "intent")

    # Run direct on all tasks
    direct_summary = direct.run(audio_path, "summarization")
    direct_sentiment = direct.run(audio_path, "sentiment")
    direct_keywords = direct.run(audio_path, "keywords")
    direct_intent = direct.run(audio_path, "intent")

    transcript = cascade_summary.get("transcript", "(No transcript)")

    return (
        transcript,
        cascade_summary.get("output", ""),
        cascade_sentiment.get("output", ""),
        cascade_keywords.get("output", ""),
        cascade_intent.get("output", ""),
        direct_summary.get("output", ""),
        direct_sentiment.get("output", ""),
        direct_keywords.get("output", ""),
        direct_intent.get("output", ""),
        f"Cascade: {cascade_summary['latency_seconds']:.2f}s | "
        f"Direct: {direct_summary['latency_seconds']:.2f}s"
    )


# Build the UI
with gr.Blocks(title="Speech Understanding Benchmark", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 Speech Understanding: Cascade vs End-to-End

    Upload an audio file and see how two different AI architectures understand it side-by-side.

    - **Cascade (left):** faster-whisper transcribes, then DeepSeek analyzes the text
    - **Direct (right):** Gemini 2.5 Flash listens to the audio directly, including tone and emotion
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            noise_radio = gr.Radio(
                choices=NOISE_LABELS,
                value="Clean",
                label="🎛️ Noise Level (test robustness)",
            )
            run_btn = gr.Button("▶ Run Comparison", variant="primary", size="lg")
            latency_text = gr.Textbox(label="⏱️ Latency", interactive=False)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔧 Cascade (ASR → Text LLM)")
            cascade_transcript = gr.Textbox(
                label="📝 Transcript", lines=3, interactive=False,
                placeholder="ASR output will appear here..."
            )
            cascade_summary_out = gr.Textbox(
                label="📊 Summary", lines=4, interactive=False
            )
            cascade_sentiment_out = gr.Textbox(
                label="😊 Sentiment", lines=2, interactive=False
            )
            cascade_keywords_out = gr.Textbox(
                label="🔑 Keywords", lines=2, interactive=False
            )
            cascade_intent_out = gr.Textbox(
                label="🎯 Intent", lines=2, interactive=False
            )

        with gr.Column():
            gr.Markdown("### 🚀 Direct (Gemini 2.5 Flash)")
            direct_summary_out = gr.Textbox(
                label="📊 Summary", lines=4, interactive=False
            )
            direct_sentiment_out = gr.Textbox(
                label="😊 Sentiment", lines=2, interactive=False
            )
            direct_keywords_out = gr.Textbox(
                label="🔑 Keywords", lines=2, interactive=False
            )
            direct_intent_out = gr.Textbox(
                label="🎯 Intent", lines=2, interactive=False
            )

    run_btn.click(
        fn=process_audio,
        inputs=[audio_input, noise_radio],
        outputs=[
            cascade_transcript, cascade_summary_out, cascade_sentiment_out,
            cascade_keywords_out, cascade_intent_out,
            direct_summary_out, direct_sentiment_out, direct_keywords_out,
            direct_intent_out, latency_text,
        ],
    )

    gr.Markdown("""
    ---
    ### Architecture Comparison

    | | 🔧 Cascade (ASR + LLM) | 🚀 Direct (Gemini Audio) |
    |---|------------------------|---------------------------|
    | **How it works** | Whisper transcribes audio → DeepSeek reads text | Gemini listens to audio directly |
    | **Speech cues** | Lost (tone, pace, emotion not in text) | Captured (prosody, tone, emphasis) |
    | **Cost per run** | ~$0.0005 (DeepSeek) | Free (Gemini tier) |
    | **Latency** | ~2-3s | ~3-5s |
    | **Best for** | Factual tasks, cost-sensitive apps | Emotion-heavy, nuanced understanding |
    """)

if __name__ == "__main__":
    demo.launch(share=False)
