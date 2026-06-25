"""Generate 20 realistic English speech samples using Microsoft Edge TTS (free, no VPN needed).

Each sample has a known transcript → perfect for benchmark evaluation.
"""
import asyncio
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from src.config import PROCESSED_DIR

# 20 diverse TED-style talks across 5 categories
SAMPLES = [
    # Technology
    ("AI is transforming healthcare through faster diagnosis and personalized treatment plans. Machine learning models can now detect diseases from medical images with accuracy rivaling human doctors, potentially saving millions of lives each year. However, we must ensure these systems are trained on diverse data to avoid bias.", "en-US-AriaNeural", "tech_ai_healthcare"),
    ("Quantum computing represents a fundamental shift in how we process information. Unlike classical bits that are either zero or one, quantum bits can exist in multiple states simultaneously. This property, called superposition, allows quantum computers to solve certain problems exponentially faster than any classical computer ever could.", "en-US-GuyNeural", "tech_quantum"),
    ("The future of education lies in personalized learning powered by artificial intelligence. Every student learns differently — some are visual learners, others prefer reading, and some learn best through hands-on practice. AI tutors can adapt to each student's unique learning style in real time.", "en-GB-SoniaNeural", "tech_education"),
    ("Blockchain technology extends far beyond cryptocurrency. Its ability to create tamper-proof, decentralized records has applications in supply chain management, digital identity verification, and even voting systems. The key innovation is trust without a central authority.", "en-US-JennyNeural", "tech_blockchain"),

    # Science
    ("Climate change is the defining challenge of our generation. Global temperatures have risen by approximately one point two degrees Celsius since pre-industrial times, and we are already seeing the effects: more frequent extreme weather events, rising sea levels, and disrupted ecosystems. The scientific consensus is clear — we must act now.", "en-GB-RyanNeural", "science_climate"),
    ("The human brain contains approximately eighty-six billion neurons, each connected to thousands of others, forming a network of unimaginable complexity. Despite decades of research, we are only beginning to understand how this three-pound organ generates consciousness, memory, and emotion.", "en-US-ChristopherNeural", "science_brain"),
    ("Space exploration has entered a new golden age. Private companies are developing reusable rockets that dramatically reduce the cost of reaching orbit, while NASA plans to return humans to the moon and eventually send them to Mars. The James Webb Space Telescope is already revealing secrets of the early universe.", "en-US-EricNeural", "science_space"),
    ("CRISPR gene editing technology gives scientists the ability to modify DNA with unprecedented precision. This breakthrough has enormous potential for treating genetic diseases, developing drought-resistant crops, and even fighting cancer. But it also raises profound ethical questions about designer babies and genetic inequality.", "en-GB-LibbyNeural", "science_crispr"),

    # Business
    ("The rise of remote work has fundamentally changed how companies operate. Studies show that many workers are actually more productive when working from home, but the loss of in-person collaboration and spontaneous water-cooler conversations remains a concern. The future is likely a hybrid model that combines the best of both worlds.", "en-US-SaraNeural", "business_remote"),
    ("Startup culture celebrates failure as a learning opportunity, but the reality is more complex. While it is true that most successful entrepreneurs failed multiple times before succeeding, the romanticization of failure ignores the real human costs: financial stress, damaged relationships, and mental health struggles.", "en-US-DavisNeural", "business_startup"),
    ("Sustainable investing has grown from a niche strategy to a mainstream financial movement. Investors are increasingly demanding that companies disclose their environmental impact, and funds that screen for ESG criteria now manage trillions of dollars. The question is no longer whether sustainability matters, but how to measure it accurately.", "en-GB-MaisieNeural", "business_esg"),
    ("The gig economy promised freedom and flexibility, but for many workers it has delivered insecurity and unpredictability. Ride-share drivers, food delivery couriers, and freelance workers often lack basic protections like health insurance, paid sick leave, and retirement benefits. We need new social contracts for the twenty-first century workforce.", "en-US-AnaNeural", "business_gig"),

    # Society & Culture
    ("Social media has reshaped how we form opinions and consume information. Algorithmic curation creates echo chambers where we rarely encounter views that challenge our own, while engagement optimization rewards outrage and sensationalism over thoughtful discourse. We need to understand these dynamics to protect democratic debate.", "en-US-MichelleNeural", "society_socialmedia"),
    ("The concept of universal basic income has moved from utopian fantasy to serious policy proposal in recent years. Pilot programs in several countries have shown promising results: recipients report better mental health, more entrepreneurial activity, and improved educational outcomes. But funding such programs at national scale remains deeply challenging.", "en-GB-ThomasNeural", "society_ubi"),
    ("Language is more than a communication tool — it shapes how we perceive reality itself. Different languages carve up the world in different ways. Some have no words for left and right, using cardinal directions instead. Others have dozens of words for snow or rice. Each language represents a unique way of understanding human experience.", "en-US-JaneNeural", "society_language"),
    ("Urban design profoundly affects our daily well-being. Cities designed around cars produce pollution, noise, and social isolation, while walkable neighborhoods with green spaces promote physical activity, community connection, and mental health. The happiest cities in the world share common design principles we can learn from.", "en-GB-MaisieNeural", "society_urban"),

    # Personal Development
    ("The most effective leaders share a surprising trait: they listen more than they talk. Active listening builds trust, surfaces hidden problems, and makes team members feel valued. In a world that rewards loudness and quick opinions, the quiet skill of genuine listening is an underrated superpower.", "en-US-ChristopherNeural", "growth_leadership"),
    ("Developing a growth mindset — the belief that abilities can be developed through effort — transforms how we approach challenges. People with growth mindsets see failure as feedback, not as a verdict on their worth. They embrace difficult tasks because they know struggle is the path to mastery.", "en-GB-SoniaNeural", "growth_mindset"),
    ("Habits, not goals, determine our long-term success. Setting a goal to write a book is easy, but establishing a daily writing habit is what actually produces pages. The key is to start so small that failure is almost impossible — write just fifty words a day, and the momentum will carry you forward.", "en-US-JennyNeural", "growth_habits"),
    ("The ability to focus deeply is becoming a competitive advantage in an age of constant distraction. Research shows that after an interruption, it takes an average of twenty-three minutes to return to the original task. Protecting blocks of uninterrupted time — what some call deep work — is essential for producing meaningful output.", "en-US-GuyNeural", "growth_focus"),
]


async def generate_one(text: str, voice: str, name: str, out_dir: Path) -> dict:
    """Generate one speech sample and return its metadata."""
    import edge_tts

    mp3_path = out_dir / f"{name}.mp3"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(mp3_path))

    # Convert MP3 to WAV via soundfile (librosa backend)
    import librosa
    audio, sr = librosa.load(str(mp3_path), sr=16000, mono=True)
    audio = audio.astype(np.float32)

    wav_path = out_dir / f"{name}.wav"
    sf.write(str(wav_path), audio, 16000)

    # Remove MP3 to save space
    mp3_path.unlink()

    duration = len(audio) / 16000
    return {
        "audio_path": str(wav_path),
        "transcript": text,
        "speaker": voice,
        "duration": duration,
        "topic": name,
    }


async def main():
    out_dir = PROCESSED_DIR / "tts_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(SAMPLES)} speech samples with Microsoft Edge TTS...")
    print("Voices: US English (Aria, Guy, Jenny, etc.) + UK English (Sonia, Ryan, etc.)")
    print()

    entries = []
    for i, (text, voice, name) in enumerate(SAMPLES):
        print(f"  [{i+1:2d}/{len(SAMPLES)}] {name} ({voice})...", end=" ", flush=True)
        entry = await generate_one(text, voice, name, out_dir)
        entries.append(entry)
        print(f"{entry['duration']:.1f}s")

    # Save metadata
    with open(out_dir / "index.json", "w") as f:
        json.dump(entries, f, indent=2)

    total_dur = sum(e["duration"] for e in entries)
    print(f"\nDone! {len(entries)} samples, {total_dur:.0f}s total ({total_dur/60:.1f} min)")
    print(f"Saved to: {out_dir}")
    print(f"  Audio: {out_dir}/*.wav")
    print(f"  Index: {out_dir}/index.json")


if __name__ == "__main__":
    asyncio.run(main())
