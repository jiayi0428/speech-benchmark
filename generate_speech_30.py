"""Generate 30 diverse TTS samples with varied intents (NOT all inform)."""
import asyncio, json, numpy as np, soundfile as sf
from pathlib import Path
from src.config import PROCESSED_DIR

# 30 samples across 4 intent categories, diverse topics, varied sentiment
SAMPLES = [
    # ====== PERSUADE (8) ======
    (
        "We need to talk about how social media is destroying our attention spans. Every notification, every infinite scroll, every autoplay video is engineered to keep you hooked. The average person now checks their phone 144 times a day. That's not connectivity — that's addiction. Delete the apps that don't serve you. Reclaim your focus.",
"en-US-GuyNeural", "persuade_socialmedia"
    ),
    (
        "Learning a second language is the single best investment you can make in your brain. Studies show bilingualism delays dementia by up to five years, improves executive function, and even changes how you perceive color. I'm not exaggerating — the cognitive benefits are backed by decades of neuroscience. Start today. Even fifteen minutes a day makes a difference.",
        "en-GB-SoniaNeural", "persuade_language"
    ),
    (
        "Here's why you should stop using plastic water bottles immediately. A single plastic bottle takes 450 years to decompose. Every piece of plastic ever made still exists somewhere on this planet. And microplastics are now in our bloodstream. Switch to a reusable bottle. It costs twenty dollars and lasts for years. The planet cannot afford your convenience.",
        "en-US-MichelleNeural", "persuade_plastic"
    ),
    (
        "Public libraries are the most underrated institutions in modern society. They provide free access to knowledge, internet, community spaces, and lifelong learning — all without asking for a single dollar. Yet we chronically underfund them while spending billions on things that matter far less. Fund your local library. It's one of the few truly democratic spaces we have left.",
        "en-US-AnaNeural", "persuade_libraries"
    ),
    (
        "If you're not getting enough sleep, you're sabotaging everything else you do. Sleep deprivation impairs judgment as much as being legally drunk. Your memory consolidation happens during deep sleep. Your immune system repairs itself. You cannot outwork bad sleep with more coffee. Prioritize eight hours. Your future self will thank you.",
        "en-US-GuyNeural", "persuade_sleep"
    ),
    (
        "We should all learn to cook at least five simple meals from scratch. Not for Instagram, not to impress anyone — but because cooking is a fundamental life skill that affects your health, your budget, and your independence. Processed food is engineered to be addictive and nutritionally empty. When you cook, you control exactly what goes into your body.",
        "en-GB-RyanNeural", "persuade_cooking"
    ),
    (
        "Reading fiction makes you a better person. No, really. Research shows that people who read literary fiction score higher on empathy tests and theory of mind assessments. When you read a novel, you practice understanding characters whose experiences are nothing like yours. That's literally empathy training. Put down the self-help book and pick up a novel.",
        "en-US-JennyNeural", "persuade_fiction"
    ),
    (
        "We are over-medicalizing normal human emotions. Sadness is not always depression. Worry is not always anxiety. Being energetic is not always ADHD. The pharmaceutical industry profits when we pathologize the full range of human experience. Sometimes you're just having a bad day — and that's okay. Not everything needs a diagnosis and a prescription.",
        "en-GB-LibbyNeural", "persuade_overmedicalize"
    ),

    # ====== ENTERTAIN (8) ======
    (
        "So I tried to assemble a piece of IKEA furniture yesterday. The instructions had no words — just a happy cartoon man pointing at things. Four hours later, I had something that looked vaguely like a bookshelf, except I had six screws left over and the doors opened in opposite directions. The cartoon man was still smiling. I was not.",
        "en-US-ChristopherNeural", "entertain_ikea"
    ),
    (
        "My grandmother once told me she met my grandfather because he accidentally threw a sandwich at her head in a crowded cafeteria. He was trying to toss it to his friend across the room, and his aim was apparently terrible. She turned around, furious, holding a tuna sandwich. He was so embarrassed he offered to buy her dinner. Sixty-two years of marriage started with airborne seafood.",
        "en-US-JaneNeural", "entertain_grandma"
    ),
    (
        "Let me tell you about the time I tried to impress a date by cooking a fancy French dinner. I had watched exactly one YouTube video and felt confident. The recipe called for flambé, which I now know means 'set on fire.' The kitchen did not burn down, but the smoke alarm went off, the cat hid under the sofa for three hours, and my date suggested we order pizza. We've been married for eight years now. She still doesn't let me flambé.",
        "en-GB-ThomasNeural", "entertain_flambe"
    ),
    (
        "Cats are just tiny, judgmental roommates who don't pay rent. Mine stares at me while I eat as if I've personally offended her ancestors. She knocks things off tables not because she's curious, but because she wants to remind me that gravity still works and she's in charge. Yesterday she sat in a cardboard box for four hours, ignoring a sixty-dollar cat bed. I respect her commitment to chaos.",
        "en-US-JennyNeural", "entertain_cats"
    ),
    (
        "The autocorrect on my phone has developed a personality, and it's not a helpful one. I tried to text my boss 'I'll be there soon' and it sent 'I'll be there spoon.' I tried to tell my mother I was feeling grateful and it changed it to 'feeling regretful.' My phone is either trying to ruin my life or it knows something about my subconscious that I don't.",
        "en-US-SaraNeural", "entertain_autocorrect"
    ),
    (
        "Parallel parking is the ultimate test of human character. You either emerge a hero, sliding perfectly into the spot in one smooth motion while imaginary spectators applaud, or you spend five minutes doing a seventeen-point turn while pedestrians judge you silently. There is no middle ground. Every successful parallel park deserves a certificate of achievement.",
        "en-US-GuyNeural", "entertain_parking"
    ),
    (
        "I've discovered that adulthood is mostly just Googling things you feel like you should already know. How long to boil an egg. What's the difference between a deduction and a credit. Is it normal for a plant to turn yellow. Nobody actually knows anything. We're all just pretending and hoping nobody notices. The secret is out.",
        "en-US-AriaNeural", "entertain_adulthood"
    ),
    (
        "My fitness tracker has become passive-aggressive. It buzzes at ten in the morning: 'You have 250 steps. Great start!' It's not a great start. It's a judgment. At eleven it suggests I breathe mindfully. By two PM it's just showing me a sad face emoji. I did not ask for a wrist-mounted guilt machine, yet here we are.",
        "en-GB-MaisieNeural", "entertain_fitness"
    ),

    # ====== DESCRIBE (8) ======
    (
        "Picture the Grand Canyon at sunrise. The first light hits the top of the rim, turning the red rock to molten gold. Layer after layer of geological history — two billion years of Earth's memory — stacked in bands of crimson, ochre, and pale limestone. A river, impossibly small from this height, snakes through the bottom, still carving, still patient. The air is cool and thin. A hawk circles below you, not above.",
        "en-US-ChristopherNeural", "describe_grandcanyon"
    ),
    (
        "The taste of a perfectly ripe mango is unlike anything else in nature. You bite through the golden flesh and it's simultaneously sweet and tart, with a floral fragrance that fills your entire head. The texture is silky, almost custard-like. Juice runs down your chin and you don't care. It tastes like sunshine condensed into fruit form. You close your eyes involuntarily. That's how good it is.",
        "en-US-AriaNeural", "describe_mango"
    ),
    (
        "The Northern Lights appear without warning. One moment the Arctic sky is black and silent. The next, a ribbon of green unfurls across the horizon like a curtain being drawn by invisible hands. It pulses, it breathes, it shifts from emerald to violet. The cold is forgotten. The world shrinks to just you and this impossibly beautiful phenomenon that humans have been staring at in wonder for thousands of years.",
        "en-GB-SoniaNeural", "describe_aurora"
    ),
    (
        "Walking through a dense bamboo forest in Kyoto is like entering another dimension. The stalks rise thirty, forty feet on either side, filtering sunlight into pale green stripes. The only sound is the hollow knock of bamboo swaying against bamboo — a wooden wind chime played by nature. The path ahead curves and disappears. You walk slowly, because rushing would somehow be disrespectful to the centuries of stillness here.",
        "en-US-EricNeural", "describe_bamboo"
    ),
    (
        "A thunderstorm approaches from the west. First the air changes — it gets heavy, charged, smelling of ozone and wet earth. The wind picks up, whipping leaves into spirals. Then the sky darkens, not gradually but like a dimmer switch being turned down fast. The first fat raindrops hit the pavement with audible splats. A flash of lightning illuminates the clouds from within, and seven seconds later, the thunder rolls through your chest.",
        "en-US-ChristopherNeural", "describe_thunderstorm"
    ),
    (
        "The smell of a bakery at five in the morning is the most comforting thing in the world. Warm yeast, caramelizing sugar, toasted nuts, and fresh butter create an invisible cloud that wraps around you the moment you walk in. The baker, dusted in flour, pulls a tray of croissants from the oven — they're golden, flaky, exhaling steam. You buy one and it's still too hot to eat, but you don't care.",
        "en-GB-MaisieNeural", "describe_bakery"
    ),
    (
        "An ancient olive tree stands alone on a hill in Tuscany. Its trunk is so gnarled and twisted it looks more like stone than wood — each groove telling a story of drought survived, of centuries endured. The leaves are silver-green, shimmering when a breeze passes. Underneath, the shade is dappled and cool. This tree was here before cars, before electricity, before the country it stands in even had a name.",
        "en-US-JaneNeural", "describe_olivetree"
    ),
    (
        "The moment you step off a plane in a foreign country, there's a sensory overload that's both exhilarating and disorienting. The air smells different — maybe of spices you can't name, or tropical flowers, or diesel and street food. The announcements are in a language you barely understand. The money looks like colorful monopoly bills. Everything is unfamiliar, which means everything is an adventure waiting to happen.",
        "en-US-SaraNeural", "describe_travel"
    ),

    # ====== QUESTION (6) ======
    (
        "What does it actually mean to live a good life? Is it about achievement — the promotions, the awards, the things you can list on a resume? Or is it about the moments that nobody sees? The laughter with old friends, the quiet Sunday mornings, the feeling of sunlight on your face. We spend decades chasing external markers of success. But when people look back on their lives, they rarely wish they had worked more.",
        "en-US-GuyNeural", "question_goodlife"
    ),
    (
        "Why do we dream? Every night, your brain constructs entire worlds, populates them with people you've never met, and tells you stories that make no logical sense — and you accept them completely until you wake up. Is dreaming just the brain's way of filing memories? Is it problem-solving? Is it a glimpse into our subconscious fears and desires? After thousands of years of wondering, we still don't have a definitive answer.",
        "en-GB-RyanNeural", "question_dreams"
    ),
    (
        "If you could send a one-sentence message to every single person on Earth, what would you say? You have seven billion listeners for exactly one moment. Do you inspire them? Warn them? Make them laugh? The terrifying thing is that most of us can't even decide what to say to our own families at dinner, let alone to the entire human species. Maybe the question itself tells us something about the limits of universal truth.",
        "en-US-MichelleNeural", "question_message"
    ),
    (
        "How much of your personality is actually you, and how much is just the people you happened to grow up around? If you had been born in a different country, to different parents, speaking a different language — would you still be you? Your taste in music, your political beliefs, your sense of humor — how much of it is freely chosen versus absorbed from your environment? The unsettling answer might be: less than you think.",
        "en-US-AnaNeural", "question_personality"
    ),
    (
        "What would happen if we treated every conversation as if it might be our last? Not in a morbid way, but in a way that made us actually listen, actually pay attention, actually say the things we've been putting off. How many 'I love you's go unspoken? How many apologies never get delivered? We all know life is short, but we live as if we have infinite tomorrows. What if we didn't?",
        "en-GB-ThomasNeural", "question_lastwords"
    ),
    (
        "Is technology bringing us closer together or pushing us further apart? We can video call someone on the other side of the planet instantly, yet we've never been lonelier as a species. We have more ways to communicate than ever before, yet the quality of our conversations seems to be deteriorating. Are we connecting, or are we just distracting ourselves from the absence of real connection?",
        "en-US-EricNeural", "question_technology"
    ),
]

FALLBACK_VOICES = ["en-US-AriaNeural","en-US-GuyNeural","en-GB-SoniaNeural","en-GB-RyanNeural","en-US-JennyNeural"]

async def generate_one(text, voice, name, out_dir):
    import edge_tts
    mp3_path = out_dir / f"{name}.mp3"
    voices_to_try = [voice] + [v for v in FALLBACK_VOICES if v != voice]
    last_error = None
    used_voice = voice
    for v in voices_to_try:
        try:
            communicate = edge_tts.Communicate(text, v)
            await communicate.save(str(mp3_path))
            used_voice = v
            break
        except Exception as e:
            last_error = e
            continue
    else:
        raise last_error
    import librosa
    audio, sr = librosa.load(str(mp3_path), sr=16000, mono=True)
    audio = audio.astype(np.float32)
    wav_path = out_dir / f"{name}.wav"
    sf.write(str(wav_path), audio, 16000)
    mp3_path.unlink()
    duration = len(audio) / 16000
    return {
        "audio_path": str(wav_path),
        "transcript": text,
        "speaker": used_voice,
        "duration": duration,
        "topic": name,
        "intent": name.split("_")[0],
        "sentiment_hint": (
            "negative" if any(w in name for w in ["plastic","socialmedia","overmedicalize","parking","adulthood","autocorrect","fitness","lastwords","technology"])
            else "positive" if any(w in name for w in ["language","libraries","sleep","cooking","fiction","grandma","flambe","mango","aurora","bakery","olivetree","goodlife","dreams","personality"])
            else "neutral"
        ),
    }

async def main():
    out_dir = PROCESSED_DIR / "tts_samples"
    existing = set(p.stem for p in out_dir.glob("*.wav"))

    # Load existing index
    index_path = out_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            entries = json.load(f)
    else:
        entries = []

    print(f"Generating {len(SAMPLES)} diverse TTS samples...")
    print(f"Intent distribution: persuade={sum(1 for _,_,n in SAMPLES if n.startswith('persuade'))}, "
          f"entertain={sum(1 for _,_,n in SAMPLES if n.startswith('entertain'))}, "
          f"describe={sum(1 for _,_,n in SAMPLES if n.startswith('describe'))}, "
          f"question={sum(1 for _,_,n in SAMPLES if n.startswith('question'))}")
    print()

    for i, (text, voice, name) in enumerate(SAMPLES):
        if name in existing:
            print(f"  [{i+1:2d}/{len(SAMPLES)}] SKIP {name} (exists)")
            continue
        print(f"  [{i+1:2d}/{len(SAMPLES)}] {name} ({voice})...", end=" ", flush=True)
        entry = await generate_one(text, voice, name, out_dir)
        entries.append(entry)
        print(f"{entry['duration']:.1f}s  intent={entry['intent']}  sentiment_hint={entry['sentiment_hint']}")

    with open(index_path, "w") as f:
        json.dump(entries, f, indent=2)

    # Stats
    intents = {}
    for e in entries:
        intent = e.get("intent", "unknown")
        intents[intent] = intents.get(intent, 0) + 1
    total_dur = sum(e["duration"] for e in entries)
    print(f"\nDone! {len(entries)} total samples, {total_dur:.0f}s ({total_dur/60:.1f} min)")
    print(f"Intent distribution: {intents}")
    print(f"All 30 new samples generated (skipped {sum(1 for _,_,n in SAMPLES if n in existing)} existing)")

if __name__ == "__main__":
    asyncio.run(main())
