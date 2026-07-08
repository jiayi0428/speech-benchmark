"""Create ground truth annotations for all 50 TTS samples and save to ground_truth.json."""
import json
from pathlib import Path

# Load existing ground truth
gt_path = Path("D:/speech-benchmark/data/ground_truth.json")
with open(gt_path) as f:
    gt = json.load(f)

# Load sample transcripts from index.json
with open("D:/speech-benchmark/data/processed/tts_samples/index.json") as f:
    index = json.load(f)

transcripts = {Path(e["audio_path"]).stem: e["transcript"] for e in index}

# ====== Ground Truth for 30 new samples ======
new_gt = {
    # PERSUADE (8) — all neutral/negative/positive sentiment
    "persuade_socialmedia": {
        "summary": "Social media is engineered to be addictive, destroying attention spans, and people should delete unhelpful apps to reclaim their focus.",
        "sentiment": "negative",
        "keywords": ["social media", "attention span", "addiction", "notifications", "focus", "dopamine"],
        "intent": "persuade",
    },
    "persuade_language": {
        "summary": "Learning a second language provides significant cognitive benefits including delayed dementia and improved executive function, and even fifteen minutes a day makes a difference.",
        "sentiment": "positive",
        "keywords": ["language learning", "bilingualism", "dementia", "cognitive benefits", "neuroscience", "executive function"],
        "intent": "persuade",
    },
    "persuade_plastic": {
        "summary": "Plastic bottles take 450 years to decompose and microplastics are entering human bloodstreams, so people should switch to reusable bottles immediately.",
        "sentiment": "negative",
        "keywords": ["plastic pollution", "microplastics", "reusable bottle", "decomposition", "environment", "convenience"],
        "intent": "persuade",
    },
    "persuade_libraries": {
        "summary": "Public libraries provide free access to knowledge and community spaces but are chronically underfunded, and citizens should support local library funding.",
        "sentiment": "positive",
        "keywords": ["public libraries", "funding", "knowledge", "community", "democratic spaces", "free access"],
        "intent": "persuade",
    },
    "persuade_sleep": {
        "summary": "Sleep deprivation impairs judgment as much as being drunk, and prioritizing eight hours of sleep is essential for memory, immunity, and overall health.",
        "sentiment": "positive",
        "keywords": ["sleep", "sleep deprivation", "memory consolidation", "immune system", "judgment", "health"],
        "intent": "persuade",
    },
    "persuade_cooking": {
        "summary": "Everyone should learn to cook at least five simple meals because cooking controls health, budget, and independence, unlike processed food which is addictive and nutritionally empty.",
        "sentiment": "positive",
        "keywords": ["cooking", "processed food", "health", "independence", "life skills", "nutrition"],
        "intent": "persuade",
    },
    "persuade_fiction": {
        "summary": "Reading literary fiction improves empathy and theory of mind, functioning as empathy training through understanding diverse characters.",
        "sentiment": "positive",
        "keywords": ["fiction", "reading", "empathy", "theory of mind", "literary fiction", "novels"],
        "intent": "persuade",
    },
    "persuade_overmedicalize": {
        "summary": "Normal human emotions like sadness and worry are being over-medicalized into depression and anxiety, partly driven by pharmaceutical industry profits.",
        "sentiment": "negative",
        "keywords": ["over-medicalization", "emotions", "pharmaceutical industry", "depression", "anxiety", "diagnosis"],
        "intent": "persuade",
    },

    # ENTERTAIN (8) — humorous storytelling, most are neutral or positive
    "entertain_ikea": {
        "summary": "The speaker humorously describes a disastrous attempt to assemble IKEA furniture, ending up with extra screws and backward doors while the cartoon instructions remained cheerfully unhelpful.",
        "sentiment": "neutral",
        "keywords": ["IKEA", "furniture assembly", "instructions", "cartoon", "DIY failure", "humor"],
        "intent": "entertain",
    },
    "entertain_grandma": {
        "summary": "A grandmother met her husband when he accidentally threw a tuna sandwich at her head in a cafeteria, leading to sixty-two years of marriage.",
        "sentiment": "positive",
        "keywords": ["grandmother", "love story", "sandwich", "cafeteria", "marriage", "serendipity"],
        "intent": "entertain",
    },
    "entertain_flambe": {
        "summary": "The speaker's attempt to impress a date with flambe cooking triggered the smoke alarm and hid the cat, but they ended up married despite the kitchen disaster.",
        "sentiment": "positive",
        "keywords": ["flambe", "cooking disaster", "date", "smoke alarm", "cat", "marriage"],
        "intent": "entertain",
    },
    "entertain_cats": {
        "summary": "Cats are described as tiny judgmental roommates who knock things off tables and ignore expensive beds in favor of cardboard boxes.",
        "sentiment": "neutral",
        "keywords": ["cats", "pets", "judgmental", "cardboard box", "cat bed", "humor"],
        "intent": "entertain",
    },
    "entertain_autocorrect": {
        "summary": "The speaker's phone autocorrect has developed an unhelpful personality, changing innocent messages into embarrassing ones and potentially revealing subconscious truths.",
        "sentiment": "negative",
        "keywords": ["autocorrect", "phone", "texting", "typos", "embarrassment", "technology"],
        "intent": "entertain",
    },
    "entertain_parking": {
        "summary": "Parallel parking is described as the ultimate test of human character where one either emerges a hero in one smooth motion or performs a humiliating seventeen-point turn.",
        "sentiment": "negative",
        "keywords": ["parallel parking", "driving", "humiliation", "pedestrians", "achievement", "humor"],
        "intent": "entertain",
    },
    "entertain_adulthood": {
        "summary": "Adulthood is revealed to be mostly Googling basic life skills while pretending to have everything figured out, exposing the collective secret that nobody actually knows what they're doing.",
        "sentiment": "neutral",
        "keywords": ["adulthood", "Googling", "pretending", "life skills", "imposter syndrome", "humor"],
        "intent": "entertain",
    },
    "entertain_fitness": {
        "summary": "The speaker's fitness tracker has become passive-aggressive, progressing from sarcastic encouragement to a sad face emoji by afternoon.",
        "sentiment": "negative",
        "keywords": ["fitness tracker", "passive-aggressive", "steps", "guilt", "technology", "wearable"],
        "intent": "entertain",
    },

    # DESCRIBE (8) — vivid sensory descriptions, neutral or positive
    "describe_grandcanyon": {
        "summary": "At sunrise, the Grand Canyon's red rock turns molten gold, revealing two billion years of geological layers while a hawk circles below and the Colorado River continues carving through time.",
        "sentiment": "neutral",
        "keywords": ["Grand Canyon", "sunrise", "geological layers", "Colorado River", "hawk", "limestone"],
        "intent": "describe",
    },
    "describe_mango": {
        "summary": "A perfectly ripe mango delivers a silky, custard-like texture with a sweet-tart floral flavor so intense it makes you close your eyes involuntarily, like sunshine condensed into fruit form.",
        "sentiment": "positive",
        "keywords": ["mango", "ripe fruit", "sweet", "tart", "floral", "juicy", "tropical"],
        "intent": "describe",
    },
    "describe_aurora": {
        "summary": "The Northern Lights appear as ribbons of green unfurling across the Arctic sky, pulsing and shifting from emerald to violet, a phenomenon that has captivated humans for thousands of years.",
        "sentiment": "positive",
        "keywords": ["Northern Lights", "aurora", "Arctic", "green ribbon", "emerald", "violet", "sky"],
        "intent": "describe",
    },
    "describe_bamboo": {
        "summary": "Walking through a dense bamboo forest in Kyoto feels like entering another dimension, with forty-foot stalks filtering sunlight into pale green stripes accompanied by the hollow knocking sound of bamboo swaying.",
        "sentiment": "neutral",
        "keywords": ["bamboo forest", "Kyoto", "sunlight", "stillness", "nature", "Japan", "zen"],
        "intent": "describe",
    },
    "describe_thunderstorm": {
        "summary": "A thunderstorm approaches with heavy ozone-scented air and darkening sky, followed by fat raindrops, lightning illuminating clouds from within, and thunder rolling through the chest seven seconds later.",
        "sentiment": "neutral",
        "keywords": ["thunderstorm", "lightning", "thunder", "rain", "ozone", "storm", "atmosphere"],
        "intent": "describe",
    },
    "describe_bakery": {
        "summary": "A bakery at five in the morning smells of warm yeast and caramelizing sugar, where a flour-dusted baker pulls golden croissants from the oven that are too hot to eat but impossible to resist.",
        "sentiment": "positive",
        "keywords": ["bakery", "croissants", "yeast", "baker", "morning", "fresh bread", "aroma"],
        "intent": "describe",
    },
    "describe_olivetree": {
        "summary": "An ancient olive tree in Tuscany has a trunk so gnarled it resembles stone, with silver-green leaves shimmering in the breeze, having stood on that hill since before the country had a name.",
        "sentiment": "positive",
        "keywords": ["olive tree", "Tuscany", "ancient", "trunk", "silver-green", "centuries", "Italy"],
        "intent": "describe",
    },
    "describe_travel": {
        "summary": "Stepping off a plane in a foreign country triggers a sensory explosion of unfamiliar smells, incomprehensible language, strange currency, and the exhilarating realization that everything is an adventure.",
        "sentiment": "positive",
        "keywords": ["travel", "foreign country", "sensory overload", "adventure", "culture shock", "exploration"],
        "intent": "describe",
    },

    # QUESTION (6) — philosophical reflection
    "question_goodlife": {
        "summary": "What constitutes a good life — is it professional achievements or quiet private moments? People rarely wish they had worked more when reflecting on their lives.",
        "sentiment": "neutral",
        "keywords": ["good life", "achievement", "happiness", "work-life balance", "meaning", "reflection"],
        "intent": "question",
    },
    "question_dreams": {
        "summary": "Why do humans dream every night, constructing elaborate worlds that feel completely real? Despite millennia of inquiry, science still lacks a definitive explanation for dreaming.",
        "sentiment": "neutral",
        "keywords": ["dreams", "subconscious", "sleep", "memory", "brain", "neuroscience", "mystery"],
        "intent": "question",
    },
    "question_message": {
        "summary": "If you could send one sentence to every person on Earth, what would you say? The difficulty of answering reveals something about the limits of universal truth.",
        "sentiment": "neutral",
        "keywords": ["message", "humanity", "universal truth", "communication", "philosophy", "thought experiment"],
        "intent": "question",
    },
    "question_personality": {
        "summary": "How much of one's personality is truly self-determined versus absorbed from environment — would you still be you if born in a different country to different parents?",
        "sentiment": "neutral",
        "keywords": ["personality", "identity", "nature vs nurture", "culture", "upbringing", "free will"],
        "intent": "question",
    },
    "question_lastwords": {
        "summary": "What would happen if we treated every conversation as potentially our last, finally saying the things we've been postponing and delivering apologies and expressions of love?",
        "sentiment": "negative",
        "keywords": ["last words", "mortality", "conversation", "apology", "love", "regret", "urgency"],
        "intent": "question",
    },
    "question_technology": {
        "summary": "Is technology bringing humanity closer together or further apart? Despite unprecedented connectivity, loneliness is epidemic and conversation quality appears to be deteriorating.",
        "sentiment": "negative",
        "keywords": ["technology", "loneliness", "connection", "social media", "communication", "isolation"],
        "intent": "question",
    },
}

# Merge into ground truth (overwrite if exists, add if new)
gt.update(new_gt)

with open(gt_path, "w") as f:
    json.dump(gt, f, indent=2)

# Report
intents = {}
for k, v in gt.items():
    intents[v.get("intent", "unknown")] = intents.get(v.get("intent", "unknown"), 0) + 1
sentiments = {}
for k, v in gt.items():
    sentiments[v.get("sentiment", "unknown")] = sentiments.get(v.get("sentiment", "unknown"), 0) + 1

print(f"Ground truth updated: {len(gt)} total entries")
print(f"Intent distribution: {intents}")
print(f"Sentiment distribution: {sentiments}")
print(f"\nNew entries: {list(new_gt.keys())}")
