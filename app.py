import gradio as gr
from transformers import pipeline
import re

print("===== Application Startup =====")
print("Loading AI models...")

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("All models loaded!")

def extractive_summary(text, num_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= num_sentences:
        return text
    words = re.findall(r'\w+', text.lower())
    freq = {}
    for w in words:
        if len(w) > 4:
            freq[w] = freq.get(w, 0) + 1
    scores = [(sum(freq.get(w.lower(), 0) for w in re.findall(r'\w+', s)), s) for s in sentences]
    return " ".join([s for _, s in sorted(scores, reverse=True)[:num_sentences]])

def analyze_sentiment(text):
    if not text.strip(): return "Please enter some text!"
    r = sentiment_model(text)[0]
    emoji = "😊" if r['label'] == "POSITIVE" else "😔"
    return f"{emoji} **{r['label']}**\nConfidence: {r['score']*100:.1f}%"

def summarize_text(text):
    if not text.strip(): return "Please enter some text!"
    if len(text.split()) < 20: return "Please enter at least 20 words."
    s = extractive_summary(text)
    return f"📝 **Summary ({len(s.split())} words from {len(text.split())}):**\n\n{s}"

def detect_emotion(text):
    if not text.strip(): return "Please enter some text!"
    results = sorted(emotion_model(text)[0], key=lambda x: x['score'], reverse=True)
    emoji_map = {"joy":"😄","sadness":"😢","anger":"😠","fear":"😨","surprise":"😲","disgust":"🤢","neutral":"😐"}
    out = "🎭 **Emotion Analysis:**\n\n"
    for r in results[:4]:
        l = r['label'].lower()
        bar = "█"*int(r['score']*10) + "░"*(10-int(r['score']*10))
        out += f"{emoji_map.get(l,'🔵')} {l.capitalize()}: {bar} {r['score']*100:.1f}%\n"
    return out

def extract_keywords(text):
    if not text.strip(): return "Please enter some text!"
    try:
        entities = ner_model(text)
        kw = list(set([e['word'] for e in entities if len(e['word']) > 2]))
        if kw: return "🎯 **Keywords:**\n\n" + ", ".join(kw[:20])
    except: pass
    words = list(set(re.findall(r'\b[A-Z][a-z]+\b', text)))[:10]
    return "🎯 **Keywords:**\n\n" + (", ".join(words) if words else "No keywords found.")

def detect_language(text):
    if not text.strip(): return "Please enter some text!"
    labels = ["English","French","Spanish","German","Hindi","Italian","Portuguese","Arabic"]
    r = classifier(text, candidate_labels=labels)
    out = f"🌐 **Detected: {r['labels'][0]}** ({r['scores'][0]*100:.1f}%)\n\n"
    for l, s in zip(r['labels'][:5], r['scores'][:5]):
        out += f"{l}: {'█'*int(s*20)}{'░'*(20-int(s*20))} {s*100:.1f}%\n"
    return out

def detect_fake_news(text):
    if not text.strip(): return "Please enter some text!"
    labels = ["reliable news","fake news","satire","misleading content","opinion"]
    r = classifier(text, candidate_labels=labels)
    em = {"reliable news":"✅","fake news":"❌","satire":"😏","misleading content":"⚠️","opinion":"💬"}
    out = f"{em.get(r['labels'][0],'🔍')} **{r['labels'][0].upper()}** ({r['scores'][0]*100:.1f}%)\n\n"
    for l, s in zip(r['labels'], r['scores']):
        out += f"{em.get(l,'🔵')} {l}: {s*100:.1f}%\n"
    return out

def text_to_bullets(text):
    if not text.strip(): return "Please enter some text!"
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 15]
    return "💬 **Key Points:**\n\n" + "\n".join(f"• {s}" for s in sentences[:8]) if sentences else "Text too short."

def check_grammar(text):
    if not text.strip(): return "Please enter some text!"
    issues = []
    if not text[0].isupper(): issues.append("• Start with a capital letter")
    if text[-1] not in '.!?': issues.append("• End with proper punctuation")
    repeated = re.findall(r'\b(\w+)\s+\1\b', text.lower())
    if repeated: issues.append(f"• Repeated words: {', '.join(set(repeated))}")
    return "✅ **No major issues!**" if not issues else "🔤 **Grammar Issues:**\n\n" + "\n".join(issues)

def analyze_resume(text):
    if not text.strip(): return "Please paste your resume!"
    sections = {"Skills": bool(re.search(r'\bskill', text.lower())), "Experience": bool(re.search(r'\bexperience|\bwork', text.lower())),
                "Education": bool(re.search(r'\beducation|\bdegree', text.lower())), "Projects": bool(re.search(r'\bproject', text.lower())),
                "Contact": bool(re.search(r'\bemail|\bphone|\blinkedin', text.lower()))}
    r = classifier(text[:500], candidate_labels=["strong resume","average resume","needs improvement"])
    out = f"📄 **{r['labels'][0].upper()}** ({r['scores'][0]*100:.1f}%)\n\n"
    for s, found in sections.items(): out += f"{'✅' if found else '❌'} {s}\n"
    missing = [s for s, f in sections.items() if not f]
    if missing: out += f"\n💡 Add: {', '.join(missing)}"
    return out

with gr.Blocks(title="🌟 All-in-One AI Hub", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌟 All-in-One AI Hub\n### Built by Malathi Thirupathi | Powered by Hugging Face 🤗\n---")
    with gr.Tabs():
        with gr.Tab("😊 Sentiment"):
            i1 = gr.Textbox(lines=3, label="Your Text", placeholder="Type any sentence...")
            o1 = gr.Markdown()
            gr.Button("🔍 Analyze", variant="primary").click(analyze_sentiment, i1, o1)
            gr.Examples(["I love learning AI!", "This is terrible."], i1)
        with gr.Tab("📝 Summarizer"):
            i2 = gr.Textbox(lines=6, label="Long Text", placeholder="Paste a long paragraph...")
            o2 = gr.Markdown()
            gr.Button("📝 Summarize", variant="primary").click(summarize_text, i2, o2)
        with gr.Tab("🎭 Emotion"):
            i3 = gr.Textbox(lines=3, label="Your Text", placeholder="Type any sentence...")
            o3 = gr.Markdown()
            gr.Button("🎭 Detect Emotion", variant="primary").click(detect_emotion, i3, o3)
            gr.Examples(["I just got promoted! Best day ever!", "I am so angry right now."], i3)
        with gr.Tab("🎯 Keywords"):
            i4 = gr.Textbox(lines=5, label="Your Text", placeholder="Paste any text...")
            o4 = gr.Markdown()
            gr.Button("🎯 Extract Keywords", variant="primary").click(extract_keywords, i4, o4)
        with gr.Tab("🌐 Lang Detect"):
            i5 = gr.Textbox(lines=3, label="Text", placeholder="Type text in any language...")
            o5 = gr.Markdown()
            gr.Button("🌐 Detect Language", variant="primary").click(detect_language, i5, o5)
            gr.Examples(["Bonjour comment allez vous", "Hola como estas"], i5)
        with gr.Tab("🤔 Fake News"):
            i6 = gr.Textbox(lines=5, label="News Text", placeholder="Paste a news headline...")
            o6 = gr.Markdown()
            gr.Button("🤔 Analyze", variant="primary").click(detect_fake_news, i6, o6)
        with gr.Tab("💬 Bullets"):
            i7 = gr.Textbox(lines=5, label="Your Text", placeholder="Paste any paragraph...")
            o7 = gr.Markdown()
            gr.Button("💬 Convert", variant="primary").click(text_to_bullets, i7, o7)
        with gr.Tab("🔤 Grammar"):
            i8 = gr.Textbox(lines=4, label="Your Text", placeholder="Type your text...")
            o8 = gr.Markdown()
            gr.Button("🔤 Check Grammar", variant="primary").click(check_grammar, i8, o8)
        with gr.Tab("📄 Resume"):
            i9 = gr.Textbox(lines=8, label="Resume Text", placeholder="Paste your resume...")
            o9 = gr.Markdown()
            gr.Button("📄 Analyze Resume", variant="primary").click(analyze_resume, i9, o9)
        with gr.Tab("ℹ️ About"):
            gr.Markdown("## About\nAll-in-One AI Hub with 9 AI features.\n\n**Developer:** Malathi Thirupathi | AI/ML Trainee\n\n**Models:** DistilBERT, RoBERTa, BERT-NER, BART-MNLI")

demo.launch()
