import gradio as gr
from transformers import pipeline

# Load models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
lang_pipeline = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
fake_pipeline = pipeline("text-classification", model="hamzab/roberta-fake-news-classification")
grammar_pipeline = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text."
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)
    emoji = "😊" if label == "POSITIVE" else "😞"
    return f"{emoji} {label}\nConfidence: {score}%"

def summarize_text(text):
    if not text.strip():
        return "Please enter some text."
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return "Text too short to summarize."
    summary = ". ".join(sentences[:3]) + "."
    return f"Summary:\n{summary}"

def extract_keywords(text):
    if not text.strip():
        return "Please enter some text."
    entities = ner_pipeline(text)
    if not entities:
        return "No keywords found."
    result = ""
    for ent in entities:
        result += f"🔑 {ent['word']} => {ent['entity_group']} ({round(ent['score']*100, 1)}%)\n"
    return result.strip()

def detect_language(text):
    if not text.strip():
        return "Please enter some text."
    result = lang_pipeline(text)[0]
    lang = result["label"].upper()
    score = round(result["score"] * 100, 2)
    return f"🌐 Language: {lang}\nConfidence: {score}%"

def check_fake_news(text):
    if not text.strip():
        return "Please enter some text."
    result = fake_pipeline(text)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)
    if label.upper() in ["FAKE", "LABEL_0"]:
        return f"🚨 FAKE NEWS Detected!\nConfidence: {score}%"
    else:
        return f"✅ Likely REAL News\nConfidence: {score}%"

def make_bullets(text):
    if not text.strip():
        return "Please enter some text."
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return "Text too short."
    bullets = "\n".join([f"* {s}" for s in sentences])
    return f"Bullet Points:\n{bullets}"

def check_grammar(text):
    if not text.strip():
        return "Please enter some text."
    result = grammar_pipeline(f"grammar: {text}", max_length=512)[0]["generated_text"]
    return f"Corrected Text:\n{result}"

def score_resume(text):
    if not text.strip():
        return "Please enter resume text."
    keywords = ["experience", "skills", "education", "project", "python", "java", "sql", "team", "leadership", "communication"]
    text_lower = text.lower()
    found = [k for k in keywords if k in text_lower]
    score = min(len(found) * 10, 100)
    return f"📄 Resume Score: {score}/100\n\nFound keywords: {', '.join(found) if found else 'None'}\nTip: Add more skills, projects and experience!"

custom_css = """
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 40px 20px;
    text-align: center;
    margin-bottom: 10px;
}
.header-title {
    font-size: 2.2em;
    font-weight: bold;
    color: #ff6b35 !important;
}
.info-box {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    line-height: 1.8;
}
"""

with gr.Blocks(css=custom_css, title="All-in-One AI Hub") as demo:

    gr.HTML("""
    <div class="header-banner">
        <div class="header-title">🔥 All-in-One AI Hub</div>
        <div style="color:#ffffff; font-size:1em; margin-top:10px; background:rgba(255,255,255,0.15); border-radius:20px; padding:6px 20px; display:inline-block;">
            User Input → Hugging Face Model → AI Output
        </div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("😊 Sentiment"):
            gr.HTML('<div class="info-box"><b>What it does:</b> Detects if text is Positive or Negative<br><b>Flow:</b> You type → DistilBERT model → POSITIVE or NEGATIVE result<br><b>Use case:</b> Analyzing reviews, feedback, comments</div>')
            with gr.Row():
                s_input = gr.Textbox(label="Enter your text", placeholder="I love learning AI!", lines=5)
                s_output = gr.Textbox(label="AI Result", lines=5)
            gr.Button("🔍 Analyze Sentiment", variant="primary").click(analyze_sentiment, inputs=s_input, outputs=s_output)
            gr.Examples(["I love this product!", "This is terrible.", "Today was okay."], inputs=s_input)

        with gr.Tab("📝 Summarizer"):
            gr.HTML('<div class="info-box"><b>What it does:</b> Shortens long text into key points<br><b>Flow:</b> You paste long text → Model extracts key sentences → Summary<br><b>Use case:</b> Summarizing articles, essays, reports</div>')
            with gr.Row():
                sum_input = gr.Textbox(label="Enter your text", placeholder="Paste a long paragraph here...", lines=5)
                sum_output = gr.Textbox(label="AI Result", lines=5)
            gr.Button("📝 Summarize Text", variant="primary").click(summarize_text, inputs=sum_input, outputs=sum_output)

        with gr.Tab("🎯 Keywords"):
            gr.HTML('<div class="info-box"><b>What it does:</b> Extracts named entities — people, places, organizations<br><b>Flow:</b> You type → BERT NER model → Keywords + entity type<br><b>Use case:</b> Extracting important names from articles</div>')
            with gr.Row():
                k_input = gr.Textbox(label="Enter your text", placeholder="Elon Musk founded Tesla in California.", lines=5)
                k_output = gr.Textbox(label="AI Result", lines=5)
            gr.Button("🔍 Extract Keywords", variant="primary").click(extract_keywords, inputs=k_input, outputs=k_output)
            gr.Examples(["Elon Musk founded Tesla in California.", "Apple was created by Steve Jobs."], inputs=k_input)

        with gr.Tab("🌐 Lang Detect"):
            gr.HTML('<div class="info-box"><b>What it does:</b> Detects which language the text is written in<br><b>Flow:</b> You type → XLM-RoBERTa model → Language + confidence<br><b>Use case:</b> Auto-detecting language of user input</div>')
            with gr.Row():
                l_input = gr.Textbox(label="Enter your text", placeholder="Type in any language...", lines=5)
                l_output = gr.Textbox(label="AI Result", lines=5)
            gr.Button("🌐 Detect Language", variant="primary").click(detect_language, inputs=l_input, outputs=l_output)
            gr.Examples(["Bonjour le monde", "Hola mundo", "Thank you"], inputs=l_input)

        with gr.Tab("🤔 Fake News"):
            gr.HTML('<div class="info-box"><b>What it does:</b> Detects if a news headline is Real or Fake<br><b>Flow:</b> You paste headline → RoBERTa model → REAL or FAKE result<br><b>Use case:</b> Fact-checking news headlines</div>')
            with gr.Row():
                f_input = gr.Textbox(label="Enter news headline", placeholder="Paste a news headline here...", lines=5)
                f_output = gr.Textbox(label="AI Result", lines=5)
            gr.Button("🔍 Check News", variant="primary").click(check_fake_news, inputs=f_input, outputs=f_output)

        with gr.Tab("💬 Bullets"):
            gr.HTML('<div class="info-box"><b>What it does:</b> Converts paragraph into bullet points<br><b>Flow:</b> You paste text → Splits into sentences → Bullet list<br><b>Use case:</b> Making notes, summaries, presentations</div>')
            with gr.Row():
                b_input = gr.Textbox(label="Enter your text", placeholder="Type a paragraph...", lines=5)
                b_output = gr.Textbox(label="AI Result", lines=5)
            gr.Button("💬 Make Bullet Points", variant="primary").click(make_bullets, inputs=b_input, outputs=b_output)

        with gr.Tab("🔤 Grammar"):
            gr.HTML('<div class="info-box"><b>What it does:</b> Fixes grammar errors in your text<br><b>Flow:</b> You type → T5 grammar model → Corrected sentence<br><b>Use case:</b> Proofreading emails, essays, messages</div>')
            with gr.Row():
                g_input = gr.Textbox(label="Enter your text", placeholder="He go to school yesterday.", lines=5)
                g_output = gr.Textbox(label="AI Result", lines=5)
            gr.Button("🔤 Check Grammar", variant="primary").click(check_grammar, inputs=g_input, outputs=g_output)
            gr.Examples(["He go to school yesterday.", "She don't like apples."], inputs=g_input)

        with gr.Tab("📄 Resume"):
            gr.HTML('<div class="info-box"><b>What it does:</b> Scores your resume based on keywords<br><b>Flow:</b> You paste resume → Keyword analysis → Score out of 100<br><b>Use case:</b> Checking resume strength before applying</div>')
            with gr.Row():
                r_input = gr.Textbox(label="Paste your resume", placeholder="Paste your resume text here...", lines=5)
                r_output = gr.Textbox(label="AI Result", lines=5)
            gr.Button("📄 Score Resume", variant="primary").click(score_resume, inputs=r_input, outputs=r_output)

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## 🔥 All-in-One AI Hub
**Built by:** Malathi Thirupathi
**Powered by:** Hugging Face Transformers

| Tab | Feature |
|-----|---------|
| 😊 Sentiment | Positive/Negative detector |
| 📝 Summarizer | Long text to short summary |
| 🎯 Keywords | Named entity extractor |
| 🌐 Lang Detect | Any language detector |
| 🤔 Fake News | Real/Fake news detector |
| 💬 Bullets | Text to bullet points |
| 🔤 Grammar | Grammar correction |
| 📄 Resume | Resume scorer |
            """)

demo.launch()