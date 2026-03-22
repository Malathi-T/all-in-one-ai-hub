import gradio as gr
from transformers import pipeline

# ══════════════════════════════════════════════════════
#   🌟 ALL-IN-ONE AI HUB
#   Built by: Malathi
#   Platform: Hugging Face Spaces
#   Framework: Gradio
# ══════════════════════════════════════════════════════

print("Loading AI models...")

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
translator_model = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

print("All 4 models loaded!")

def analyze_sentiment(text):
    if not text.strip(): return "Please enter some text."
    result = sentiment_model(text)[0]
    label = result['label']
    score = round(result['score'] * 100, 2)
    emoji = "😊" if label == "POSITIVE" else "😔"
    bar = "█" * int(score/5) + "░" * (20 - int(score/5))
    return f"{emoji}  {label}\n\nConfidence: {score}%\n[{bar}]\n\n━━━━━━━━━━━━━━━━━━\n📌 How it works:\nYour text → DistilBERT model → Prediction\n🤖 Model: distilbert-base-uncased-finetuned-sst-2-english"

def summarize_text(text):
    if not text.strip(): return "Please enter some text."
    words = len(text.split())
    if words < 30: return f"Too short! You entered {words} words. Need at least 30."
    try:
        max_len = min(130, max(30, words // 3))
        min_len = min(25, max_len - 5)
        result = summarizer_model(text, max_length=max_len, min_length=min_len, do_sample=False)
        summary = result[0]['summary_text']
        reduction = round((1 - len(summary.split()) / words) * 100)
        return f"📝  SUMMARY\n\n{summary}\n\n━━━━━━━━━━━━━━━━━━\nOriginal: {words} words\nSummary: {len(summary.split())} words\nReduced: {reduction}%\n\n📌 How it works:\nYour text → DistilBART model → Summary\n🤖 Model: sshleifer/distilbart-cnn-12-6"
    except Exception as e:
        return f"Error: {str(e)}"

def translate_text(text):
    if not text.strip(): return "Please enter English text."
    try:
        result = translator_model(text)
        translation = result[0]['translation_text']
        return f"🌍  TRANSLATION\n\n🇬🇧 English:\n{text}\n\n🇫🇷 French:\n{translation}\n\n━━━━━━━━━━━━━━━━━━\n📌 How it works:\nEnglish → Helsinki-NLP model → French\n🤖 Model: Helsinki-NLP/opus-mt-en-fr"
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(context, question):
    if not context.strip(): return "Please enter a context paragraph."
    if not question.strip(): return "Please enter a question."
    try:
        result = qa_model(question=question, context=context)
        answer = result['answer']
        score = round(result['score'] * 100, 2)
        return f"🤖  ANSWER\n\n{answer}\n\n━━━━━━━━━━━━━━━━━━\nConfidence: {score}%\n\n📌 How it works:\nContext + Question → DistilBERT → Answer\n🤖 Model: distilbert-base-cased-distilled-squad"
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(
    title="🌟 All-in-One AI Hub",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan"),
    css="""
        .gradio-container { max-width: 900px !important; margin: auto !important; }
    """
) as demo:

    gr.HTML("""
        <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b,#164e63);
                    border-radius:16px;padding:28px;text-align:center;
                    color:white;margin-bottom:20px;">
            <div style="font-size:2rem;font-weight:bold;margin-bottom:8px;">
                🌟 All-in-One AI Hub
            </div>
            <div style="opacity:0.75;margin-bottom:12px;">
                Powered by Hugging Face Pre-trained Models · Built with Gradio
            </div>
            <div style="background:rgba(255,255,255,0.15);border-radius:20px;
                        padding:8px 20px;display:inline-block;font-size:0.9rem;">
                User Input → Hugging Face Model → AI Output
            </div>
        </div>
    """)

    with gr.Tab("😊 Sentiment Analysis"):
        gr.HTML("""<div style="background:#e0e7ff;border-radius:10px;padding:12px;
                    color:#3730a3;margin-bottom:14px;">
            <b>📌 What it does:</b> Detects if text is Positive or Negative<br>
            <b>🔄 Flow:</b> You type → DistilBERT model → POSITIVE or NEGATIVE result<br>
            <b>🎯 Use case:</b> Analyzing reviews, feedback, comments
        </div>""")
        with gr.Row():
            with gr.Column():
                s_in = gr.Textbox(label="✏️ Enter your text", placeholder="I love learning AI!", lines=5)
                s_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
            with gr.Column():
                s_out = gr.Textbox(label="🤖 AI Result", lines=10)
        gr.Examples(
            examples=[
                ["I absolutely love this course! It is amazing and very helpful."],
                ["This is terrible. I am very disappointed with the results."],
                ["The weather today is nice and I feel great!"],
                ["I hate when things do not work. Very frustrating."],
            ],
            inputs=s_in
        )
        s_btn.click(fn=analyze_sentiment, inputs=s_in, outputs=s_out)

    with gr.Tab("📝 Text Summarizer"):
        gr.HTML("""<div style="background:#e0e7ff;border-radius:10px;padding:12px;
                    color:#3730a3;margin-bottom:14px;">
            <b>📌 What it does:</b> Converts long text into a short summary<br>
            <b>🔄 Flow:</b> You paste text → DistilBART model → Short summary<br>
            <b>🎯 Use case:</b> Summarizing articles, notes, study material
        </div>""")
        with gr.Row():
            with gr.Column():
                t_in = gr.Textbox(label="✏️ Paste long text (min 30 words)", placeholder="Paste a long paragraph here...", lines=10)
                t_btn = gr.Button("📝 Summarize", variant="primary", size="lg")
            with gr.Column():
                t_out = gr.Textbox(label="🤖 AI Summary", lines=10)
        gr.Examples(
            examples=[["""Machine learning is a branch of artificial intelligence that enables
            computers to learn from data without being explicitly programmed.
            Instead of writing rules manually, machine learning algorithms find
            patterns in large datasets and use those patterns to make predictions.
            There are three main types: supervised learning, unsupervised learning,
            and reinforcement learning. Supervised learning uses labeled data,
            while unsupervised learning finds hidden patterns in unlabeled data.
            Machine learning is used in image recognition, natural language processing,
            fraud detection, recommendation systems, and medical diagnosis."""]],
            inputs=t_in
        )
        t_btn.click(fn=summarize_text, inputs=t_in, outputs=t_out)

    with gr.Tab("🌍 Translator EN → FR"):
        gr.HTML("""<div style="background:#e0e7ff;border-radius:10px;padding:12px;
                    color:#3730a3;margin-bottom:14px;">
            <b>📌 What it does:</b> Translates English text to French<br>
            <b>🔄 Flow:</b> You type English → Helsinki-NLP model → French text<br>
            <b>🎯 Use case:</b> Language learning, translating content
        </div>""")
        with gr.Row():
            with gr.Column():
                tr_in = gr.Textbox(label="✏️ English Text", placeholder="Hello! I am learning AI.", lines=5)
                tr_btn = gr.Button("🌍 Translate to French", variant="primary", size="lg")
            with gr.Column():
                tr_out = gr.Textbox(label="🤖 French Translation", lines=10)
        gr.Examples(
            examples=[
                ["Hello! My name is Malathi and I am learning Artificial Intelligence."],
                ["Hugging Face is a platform for sharing AI models."],
                ["I built an AI application using Python and Gradio."],
                ["Thank you for visiting my All-in-One AI Hub!"],
            ],
            inputs=tr_in
        )
        tr_btn.click(fn=translate_text, inputs=tr_in, outputs=tr_out)

    with gr.Tab("🤖 Q&A Bot"):
        gr.HTML("""<div style="background:#e0e7ff;border-radius:10px;padding:12px;
                    color:#3730a3;margin-bottom:14px;">
            <b>📌 What it does:</b> Answers questions based on a paragraph you provide<br>
            <b>🔄 Flow:</b> Context + Question → DistilBERT → Finds exact answer<br>
            <b>🎯 Use case:</b> Study assistant, reading comprehension help
        </div>""")
        with gr.Row():
            with gr.Column():
                qa_ctx = gr.Textbox(label="📄 Context Paragraph", placeholder="Paste any paragraph here...", lines=7)
                qa_q = gr.Textbox(label="❓ Your Question", placeholder="Ask a question about the paragraph...", lines=2)
                qa_btn = gr.Button("🤖 Get Answer", variant="primary", size="lg")
            with gr.Column():
                qa_out = gr.Textbox(label="🤖 AI Answer", lines=10)
        gr.Examples(
            examples=[
                ["Python is a high-level programming language created by Guido van Rossum in 1991. It is known for its simple and readable syntax. Python is widely used in web development, data science, artificial intelligence, and automation.", "Who created Python?"],
                ["Hugging Face is a company founded in 2016 that provides tools for building machine learning applications. It is best known for its Transformers library and Model Hub which has thousands of pre-trained models.", "What is Hugging Face best known for?"],
                ["Gradio is an open-source Python library that allows developers to quickly build web interfaces for machine learning models. With just a few lines of code, you can create interactive demos that can be shared through a public link.", "What is Gradio used for?"],
            ],
            inputs=[qa_ctx, qa_q]
        )
        qa_btn.click(fn=answer_question, inputs=[qa_ctx, qa_q], outputs=qa_out)

    with gr.Tab("ℹ️ About"):
        gr.HTML("""
            <div style="max-width:720px;margin:auto;padding:10px;">
                <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);
                            border-radius:16px;padding:24px;color:white;
                            text-align:center;margin-bottom:20px;">
                    <h2>🌟 All-in-One AI Hub</h2>
                    <p style="opacity:0.75;">Built with Hugging Face + Gradio</p>
                </div>
                <div style="background:#e0e7ff;border-radius:12px;padding:16px;margin-bottom:14px;">
                    <h3 style="color:#3730a3;">🤗 What is Hugging Face?</h3>
                    <p style="line-height:1.8;color:#1e1b4b;">
                        GitHub → sharing <b>code</b><br>
                        Hugging Face → sharing <b>AI models</b><br><br>
                        A Space = small web app that runs an AI model live on the internet.<br>
                        Flow: <b>User Input → API → Model → Output</b>
                    </p>
                </div>
                <div style="background:#fff7ed;border-radius:12px;padding:16px;margin-bottom:14px;">
                    <h3 style="color:#92400e;">🤖 Models Used</h3>
                    <p style="line-height:2;color:#1e1b4b;">
                        😊 Sentiment → distilbert-base-uncased-finetuned-sst-2-english<br>
                        📝 Summarizer → sshleifer/distilbart-cnn-12-6<br>
                        🌍 Translator → Helsinki-NLP/opus-mt-en-fr<br>
                        🤖 Q&A Bot → distilbert-base-cased-distilled-squad
                    </p>
                </div>
                <div style="background:#fdf2f8;border-radius:12px;padding:16px;">
                    <h3 style="color:#9d174d;">👩‍💻 Built By</h3>
                    <p style="line-height:1.8;color:#1e1b4b;">
                        <b>Malathi</b> — Student Developer<br>
                        🎓 Learning AI & Machine Learning<br>
                        🚀 Happy Coding Learning Platform · March 2026
                    </p>
                </div>
            </div>
        """)

if __name__ == "__main__":
    demo.launch()
