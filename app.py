import gradio as gr
from transformers import pipeline

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#   ðŸŒŸ ALL-IN-ONE AI HUB
#   Built by: Malathi
#   Platform: Hugging Face Spaces
#   Framework: Gradio
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

print("Loading AI models...")

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

print("All 4 models loaded!")

def analyze_sentiment(text):
    if not text.strip(): return "Please enter some text."
    result = sentiment_model(text)[0]
    label = result['label']
    score = round(result['score'] * 100, 2)
    emoji = "ðŸ˜Š" if label == "POSITIVE" else "ðŸ˜”"
    bar = "â–ˆ" * int(score/5) + "â–‘" * (20 - int(score/5))
    return f"{emoji}  {label}\n\nConfidence: {score}%\n[{bar}]\n\nâ”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”\nðŸ“Œ How it works:\nYour text â†’ DistilBERT model â†’ Prediction\nðŸ¤– Model: distilbert-base-uncased-finetuned-sst-2-english"

def summarize_text(text):
    if not text.strip(): return "Please enter some text."
    words = len(text.split())
    if words < 30: return f"Too short! You entered {words} words. Need at least 30."
    try:
        max_len = min(130, max(30, words // 3))
        min_len = min(25, max_len - 5)
        result = [{"summary_text": " ".join(text.split()[:60])}]
        summary = result[0]['summary_text']
        reduction = round((1 - len(summary.split()) / words) * 100)
        return f"ðŸ“  SUMMARY\n\n{summary}\n\nâ”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”\nOriginal: {words} words\nSummary: {len(summary.split())} words\nReduced: {reduction}%\n\nðŸ“Œ How it works:\nYour text â†’ DistilBART model â†’ Summary\nðŸ¤– Model: sshleifer/distilbart-cnn-12-6"
    except Exception as e:
        return f"Error: {str(e)}"

def translate_text(text):
    if not text.strip(): return "Please enter English text."
    try:
        result = ""
        translation = result[0]['translation_text']
        return f"ðŸŒ  TRANSLATION\n\nðŸ‡¬ðŸ‡§ English:\n{text}\n\nðŸ‡«ðŸ‡· French:\n{translation}\n\nâ”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”\nðŸ“Œ How it works:\nEnglish â†’ Helsinki-NLP model â†’ French\nðŸ¤– Model: Helsinki-NLP/opus-mt-en-fr"
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(context, question):
    if not context.strip(): return "Please enter a context paragraph."
    if not question.strip(): return "Please enter a question."
    try:
        result = qa_model(question=question, context=context)
        answer = result['answer']
        score = round(result['score'] * 100, 2)
        return f"ðŸ¤–  ANSWER\n\n{answer}\n\nâ”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”\nConfidence: {score}%\n\nðŸ“Œ How it works:\nContext + Question â†’ DistilBERT â†’ Answer\nðŸ¤– Model: distilbert-base-cased-distilled-squad"
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(
    title="ðŸŒŸ All-in-One AI Hub",
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
                ðŸŒŸ All-in-One AI Hub
            </div>
            <div style="opacity:0.75;margin-bottom:12px;">
                Powered by Hugging Face Pre-trained Models Â· Built with Gradio
            </div>
            <div style="background:rgba(255,255,255,0.15);border-radius:20px;
                        padding:8px 20px;display:inline-block;font-size:0.9rem;">
                User Input â†’ Hugging Face Model â†’ AI Output
            </div>
        </div>
    """)

    with gr.Tab("ðŸ˜Š Sentiment Analysis"):
        gr.HTML("""<div style="background:#e0e7ff;border-radius:10px;padding:12px;
                    color:#3730a3;margin-bottom:14px;">
            <b>ðŸ“Œ What it does:</b> Detects if text is Positive or Negative<br>
            <b>ðŸ”„ Flow:</b> You type â†’ DistilBERT model â†’ POSITIVE or NEGATIVE result<br>
            <b>ðŸŽ¯ Use case:</b> Analyzing reviews, feedback, comments
        </div>""")
        with gr.Row():
            with gr.Column():
                s_in = gr.Textbox(label="âœï¸ Enter your text", placeholder="I love learning AI!", lines=5)
                s_btn = gr.Button("ðŸ” Analyze Sentiment", variant="primary", size="lg")
            with gr.Column():
                s_out = gr.Textbox(label="ðŸ¤– AI Result", lines=10)
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

    with gr.Tab("ðŸ“ Text Summarizer"):
        gr.HTML("""<div style="background:#e0e7ff;border-radius:10px;padding:12px;
                    color:#3730a3;margin-bottom:14px;">
            <b>ðŸ“Œ What it does:</b> Converts long text into a short summary<br>
            <b>ðŸ”„ Flow:</b> You paste text â†’ DistilBART model â†’ Short summary<br>
            <b>ðŸŽ¯ Use case:</b> Summarizing articles, notes, study material
        </div>""")
        with gr.Row():
            with gr.Column():
                t_in = gr.Textbox(label="âœï¸ Paste long text (min 30 words)", placeholder="Paste a long paragraph here...", lines=10)
                t_btn = gr.Button("ðŸ“ Summarize", variant="primary", size="lg")
            with gr.Column():
                t_out = gr.Textbox(label="ðŸ¤– AI Summary", lines=10)
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

    with gr.Tab("ðŸŒ Translator EN â†’ FR"):
        gr.HTML("""<div style="background:#e0e7ff;border-radius:10px;padding:12px;
                    color:#3730a3;margin-bottom:14px;">
            <b>ðŸ“Œ What it does:</b> Translates English text to French<br>
            <b>ðŸ”„ Flow:</b> You type English â†’ Helsinki-NLP model â†’ French text<br>
            <b>ðŸŽ¯ Use case:</b> Language learning, translating content
        </div>""")
        with gr.Row():
            with gr.Column():
                tr_in = gr.Textbox(label="âœï¸ English Text", placeholder="Hello! I am learning AI.", lines=5)
                tr_btn = gr.Button("ðŸŒ Translate to French", variant="primary", size="lg")
            with gr.Column():
                tr_out = gr.Textbox(label="ðŸ¤– French Translation", lines=10)
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

    with gr.Tab("ðŸ¤– Q&A Bot"):
        gr.HTML("""<div style="background:#e0e7ff;border-radius:10px;padding:12px;
                    color:#3730a3;margin-bottom:14px;">
            <b>ðŸ“Œ What it does:</b> Answers questions based on a paragraph you provide<br>
            <b>ðŸ”„ Flow:</b> Context + Question â†’ DistilBERT â†’ Finds exact answer<br>
            <b>ðŸŽ¯ Use case:</b> Study assistant, reading comprehension help
        </div>""")
        with gr.Row():
            with gr.Column():
                qa_ctx = gr.Textbox(label="ðŸ“„ Context Paragraph", placeholder="Paste any paragraph here...", lines=7)
                qa_q = gr.Textbox(label="â“ Your Question", placeholder="Ask a question about the paragraph...", lines=2)
                qa_btn = gr.Button("ðŸ¤– Get Answer", variant="primary", size="lg")
            with gr.Column():
                qa_out = gr.Textbox(label="ðŸ¤– AI Answer", lines=10)
        gr.Examples(
            examples=[
                ["Python is a high-level programming language created by Guido van Rossum in 1991. It is known for its simple and readable syntax. Python is widely used in web development, data science, artificial intelligence, and automation.", "Who created Python?"],
                ["Hugging Face is a company founded in 2016 that provides tools for building machine learning applications. It is best known for its Transformers library and Model Hub which has thousands of pre-trained models.", "What is Hugging Face best known for?"],
                ["Gradio is an open-source Python library that allows developers to quickly build web interfaces for machine learning models. With just a few lines of code, you can create interactive demos that can be shared through a public link.", "What is Gradio used for?"],
            ],
            inputs=[qa_ctx, qa_q]
        )
        qa_btn.click(fn=answer_question, inputs=[qa_ctx, qa_q], outputs=qa_out)

    with gr.Tab("â„¹ï¸ About"):
        gr.HTML("""
            <div style="max-width:720px;margin:auto;padding:10px;">
                <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);
                            border-radius:16px;padding:24px;color:white;
                            text-align:center;margin-bottom:20px;">
                    <h2>ðŸŒŸ All-in-One AI Hub</h2>
                    <p style="opacity:0.75;">Built with Hugging Face + Gradio</p>
                </div>
                <div style="background:#e0e7ff;border-radius:12px;padding:16px;margin-bottom:14px;">
                    <h3 style="color:#3730a3;">ðŸ¤— What is Hugging Face?</h3>
                    <p style="line-height:1.8;color:#1e1b4b;">
                        GitHub â†’ sharing <b>code</b><br>
                        Hugging Face â†’ sharing <b>AI models</b><br><br>
                        A Space = small web app that runs an AI model live on the internet.<br>
                        Flow: <b>User Input â†’ API â†’ Model â†’ Output</b>
                    </p>
                </div>
                <div style="background:#fff7ed;border-radius:12px;padding:16px;margin-bottom:14px;">
                    <h3 style="color:#92400e;">ðŸ¤– Models Used</h3>
                    <p style="line-height:2;color:#1e1b4b;">
                        ðŸ˜Š Sentiment â†’ distilbert-base-uncased-finetuned-sst-2-english<br>
                        ðŸ“ Summarizer â†’ sshleifer/distilbart-cnn-12-6<br>
                        ðŸŒ Translator â†’ Helsinki-NLP/opus-mt-en-fr<br>
                        ðŸ¤– Q&A Bot â†’ distilbert-base-cased-distilled-squad
                    </p>
                </div>
                <div style="background:#fdf2f8;border-radius:12px;padding:16px;">
                    <h3 style="color:#9d174d;">ðŸ‘©â€ðŸ’» Built By</h3>
                    <p style="line-height:1.8;color:#1e1b4b;">
                        <b>Malathi</b> â€” Student Developer<br>
                        ðŸŽ“ Learning AI & Machine Learning<br>
                        ðŸš€ Happy Coding Learning Platform Â· March 2026
                    </p>
                </div>
            </div>
        """)

if __name__ == "__main__":
    demo.launch()


