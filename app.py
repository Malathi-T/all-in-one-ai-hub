@"
import gradio as gr
from transformers import pipeline

print('Loading models...')
sentiment_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
emotion_model = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None)
qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
lang_model = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')
ner_model = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', aggregation_strategy='simple')
zero_shot = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
print('All models loaded!')

def analyze_sentiment(text):
    if not text.strip(): return 'Please enter text.'
    r = sentiment_model(text)[0]
    score = round(r['score']*100, 2)
    emoji = '😊' if r['label']=='POSITIVE' else '😔'
    bar = '█'*int(score/5) + '░'*(20-int(score/5))
    return f"{emoji} {r['label']}\n\nConfidence: {score}%\n[{bar}]"

def answer_question(context, question):
    if not context.strip() or not question.strip(): return 'Please enter context and question.'
    r = qa_model(question=question, context=context)
    return f"🤖 ANSWER\n\n{r['answer']}\n\nConfidence: {round(r['score']*100,2)}%"

def detect_emotion(text):
    if not text.strip(): return 'Please enter text.'
    results = sorted(emotion_model(text)[0], key=lambda x: x['score'], reverse=True)
    emojis = {'joy':'😄','sadness':'😢','anger':'😠','fear':'😨','surprise':'😲','disgust':'🤢','neutral':'😐'}
    out = '🎭 EMOTION ANALYSIS\n\n'
    for r in results:
        lbl = r['label'].lower()
        sc = round(r['score']*100,1)
        bar = '█'*int(sc/5)+'░'*(20-int(sc/5))
        out += f"{emojis.get(lbl,'🔵')} {lbl.upper():10} {sc:5.1f}%  [{bar}]\n"
    return out

def detect_language(text):
    if not text.strip(): return 'Please enter text.'
    results = lang_model(text, top_k=5)
    names = {'en':'English 🇬🇧','fr':'French 🇫🇷','de':'German 🇩🇪','es':'Spanish 🇪🇸','it':'Italian 🇮🇹','hi':'Hindi 🇮🇳','ar':'Arabic 🇸🇦','zh':'Chinese 🇨🇳','ja':'Japanese 🇯🇵','ru':'Russian 🇷🇺'}
    out = '🌐 LANGUAGE DETECTION\n\n'
    for r in results:
        sc = round(r['score']*100,1)
        nm = names.get(r['label'], r['label'])
        bar = '█'*int(sc/5)+'░'*(20-int(sc/5))
        out += f"{nm:22} {sc:5.1f}%  [{bar}]\n"
    return out

def extract_keywords(text):
    if not text.strip(): return 'Please enter text.'
    entities = ner_model(text)
    stop = {'the','a','an','is','are','was','in','on','at','to','for','of','and','or','but','it','this','that'}
    freq = {}
    for w in text.split():
        w2 = w.lower().strip('.,!?')
        if len(w2)>3 and w2 not in stop:
            freq[w2] = freq.get(w2,0)+1
    top = sorted(freq.items(), key=lambda x:x[1], reverse=True)[:8]
    out = '🎯 KEYWORDS\n\n📍 Named Entities:\n'
    seen = set()
    for e in entities[:8]:
        if e['word'] not in seen:
            out += f"  • {e['word']:20} [{e['entity_group']}]\n"
            seen.add(e['word'])
    out += '\n📊 Top Keywords:\n'
    for word,count in top:
        out += f"  • {word:20} {'█'*min(count*2,10)} ({count}x)\n"
    return out

def check_grammar(text):
    if not text.strip(): return 'Please enter text.'
    issues = []
    if text[0].islower(): issues.append('❌ Start with a capital letter')
    if not text.rstrip().endswith(('.','!','?')): issues.append('❌ End with punctuation')
    if '  ' in text: issues.append('❌ Remove double spaces')
    out = f'🔤 GRAMMAR CHECK\n\nWords: {len(text.split())}\n\n'
    if not issues: return out + '✅ No issues found! Great writing! 🌟'
    return out + '\n'.join(issues)

def analyze_resume(text):
    if not text.strip(): return 'Please paste resume text.'
    sections = {
        'Contact Info': any(k in text.lower() for k in ['email','phone','linkedin']),
        'Education': any(k in text.lower() for k in ['education','university','degree']),
        'Experience': any(k in text.lower() for k in ['experience','worked','company']),
        'Skills': any(k in text.lower() for k in ['skills','python','javascript']),
        'Projects': any(k in text.lower() for k in ['project','built','developed']),
    }
    score = min(100, sum(sections.values())*18)
    grade = 'A+' if score>=90 else 'A' if score>=80 else 'B' if score>=60 else 'C'
    bar = '█'*(score//5)+'░'*(20-score//5)
    out = f'📄 RESUME SCORE\n\nScore: {score}/100  Grade: {grade}\n[{bar}]\n\n'
    for s,f in sections.items():
        out += f"  {'✅' if f else '❌'} {s}\n"
    return out

def detect_fake_news(text):
    if not text.strip(): return 'Please enter news text.'
    result = zero_shot(text, candidate_labels=['real news','fake news','satire','opinion'])
    out = '🤔 FAKE NEWS ANALYSIS\n\n'
    for label,score in zip(result['labels'],result['scores']):
        sc = round(score*100,1)
        bar = '█'*int(sc/5)+'░'*(20-int(sc/5))
        out += f"  {label:15} {sc:5.1f}%  [{bar}]\n"
    top = result['labels'][0]
    verdict = '🔴 LIKELY FAKE' if 'fake' in top else '🟡 SATIRE' if 'satire' in top else '🟢 LIKELY REAL'
    return out + f'\nVerdict: {verdict}'

def text_to_bullets(text):
    if not text.strip(): return 'Please enter text.'
    sents = [s.strip() for s in text.replace('!','.').replace('?','.').split('.') if len(s.strip())>15]
    out = '💬 BULLET POINTS\n\n'
    for s in sents[:8]: out += f'  • {s}\n'
    return out

def translate_text(text, lang):
    if not text.strip(): return 'Please enter English text.'
    codes = {'French 🇫🇷':'fr','German 🇩🇪':'de','Spanish 🇪🇸':'es','Italian 🇮🇹':'it','Hindi 🇮🇳':'hi','Arabic 🇸🇦':'ar','Portuguese 🇵🇹':'pt','Russian 🇷🇺':'ru'}
    code = codes.get(lang,'fr')
    try:
        tr = pipeline('translation', model=f'Helsinki-NLP/opus-mt-en-{code}')
        return f"🌍 TRANSLATION\n\n🇬🇧 English:\n{text}\n\n{lang.split()[-1]} {lang.split()[0]}:\n{tr(text)[0]['translation_text']}"
    except Exception as e:
        return f'Error: {str(e)}'

with gr.Blocks(title='🌟 All-in-One AI Hub', theme=gr.themes.Soft(primary_hue='indigo')) as demo:
    gr.HTML('<div style="background:linear-gradient(135deg,#0f172a,#1e1b4b,#164e63);border-radius:16px;padding:24px;text-align:center;color:white;margin-bottom:20px;"><div style="font-size:2rem;font-weight:bold;">🌟 All-in-One AI Hub</div><div style="opacity:0.75;">12 AI Features · Hugging Face · By Malathi</div></div>')

    with gr.Tab('😊 Sentiment'):
        with gr.Row():
            with gr.Column():
                s_in=gr.Textbox(label='Enter text',lines=4)
                gr.Button('Analyze',variant='primary').click(analyze_sentiment,s_in,gr.Textbox(label='Result',lines=7))

    with gr.Tab('🤖 Q&A Bot'):
        with gr.Row():
            with gr.Column():
                q_ctx=gr.Textbox(label='Context paragraph',lines=5)
                q_q=gr.Textbox(label='Your question',lines=2)
                q_btn=gr.Button('Get Answer',variant='primary')
            with gr.Column():
                q_out=gr.Textbox(label='Answer',lines=7)
        q_btn.click(answer_question,[q_ctx,q_q],q_out)

    with gr.Tab('🎭 Emotions'):
        with gr.Row():
            with gr.Column():
                e_in=gr.Textbox(label='Enter text',lines=4)
                gr.Button('Detect',variant='primary').click(detect_emotion,e_in,gr.Textbox(label='Emotions',lines=10))

    with gr.Tab('🌐 Lang Detect'):
        with gr.Row():
            with gr.Column():
                l_in=gr.Textbox(label='Enter text in any language',lines=4)
                gr.Button('Detect Language',variant='primary').click(detect_language,l_in,gr.Textbox(label='Result',lines=8))

    with gr.Tab('🎯 Keywords'):
        with gr.Row():
            with gr.Column():
                k_in=gr.Textbox(label='Enter text',lines=5)
                k_btn=gr.Button('Extract',variant='primary')
            with gr.Column():
                k_out=gr.Textbox(label='Keywords',lines=8)
        k_btn.click(extract_keywords,k_in,k_out)

    with gr.Tab('🔤 Grammar'):
        with gr.Row():
            with gr.Column():
                gr_in=gr.Textbox(label='Enter text',lines=5)
                gr_btn=gr.Button('Check Grammar',variant='primary')
            with gr.Column():
                gr_out=gr.Textbox(label='Report',lines=8)
        gr_btn.click(check_grammar,gr_in,gr_out)

    with gr.Tab('📄 Resume'):
        with gr.Row():
            with gr.Column():
                r_in=gr.Textbox(label='Paste resume',lines=10)
                r_btn=gr.Button('Analyze Resume',variant='primary')
            with gr.Column():
                r_out=gr.Textbox(label='Score',lines=10)
        r_btn.click(analyze_resume,r_in,r_out)

    with gr.Tab('🤔 Fake News'):
        with gr.Row():
            with gr.Column():
                f_in=gr.Textbox(label='Paste news',lines=5)
                f_btn=gr.Button('Check News',variant='primary')
            with gr.Column():
                f_out=gr.Textbox(label='Analysis',lines=8)
        f_btn.click(detect_fake_news,f_in,f_out)

    with gr.Tab('💬 Bullets'):
        with gr.Row():
            with gr.Column():
                b_in=gr.Textbox(label='Enter text',lines=7)
                b_btn=gr.Button('Convert',variant='primary')
            with gr.Column():
                b_out=gr.Textbox(label='Bullet Points',lines=7)
        b_btn.click(text_to_bullets,b_in,b_out)

    with gr.Tab('🌍 Translator'):
        with gr.Row():
            with gr.Column():
                t_in=gr.Textbox(label='English text',lines=4)
                t_lang=gr.Dropdown(choices=['French 🇫🇷','German 🇩🇪','Spanish 🇪🇸','Italian 🇮🇹','Hindi 🇮🇳','Arabic 🇸🇦','Portuguese 🇵🇹','Russian 🇷🇺'],value='French 🇫🇷',label='Target Language')
                t_btn=gr.Button('Translate',variant='primary')
            with gr.Column():
                t_out=gr.Textbox(label='Translation',lines=8)
        t_btn.click(translate_text,[t_in,t_lang],t_out)

demo.launch()
"@ | Out-File -FilePath app.py -Encoding UTF8