
import os
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import gradio as gr
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ---------- Environment & Clients ----------
def setup_openai():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
    return OpenAI(api_key=api_key)

# PDF text extraction
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

# ---------- RAG Functions ----------
def load_and_partition_text(file_path, chunk_size=300, chunk_overlap=50):
    if file_path.lower().endswith(".pdf"):
        content = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    sections = content.split("####")
    partitions = {}
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        header, body = sec.split("\n", 1) if "\n" in sec else (sec, "")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", ".", "。"]
        )
        docs = [Document(page_content=body)]
        splits = splitter.split_documents(docs)
        segments = [f"{header}\n{d.page_content}" for d in splits]
        partitions[header] = segments
    return partitions


def initialize_rag(partitions, model_name="intfloat/multilingual-e5-base"):
    device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
    embed_model = SentenceTransformer(model_name, device=device)
    indexes, segments_map = {}, {}
    for sec, texts in partitions.items():
        embeddings = embed_model.encode(texts, show_progress_bar=False)
        dim = embeddings.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(np.array(embeddings, dtype="float32"))
        indexes[sec] = idx
        segments_map[sec] = texts
    return embed_model, indexes, segments_map


def query_rag(query, embed_model, indexes, segments_map, top_k=3, threshold=1.5):
    q_emb = embed_model.encode([query])
    results = []
    for sec, idx in indexes.items():
        dists, ids = idx.search(np.array(q_emb, dtype="float32"), k=top_k)
        for dist, i in zip(dists[0], ids[0]):
            if i >= 0 and dist < threshold:
                results.append(segments_map[sec][i])
    return results


def generate_answer(openai_client, query, contexts, model_name="gpt-4.1", max_tokens=1000):
    context = "\n".join(contexts)[:2000]
    messages = [
        {"role": "system", "content": "你是論文閱讀助理，請根據上下文回答問題。"},
        {"role": "user", "content": f"問題: {query}\n上下文: {context}"}
    ]
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

# ---------- Setup ----------
openai_client = setup_openai()
# Store RAG instances per file path
dag = {}

# Function to handle file uploads and initialize RAG for each

def handle_upload(files):
    choices = []
    for f in files:
        path = f.name
        partitions = load_and_partition_text(path)
        embed_model, indexes, segments_map = initialize_rag(partitions)
        dag[path] = (embed_model, indexes, segments_map)
        choices.append(path)
    return gr.update(choices=choices, value=choices[0] if choices else None)

# Chat function using selected file

def chat(query, selected_file):
    if selected_file not in dag:
        return "請先上傳並選擇 PDF 檔。"
    embed_model, indexes, segments_map = dag[selected_file]
    contexts = query_rag(query, embed_model, indexes, segments_map)
    return generate_answer(openai_client, query, contexts)

# ---------- Gradio Interface ----------
with gr.Blocks() as demo:
    gr.Markdown("### AIITNTPU RAG + GPT 查詢介面")
    with gr.Row():
        upload = gr.File(label="上傳 PDF 檔案", file_types=['.pdf'], file_count="multiple")
        selector = gr.Dropdown(label="選擇要查詢的 PDF", choices=[])
    upload.change(handle_upload, inputs=upload, outputs=selector)
    with gr.Row():
        txt = gr.Textbox(label="輸入查詢內容", placeholder="請輸入你的問題...")
        btn = gr.Button("送出")
    out = gr.Textbox(label="回答結果")
    btn.click(chat, inputs=[txt, selector], outputs=out)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)