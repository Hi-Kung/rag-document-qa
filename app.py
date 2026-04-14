import os
import io
import requests
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── 配置 ────────────────────────────────────────────
SILICONFLOW_KEY = os.environ.get("SILICONFLOW_API_KEY")
BASE_URL        = "https://api.siliconflow.cn/v1"

@st.cache_resource
def get_models():
    emb = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=SILICONFLOW_KEY,
        openai_api_base=BASE_URL
    )
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        openai_api_key=SILICONFLOW_KEY,
        openai_api_base=BASE_URL,
        temperature=0
    )
    return emb, llm

# ── 文件解析 ────────────────────────────────────────
def parse_uploaded_file(uploaded_file):
    """解析上传的文件，返回 Document 对象，失败返回 None。"""
    filename = uploaded_file.name
    data = uploaded_file.read()
    if filename.endswith('.txt'):
        try:    text = data.decode('utf-8')
        except: text = data.decode('gbk', errors='ignore')
    elif filename.endswith('.pdf'):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = '\n\n'.join([p.extract_text() or '' for p in reader.pages])
    else:
        return None
    text = text.strip()
    if not text:
        return None
    return Document(page_content=text, metadata={'source': filename})

def build_from_documents(docs_list):
    """
    从 Document 列表构建内存向量库。
    （不设 persist_directory），用户每次上传自动重建，
    同时返回 chunks 供混合检索使用。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_documents(docs_list)
    emb, _ = get_models()
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=None   # 内存模式
    )
    return vs, chunks, len(chunks)

# ── 混合检索：加权 RRF 融合 ──────────────────────────
def hybrid_retrieve(question: str, vectorstore, all_chunks: list, k: int = 3) -> list:
    """
    向量检索（权重0.7）+ BM25关键词检索（权重0.3），加权RRF融合。
    两路各召回 k*2 条，融合后取前 k 条，互不挤占。
    """
    fetch_k = k * 2

    # 路一：向量检索
    vec_docs = vectorstore.as_retriever(
        search_kwargs={"k": fetch_k}
    ).invoke(question)

    # 路二：BM25 关键词检索（用于精确词命中，比如人名、地名）
    bm25_docs = []
    if all_chunks:
        try:
            from langchain_community.retrievers import BM25Retriever
            bm25_docs = BM25Retriever.from_documents(
                all_chunks, k=fetch_k
            ).invoke(question)
        except Exception:
            pass

    # 加权 RRF 融合
    VEC_W, BM25_W = 0.7, 0.3
    scores, doc_map = {}, {}
    for rank, doc in enumerate(vec_docs):
        key = doc.page_content[:80]
        scores[key]  = scores.get(key, 0) + VEC_W / (rank + 60)
        doc_map[key] = doc
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:80]
        scores[key]  = scores.get(key, 0) + BM25_W / (rank + 60)
        doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    mixed_docs = [doc_map[key] for key in sorted_keys[:k]]
    reranked_docs = rerank_docs(question, mixed_docs, top_n=k)
    return reranked_docs

def rerank_docs(query: str, docs: list, top_n=3):
    """
    对文档列表进行 reranking，返回前 top_n 条。
    每个文档的 metadata 中会添加 rerank_score 字段
    """
    if not docs:
        return docs
    resp = requests.post(
        "https://api.siliconflow.cn/v1/rerank",
        headers={"Authorization": f"Bearer {SILICONFLOW_KEY}",
                 "Content-Type": "application/json"},
        json={"model": "BAAI/bge-reranker-v2-m3", "query": query,
             "documents": [d.page_content for d in docs],
             "top_n": top_n, "return_documents": True}
    )
    reranked = []
    for item in resp.json().get("results", []):
        doc = docs[item["index"]]
        doc.metadata["rerank_score"] = round(item["relevance_score"], 4)
        reranked.append(doc)
    return reranked

# ── RAG 问答 ────────────────────────────────────────
RAG_PROMPT = """你是文档问答助手。请严格基于以下文档内容回答问题。
如文档中没有相关信息，请说"文档中未提及此内容"，不要编造。

文档内容：
{context}

对话历史：
{history}

当前问题：{question}

回答："""

def rag_answer(question, vectorstore, all_chunks, chat_history, k=3):
    relevant_docs = hybrid_retrieve(question, vectorstore, all_chunks, k)
    context = "\n\n---\n\n".join([
        f"[{d.metadata.get('source','?')}]\n{d.page_content}"
        for d in relevant_docs
    ])
    recent = chat_history[-6:]
    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}" for m in recent
    ]) if recent else "（无历史）"

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    _, llm = get_models()
    response = llm.invoke(prompt.format_messages(
        context=context, history=history_text, question=question
    ))
    return response.content, relevant_docs

# ── 页面配置 ────────────────────────────────────────
st.set_page_config(page_title="文档智能问答", page_icon="📄", layout="wide")

for key, default in [
    ('vectorstore', None), ('all_chunks', []),
    ('messages', []), ('uploaded_names', []), ('chunk_count', 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 侧边栏：文件上传 ─────────────────────────────────
with st.sidebar:
    st.title('📄 上传文档')
    st.caption('支持 .txt 和 .pdf，可多选')

    uploaded_files = st.file_uploader(
        '选择文件',
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        label_visibility='collapsed'
    )

    if uploaded_files:
        new_names = sorted([f.name for f in uploaded_files])
        if new_names != st.session_state.uploaded_names:
            with st.spinner(f'正在解析 {len(uploaded_files)} 个文件并建库...'):
                docs, failed = [], []
                for uf in uploaded_files:
                    doc = parse_uploaded_file(uf)
                    if doc:  docs.append(doc)
                    else:    failed.append(uf.name)

                if docs:
                    vs, chunks, n_chunks = build_from_documents(docs)
                    st.session_state.vectorstore    = vs
                    st.session_state.all_chunks     = chunks
                    st.session_state.uploaded_names = new_names
                    st.session_state.chunk_count    = n_chunks
                    st.session_state.messages       = []
                    st.success(f'建库完成！{len(docs)} 文件，{n_chunks} 片段')
                    if failed:
                        st.warning(f'以下文件解析失败：{failed}')
                else:
                    st.error('所有文件解析失败，请检查文件格式')

    st.markdown('---')
    if st.session_state.uploaded_names:
        st.caption('当前知识库：')
        for name in st.session_state.uploaded_names:
            st.caption(f'  📄 {name}')
        st.caption(f'共 {st.session_state.chunk_count} 个片段')

    st.markdown('---')
    k_value = st.slider('检索片段数 k', 1, 6, 3)

    if st.button('🗑️ 清空对话', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── 主区域 ───────────────────────────────────────────
st.title('📄 文档智能问答')

if st.session_state.vectorstore is None:
    st.info('👈 请先在左侧上传文档（支持 PDF 和 TXT），上传后即可提问。')
    st.markdown("""
**使用方式：**
1. 在左侧上传一个或多个文件
2. 等待系统自动建立知识库（通常 10–30 秒）
3. 在下方输入框提问
4. 系统将基于你的文档内容回答，并展示参考段落
""")
else:
    names_str = '、'.join(st.session_state.uploaded_names)
    st.caption(f'当前文档：{names_str} · {st.session_state.chunk_count} 个片段')

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg['role'] == 'assistant' and msg.get('sources'):
            sources = msg['sources']
            with st.expander(f'📚 参考来源（{len(sources)} 段）'):
                for i, doc in enumerate(sources, 1):
                    src = doc.metadata.get('source', '未知')
                    rerank_score = doc.metadata.get('rerank_score', 0)
                    st.markdown(f'**[{i}] `{src} (rerank score:{rerank_score:.4f})`**')
                    st.caption(doc.page_content[:150] + '...')
                    if i < len(sources):
                        st.divider()

if prompt := st.chat_input('基于上传文档提问...'):
    if not st.session_state.vectorstore:
        st.warning('请先上传文档！')
        st.stop()

    st.session_state.messages.append({'role': 'user', 'content': prompt, 'sources': []})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner('正在检索并生成回答...'):
            answer, sources = rag_answer(
                prompt,
                st.session_state.vectorstore,
                st.session_state.all_chunks,
                st.session_state.messages,
                k=k_value
            )
        st.markdown(answer)
        if sources:
            with st.expander(f'📚 参考来源（{len(sources)} 段）'):
                for i, doc in enumerate(sources, 1):
                    src = doc.metadata.get('source', '未知')
                    rerank_score = doc.metadata.get('rerank_score', 0)
                    st.markdown(f'**[{i}] `{src} (rerank score:{rerank_score:.4f})`**')
                    st.caption(doc.page_content[:150] + '...')
                    if i < len(sources)-1:
                        st.divider()

    st.session_state.messages.append({
        'role': 'assistant', 'content': answer, 'sources': sources
    })
