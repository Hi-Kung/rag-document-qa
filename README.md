# 📄 文档智能问答系统

基于 RAG（检索增强生成）技术的文档问答应用，支持上传 PDF / TXT 文件后进行自然语言提问，系统将严格基于文档内容生成回答。

🔗 **在线体验**：https://huggingface.co/spaces/Hi-Kung/rag-document-qa

---

## ✨ 功能特性

- **多文件上传**：同时上传多个 PDF 或 TXT 文件，自动合并为统一知识库
- **混合检索（Hybrid RAG）**：向量语义检索（权重 0.7）+ BM25 关键词检索（权重 0.3），通过加权 RRF 融合，兼顾语义理解与精确词命中
- **Reranking 精排**：使用 `BAAI/bge-reranker-v2-m3` 对召回文档二次精排，提升答案相关性
- **多轮对话**：保留最近 6 条对话历史，支持上下文追问
- **来源溯源**：每条回答附带参考段落及 rerank 得分，结果透明可查
- **可调参数**：侧边栏滑块实时调整检索片段数（k = 1~6）

---

## 🏗️ 技术架构

```
用户上传文件
    │
    ▼
文件解析（pypdf / 纯文本）
    │
    ▼
文本分块（RecursiveCharacterTextSplitter, chunk=400, overlap=50）
    │
    ▼
向量入库（Chroma 内存模式 + BAAI/bge-m3 Embeddings）
    │
    ▼
用户提问
    ├── 向量检索（top 2k 条）
    └── BM25 检索（top 2k 条）
            │
            ▼
      加权 RRF 融合（取前 k 条）
            │
            ▼
      Reranking（bge-reranker-v2-m3）
            │
            ▼
      LLM 生成回答（DeepSeek-V3）
            │
            ▼
      展示答案 + 参考来源
```

## 🖥️ 界面截图

![](./screenshot/qa_demo.png)

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. 安装依赖

```bash
pip install streamlit langchain langchain-community langchain-openai \
            langchain-text-splitters chromadb pypdf rank-bm25 requests
```

### 3. 配置 API Key

本项目使用 [SiliconFlow](https://siliconflow.cn/) 提供的模型服务（兼容 OpenAI 接口）。

```bash
export SILICONFLOW_API_KEY="your_api_key_here"
```

或在项目根目录创建 `.env` 文件：

```
SILICONFLOW_API_KEY=your_api_key_here
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

---

## 📦 依赖说明

| 依赖 | 用途 |
|------|------|
| `streamlit` | Web UI 框架 |
| `langchain` / `langchain-community` | RAG 链路、BM25 检索 |
| `langchain-openai` | 调用兼容 OpenAI 接口的模型 |
| `chromadb` | 向量数据库（内存模式） |
| `pypdf` | PDF 文本提取 |
| `rank-bm25` | BM25 关键词检索 |
| `requests` | 调用 SiliconFlow Rerank API |

---

## 🤖 使用的模型

| 模型 | 用途 | 提供方 |
|------|------|--------|
| `BAAI/bge-m3` | 文本向量化（Embedding） | SiliconFlow |
| `BAAI/bge-reranker-v2-m3` | 文档精排（Rerank） | SiliconFlow |
| `deepseek-ai/DeepSeek-V3` | 答案生成（LLM） | SiliconFlow |

---

## 💡 使用方式

1. 在左侧侧边栏上传一个或多个 `.pdf` / `.txt` 文件
2. 等待系统自动解析并建立知识库（通常 10–30 秒）
3. 在底部输入框输入问题
4. 系统返回答案，点击「📚 参考来源」可查看检索到的原文段落及 rerank 得分

> ⚠️ 系统将严格基于文档内容回答，若文档中无相关信息会明确告知，不会编造内容。

---

## ⚙️ 可配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `chunk_size` | 文本分块大小（字符数） | 400 |
| `chunk_overlap` | 分块重叠大小 | 50 |
| `k`（侧边栏滑块） | 最终返回的检索片段数 | 3 |
| `fetch_k` | 单路召回数量（= k × 2） | 6 |
| 向量检索权重 | RRF 融合中向量检索的权重 | 0.7 |
| BM25 检索权重 | RRF 融合中关键词检索的权重 | 0.3 |

---

## 📝 注意事项

- 向量库采用**内存模式**，每次重新上传文件会重建知识库，刷新页面后数据不保留
- 对话历史仅保留最近 **6 条**参与上下文构建
- 文件解析仅支持 `.txt`（UTF-8 / GBK）和 `.pdf` 格式
- 需要有效的 SiliconFlow API Key 才能正常使用

---

## 📄 License

MIT License
