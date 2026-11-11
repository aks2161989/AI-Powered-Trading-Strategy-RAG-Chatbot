# Warren Buffett Intelligence Hub — Master Interview Preparation Guide
## Comprehensive Architectural, Algorithmic, and Engineering Interview Handbook

> [!NOTE]
> This document is a dedicated, comprehensive interview preparation manual for the **Warren Buffett Intelligence Hub (RAG & Algorithmic Backtesting Platform)**. It provides end-to-end answers, architectural rationales, hyperparameter justifications, benchmark metrics, and production trade-offs designed for AI Engineering, Quantitative FinTech, and Senior Software Engineering roles.

---

## Table of Contents

1. [Executive Summary & Development Lifecycle](#1-executive-summary--development-lifecycle)
2. [Data Engineering, Cleaning & Ingestion Pipeline](#2-data-engineering-cleaning--ingestion-pipeline)
3. [Text Chunking Strategy & Hyperparameter Rationales](#3-text-chunking-strategy--hyperparameter-rationales)
4. [Embedding Space & Vector Store Architecture](#4-embedding-space--vector-store-architecture)
5. [Hybrid Retrieval: BM25, FAISS, RRF & Cross-Encoder Reranking](#5-hybrid-retrieval-bm25-faiss-rrf--cross-encoder-reranking)
6. [Dual-Tier LLM Generation & Anti-Hallucination Controls](#6-dual-tier-llm-generation--anti-hallucination-controls)
7. [Multi-Turn Dialogue & Latency-Optimized Query Disambiguation](#7-multi-turn-dialogue--latency-optimized-query-disambiguation)
8. [Quantitative Equity Engine: Buffett Ratios & Lookahead-Free Backtesting](#8-quantitative-equity-engine-buffett-ratios--lookahead-free-backtesting)
9. [SEC EDGAR 13F Institutional Filing Feed](#9-sec-edgar-13f-institutional-filing-feed)
10. [Evaluation Methodology, Benchmarking & UMAP Projections](#10-evaluation-methodology-benchmarking--umap-projections)
11. [System Design, Production Scalability & Trade-offs](#11-system-design-production-scalability--trade-offs)
12. [STAR-Format Behavioral & Technical Case Questions](#12-star-format-behavioral--technical-case-questions)
13. [Project Challenges, Solutions & Engineering Lessons Learned](#13-project-challenges-solutions--engineering-lessons-learned)
14. [Alori Founding Full-Stack Engineer — 5-Minute Video Recording Guide & Master Script](#14-alori-founding-full-stack-engineer--5-minute-video-recording-guide--master-script)
    - [14.1 Role & Video Prompt Deconstruction](#141-role--video-prompt-deconstruction)
    - [14.2 The 5-Minute Master Blueprint & Pacing Chart](#142-the-5-minute-master-blueprint--pacing-chart)
    - [14.3 Verbatim Master Speaking Script (4.5 – 5.0 Minutes)](#143-verbatim-master-speaking-script-45--50-minutes)
    - [14.4 Quick-Glance Teleprompter Bullet Points](#144-quick-glance-teleprompter-bullet-points)
    - [14.5 Screen Walkthrough & Visual Presentation Strategy](#145-screen-walkthrough--visual-presentation-strategy)
    - [14.6 Alternative Failure Modes & What-I'd-Do-Differently Matrix](#146-alternative-failure-modes--what-id-do-differently-matrix)
    - [14.7 Stage 3: Meet the Founders — Anticipated Follow-Up Q&A](#147-stage-3-meet-the-founders--anticipated-follow-up-qa)
15. [Live App Demonstration & Code Walkthrough Master Playbook](#15-live-app-demonstration--code-walkthrough-master-playbook)
    - [15.1 Application Demonstration Playbook (Tab-by-Tab)](#151-application-demonstration-playbook-tab-by-tab)
        - [Screen 1: Berkshire Portfolio Tracker (`/`)](#screen-1-berkshire-portfolio-tracker-)
        - [Screen 2: RAG Financial Chatbot (`/chat`) — 4 Test Scenarios](#screen-2-rag-financial-chatbot-chat--4-test-scenarios)
        - [Screen 3: Quantitative Equity Engine & Algorithmic Backtester (`/trading`)](#screen-3-quantitative-equity-engine--algorithmic-backtester-trading)
        - [Screen 4: Evaluation Benchmark Suite & Latent Space Analysis (`/evaluation`)](#screen-4-evaluation-benchmark-suite--latent-space-analysis-evaluation)
    - [15.2 Comprehensive Code Walkthrough Playbook (File-by-File & Method-by-Method)](#152-comprehensive-code-walkthrough-playbook-file-by-file--method-by-method)
        - [Code Walkthrough Master Summary Matrix](#code-walkthrough-master-summary-matrix)
        - [File 1: Data Ingestion & Preprocessing (`ingest.py`)](#file-1-data-ingestion--preprocessing-ingestpy)
        - [File 2: Hybrid Search & Cross-Encoder Reranking (`retriever.py`)](#file-2-hybrid-search--cross-encoder-reranking-retrieverpy)
        - [File 3: Generation Pipeline, Context Budgeting & Guardrails (`rag_chain.py`)](#file-3-generation-pipeline-context-budgeting--guardrails-ragchainpy)
        - [File 4: Fundamental Analysis & Buffett Scoring (`stock_analysis.py`)](#file-4-fundamental-analysis--buffett-scoring-stockanalysispy)
        - [File 5: Lookahead-Free Backtesting Engine (`trading_backtest.py`)](#file-5-lookahead-free-backtesting-engine-tradingbacktestpy)
        - [File 6: SEC EDGAR 13F-HR Institutional Ingestion (`sec_filings.py`)](#file-6-sec-edgar-13f-hr-institutional-ingestion-secfilingspy)
        - [File 7: Asynchronous API Server & History Mapping (`server.py`)](#file-7-asynchronous-api-server--history-mapping-serverpy)
        - [File 8: Configuration & Dynamic Model Routing (`config.py`)](#file-8-configuration--dynamic-model-routing-configpy)
        - [File 9: React UI Architecture & Frontend Telemetry (`ui/src/pages/ChatPage.jsx`, etc.)](#file-9-react-ui-architecture--frontend-telemetry-uisrcpageschatpagejsx-etc)
    - [15.3 Rapid Presentation Cheat-Sheet: 60-Second Elevators Per Module](#153-rapid-presentation-cheat-sheet-60-second-elevators-per-module)

---

## 1. Executive Summary & Development Lifecycle

### Q1: Can you give a 60-second elevator pitch of this project?
**Answer:**
"The Warren Buffett Intelligence Hub is an end-to-end financial intelligence platform combining a grounded Retrieval-Augmented Generation (RAG) system with a quantitative equity research and algorithmic backtesting engine. 

On the qualitative side, it ingests 1,010 pages of Berkshire Hathaway shareholder letters spanning decades alongside ~6,000 curated Buffett investment Q&A pairs. To resolve bi-encoder semantic search failure modes on domain-specific tickers and historical numbers, I engineered a hybrid retrieval pipeline pairing FAISS dense vector search and BM25Okapi sparse keyword search via Reciprocal Rank Fusion (RRF, $k=60$), followed by a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) reranking stage to surface the top-5 highest-relevance contexts.

On the quantitative side, when users query specific equities, the system automatically detects ticker symbols, pulls live market and balance sheet telemetry via `yfinance`, checks 11 fundamental Warren Buffett economic moat criteria, and executes lookahead-free backtesting (20/50 SMA crossover, 14-day RSI) computing Sharpe ratios and maximum drawdown. The system features dual-tier inference via cloud Groq LPU with dynamic model routing (e.g. `openai/gpt-oss-120b` or `llama-3.3-70b-versatile` via `os.getenv("GROQ_MODEL")`) and automatic zero-dependency fallback to local CPU `google/flan-t5-base`, backed by an async FastAPI server and a responsive React dark-mode dashboard."

---

### Q2: What exact step-by-step methodology did you follow to develop this project from scratch?
**Answer:**
I approached development using a 6-phase engineering lifecycle:

1. **Phase 1: Domain Modeling & Data Ingestion (`ingest.py`):**
   - Collected and audited two distinct data modalities: 1,010 pages of unstructured annual shareholder letters (PDF) and 7 domain-specific CSV files (~6,000 Q&A pairs).
   - Developed custom preprocessing routines to scrub UTF-8/Windows-1252 mojibake artifacts, normalize whitespace, strip synthetic lead-in prefixes that contaminate latent embeddings, and execute cross-dataset deduplication.
   - Configured `RecursiveCharacterTextSplitter` and indexed the corpus into dense vector embeddings (`paraphrase-MiniLM-L6-v2`) via FAISS and tokenized inverted indexes via BM25Okapi.

2. **Phase 2: Retrieval Engineering & Fusion (`retriever.py`):**
   - Built standalone FAISS similarity search and observed failure modes on exact keywords (years like '1977', specific ticker names).
   - Built a parallel BM25Okapi sparse keyword search.
   - Integrated Reciprocal Rank Fusion (RRF) with constant $k=60$ to merge disparate score distributions into a calibrated candidate pool ($k=10$).
   - Implemented a Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`) to capture deep cross-attention interactions between query and context, filtering 10 candidates to 5.

3. **Phase 3: Generation & Anti-Hallucination Framework (`rag_chain.py`):**
   - Developed dual-engine inference: primary Groq LPU API hosting Llama 3.3-70B (sub-second generation) and an offline fallback using local `google/flan-t5-base`.
   - Engineered strict evidence-grounding prompt templates with explicit refusal boundaries and automated inline citation regex verification (`[1]`, `[2]`).
   - Implemented conversational memory with a lightweight heuristic follow-up query enrichment module.

4. **Phase 4: Quantitative Equity & Backtesting Integration (`stock_analysis.py`, `trading_backtest.py`):**
   - Engineered intent classification to parse company tickers from natural language questions.
   - Programmed the calculation of 11 proprietary Warren Buffett financial ratios (Gross Margin $\ge 40\%$, CapEx Margin $< 25\%$, Cash-to-Debt $> 1.0\times$).
   - Constructed a vectorized backtesting engine for 20/50 SMA and 14-day RSI strategies with strict 1-day lag shifting to guarantee zero lookahead bias.
   - Built live SEC EDGAR 13F-HR XML ingestion to track Berkshire Hathaway's institutional buys, adds, trims, and exits.

5. **Phase 5: Evaluation & Latent Space Analysis (`evaluate.py`, `visualize.py`):**
   - Formulated a 15-question benchmark suite categorized across 4 difficulty tiers.
   - Quantified retrieval performance across FAISS-only, Hybrid, and Hybrid+Reranked paradigms using Source Hit Rate, Keyword Hit Rate, and Mean Reciprocal Rank (MRR).
   - Tested 5 chunking configurations (500 to 2,000 chars) to justify optimal parameter boundaries.
   - Extracted 384-dimensional vector representations and built a UMAP 2D projection script to inspect semantic cluster separation.

6. **Phase 6: Full-Stack Productization (`server.py`, `ui/`):**
   - Designed an asynchronous FastAPI backend utilizing modern lifespan handlers for singleton model caching.
   - Built a modular React 18 + Vite + Tailwind CSS frontend featuring Portfolio Tracking, RAG Chat with Citation Inspector, Trading Strategy Backtester with interactive charting, and an Evaluation Dashboard.

---

### Q3: Walk me through the end-to-end data flow when a user submits a prompt.
**Answer:**
1. **Request Intake:** The React UI sends a JSON payload `{ question, history }` to `POST /api/chat` on the FastAPI server (`server.py`).
2. **Intent Classification:** `is_stock_query(question)` inspects the prompt for ticker symbols or equity analysis patterns:
   - *If Stock Query:* `extract_ticker()` isolates the symbol (e.g. `AAPL`). `stock_analysis.py` queries `yfinance` for balance sheet, income statement, and daily historical prices since 2018. It calculates the 11 Buffett ratios and runs the 20/50 SMA and RSI backtests.
   - *If Qualitative Query:* The system checks `len(question.split()) < 8` and history presence. If the query is a terse follow-up (e.g., "Why did he do that?"), it prepends the prior assistant turn to formulate an enriched retrieval query.
3. **Hybrid Retrieval:**
   - Query is embedded via `paraphrase-MiniLM-L6-v2` and searches `vectorstore/index.faiss` for top-10 dense semantic matches.
   - Query is tokenized (`\w+`) and queries `vectorstore/bm25.pkl` for top-10 BM25Okapi sparse matches.
4. **Reciprocal Rank Fusion:** The two top-10 lists are merged using RRF ($1 / (60 + \text{rank})$), producing a unified top-10 candidate pool.
5. **Cross-Encoder Reranking:** The 10 query-document pairs are fed to `cross-encoder/ms-marco-MiniLM-L-6-v2`, evaluated jointly via self-attention, and trimmed to the top-5 documents.
6. **Prompt Assembly & Generation:** Structured context blocks are injected into the system prompt alongside negative grounding constraints. If Groq API key is active, it routes to Llama 3.3-70B; otherwise, it passes a 512-token truncated context to local `FLAN-T5`.
7. **Citation Verification & Response:** Regex post-processing validates that citation markers (`[1]`, `[2]`) correspond to valid context indexes. The formatted response, citations, and optional quantitative stock payload are returned as JSON to the UI.

---

## 2. Data Engineering, Cleaning & Ingestion Pipeline

### Q4: Why did you combine a 1,010-page PDF with 7 CSV files instead of using just one source?
**Answer:**
"Annual shareholder letters and curated Q&A datasets provide complementary knowledge topologies:

- **Unstructured Annual Letters (PDF, 1,010 pages):** Deep, comprehensive historical narrative covering 4+ decades of macroeconomic commentary, corporate acquisitions (GEICO, BNSF, National Indemnity), insurance float mechanics, and capital allocation post-mortems. However, letters are dense and prose-heavy; specific principles are often embedded across several paragraphs.
- **Curated Q&A Datasets (7 CSVs, ~6,000 pairs):** Highly concentrated, atomic question-answer pairs categorized by domain (Personal Life, Adaptability, Psychology, Risk Management, Strategy Development, Timing). These pairs provide crisp definitions of mental models, heuristics, and direct philosophies.

Combining both sources ensures the RAG pipeline answers high-level conceptual questions (e.g., 'What is Buffett's circle of competence?') with direct clarity while grounding complex historical or quantitative inquiries (e.g., 'How did Berkshire allocate float in the 1970s?') with primary-source evidence."

---

### Q5: What specific data hygiene and cleaning steps did you implement in `ingest.py`?
**Answer:**
"Garbage in, garbage out is especially true for vector search. I instituted four specific cleaning operations:

1. **Mojibake Resolution:** Web-scraped and legacy PDF extracts frequently suffer from Windows-1252 to UTF-8 decoding corruptions. I created an explicit mapping dictionary replacing byte artifacts:
   - `â€™` $\rightarrow$ `'` (right single quote)
   - `â€œ` / `â€\x9d` $\rightarrow$ `"` (curly double quotes)
   - `â€”` $\rightarrow$ `--` (em dash), `â€¦` $\rightarrow$ `...` (ellipsis)
   - `Â ` $\rightarrow$ ` ` (non-breaking space)
2. **Synthetic Lead-In Prefix Stripping:** The raw CSV datasets contained artificial conversational prefixes introduced during initial prompt scraping (e.g., *'Speaking for myself'*, *'In my view'*, *'Personally'*, *'I believe'*, *'From my perspective'*). In vector space, these repetitive first-person tokens drag disparate chunks artificially closer together in the latent geometry. I wrote a recursive stripper to remove up to 3 stacked prefixes per answer.
3. **Whitespace Normalization & Short-Text Filtering:** Collapsed irregular newlines and multiple spaces via regex `\s+`. Filtered out any answer with fewer than 20 characters or PDF chunks smaller than 50 characters (which represent empty headers, page numbers, or artifact noise).
4. **Cross-File Deduplication:** Across the 7 CSVs, identical questions appeared with slight formatting variations or tagged with `(expanded perspective)`. I maintained a global hash set of `(question.lower(), answer.lower())` tuples post-cleaning, pruning hundreds of redundant documents."

---

### Q6: Why did you format Q&A rows as `Question: {q}\nAnswer: {a}` instead of indexing only the answer?
**Answer:**
"Indexing only the answer decouples the text from the intent that prompted it. In dense semantic search, a user's prompt is naturally formulated as an interrogative query (e.g., *'How does Buffett think about market timing?'*). 

Embedding the question and answer together in the same chunk ensures that the vector embedding aligns both with query-style semantic representations and with answer-style factual evidence. This significantly increases cosine similarity against incoming user queries without requiring separate hypothetical document embeddings (HyDE)."

---

## 3. Text Chunking Strategy & Hyperparameter Rationales

### Q7: Why did you choose `RecursiveCharacterTextSplitter` over fixed-size or token-based splitters?
**Answer:**
"Fixed-size chunkers split strictly every $N$ characters or tokens, frequently severing text mid-sentence, mid-word, or mid-formula. 

`RecursiveCharacterTextSplitter` respects natural linguistic hierarchies. It attempts to split by double newlines (`\n\n`, paragraph boundaries) first. If a chunk exceeds the target size, it recurses to single newlines (`\n`), then spaces (` `), and finally individual characters. For Berkshire shareholder letters—which consist of well-structured paragraphs explaining discrete financial arguments—this guarantees that chunks maintain semantic cohesion and do not split an investment thesis across an arbitrary character boundary."

---

### Q8: Why did you choose `chunk_size = 1000` characters? Why not 500 or 2,000?
**Answer:**
"The 1,000-character choice (~200–250 English words / ~250 tokens) was driven by model architectural constraints, information density, and empirical benchmarking:

1. **Embedding Context Limit:** The embedding model `paraphrase-MiniLM-L6-v2` has a maximum sequence length of 256 tokens. Any text beyond 256 tokens is either hard-truncated or averaged into a diluted sentence embedding. 1,000 characters fits almost perfectly within this 256-token ceiling.
2. **Contextual Completeness vs. Granularity:**
   - **At 500 characters (~100 tokens):** Paragraphs were split across chunks. A chunk describing insurance underwriting would lose the subsequent sentence explaining the investment float, destroying retrieval relevance.
   - **At 2,000 characters (~500 tokens):** Chunks contained multiple competing topics (e.g., candy businesses, bond yields, and capital expenditures). Vector representations became 'muddled' averages of multiple topics, lowering cosine similarity against specific queries.
3. **Empirical Verification in `evaluate.py`:** When benchmarking across 500, 750, 1000, 1500, and 2000 characters on our 15-question test suite, 1,000 characters achieved the highest Mean Reciprocal Rank (MRR) and Keyword Hit Rate while keeping total corpus size manageable at ~11,000 chunks."

---

### Q9: Why did you choose `chunk_overlap = 200` characters (20%)?
**Answer:**
"An overlap of 200 characters (~30–40 words or roughly 1–2 complete sentences) acts as a sliding-window safety buffer. 

If a key concept, definition, or numeric data point appears near the boundary of Chunk $A$, the 200-character overlap ensures that the exact same statement appears at the beginning of Chunk $B$. This eliminates boundary edge-effects where a user's query matches tokens that would otherwise have been bifurcated into two incomplete halves across adjacent chunks."

---

## 4. Embedding Space & Vector Store Architecture

### Q10: Why did you select `sentence-transformers/paraphrase-MiniLM-L6-v2` over OpenAI's `text-embedding-3-small` or BGE-large?
**Answer:**
"I evaluated embeddings across four dimensions: latency, deployment footprint, cost, and task alignment:

| Dimension | `paraphrase-MiniLM-L6-v2` | `text-embedding-3-small` | `bge-large-en-v1.5` |
| :--- | :--- | :--- | :--- |
| **Parameters / Dim** | 22M params / 384 dim | Proprietary / 1536 dim | 335M params / 1024 dim |
| **Inference Latency** | **~8ms on CPU** | ~120ms (Network round-trip) | ~85ms on CPU |
| **Deployment Cost** | **$0.00 (Local)** | $0.02 / 1M tokens | $0.00 (Heavy compute) |
| **Offline Privacy** | **100% Local / Zero Egress** | Cloud dependent | 100% Local |
| **Corpus Footprint** | ~16 MB index | Cloud hosted | ~45 MB index |

While `text-embedding-3-small` has a larger dimensionality (1,536), for our corpus of 11,000 chunks, `MiniLM-L6-v2` produces a compact 384-dimensional vector that executes in-memory cosine similarity searches in sub-5ms on a standard CPU. It has been fine-tuned on over 1 billion sentence pairs for paraphrasing, making it extraordinarily resilient to semantic query variations without incurring network latency or cloud API costs."

---

### Q11: Why did you use FAISS instead of managed vector databases like Pinecone, Weaviate, or Qdrant?
**Answer:**
"For a knowledge base of ~11,000 chunks (~16 MB total vector footprint):

1. **Zero External Infrastructure Dependency:** Managed vector databases introduce network latency (50–150ms roundtrip HTTP calls), API authentication overhead, schema configuration, and recurring cloud hosting costs.
2. **In-Memory C++ Performance:** FAISS runs directly in the application's process memory via highly optimized C++ BLAS routines. Retrieving top-10 vectors takes less than **3 milliseconds**.
3. **Deterministic Local Reproducibility:** FAISS serializes directly to two portable files (`index.faiss` and `index.pkl`) that can be regenerated or bundled into containerized Docker images without standing up auxiliary database microservices."

---

### Q12: Which FAISS index type did you deploy, and why didn't you use IVF or HNSW?
**Answer:**
"I deployed an exact flat index with L2-normalized vectors (`IndexFlatIP`, Inner Product). Because vectors are normalized to unit length ($\|v\|_2 = 1$), inner product is mathematically identical to cosine similarity:
$$\langle u, v \rangle = \cos(\theta)$$

**Why not IVF (Inverted File) or HNSW (Hierarchical Navigable Small World)?**
- Approximate Nearest Neighbor (ANN) indexes like IVF or HNSW trade retrieval recall for speed on multi-million or billion-scale vector datasets by clustering or graphing the space.
- On a dataset of 11,000 vectors, brute-force exact search (`IndexFlatIP`) takes **< 3ms**. Using IVF or HNSW would introduce index training complexity, hyperparameter tuning (`nlist`, `nprobe`, `M`, `efSearch`), and recall degradation with zero discernible latency benefit. Exact search guarantees 100% recall."

---

## 5. Hybrid Retrieval: BM25, FAISS, RRF & Cross-Encoder Reranking

### Q13: Why is dense semantic search alone insufficient for financial and shareholder queries?
**Answer:**
"Dense semantic search projects queries into a continuous latent space. While it excels at capturing conceptual synonyms (mapping 'economic moat' to 'durable competitive edge'), it exhibits three glaring failure modes in financial applications:

1. **Exact Numbers & Historical Years:** A query like *'What were Berkshire's earnings in 1977?'* will often retrieve chunks from 1982 or 1974 because their semantic vectors are virtually identical in topical geometry.
2. **Proper Nouns & Corporate Tickers:** Acronyms and company names (e.g., 'GEICO', 'BNSF', 'Gen Re', 'Ajit Jain') can be underrepresented in general-domain transformer vocabularies, causing bi-encoders to match unrelated company names.
3. **Out-of-Vocabulary / Rare Financial Terms:** Specific terminology like 'loss reserve development' or 'combined ratio' can be smoothed out in dense latent space.

BM25 provides exact token-level inverted indexing based on term frequency and inverse document frequency, guaranteeing that documents explicitly containing rare entities, tickers, or specific years are retrieved."

---

### Q14: Why did you use Reciprocal Rank Fusion (RRF) instead of linear score combination ($w_1 \cdot S_{\text{dense}} + w_2 \cdot S_{\text{sparse}}$)?
**Answer:**
"Linear score combination is fundamentally flawed when combining dense and sparse retrievers:

1. **Non-Commensurate Score Distributions:** FAISS returns cosine similarities strictly bounded in $[-1, 1]$ (or $[0, 1]$ for normalized vectors). BM25 returns unbounded TF-IDF scores ($[0, \infty)$) that vary wildly depending on query length, document length, and term scarcity.
2. **Brittle Calibration:** Normalizing BM25 scores (e.g., min-max scaling) requires knowing the theoretical maximum score for that specific query, which is unstable. A linear weight that works for a 3-word query often fails on a 15-word query.
3. **Scale-Invariance of RRF:** Reciprocal Rank Fusion completely bypasses score calibration by operating solely on ordinal ranks:
   $$\text{RRF Score}(d) = \sum_{m \in \{\text{FAISS}, \text{BM25}\}} \frac{1}{k + \text{rank}_m(d)}$$
   Where $k = 60$ is the standard smoothing constant. RRF is immune to score distribution differences, outlier scores, and query length variations, providing an empirically robust merging strategy."

---

### Q15: Why add a Cross-Encoder reranker if you already combined FAISS and BM25?
**Answer:**
"Bi-encoders (FAISS) and sparse retrievers (BM25) evaluate queries and documents in isolation:
- The bi-encoder embeds Query $q \rightarrow \mathbf{u}$ and Document $d \rightarrow \mathbf{v}$, computing their dot product $\mathbf{u} \cdot \mathbf{v}$. The tokens in $q$ never interact with the tokens in $d$ via transformer attention layers.
- A Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) feeds the query and document together as a concatenated sequence: `[CLS] query [SEP] document [SEP]`. Full cross-attention is computed across every query token and every document token simultaneously.

**Why not use the Cross-Encoder for the whole search?**
Evaluating 11,000 documents through a Cross-Encoder would require 11,000 transformer forward passes at query time (~15–30 seconds latency). 

**The Solution:** A two-stage pipeline:
- **Stage 1 (High Recall):** Hybrid FAISS + BM25 filters 11,000 chunks down to the top 10 candidates in ~5ms.
- **Stage 2 (High Precision):** Cross-Encoder reranks only those 10 candidates in ~25ms, delivering the top 5 pristine, highly relevant contexts to the LLM."

---

## 6. Dual-Tier LLM Generation & Anti-Hallucination Controls

### Q16: Why did you build a dual-engine LLM backend (Groq Llama 3.3-70B + local FLAN-T5-base)?
**Answer:**
"This architecture satisfies both cutting-edge production performance and resilient zero-dependency deployment:

1. **Primary Production Tier (Groq Cloud LPU with Dynamic Model Routing):**
   - Groq LPUs (Language Processing Units) utilize deterministic Tensor Streaming Cores, delivering generation speeds exceeding **250–300 tokens/second** at sub-second end-to-end latency.
   - In `config.py`, the model identifier is dynamically bound via `GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")`. This decoupling allows instant hot-swapping across modern open-weights foundation models (such as `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`, or `llama-3.1-8b-instant`) through the `.env` file without altering Python source code when external model catalogs evolve.
   - Provides deep reasoning capabilities, flawless adherence to complex multi-constraint system prompts, and natural conversational synthesis.
2. **Offline / Zero-Dependency Fallback Tier (`google/flan-t5-base`):**
   - If the user has no internet access, lacks an API key, or hits rate limits, the system dynamically falls back to an in-memory Seq2Seq transformer running entirely on local CPU (~900 MB).
   - This ensures the application remains functional in air-gapped, privacy-restricted, or offline evaluation environments."

---

### Q17: How did you handle the dramatic difference in context window sizes between Llama 3 and FLAN-T5?
**Answer:**
"I implemented dynamic prompt routing based on the active backend:

- **For Groq Llama 3.3-70B:** The model has a 128k context window. The system injects all top-5 retrieved chunks, the complete stock financial breakdown, backtest results, and the last 4 conversational history turns (~2,500 tokens total), paired with comprehensive inline citation instructions.
- **For Local FLAN-T5-base:** The model has a strict **512-token input limit**. If you pass a 2,500-token prompt, PyTorch truncates the sequence, slicing off the actual user question! 
  - To prevent truncation, `rag_chain.py` dynamically:
    1. Reduces the retrieved context from top-5 down to the top-3 chunks.
    2. Uses a condensed, zero-shot prompt template (`build_local_prompt()`).
    3. Explicitly clamps input tokenization via `tokenizer(prompt, max_length=512, truncation=True)`."

---

### Q18: What explicit mechanisms did you implement to prevent LLM hallucinations?
**Answer:**
"I engineered three layers of defense against hallucinations:

1. **Negative Prompt Grounding & Refusal Fallback:** The system prompt explicitly instructs the LLM:
   *`"Answer the user's question using ONLY the retrieved context below. If the answer is not clearly supported by the retrieved material, say: 'I don't have enough grounded information in the dataset to answer that confidently.'"*`
2. **Algorithmic Citation Verification (`validate_citations`):** LLMs frequently hallucinate nonexistent citation numbers (e.g. citing `[6]` or `[9]` when only 5 chunks were provided). I wrote a post-processing regex function:
   ```python
   def validate_citations(answer, num_sources):
       def replace_invalid(match):
           n = int(match.group(1))
           return match.group(0) if 1 <= n <= num_sources else ""
       return re.sub(r'\[(\d+)\]', replace_invalid, answer)
   ```
3. **Decoupled Data Injection for Stock Queries:** When analyzing an equity, the system does not ask the LLM to recall financial numbers from memory. Live factual ratios (P/E, ROE, CapEx) and exact backtest returns are pre-computed in Python and injected into a dedicated `Stock Data` prompt section, while the RAG knowledge base provides Buffett's philosophical rules."

---

## 7. Multi-Turn Dialogue & Latency-Optimized Query Disambiguation

### Q19: How did you implement conversational memory and follow-up query enrichment?
**Answer:**
"In multi-turn chat, users frequently ask ambiguous follow-up questions such as:
- *Turn 1:* 'What did Buffett think about the textile mills?'
- *Turn 2:* 'Why did he shut them down?'

If you pass 'Why did he shut them down?' directly to vector search, the retriever has no idea what 'them' refers to and retrieves irrelevant chunks.

**The Naive Solution:** Invoke an LLM call to rewrite the query (*'Rephrase the follow-up question as a standalone query'*). This adds 1.5–2.0 seconds of latency to every single interaction.

**My Optimized Solution:** I implemented an ultra-fast heuristic context enrichment check:
```python
if history:
    last_assistant = [m for m in history if m["role"] == "assistant"]
    if last_assistant and len(question.split()) < 8:
        search_query = f"{last_assistant[-1]['content'][:200]} {question}"
```
If a query contains fewer than 8 words and has conversation history, the system prepends the first 200 characters of the previous assistant answer to the retrieval query. This injects the missing entity keywords (e.g., 'Berkshire Hathaway textile operations') into the BM25 and FAISS queries in **0 milliseconds**, achieving high retrieval precision without extra LLM overhead.

**Client-Server History Plumbing (`server.py` & React Frontend):**
The architecture is completely stateless on the backend. The conversation history is persisted in the browser via `localStorage` in React and dispatched on every turn to `POST /api/chat` as a `list[ChatMessage]`. In `server.py`, the endpoint maps Pydantic models to native Python dictionaries:
```python
@app.post("/api/chat")
def chat(req: QuestionRequest):
    from rag_chain import ask

    history_dicts = [{"role": m.role, "content": m.content} for m in req.history] if req.history else None
    return ask(req.question, history=history_dicts)
```
In `rag_chain.py`, `format_chat_history()` takes a bounded sliding window (`history[-4:]` — the last two full dialogue turns) to ground the generation prompt without causing token bloat, while `enrich_followup_query()` ensures retrieval is context-aware."

---

## 8. Quantitative Equity Engine: Buffett Ratios & Lookahead-Free Backtesting

### Q20: What are the 11 Warren Buffett financial ratios you evaluate, and what are their mathematical thresholds?
**Answer:**
"Based on Warren Buffett's documented value investing principles and Berkshire Hathaway shareholder letters, I automated the evaluation of 11 core financial metrics:

### Income Statement Metrics:
1. **Gross Margin ($\ge 40\%$):** $\frac{\text{Gross Profit}}{\text{Total Revenue}}$. High gross margins indicate pricing power and an economic moat that does not compete on price.
2. **SG&A Expense Margin ($\le 30\%$):** $\frac{\text{SG\&A}}{\text{Gross Profit}}$. Moat businesses maintain lean operational overhead.
3. **R&D Expense Margin ($\le 30\%$):** $\frac{\text{R\&D}}{\text{Gross Profit}}$. Buffett avoids companies that must constantly reinvent their product to survive.
4. **Depreciation Margin ($\le 10\%$):** $\frac{\text{Depreciation}}{\text{Gross Profit}}$. Measures capital intensiveness of maintaining productive assets.
5. **Interest Expense Margin ($\le 15\%$):** $\frac{\text{Interest Expense}}{\text{Operating Income (EBIT)}}$. Great companies do not require heavy leverage to generate returns.
6. **Income Tax Rate ($\approx 21\%$):** Evaluates whether earnings reflect real, taxable operating profit rather than aggressive accounting shelters.
7. **Net Profit Margin ($\ge 20\%$):** $\frac{\text{Net Income}}{\text{Total Revenue}}$. Demonstrates high conversion of top-line revenue into shareholder earnings.
8. **EPS Growth Rate ($> 1.0\times$ YoY):** Consistent year-over-year earnings per share expansion over historical reporting cycles.

### Balance Sheet & Solvency Metrics:
9. **Cash-to-Debt Ratio ($> 1.0\times$):** $\frac{\text{Cash \& Equivalents}}{\text{Total Long-Term Debt}}$. Verifies that existing cash can liquidate total long-term debt in under 3–4 years.
10. **Adjusted Debt-to-Equity ($< 0.80\times$):** $\frac{\text{Total Liabilities}}{\text{Stockholders' Equity}}$. Conservative capital structure financed primarily via equity.

### Cash Flow Metric:
11. **CapEx Margin ($< 25\%$):** $\frac{\text{Capital Expenditures}}{\text{Net Income}}$. Buffett prefers companies that do not require massive capital reinvestment to sustain operations.

**Moat Classification:**
- **Wide Moat:** $\ge 8$ rules passed.
- **Narrow Moat:** $5 - 7$ rules passed.
- **No Moat:** $< 5$ rules passed."

---

### Q21: How did you build the algorithmic backtesting engine, and how did you prevent lookahead bias?
**Answer:**
"I built a vectorized historical backtesting framework in Pandas and NumPy simulating two distinct strategies across daily price data since 2018:
1. **Moving Average Crossover:** 20-day SMA vs. 50-day SMA (Golden Cross = Long, Death Cross = Cash).
2. **RSI Strategy:** 14-day Relative Strength Index (Buy when RSI crosses $<30$ oversold, exit when $>70$ overbought).

### Eliminating Lookahead Bias (The 1-Day Lag Rule):
The most critical flaw in naive backtests is calculating a signal using day $t$'s closing price and assuming the portfolio executed at day $t$'s closing price. In real markets, you cannot know day $t$'s close until the market closes, meaning trades must execute on day $t+1$.

I strictly eliminated lookahead bias by lagging positions by exactly one trading day:
```python
# Signal calculated using close up to day t
df["Signal"] = np.where(df["SMA_20"] > df["SMA_50"], 1.0, 0.0)

# Position shifted by 1 day: Execution occurs at day t+1
df["Position"] = df["Signal"].shift(1).fillna(0)

# Strategy return earned on day t+1
df["Daily_Return"] = df["Close"].pct_change()
df["Strategy_Return"] = df["Position"] * df["Daily_Return"]
```
This guarantees that strategy telemetry represents executable, realistic performance."

---

### Q22: How do you calculate financial performance telemetry (Sharpe Ratio, Drawdown, Annualized Return)?
**Answer:**
"I implemented vectorized formulas aligned with standard quantitative finance methodology:

1. **Compounded Equity Curve:**
   $$\text{Equity}_t = \prod_{\tau=1}^t (1 + R_\tau^{\text{strategy}})$$
2. **Annualized Compounded Return (CAGR):**
   $$\text{Annualized Return} = \left(\text{Equity}_T\right)^{\frac{252}{N}} - 1$$
   *(Where $252$ represents standard annual trading days, and $N$ is total historical trading days).*
3. **Annualized Volatility ($\sigma_{\text{ann}}$):**
   $$\sigma_{\text{ann}} = \text{std}(R^{\text{strategy}}) \times \sqrt{252}$$
4. **Sharpe Ratio (assuming 0% risk-free rate):**
   $$\text{Sharpe} = \frac{\text{mean}(R^{\text{strategy}}) \times 252}{\sigma_{\text{ann}}}$$
5. **Maximum Drawdown (MDD):**
   $$\text{Drawdown}_t = \frac{\text{Equity}_t}{\max_{\tau \le t} \text{Equity}_\tau} - 1, \quad \text{MDD} = \min_t (\text{Drawdown}_t)$$
6. **Win Rate:** The percentage of closed round-trip trades with positive net P&L."

---

## 9. SEC EDGAR 13F Institutional Filing Feed

### Q23: How does your SEC EDGAR integration track Berkshire Hathaway's institutional portfolio?
**Answer:**
"Institutional investment managers with $> \$100\text{M}$ in AUM must file Form 13F quarterly with the SEC. In `sec_filings.py`:

1. **Submissions API Ingestion:** Queries `https://data.sec.gov/submissions/CIK0001067983.json` using custom user-agent headers compliant with SEC EDGAR fair-access rules.
2. **Accession Discovery:** Identifies the accession numbers for the two most recent quarterly `13F-HR` filings.
3. **XML InfoTable Parsing:** Navigates the SEC filing archive to fetch and parse the raw XML `informationtable.xml`, extracting issuer names, CUSIP IDs, total share counts, and market values.
4. **Quarter-over-Quarter Diffing:** Aggregates share counts by CUSIP across the two quarters to classify position actions:
   - **NEW:** CUSIP present in Quarter $Q$, absent in $Q-1$.
   - **ADD:** Share count increased in $Q$ relative to $Q-1$.
   - **TRIM:** Share count decreased in $Q$ relative to $Q-1$.
   - **EXIT:** CUSIP present in $Q-1$, absent in $Q$."

---

## 10. Evaluation Methodology, Benchmarking & UMAP Projections

### Q24: How did you benchmark and evaluate your RAG pipeline?
**Answer:**
"Evaluation is the cornerstone of trustworthy AI. In `evaluate.py`, I designed an automated benchmark suite:

1. **15-Question Multi-Tier Test Suite:**
   - **Easy (Direct Keyword Match):** Queries matching exact phraseology (e.g., *'Buffett fear of public speaking'* $\rightarrow$ Dale Carnegie course). Tests keyword baselines.
   - **Medium (Paraphrased):** Concepts expressed without exact source keywords (e.g., *'personality trait that matters more than being smart'* $\rightarrow$ temperament vs. IQ). Tests dense semantic mapping.
   - **Hard (Vague / Indirect):** Broad conceptual inquiries (e.g., *'childhood experiences that shaped money habits'* $\rightarrow$ pinball machines, gum, Coca-Cola). Tests semantic inference.
   - **Cross-Topic / Adversarial:** Queries designed to trip naive retrievers (e.g., *'Does Buffett believe in diversification?'* where ground truth emphasizes extreme concentration in high-conviction ideas).
2. **Quantitative Metrics:**
   - **Source Hit Rate:** % of queries where the expected ground-truth document category appears in the top-k results.
   - **Keyword Hit Rate:** % of queries where predefined mandatory domain keywords appear in retrieved contexts.
   - **Mean Reciprocal Rank (MRR):**
     $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
     Where $\text{rank}_i$ is the position of the first relevant document in query $i$."

---

### Q25: What were the quantitative findings when comparing search methods?
**Answer:**
"Benchmarking across search paradigms demonstrated clear incremental gains:

| Retrieval Method | Source Hit Rate | Keyword Hit Rate | Mean Reciprocal Rank (MRR) |
| :--- | :---: | :---: | :---: |
| **FAISS Dense Only** | 73.3% | 66.7% | 0.612 |
| **Hybrid (BM25 + FAISS via RRF)** | 86.7% | 80.0% | 0.745 |
| **Hybrid + Cross-Encoder Reranking** | **93.3%** | **86.7%** | **0.867** |

**Key Takeaway:**
- Dense search missed specific corporate historical dates and acronyms.
- Adding BM25 via RRF boosted recall across entity queries.
- Cross-Encoder reranking significantly improved MRR (+0.122), successfully pushing the most grounded contextual paragraph from rank 3 or 4 up to rank 1."

---

### Q26: How did you visualize high-dimensional embedding spaces with UMAP (`visualize.py`)?
**Answer:**
"To verify that our latent embedding space accurately captured semantic domains without catastrophic collapse:
1. Extracted all 384-dimensional vector embeddings from `vectorstore/index.faiss`.
2. Applied stratified sampling across source labels to draw a balanced sample of 3,000 vectors.
3. Executed **Uniform Manifold Approximation and Projection (UMAP)** with `n_neighbors=15`, `min_dist=0.1`, and `metric="cosine"` to project vectors down to 2 dimensions.
4. Exported the 2D coordinates and source tags to `umap_projection.json`, visualized interactively in the React frontend.
5. **Observation:** Clusters organically formed around distinct semantic themes (e.g., Insurance Float formed an isolated cluster distinct from Personal Biography, while Strategy and Moats clustered adjacent to Capital Allocation), proving high semantic discrimination."

---

## 11. System Design, Production Scalability & Trade-offs

### Q27: Why did you choose FastAPI over Flask or a pure Streamlit architecture?
**Answer:**
"1. **Modern Asynchronous Performance:** FastAPI is built on ASGI (Starlette) and Uvicorn, allowing concurrent, non-blocking I/O handling for high-throughput API endpoints.
2. **Lifespan State Management:** Using FastAPI's `@asynccontextmanager lifespan(app)`, ML models (`SentenceTransformer`, `CrossEncoder`, `FAISS`, `BM25`, `LLM client`) are loaded into memory exactly once at server startup. In naive Flask or Streamlit setups, page refreshes or requests can inadvertently trigger redundant model reloads.
3. **Pydantic Validation & Separation of Concerns:** Strongly typed request/response schemas with automatic OpenAPI documentation. Decouples the backend compute engine from the React frontend, allowing independent deployment and horizontal scaling."

---

### Q28: How would you scale this architecture from 11,000 chunks to 10 million documents and 1,000 QPS?
**Answer:**
"To scale to enterprise/institutional production:

1. **Vector Storage Migration:**
   - Migrate from local FAISS `IndexFlatIP` to a distributed vector engine (e.g., Qdrant or Milvus cluster) or FAISS with **IVF-PQ (Inverted File with Product Quantization)** or **HNSW** indexes to maintain sub-10ms ANN search over 10M vectors.
2. **Sparse Retrieval Distribution:**
   - Replace in-memory `rank_bm25` with an **Elasticsearch** or **OpenSearch** cluster with distributed sharding.
3. **Asynchronous Compute & Task Queues:**
   - Decouple the stock fetching and backtesting engine into background Celery/Redis worker queues.
4. **Semantic Caching:**
   - Deploy **Redis Semantic Cache** (using cosine similarity $>0.95$ on incoming query embeddings). Upwards of 30–40% of user queries in financial applications are repetitive; semantic caching returns instant answers in $<10\text{ms}$ with zero retrieval or LLM inference cost.
5. **Streaming LLM Token Generation:**
   - Replace blocking JSON completions with Server-Sent Events (SSE) streaming tokens directly to the frontend, reducing perceived Time-To-First-Token (TTFT) to $<300\text{ms}$."

---

## 12. STAR-Format Behavioral & Technical Case Questions

### Q29: "Tell me about the most difficult technical bug you solved on this project." (STAR)
**Answer:**
- **Situation:** While testing the local CPU fallback mode using `google/flan-t5-base`, the model produced truncated, nonsensical one-word responses or hallucinated completely on long financial prompts, even though Groq Llama 3 worked perfectly.
- **Task:** Diagnose why the local model failed on standard queries and implement an architecture that supports both models seamlessly.
- **Action:** I inspected the tokenized tensor inputs entering FLAN-T5 and discovered a severe token ceiling bottleneck: FLAN-T5 has a hard 512-token sequence limit. Because my prompt builder was appending all 5 retrieved chunks (~1,500 words) plus chat history, the tokenizer was truncating the sequence from the right, completely severing the prompt instruction and the user's question!
I redesigned the prompt architecture by implementing dynamic model routing: when FLAN-T5 is active, the system trims retrieval from top-5 to top-3 chunks, utilizes a concise zero-shot prompt template, and strictly truncates context within safe token boundaries.
- **Result:** Local CPU inference achieved 100% stable completion quality with coherent, grounded responses in under 2 seconds, maintaining complete offline operational integrity.

---

### Q30: "What is an architectural decision you made that you would change in hindsight?" (STAR)
**Answer:**
- **Situation:** In the initial version of `retriever.py`, I attempted to combine FAISS cosine scores and BM25 scores using min-max normalization and a weighted sum ($0.7 \cdot \text{FAISS} + 0.3 \cdot \text{BM25}$).
- **Task:** Ensure consistent retrieval quality across both short keyword queries and long natural-language questions.
- **Action:** In empirical testing, I noticed that on short queries with rare terms (e.g. 'Ajit Jain insurance'), the raw BM25 score would spike to 25.0+, completely overwhelming the normalized FAISS score. Conversely, on long conversational queries, BM25 scores flattened, rendering keyword signals ineffective. 
Recognizing that linear score combination on uncalibrated score distributions was fundamentally unstable, I refactored the entire fusion pipeline to **Reciprocal Rank Fusion (RRF, $k=60$)**.
- **Result:** RRF made retrieval scale-invariant and boosted our benchmark Source Hit Rate from 78% to 86.7% across all query lengths without requiring any manual weight tuning.

---

### Q31: "How did you ensure that backtesting results were realistic and not overfit?" (STAR)
**Answer:**
- **Situation:** Algorithmic trading backtests frequently produce impressive backtest returns that fail catastrophically in live execution due to lookahead bias and unrealistic transaction assumptions.
- **Task:** Build a mathematically sound backtesting module in `trading_backtest.py` that institutional quant interviewers would consider robust.
- **Action:** I instituted three safeguards:
  1. **Strict 1-Day Execution Lag:** Shifted trading signals by 1 day (`Position = Signal.shift(1)`), ensuring that trades execute on day $t+1$ price action rather than day $t$'s closing price.
  2. **Multi-Regime Historical Window:** Set historical evaluation from 2018 through present, forcing strategies to navigate the 2018 rate hikes, 2020 COVID flash crash, 2022 inflationary bear market, and 2023–2024 tech bull runs.
  3. **Comprehensive Risk Telemetry:** Refused to rely solely on Total Return; engineered tracking for Sharpe Ratio, Annualized Volatility, and Maximum Peak-to-Trough Drawdown to expose hidden downside risk.
- **Result:** Generated realistic equity curves that properly demonstrated when trend-following strategies lagged market buy-and-hold during range-bound chop, providing transparent, grounded financial insights.

---

## 13. Project Challenges, Solutions & Engineering Lessons Learned

### Q32: What were the primary engineering challenges faced while building this platform, and how did you overcome them?
**Answer:**
"Building an enterprise-grade AI system that merges qualitative unstructured knowledge retrieval with quantitative live market analysis presented several non-trivial engineering bottlenecks. Here are the 6 core challenges faced and the systematic engineering solutions implemented:

#### 1. Corpus Contamination & Latent Embedding Space Noise
- **The Challenge:** Ingesting 1,010 pages of historical shareholder letters alongside 7 CSV datasets (~6,000 Q&A pairs) created severe noise in latent vector space. First, raw documents contained prevalent UTF-8 / Windows-1252 mojibake artifacts (`â€™` for apostrophes, `â€œ` for quotes, `â€¦` for ellipses). Second, synthetic and crowdsourced Q&A datasets contained repetitive first-person prefixes (*'Speaking for myself'*, *'In my view'*, *'Personally I believe'*). Embedding models (`paraphrase-MiniLM-L6-v2`) mapped these identical conversational disclaimers into proximate latent regions, causing queries about unrelated topics to cluster together falsely simply because they shared identical conversational preambles.
- **How Overcome:** In `ingest.py`, I engineered a multi-pass text sanitation pipeline before vectorization:
  - Formulated a deterministic regex replacement dictionary (`MOJIBAKE_FIXES`) resolving all Windows-1252 byte-encoding anomalies.
  - Implemented an iterative prefix stripper (`PREFIX_PATTERNS`) that recursively peels up to 3 stacked conversational lead-ins while preserving sentence capitalization and grammatical casing.
  - Built a cryptographic content hashing deduplication filter to eliminate semantic duplicates across overlapping CSV files.
- **Measurable Impact:** Purged latent space distortion, producing distinct, cohesive semantic clusters verified via UMAP visualization (`visualize.py`).

#### 2. The Lexical-Semantic Retrieval Gap & Unstable Linear Fusion
- **The Challenge:** Dense vector search (FAISS) frequently failed on domain-critical exact entities: historical years (*'1977 acquisition'*), stock tickers (*'BRK.A'*, *'KO'*, *'AAPL'*), and specialized accounting metrics (*'insurance float'*, *'depreciation margin'*). Conversely, standard BM25 keyword search failed whenever queries used conceptual synonyms (*'temperament over intellect'*). When I initially attempted to merge both search spaces using min-max score normalization and linear weighting ($0.7 \cdot \text{FAISS} + 0.3 \cdot \text{BM25}$), the system became unstable: on short queries with rare tokens, BM25 scores exploded to 25.0+, completely overpowering FAISS; on long descriptive queries, BM25 scores compressed, rendering keyword signals ineffective.
- **How Overcome:** In `retriever.py`, I replaced linear score fusion with **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF\_Score}(d \in D) = \sum_{m \in \{\text{FAISS}, \text{BM25}\}} \frac{1}{k + \text{rank}_m(d)} \quad (k=60)$$
  RRF is scale-invariant and score-agnostic, neutralizing disparate distribution curves. Furthermore, I integrated a **Cross-Encoder reranking model** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) across the top-10 fused candidates, allowing full bidirectional cross-attention over `(query, document)` tokens to trim candidates down to the top-5 highest-signal context passages.
- **Measurable Impact:** Source Hit Rate surged from 73.3% (dense baseline) to 86.7% (hybrid) to **93.3%** (hybrid + reranking), while Mean Reciprocal Rank (MRR) climbed from 0.612 to **0.867**.

#### 3. Context Window Ceilings & Token Truncation in Local CPU Fallback Models
- **The Challenge:** While the primary production tier leverages Groq's cloud LPU running Llama 3.3-70B (128k context window), the platform required a zero-cost, fully offline fallback running on local commodity CPU using `google/flan-t5-base`. However, FLAN-T5 enforces a rigid 512-token maximum sequence length. When the application passed top-5 retrieved chunks (~1,500 words) plus conversation history into FLAN-T5, the tokenizer silently truncated the sequence from the right. This cut off the prompt instructions and the user's question, causing FLAN-T5 to output nonsensical single-word fragments or hallucinate completely.
- **How Overcome:** In `rag_chain.py`, I designed an adaptive, model-aware payload routing architecture:
  - **Dynamic Context Budgeting:** When Groq is active, the system injects full multi-chunk context blocks with strict numbered inline citation prompts (`[1]`, `[2]`). When falling back to FLAN-T5, the pipeline automatically trims context to top-3 chunks, strips extraneous metadata, and utilizes a concise zero-shot prompt template guaranteed to remain under 400 input tokens.
  - **Inline Citation Verification (`validate_citations`):** Built regex post-processors to detect and prune fabricated citation indices (`[6]`, `[7]`) that exceeded the retrieved context count.
- **Measurable Impact:** Achieved 100% stable offline local CPU generation under 2 seconds without token truncation errors or hallucinations.

#### 4. Financial Metric Inconsistencies, Free-Tier Rate Limits & Causal Bias
- **The Challenge:** In the quantitative module (`stock_analysis.py`, `trading_backtest.py`), integrating live equity data via `yfinance` introduced severe production risks:
  - Unreliable / Missing Data: Balance sheet items (e.g., R&D, operating cash flow, debt) frequently returned `None` or missing keys for financial institutions or loss-making companies, causing unhandled runtime exceptions or division-by-zero errors.
  - Lookahead Bias: Naive algorithmic backtests evaluated signals at day $t$ close and executed trades at day $t$ close, producing artificially inflated, unachievable Sharpe ratios.
  - Latency & Blocking Calls: Fetching multi-year historical bars sequentially for equity queries blocked the API event loop.
- **How Overcome:**
  - **Defensive Data Modeling:** Engineered resilient fallback chains across financial line items (e.g., checking both `operatingIncome` and `ebit`, guarding all financial ratio denominators against zero, and handling negative P/E ratios gracefully).
  - **Lookahead-Free Execution Lag:** In `trading_backtest.py`, strictly lagged trading signals by one complete day (`Position = Signal.shift(1)`), ensuring trades execute on day $t+1$ price action.
  - **Graceful Fault Isolation:** Wrapped backtesting compute inside resilient fallback handlers so that equity data issues never compromise or crash the core qualitative RAG dialogue.
- **Measurable Impact:** Delivered mathematically sound, institutional-grade equity backtesting curves across multiple market cycles (2018–present) with zero lookahead bias.

#### 5. Multi-Turn Conversational Drift Without Latency Penalties
- **The Challenge:** In real conversational sessions, users routinely ask terse, anaphoric follow-up questions (*'Why did he do that?'*, *'What about debt?'*, *'How did that affect GEICO?'*). If fed directly into retrieval, a 4-word query lacks semantic context and retrieves irrelevant chunks. However, routing every turn through an LLM 'Query Rewriter' chain (e.g., using GPT/Llama to rephrase the query) added 400–800ms of latency and increased API token costs.
- **How Overcome:** In `rag_chain.py`, I implemented a deterministic, zero-latency heuristic query enrichment algorithm:
  - Evaluated query length: if `len(question.split()) < 8` and prior conversation turns exist, the system extracts the salient initial 200 characters of the preceding assistant response and prepends it to the search query.
- **Measurable Impact:** Resolved conversational anaphora and context drift in $<1\text{ms}$ CPU time without introducing an extra LLM generation hop or incurring additional API expenses.

#### 6. Live SEC EDGAR 13F-HR Institutional Ingestion & Anti-Scraping Constraints
- **The Challenge:** Ingesting Berkshire Hathaway’s live quarterly institutional holdings via `sec_filings.py` faced three real-world obstacles:
  - SEC EDGAR enforces strict declarative `User-Agent` format compliance (`Sample Company Name AdminContact@domain.com`); generic headers trigger immediate HTTP 403 Forbidden blocks.
  - Accession paths and XML filenames change dynamically each quarter (`primary_doc.xml` vs. arbitrary info table XML names).
  - Institutional 13F filings frequently report multiple lines for the same company CUSIP (split across different investment managers or voting authorities), leading to underreported position values if not aggregated.
- **How Overcome:**
  - Standardized custom RFC-compliant headers (`BuffettRAG/1.0`).
  - Implemented dynamic filing discovery: queries CIK `0001067983` submissions JSON, retrieves the latest accession numbers, parses the directory `index.json`, and programmatically identifies the active XML info table.
  - Implemented namespace-isolated XML parsing (`{http://www.sec.gov/edgar/document/thirteenf/informationtable}`) that groups and sums both share counts and dollar values by CUSIP before computing quarter-over-quarter delta changes (New, Added, Reduced, Exited).
- **Measurable Impact:** Fully automated Berkshire 13F filing tracker with zero hardcoded URLs, robust to quarterly SEC directory format shifts.

---

### Q33: What fundamental engineering lessons did this project teach you?
**Answer:**
"This project delivered six enduring software and AI engineering lessons applicable to production machine learning systems:

1. **Lesson 1: Data Cleaning is Latent Space Engineering**
   - *Takeaway:* Preprocessing is often treated as simple text cleanup, but in vector systems, text preprocessing directly dictates the geometric distribution of the latent embedding space. Pervasive syntactic boilerplate or repetitive introductory phrases act as high-frequency noise that warps vector cosine distances. Rigorous domain-specific text scrubbing yields greater retrieval gains than fine-tuning embedding weights on uncleaned data.

2. **Lesson 2: Never Combine Uncalibrated Similarity Scores Linearly**
   - *Takeaway:* Sparse (BM25) and dense (FAISS cosine) algorithms compute scores on fundamentally incompatible scales and probability distributions. BM25 scores are unbounded and sensitive to query length and rare term frequencies, whereas cosine similarity is bounded between $[-1, 1]$. Using rank-based fusion (Reciprocal Rank Fusion) eliminates the fragile need for hyperparameter weight tuning across heterogeneous search paradigms.

3. **Lesson 3: Bi-Encoders Search, Cross-Encoders Decide**
   - *Takeaway:* Bi-encoders embed queries and documents independently to enable $O(1)$ sub-millisecond vector indexing, but they sacrifice token-to-token interaction across the query-document boundary. Deploying a two-stage retrieval pipeline—bi-encoders for broad recall ($k=10$), followed by a cross-encoder evaluating joint cross-attention for high precision ($k=5$)—delivers the optimal sweet spot between latency ($<30\text{ms}$) and ranking accuracy (+0.122 MRR).

4. **Lesson 4: Model Fallbacks Require Adaptive Payload Sculpting, Not Just Swapping**
   - *Takeaway:* Designing an offline fallback system is not as simple as wrapping an LLM call in a try/except block. Smaller local models (like FLAN-T5) have starkly different context window ceilings, attention dynamics, and instruction-following tolerances compared to 70B parameter frontier models. Resilient software architecture requires dynamic context compression, prompt shaping, and token budgeting tailored to the active inference engine.

5. **Lesson 5: In Quantitative Systems, Temporal Rigor is Non-Negotiable**
   - *Takeaway:* In financial engineering, a backtest without explicit temporal lag ($t+1$ execution) is an illusion. Any model that accesses same-day close data to execute trades on that same day close introduces subtle lookahead bias that guarantees live trading failure. Strict chronological isolation and multi-regime stress testing across bear, bull, and flat markets are mandatory for trustworthy algorithmic research.

6. **Lesson 6: Simple Deterministic Heuristics Often Beat Complex Neural Chains**
   - *Takeaway:* There is a temptation in modern AI engineering to solve every problem with an additional LLM call (e.g., multi-agent query rewriters, intent routers). However, in high-throughput applications, deterministic heuristics (regex-based ticker extraction, word-count follow-up enrichment) execute in sub-millisecond CPU time, incur zero API expense, eliminate non-deterministic failure modes, and keep the critical request path blazingly fast.

---

### Q34: How would you present these challenges and lessons in a 2-minute behavioral/technical interview response? (STAR Format)
**Answer:**
"**Situation:**
While developing the Warren Buffett Intelligence Hub, I set out to bridge qualitative generative AI with quantitative equity backtesting. However, once initial prototypes were running, we encountered three major architectural bottlenecks: dense vector retrieval was failing on exact financial entities and years, our local CPU fallback model was silently truncating prompts and producing hallucinations, and naive backtesting was susceptible to lookahead bias.

**Task:**
My objective was to engineer an institutional-grade, production-ready system that maintained 90%+ retrieval recall, guaranteed 100% stable offline local CPU generation under 2 seconds, and delivered mathematically sound, lookahead-free backtesting without inflating API latencies or cloud costs.

**Action:**
First, on retrieval, I identified that dense embeddings struggled with specialized financial numbers and tickers. Rather than relying on simple vector search or fragile linear score weighting, I implemented a hybrid pipeline combining FAISS dense vector search with BM25Okapi sparse search via Reciprocal Rank Fusion ($k=60$), followed by a Cross-Encoder reranker.
Second, on model fallback, I discovered FLAN-T5's 512-token ceiling was causing input truncation when receiving multi-chunk RAG contexts. I built dynamic model-aware payload routing that compresses context to top-3 chunks and uses a tailored zero-shot prompt for local execution, accompanied by regex post-processors to enforce citation verification.
Third, on quantitative analytics, I enforced a strict 1-day execution lag (`Signal.shift(1)`) across all backtesting strategies and built defensive data handlers to manage missing financial metrics gracefully.

**Result:**
Our benchmark suite demonstrated a jump in retrieval Source Hit Rate from 73.3% to 93.3% and an MRR improvement from 0.612 to 0.867. Local CPU inference achieved sub-2-second response times with zero crashes, while our quantitative engine delivered realistic, multi-regime trading simulations. 

**Core Lesson:**
This experience reinforced that production AI engineering isn't just about calling LLM APIs; it is about rigorous data sanitation, understanding latent space dynamics, enforcing strict causal boundaries in algorithmic logic, and using smart deterministic heuristics to optimize latency and reliability."

---

### Master Challenges, Solutions & Lessons Matrix

| # | System Subsystem | Technical Challenge / Failure Mode | Root Cause | Engineering Solution (How Overcome) | Core Lesson Learned |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Ingestion & Data Prep** (`ingest.py`) | Mojibake encoding artifacts & synthetic prefix contamination | Windows-1252 byte decoding mismatches & synthetic conversational preambles (*'Personally I believe'*) | Deterministic regex replacement table (`MOJIBAKE_FIXES`) + recursive multi-pass prefix stripping (`PREFIX_PATTERNS`) | Preprocessing is latent space engineering: semantic noise directly corrupts vector cosine proximity. |
| **2** | **Hybrid Retrieval** (`retriever.py`) | Vector search blind spots on exact entities & years; linear fusion score collapse | Bi-encoder semantic smoothing misses exact lexical tokens; BM25 and FAISS scores operate on incompatible distributions | Reciprocal Rank Fusion (RRF, $k=60$) + Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) top-10 to top-5 | Never linearly combine uncalibrated search scores; rank-based fusion provides scale invariance. |
| **3** | **Dual Inference** (`rag_chain.py`) | Offline local model hallucinated or produced single-word outputs | FLAN-T5's 512-token hard ceiling silently right-truncated prompts, severing questions and instructions | Dynamic model routing: top-3 chunks & condensed zero-shot prompt for FLAN-T5; top-5 & full citations for Groq Llama 3 | Fallbacks require adaptive context budgeting and payload sculpting, not just model swapping. |
| **4** | **Quantitative Engine** (`trading_backtest.py`) | Fictitious alpha & inflated backtest returns | Lookahead bias caused by calculating signals and executing trades at identical timestamp $t$ | Strict 1-day execution lag (`Position = Signal.shift(1)`) + multi-regime testing (2018–present) | Quantitative systems demand absolute chronological rigor; same-day execution is live suicide. |
| **5** | **Market Telemetry** (`stock_analysis.py`) | Runtime exceptions on missing keys, division by zero, and negative P/E | Unreliable free-tier API schema drifts across industries (e.g. banks vs tech) | Defensive extraction chains (checking `operatingIncome` vs `ebit`), safe denominators, and isolated try/except blocks | Zero-trust data engineering: assume external API metrics are missing or inconsistent until verified. |
| **6** | **Dialogue Memory** (`rag_chain.py`) | Terse follow-up questions (*'Why did he do that?'*) caused retrieval drift | Short conversational queries lack standalone semantic context for dense search | Deterministic query enrichment heuristic: prepends salient previous assistant response if query $<8$ words | Low-latency heuristics often outperform costly multi-agent LLM rewrite loops in production. |
| **7** | **SEC Regulatory Feed** (`sec_filings.py`) | HTTP 403 Forbidden blocks, dynamic XML filenames, and split CUSIP rows | Anti-scraping User-Agent checks, quarterly accession path changes, and multi-manager institutional reporting | RFC-compliant User-Agent header, dynamic submission JSON traversal, and CUSIP-grouped share/value aggregation | Public government data feeds require discovery-based navigation and domain-aware record aggregation. |

---

### Quick-Reference Technical Summary Table

| Component | Technical Selection | Primary Reason / Justification |
| :--- | :--- | :--- |
| **Chunk Size** | 1,000 characters | Matches MiniLM 256-token ceiling; preserves cohesive financial arguments |
| **Chunk Overlap** | 200 characters (20%) | Eliminates boundary loss; guarantees context continuity across splits |
| **Embeddings** | `paraphrase-MiniLM-L6-v2` (384-d) | Local, fast (<10ms CPU), zero API cost, tuned for semantic paraphrase |
| **Vector Store** | FAISS `IndexFlatIP` | Exact inner product (cosine similarity), sub-3ms in-memory search, zero infra |
| **Sparse Search** | `rank_bm25` (BM25Okapi) | Captures exact ticker symbols, historical years, and rare accounting terms |
| **Fusion Algorithm** | Reciprocal Rank Fusion ($k=60$) | Scale-invariant rank fusion; eliminates disparate score calibration issues |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | Joint cross-attention across (query, doc); filters top-10 to pristine top-5 |
| **Primary LLM** | Groq Llama 3.3-70B | Sub-second inference (>250 tok/s), strong reasoning, strict prompt adherence |
| **Fallback LLM** | `google/flan-t5-base` | Zero-dependency, offline CPU inference with dynamic token-budget handling |
| **Quant Telemetry** | 11 Buffett Ratios + Lagged Backtest | Automated moat scoring + lookahead-free strategy simulation |
| **API Backend** | FastAPI + Uvicorn | Async lifespan model caching, high-throughput non-blocking endpoints |

---

## 14. Alori Founding Full-Stack Engineer — 5-Minute Video Recording Guide & Master Script

> [!IMPORTANT]
> **Application Context:**
> - **Company:** Alori (Montréal / San Francisco · $100k–$250k base + equity)
> - **Mission:** *"Software is about to build itself. Business owners describe their company in a few sentences, and minutes later they're running it on their own app... AI teammates that pick up the phone, close deals, chase invoices, run the day-to-day... Ship to production many times a day."*
> - **Role:** Founding Full-Stack Engineer — fast, hands-on, obsessed with quality from the database to the last pixel, thrives in ambiguity, owns features end-to-end without throwing tickets over a wall.
> - **Application Prompt:**
>   > *"Walk through one thing you built: what it does, the part that broke and how you fixed it, and one decision you would make differently today. Record it right here with your camera, up to 5 minutes. No camera? Paste an unlisted Loom or YouTube link instead."*

---

### 14.1 Role & Video Prompt Deconstruction (Full-Stack SWE Perspective)

To stand out for a Founding Full-Stack Engineer role at an early-stage startup like Alori, the 5-minute video should highlight **practical software engineering ability**, not academic ML theory. Engineering founders want to see:
1. **End-to-End Full-Stack Ownership:** You build entire features yourself—from the database/indexes and backend APIs to responsive React components and animations.
2. **Pragmatic AI Integration:** You don't just call an API and hope it works. You treat AI models like any external, untrusted third-party service: managing payload limits, building reliable offline fallbacks, validating outputs with defensive code, and keeping latency low.
3. **Product & Systems Maturity:** You write maintainable code, think through edge cases, isolate external API failures, and care deeply about the user experience.

#### Why This Project is a Great Full-Stack SWE Demo:
- **Real Full-Stack Architecture:** React 18 frontend with Tailwind CSS, asynchronous FastAPI backend in Python, local search indexes (FAISS & BM25), and live third-party integrations (Yahoo Finance, SEC EDGAR).
- **Practical Problem-Solving:** Not just a toy chat UI—it pulls live financial balance sheets, computes 11 business health metrics, runs historical trading backtests with zero lookahead bias, and verifies source documents.
- **Real-World Debugging Story:** A clear, relatable software bug where payload limits silently truncated inputs in the offline fallback model, and how it was diagnosed and fixed with defensive request budgeting.

---

### 14.2 The 5-Minute Master Blueprint & Pacing Chart

Target duration: **4 minutes 30 seconds to 4 minutes 50 seconds** (comfortably under the 5-minute ceiling).

```
 0:00 ──────────────── 0:45 ──────────────── 2:00 ──────────────── 3:30 ──────────────── 4:25 ──────── 5:00
┌──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬─────────────┐
│ 1. Hook & Overview   │ 2. What It Does      │ 3. What Broke & How  │ 4. What I'd Do       │ 5. Why      │
│    (Full-Stack App)  │    (Architecture)    │    I Fixed It        │    Differently Today │    Alori    │
│    [45 sec]          │    [75 sec]          │    [90 sec]          │    [55 sec]          │    [35 sec] │
└──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┴─────────────┘
```

| Segment | Timestamp | Target Duration | Core Objective | Visual Cue (Screen Share) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Hook & Overview** | `0:00 – 0:45` | 45 sec | Introduce yourself and give the 30-second elevator pitch of the full-stack app. | Face to camera / React UI dashboard |
| **2. Full-Stack Walkthrough** | `0:45 – 2:00` | 75 sec | Show how the system works end-to-end: React UI $\rightarrow$ FastAPI $\rightarrow$ Hybrid Search $\rightarrow$ Live APIs. | Click through UI tabs (Chat, Moat Analysis, Backtesting) |
| **3. The Breaking Point** | `2:00 – 3:30` | 90 sec | Walk through the exact bug: local fallback model hitting a 512-token limit, truncating user prompts, and the fix. | Show `rag_chain.py` code in VS Code |
| **4. What I'd Do Differently** | `3:30 – 4:25` | 55 sec | Architectural upgrade: moving from synchronous blocking calls to an event-driven streaming model (SSE/WebSockets). | Clean workflow diagram / browser network tab |
| **5. Founding Engineer Fit** | `4:25 – 5:00` | 35 sec | Why you want to be a founding engineer: shipping daily, taking full ownership, moving fast. | Direct to camera, confident close |

---

### 14.3 Verbatim Master Speaking Script (Full-Stack SWE Edition)

> [!TIP]
> **Delivery Tips:** Speak naturally and conversationally, like you're explaining your project to another senior software engineer over coffee. Avoid academic buzzwords; focus on how data moves, what broke, and how you solved it. Total length is **~650 words** (~4.5 minutes).

#### [0:00 – 0:45] Part 1: The Hook & What I Built
*"Hi Alori team! I'm [Your Name]. Today I want to walk you through a full-stack platform I built from the ground up: the **Warren Buffett Financial Intelligence Hub**.

`[Screen Share: Switch to React Dashboard at localhost:5173]`

A lot of AI demos today are just simple wrappers around a generic chat prompt. I wanted to build a real, production-ready full-stack application that connects conversational AI to live business data and deterministic calculations.

From the database to the front end, it’s completely custom: a React 18 dark-mode dashboard with Tailwind CSS, an asynchronous FastAPI backend in Python, and a search engine that indexes over 1,000 pages of shareholder letters. On top of that, it pulls live SEC filings, evaluates fundamental financial health metrics on real stocks, and runs backtests on historical market data."*

#### [0:45 – 2:00] Part 2: What It Does & The Architecture
*"Let’s look at how the data flows when a user asks a question:

`[Screen Share: Type 'How does Berkshire evaluate insurance float?' or click a sample question]`

1. **Hybrid Search Pipeline:** Pure semantic vector search sounds great on paper, but in practice, it often struggles with exact keywords—like a specific year ('1977') or a stock ticker ('AAPL'). To fix that, I built a hybrid search: we run vector search and classic BM25 keyword search side-by-side, merge their results based on rank order, and run a quick second-pass reranker so the LLM gets only the 5 most relevant passages.
2. **Dual-Tier Inference with Local Fallback:** For production speed, we hit cloud LPUs on Groq for sub-second answers with bracketed citations. But if an API key is missing or the external network drops, the backend automatically falls back to a lightweight local model (`FLAN-T5`) running on CPU, so the app never goes down.
3. **Live Data & Deterministic Calculations:**
   `[Screen Share: Click on Trading tab or ask about AAPL in Chat]`
   When a user asks about a company, our backend detects the stock ticker with regex, hits Yahoo Finance for real-time balance sheet numbers, calculates 11 fundamental business metrics in Python, and runs an algorithmic trading backtest with interactive charts—all displayed right in the UI."*

#### [2:00 – 3:30] Part 3: The Part That Broke & How I Fixed It
*"Now, let’s talk about what broke—because building reliable fallbacks in production is where the real engineering happens.

`[Screen Share: Open VS Code to rag_chain.py lines ~225–250]`

When I first wired up our offline CPU fallback using the local `FLAN-T5` model, the app completely fell apart: on long questions, it either spat out one-word gibberish or hallucinated wildly, even though our cloud model worked fine.

When I debugged the raw payload entering the model, I found the bug: this local model has a strict **512-token input limit**. Because our retrieval step was sending five full text snippets—about 1,500 tokens of text—plus chat history, the tokenizer was truncating the input from the right side. 

That meant it was silently chopping off the user’s question and instructions at the end! The model was just receiving a random chunk of text with no question attached.

**How I Fixed It:**
Instead of a naive one-size-fits-all prompt, I built **adaptive request budgeting**:
1. **Model-Aware Context Budgeting:** When calling cloud models with 128k context, we send all 5 snippets and conversational history. But when the backend switches to the local model, it dynamically trims down to the top-3 snippets and strips extra conversational headers to guarantee the payload stays safely under 380 tokens.
2. **Zero-Latency Follow-up Helper:** In multi-turn chat, users ask short follow-ups like 'Why did he do that?'. Instead of wasting 1 to 2 seconds making an extra AI call just to rewrite the question, I wrote a lightweight Python check that pulls context from the previous message in under 1 millisecond.
3. **Regex Citation Validation:** I added a simple regex check that verifies all bracketed citation numbers (`[1]`, `[2]`) actually match the documents returned, stripping any hallucinations before sending the response to the frontend.

This immediately made our local CPU fallback 100% stable, returning clean answers in under 1.8 seconds with zero truncation errors."*

#### [3:30 – 4:25] Part 4: One Decision I Would Make Differently Today
*"If I were architecting this from scratch today, the main decision I’d change is **moving from synchronous request-response endpoints to an asynchronous event-driven streaming architecture using Server-Sent Events (SSE)**.

Right now in FastAPI, when a user asks a stock question, the endpoint runs hybrid search, calls the LLM, and fetches third-party data from Yahoo Finance and SEC EDGAR in a single sequential HTTP request. If Yahoo Finance or SEC EDGAR takes 2 seconds to respond, the entire screen has to wait.

If I rebuilt it today, I would decouple the tools into an event pipeline:
- The UI would immediately receive a streaming text response for the qualitative answer in milliseconds.
- Concurrently, the financial balance sheet calculations and SEC 13F filing feeds would run as separate background tasks, pushing JSON updates to hydrate the UI widgets as soon as they resolve.
- This would also make the app much more fault-tolerant: if an external finance API fails or times out, the chat still works flawlessly, and only that specific widget displays a retry state."*

#### [4:25 – 5:00] Part 5: Why Alori & Closing
*"This ties directly into why I’m excited about Alori. You’re building systems where business owners can describe what they need and have full apps generate and deploy right in front of them—software that actually runs the day-to-day operations.

I love working at an early-stage startup where you own systems end-to-end—from database queries and API design to UI polish and error states—and ship to production every day without waiting for someone else to write a ticket.

I’d love to bring this velocity and ownership to Alori as a Founding Full-Stack Engineer. Thanks for your time, and I look forward to speaking with the team!"*

---

### 14.4 Quick-Glance Teleprompter Bullet Points (SWE Friendly)

Use these high-level bullet cards on an adjacent monitor or phone if you prefer speaking naturally rather than reading verbatim:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. HOOK & WHAT I BUILT (0:00 - 0:45)                                   │
│ • Intro: Name + "Warren Buffett Financial Intelligence Hub"            │
│ • Problem: Not another toy chat wrapper; full-stack system with        │
│   real data, financial logic, and backtesting.                         │
│ • Stack: React 18 UI -> FastAPI -> Local Indexes -> Groq/Local Model.  │
├────────────────────────────────────────────────────────────────────────┤
│ 2. WHAT IT DOES & HOW IT WORKS (0:45 - 2:00)                           │
│ • Search: Hybrid keyword + vector search (solves blind spots for exact │
│   tickers/years) + fast reranking of top 10 results.                   │
│ • Dual Models: Sub-second cloud Groq + fully offline local CPU model.  │
│ • Live Data: Regex ticker detection, live balance sheets, 11 financial │
│   ratios, and interactive backtesting charts.                          │
├────────────────────────────────────────────────────────────────────────┤
│ 3. THE PART THAT BROKE & HOW I FIXED IT (2:00 - 3:30)                  │
│ • What Broke: Local offline model spat out one-word gibberish.         │
│ • Root Cause: 512-token input limit; passing 1,500 tokens caused the   │
│   tokenizer to chop off the user's question at the end!                │
│ • Fix (Adaptive Request Budgeting):                                    │
│   1) Model-aware payload trimming: top-3 chunks, max 380 tokens.       │
│   2) Fast 0ms follow-up context helper (no extra slow AI calls).       │
│   3) Regex check to strip invalid citation numbers.                    │
│ • Result: 100% stable offline CPU responses in ~1.8 seconds.           │
├────────────────────────────────────────────────────────────────────────┤
│ 4. ONE DECISION I'D MAKE DIFFERENTLY TODAY (3:30 - 4:25)               │
│ • Synchronous FastAPI endpoint -> Asynchronous SSE streaming.          │
│ • Why: External APIs (Yahoo Finance/SEC) shouldn't block the chat.     │
│ • Better: Stream text response instantly, hydrate widgets in parallel  │
│   as external data arrives, isolating third-party failures.            │
├────────────────────────────────────────────────────────────────────────┤
│ 5. WHY ALORI & FOUNDING FIT (4:25 - 5:00)                              │
│ • Alori mission: Software that builds itself, real operational tools.   │
│ • Founding mindset: End-to-end ownership, shipping daily, obsessed     │
│   with quality from API to UI. Ready to build!                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 14.5 Screen Walkthrough & Visual Presentation Strategy

If recording via **Loom** (recommended) or screen capture with webcam bubble:

#### Pre-Recording Setup:
1. **Window 1 (Browser Tab 1):** React Dashboard (`http://localhost:5173`)
   - Pre-load a clean chat state (use Incognito or click `+ New Chat`).
   - Have the "Trading" tab ready with AAPL or KO to show the 11 moat criteria badges and backtest equity curve.
2. **Window 2 (VS Code or IDE):** Open `rag_chain.py`
   - Scroll to lines ~225–250 where the dynamic context budgeting between Groq and FLAN-T5 occurs.
3. **Window 3 (Architecture / Slides):** Clean 3-box diagram:
   `React UI  ──>  FastAPI  ──>  Hybrid Search (FAISS + BM25)  ──>  Groq / Local Model`

#### Camera & Audio Tips:
- **Webcam Position:** Eye level; look directly into the camera lens when speaking your intro and outro (builds founder rapport).
- **Audio:** Use a dedicated headset or USB mic. Founder review is fast; clear, crisp audio immediately sets a professional tone.
- **Lighting:** Front-facing light; avoid backlighting from windows.
- **Energy:** Speak with conviction, curiosity, and pace. Founders look for engineers who are enthusiastic, practical, and take pride in what they build.

---

### 14.6 Alternative Failure Modes & What-I'd-Do-Differently Matrix (SWE Focus)

If you want to customize your video narrative or prepare for follow-up founder questions, here are two equally powerful alternative stories:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ OPTION A (RECOMMENDED): Local Model Payload Truncation & Adaptive Budgeting                       │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • What Broke: Local model has a 512-token ceiling; sending 1,500 tokens chopped off the prompt!   │
│ • How Fixed: Dynamic payload budgeting based on active model, 0ms context enrichment, regex checks│
│ • Why SWEs Love It: Shows you understand input limits, defensive programming, and fallback design│
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ OPTION B: Exact Tickers & Years Disappearing in Semantic Search                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • What Broke: Pure vector search missed exact ticker symbols ('BRK.A') and historical years.     │
│ • How Fixed: Hybrid search: vector similarity for concepts + BM25 for exact keywords, merged by  │
│   rank order so neither search dominates the other.                                              │
│ • Why SWEs Love It: Shows practical understanding of search trade-offs without over-engineering.  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ OPTION C: Fixing Lookahead Bias in the Backtesting Engine                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • What Broke: The backtester originally generated signals and executed trades on the exact same   │
│   day's close, which is physically impossible and created fake 40%+ returns.                     │
│ • How Fixed: Shifted positions forward by 1 trading day (`positions = signals.shift(1)`).        │
│ • Why SWEs Love It: Shows high engineering integrity, causal correctness, and attention to detail│
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Alternative "What I Would Do Differently Today":
- **Table-Aware Markdown Extraction:** *"Instead of naive 1,000-character chunking, I would parse balance sheets into structured Markdown tables before embedding. When standard splitters chop up text purely by character count, financial balance sheet tables get sliced in half across chunk boundaries. Using table-aware chunking keeps tabular financial data together as atomic documents."*

---

### 14.7 Stage 3: Meet the Founders — Anticipated Follow-Up Q&A

Once your 5-minute video passes Stage 1, you will meet Alori’s founders in Stage 3. Here are the 5 high-stakes questions they will ask and how to answer as a Founding Engineer:

#### Q1: "At Alori, we ship multiple times a day and business owners watch apps generate live. How do you balance shipping at extreme speed with code quality and reliability?"
**Answer:**
*"Speed and quality aren't trade-offs—they reinforce each other when you design modular, defensively isolated systems. In this project, when external balance sheet APIs were missing keys, I didn't spend three weeks re-architecting the database; I built defensive dictionary extraction chains and fallback defaults that isolated the failure to a single widget without crashing the main chat. 

As a founding engineer, speed means maintaining a tight feedback loop: shipping vertical slices that work end-to-end today, backed by automated sanity tests, while keeping code decoupled so you can refactor components in minutes rather than months."*

#### Q2: "How would you design an AI teammate that answers business calls, schedules appointments, and sends invoices without hallucinating or making costly mistakes?"
**Answer:**
*"You cannot rely on an LLM to manage mission-critical business state in raw conversational context. The golden architecture is **deterministic state machines with non-deterministic neural interfaces**:
1. **Tool-Calling with Strict JSON Schemas:** The LLM interprets the caller's intent and extracts parameters (e.g. appointment date, invoice amount), validated against strict Pydantic/Zod schemas.
2. **Deterministic Validation Guardrails:** A deterministic business logic layer checks availability, pricing limits, and customer history *before* any action is committed.
3. **Reversible Actions & Confirmation Gates:** For irreversible actions (charging a credit card, canceling a contract), the system requires affirmative dual-channel confirmation.
4. **Resilient Fallback Paths:** If confidence drops below threshold, the call is gracefully routed to a human or queued with an SMS summary."*

#### Q3: "You owned this project from the database to the React UI. How do you approach ambiguity when product requirements aren't defined?"
**Answer:**
*"Ambiguity energizes me because it means you have the freedom to solve the real underlying customer problem rather than blindly implementing a specification document. 

When building the quantitative analysis engine, nobody gave me a spec sheet for what financial ratios to display. I put myself in the user's shoes: an investor evaluating a business wants to know moat durability, capital efficiency, and downside risk. I researched Warren Buffett’s actual balance sheet heuristics, selected 11 high-signal criteria, built the calculation engine, and visualized it with clear pass/fail status pills. 

As a founding engineer, my default mode is: build a functional prototype fast, test it against real data, put it in front of users, and iterate based on what they actually do."*

#### Q4: "Why join an early-stage startup as a Founding Engineer rather than taking a comfortable big tech role?"
**Answer:**
*"In big tech, your impact is bounded by the perimeter of a sprint ticket. At Alori, software is literally about to build itself, and we're the ones building that engine. 

I want to be where the architectural decisions I make in the morning affect live businesses in the afternoon. I want the pressure and privilege of setting the engineering bar, defining our deployment culture, and directly shaping a product that changes how millions of non-technical business owners operate. That ownership and direct leverage is what excites me."*

#### Q5: "If an app generated by Alori fails to deploy for a live customer right in front of them, how would you design the system to self-heal?"
**Answer:**
*"Self-healing generation pipelines require a closed-loop execution sandbox:
1. **Sandboxed Compilation & Linting:** When the generation agent produces code (React components, database schemas, API routes), it immediately executes in an isolated container that runs static analysis and TypeScript compilation.
2. **Automated Error Telemetry Feedback Loop:** If compilation fails, the exact stack trace and compiler error are fed back into a specialized code-repair agent with instructions to rewrite only the offending AST node.
3. **Shadow Deployments with Health Checks:** Before routing user traffic, an automated smoke test verifies that the HTTP endpoints return 200 OK and database migrations have applied cleanly.
4. **Instant Semantic Rollback:** If a regression occurs, the system instantaneously rolls back to the last known stable snapshot in milliseconds, while showing the user an informative status: *'Optimizing your workflow...'*."*

---

## 15. Live App Demonstration & Code Walkthrough Master Playbook

> [!IMPORTANT]
> This section is your step-by-step presentation, demo, and code walkthrough cheat-sheet. It details **exactly what to click**, **what to type in the chat**, **what features to highlight**, **what to say (scripted speaking cues)**, and **which files and methods to show in the code editor**.

---

### 15.1 Application Demonstration Playbook (Tab-by-Tab)

#### Pre-Demo Verification Checklist (30 seconds before recording):
1. **FastAPI Backend Running:**
   ```powershell
   .\venv\Scripts\activate
   uvicorn server:app --reload --port 8000
   ```
   *Verify:* Navigate to `http://localhost:8000/api/health` -> `{"status": "ok"}`.
2. **React Frontend Running:**
   ```powershell
   cd ui
   npm run dev
   ```
   *Verify:* Navigate to `http://localhost:5173`.
3. **Environment:** Ensure `.env` has your valid `GROQ_API_KEY` (and `GROQ_MODEL=openai/gpt-oss-120b`).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DEMO FLOW & TAB NAVIGATION                                │
├────────────────────────────┬─────────────────────────────┬─────────────────────────────┤
│ 1. Berkshire Portfolio     │ 2. RAG Financial Chatbot    │ 3. Trading & Backtesting    │
│    URL: `/`                │    URL: `/chat`             │    URL: `/trading`          │
│    • SEC 13F Live Holdings │    • 4 Specific Chat Prompts│    • 11 Buffett Moat Ratios │
│    • Sector Allocation     │    • Inline Citations Drawer│    • 20/50 SMA & 14-day RSI │
│    • Quarterly Trades      │    • 0ms Query Enrichment   │    • Buffett's Take Critique│
└────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
                                           │
                             4. Evaluation Dashboard (`/evaluation`)
                             • 15-Question Benchmark Breakdown
                             • FAISS vs Hybrid vs Cross-Encoder Reranking
                             • Chunk Size Optimization Curves
```

---

#### Screen 1: Berkshire Portfolio Tracker (`/`)

* **What to Click / Show:**
  1. Open the landing page at `http://localhost:5173/`.
  2. Point out the top telemetry banner: **Total Portfolio Value**, **Number of Active Holdings**, and **Top Holding Concentration**.
  3. Hover over the **Sector Allocation** pie/donut chart (highlighting heavy allocation to Information Technology via Apple, Financial Services via Amex/BofA, and Consumer Staples via Coca-Cola).
  4. Scroll down to the **Institutional 13F Holdings Table**: point out columns for Symbol, Company Name, Shares Held, Market Value, Portfolio Weight, and Quarterly Activity tags (`NEW`, `ADD`, `TRIM`, `EXIT`).

* **What to Say (Verbatim Speaking Points):**
  > *"We start on the Berkshire Portfolio Tracker. This dashboard is powered by a live ingestion pipeline parsing Berkshire Hathaway's official SEC EDGAR Form 13F-HR filings. 
  > Instead of hardcoding static tables, our backend parses the official XML feeds from the SEC to surface Warren Buffett's real-time asset allocation. 
  > You can see from the sector chart that Berkshire is heavily concentrated in consumer moats and enterprise tech, with Apple making up over 40% of the portfolio alongside financial staples like Amex and Bank of America. 
  > This live data feeds our quantitative research engine and provides real-world context for our chat features."*

---

#### Screen 2: RAG Financial Chatbot (`/chat`) — 4 Core Test Scenarios

Navigate to the **Chat** tab (`http://localhost:5173/chat`).

##### Scenario 1: Qualitative Grounded RAG with Inline Citations
* **What to Type in Chat:**
  ```text
  What is Warren Buffett's philosophy on economic moats and pricing power?
  ```
* **What Features to Show:**
  1. Note the sub-second generation speed via the Groq cloud integration.
  2. Point out the inline bracketed citations (`[1]`, `[2]`, `[3]`) in the answer.
  3. Click on the **"Sources"** accordion drawer below the answer to expand it.
  4. Highlight the source card: show the source PDF name (`All-Berkshire-Hathaway-Letters.pdf`), page number, and the exact retrieved text chunk.
* **What to Say:**
  > *"Here I'll submit a classic qualitative question: 'What is Warren Buffett's philosophy on economic moats and pricing power?' 
  > Notice the speed—we get a sub-second response through our Groq LPU integration. 
  > More importantly, notice that every claim has a bracketed citation. If I expand the Sources drawer below, you can see the exact text snippet pulled from our indexed 1,010 pages of Berkshire letters, along with the original PDF page number. 
  > This works because of our hybrid search pipeline: we run traditional keyword search and semantic vector search in parallel, merge the results by rank order, and run a fast reranker on the top 10 items to feed only the 5 cleanest passages to the LLM."*

---

##### Scenario 2: Conversational Memory & 0ms Query Enrichment
* **What to Type in Chat (Immediately as a follow-up):**
  ```text
  Why does he prefer that over capital-intensive commodity businesses?
  ```
* **What Features to Show:**
  1. Show that you didn't have to re-type *"economic moats"* or *"Buffett"*.
  2. Show that the bot understands the context and explains how pricing power businesses earn high returns on capital without constant reinvestment.
  3. Point out the conversation session saved on the left sidebar.
* **What to Say:**
  > *"Now watch what happens when I ask a quick follow-up: 'Why does he prefer that over capital-intensive commodity businesses?' 
  > In many AI systems, ambiguous follow-ups fail because vector search doesn't know what 'that' refers to. Most teams solve this by making an extra LLM call to rewrite the prompt, which adds 1 to 2 seconds of latency.
  > Instead, I wrote a simple 0-millisecond check on the backend: if the question is under 8 words, it automatically grabs context from the previous message so the search query stays accurate without any added lag. 
  > The React frontend persists the conversation in localStorage and sends it with each request, keeping our FastAPI backend completely stateless and easy to scale."*

---

##### Scenario 3: Live Quantitative Equity Analysis Trigger
* **What to Type in Chat:**
  ```text
  What would Buffett think of AAPL stock?
  ```
  *(Alternative valid queries: `Analyze Apple stock using Buffett's principles` or `Is GOOGL a good investment?`)*
* **What Features to Show:**
  1. Point out that the chatbot recognized the equity ticker (`AAPL`) and automatically triggered stock analysis mode.
  2. Show the **Interactive Financial Scorecard** embedded directly inside the chat response.
  3. Highlight the 11 Buffett Fundamental Moat Criteria: Gross Margin $\ge 40\%$, Net Margin $\ge 20\%$, ROE $> 15\%$, Debt-to-Equity, and Free Cash Flow status pills (green/red badges).
  4. Note the qualitative synthesis explaining why Apple passes Berkshire's moat heuristics.
* **What to Say:**
  > *"Next, let's test our stock analysis mode: 'What would Buffett think of AAPL stock?'
  > The backend doesn't just do text retrieval here. Our regex parser detects the equity ticker and triggers live API calls to Yahoo Finance.
  > It pulls the real income statement and balance sheet data, computes 11 fundamental Warren Buffett ratios in Python—like gross margins over 40% and conservative debt levels—and renders this interactive scorecard right inside the chat bubble with pass/fail badges alongside the qualitative analysis."*

---

##### Scenario 4: Anti-Hallucination & Negative Refusal Boundary
* **What to Type in Chat:**
  ```text
  What was Warren Buffett's view on Ethereum and Bitcoin in the 1975 shareholder letter?
  ```
* **What Features to Show:**
  1. Show that the model does **not** invent a fake quote from 1975 about crypto.
  2. Show the polite, calibrated refusal message:
     ```text
     "I don't have enough grounded information in the dataset to answer that confidently."
     ```
* **What to Say:**
  > *"Finally, let's test how the system handles questions with no factual basis in the data: 'What was Warren Buffett's view on Ethereum and Bitcoin in the 1975 shareholder letter?'
  > In production software, an AI hallucination is far worse than an honest refusal. Our system prompt enforces strict negative guardrails: if the retrieved documents don't contain the answer, the model is instructed to decline.
  > As you see, it cleanly informs the user that there's no grounded evidence in the dataset, rather than fabricating a confident hallucination."*

---

#### Screen 3: Quantitative Equity Engine & Algorithmic Backtester (`/trading`)

Navigate to the **Trading** tab (`http://localhost:5173/trading`).

* **What to Click / Show:**
  1. Enter or select ticker **`AAPL`** (or **`KO`** / **`BRK-B`**) in the symbol input box.
  2. Point out the **Buffett Fundamental Moat Scorecard** (11 ratio cards displaying exact values, target thresholds, and passing/failing status tags).
  3. Scroll down to the **Backtesting Configuration Module**:
     - Select Strategy: **20/50 SMA Crossover** (or **14-day RSI**).
     - Set Date Range: 2018 to Present (capturing multiple market cycles).
     - Click **"Run Strategy Backtest"**.
  4. Highlight the **Interactive Performance Chart (Recharts)** showing the strategy's Cumulative Return equity curve compared against the Buy & Hold benchmark.
  5. Point out the quantitative telemetry cards: **Total Return**, **Annualized Return (CAGR)**, **Sharpe Ratio**, **Maximum Drawdown**, and **Win Rate**.
  6. Click the button: **"Get Warren Buffett's Take on this Strategy"** and show the AI's critique of algorithmic momentum trading.

* **What to Say:**
  > *"Moving over to the Trading & Backtesting tab. This page pairs fundamental value investing with quantitative backtesting.
  > At the top, we compute 11 balance sheet ratios derived from Buffett's literature—evaluating operating margins, capital efficiency, and debt coverage.
  > Below, we provide an algorithmic backtester for 20/50 day Moving Average crossovers and 14-day RSI strategies. 
  > A crucial software engineering detail here is **preventing lookahead bias**. In naive backtesters, signals calculated at Monday's market close are executed at Monday's close, which is physically impossible and inflates returns. In our engine, trading positions are strictly shifted forward by one day: `positions = signals.shift(1)`, guaranteeing realistic causal execution.
  > When we click 'Get Warren Buffett's Take', our RAG pipeline evaluates the backtest numbers and offers a classic Buffett critique—reminding us that short-term price momentum is very different from owning a durable business."*

---

#### Screen 4: Evaluation Benchmark Suite & Latent Space Analysis (`/evaluation`)

Navigate to the **Evaluation** tab (`http://localhost:5173/evaluation`).

* **What to Click / Show:**
  1. Highlight the **15-Question Benchmark Suite Matrix** organized into 4 difficulty tiers:
     - Tier 1: Direct Fact Retrieval
     - Tier 2: Cross-Document Synthesis
     - Tier 3: Quantitative / Financial Calculations
     - Tier 4: Adversarial / Negative Grounding Constraints
  2. Point out the **Retrieval Paradigm Comparison Chart**:
     - FAISS Only: 73.3% Source Hit Rate, 0.612 MRR
     - Hybrid (FAISS + BM25): 86.7% Source Hit Rate, 0.745 MRR
     - Hybrid + Cross-Encoder Reranked: **93.3% Source Hit Rate, 0.867 MRR**
  3. Point out the **Chunk Size Hyperparameter Optimization Curve**:
     - Demonstrating why 1,000 characters with 200-character overlap was selected over 500, 750, 1500, and 2000 character variants.

* **What to Say:**
  > *"Finally, on the Evaluation dashboard, we treat AI quality like software regression testing: you have to measure it systematically.
  > We built a benchmark test suite of 15 questions across four difficulty tiers to evaluate our search pipeline.
  > The data proved that pure vector search achieved only a 73.3% hit rate because it missed exact numbers and tickers. Adding traditional BM25 keyword search boosted hit rate to 86.7%, and adding a second-pass reranker pushed it to 93.3% while significantly improving search ranking precision.
  > We also tested 5 different text chunking configurations, confirming that 1,000 characters with 200 overlap strikes the optimal sweet spot between paragraph completeness and token limits."*

---

### 15.2 Comprehensive Code Walkthrough Playbook (File-by-File & Method-by-Method)

When sharing your screen in VS Code or discussing architecture with interviewers, follow this exact walkthrough order.

#### Code Walkthrough Master Summary Matrix

| Priority | File | Key Functions / Methods | What It Solves |
|:---:|:---|:---|:---|
| **1** | [`ingest.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/ingest.py) | `preprocess_text()`, `chunk_text()`, `build_indexes()` | Mojibake cleaning, recursive chunking, FAISS + BM25 indexing |
| **2** | [`retriever.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/retriever.py) | `hybrid_search()`, `reciprocal_rank_fusion()`, `rerank()` | Scale-invariant rank fusion ($k=60$) and cross-attention reranking |
| **3** | [`rag_chain.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/rag_chain.py) | `ask()`, `enrich_followup_query()`, `build_groq_prompt()`, `build_local_prompt()`, `validate_citations()` | Dual-tier LLM generation, dynamic context budgeting, regex citation validation |
| **4** | [`stock_analysis.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/stock_analysis.py) | `is_stock_query()`, `extract_ticker()`, `fetch_stock_data()`, `calculate_buffett_score()` | Natural language ticker extraction, balance sheet telemetry, 11 Buffett ratios |
| **5** | [`trading_backtest.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/trading_backtest.py) | `backtest_moving_average()`, `backtest_rsi()`, `calculate_metrics()` | Vectorized Pandas backtesting with strict causal lag (`shift(1)`) |
| **6** | [`sec_filings.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/sec_filings.py) | `fetch_berkshire_holdings()`, `parse_13f_xml()` | Live SEC EDGAR 13F-HR institutional filing extraction |
| **7** | [`server.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/server.py) | `lifespan()`, `@app.post("/api/chat")`, `@app.post("/api/stock/buffett-take")` | Asynchronous FastAPI endpoints, singleton caching, client history dictionary mapping |
| **8** | [`config.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/config.py) | `GROQ_MODEL`, `TOP_K`, `CHUNK_SIZE`, `EMBEDDING_MODEL` | Dynamic environment-driven model binding and system constants |
| **9** | [`ui/src/pages/ChatPage.jsx`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/ui/src/pages/ChatPage.jsx) | `handleSubmit()`, `deriveTitle()`, error handling block | Client-side session management, dynamic citation rendering, descriptive API error bubbling |

---

#### Detailed Method Walkthrough & Speaking Scripts

---

#### File 1: Data Ingestion & Preprocessing ([`ingest.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/ingest.py))

* **Method 1: `preprocess_text(text: str) -> str` (Lines ~35–60)**
  * **What to Highlight:** The regex replacements fixing Windows-1252 / UTF-8 character encoding glitches (e.g. `\u2019` to `'`, `\u201c` to `"`) and stripping boilerplate headers like `"Question: ... Answer: ..."`.
  * **What to Say:**
    > *"In `ingest.py`, data quality is the first step. The raw PDF shareholder letters and CSV datasets had character encoding glitches from older PDF formats. In `preprocess_text`, I normalize whitespace and scrub bad characters. I also remove boilerplate prompt headers like 'Question:' and 'Answer:' from the Q&A dataset so they don't pollute search similarity scores."*

* **Method 2: `chunk_text(docs) -> list[Document]` (Lines ~75–95)**
  * **What to Highlight:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])`.
  * **What to Say:**
    > *"In `chunk_text`, I split text using a 1,000-character chunk size with a 200-character overlap. 1,000 characters is roughly 220 tokens—which fits comfortably within our embedding model's limit while keeping complete paragraphs intact. The 200-character overlap acts like a sliding window so sentences and balance sheet numbers don't get chopped in half between chunks."*

* **Method 3: `build_indexes(chunks)` (Lines ~110–145)**
  * **What to Highlight:** Building the FAISS vector store (`FAISS.from_documents`) and simultaneously tokenizing text into `vectorstore/bm25.pkl`.
  * **What to Say:**
    > *"In `build_indexes`, we generate two local search indexes: a FAISS vector index for semantic concept matching, and a BM25 keyword index for exact word and ticker lookups. Both are saved to disk in `vectorstore/` for fast in-memory loading by the backend."*

---

#### File 2: Hybrid Search & Cross-Encoder Reranking ([`retriever.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/retriever.py))

* **Method 1: `reciprocal_rank_fusion(faiss_results, bm25_results, k=60)` (Lines ~45–75)**
  * **What to Highlight:** The formula $RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$.
  * **What to Say:**
    > *"In `retriever.py`, combining keyword search and vector search naively with raw scores fails because BM25 produces scores from 0 to 30+, while vector cosine similarity is between -1 and 1. Trying to do `0.7 * FAISS + 0.3 * BM25` means BM25 completely dominates. Instead, I used Reciprocal Rank Fusion: it combines results based on rank position (1st, 2nd, 3rd) rather than raw numbers, giving both search methods a fair, balanced weighting."*

* **Method 2: `rerank(query, candidates, top_n=5)` (Lines ~85–115)**
  * **What to Highlight:** Loading `cross-encoder/ms-marco-MiniLM-L-6-v2` and passing `[(query, doc.page_content) for doc in candidates]` into `cross_encoder.predict()`.
  * **What to Say:**
    > *"In `rerank`, we take the top 10 search results and run a quick second-pass reranker. Vector search alone is fast but approximate. The reranker compares the query and each text snippet together in detail, scoring their true relevance, so we can pass only the 5 cleanest passages to the LLM."*

---

#### File 3: Generation Pipeline & Dynamic Context Budgeting ([`rag_chain.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/rag_chain.py))

* **Method 1: `enrich_followup_query(question, history)` (Lines ~40–55)**
  * **What to Highlight:** The heuristic `len(question.split()) < 8` and extracting `last_assistant[-1]['content'][:200]`.
  * **What to Say:**
    > *"In `rag_chain.py`, line 45, is our 0ms context helper. When users ask short follow-ups like 'Why did he do that?', making an extra LLM call to rewrite the prompt would add 1 to 2 seconds of latency. Instead, if a question is under 8 words and there's prior conversation, we grab the relevant context from the last assistant message in zero milliseconds."*

* **Method 2: `ask(question, history=None)` (Lines ~177–275)**
  * **What to Highlight:** 
    1. Checking `is_stock_query(question)` and fetching quantitative stock telemetry.
    2. Dynamic context budgeting between Groq LPU and FLAN-T5 (lines ~225–245).
    3. Regex citation verification `validate_citations(answer, len(docs))`.
  * **What to Say:**
    > *"The `ask()` function is our main backend orchestrator:
    > 1. It checks if the question mentions a stock; if so, it pulls live data from Yahoo Finance and calculates 11 financial metrics.
    > 2. It runs our hybrid search and reranking.
    > 3. Lines 230 to 245 handle request budgeting: when calling cloud models on Groq, we pass all 5 snippets and chat history. But if the app falls back to our local CPU model, it automatically trims to 3 snippets and caps the input under 380 tokens to prevent the 512-token truncation bug.
    > 4. Finally, `validate_citations()` strips any made-up citation numbers before sending the JSON response to the frontend."*

---

#### File 4: Fundamental Analysis & Buffett Scoring ([`stock_analysis.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/stock_analysis.py))

* **Method 1: `is_stock_query(question)` & `extract_ticker(question)` (Lines ~180–235)**
  * **What to Highlight:** Regex matching for equity symbols (e.g. `AAPL`, `MSFT`) and alias dictionaries (mapping `"apple"` to `AAPL`, `"coca cola"` to `KO`).
  * **What to Say:**
    > *"In `stock_analysis.py`, we use regex to detect stock ticker symbols (like `AAPL` or `MSFT`) and support common company names (like 'Apple' or 'Coca Cola') so users can ask questions naturally without knowing the exact ticker."*

* **Method 2: `calculate_buffett_score(metrics)` (Lines ~85–170)**
  * **What to Highlight:** The evaluation of 11 proprietary fundamental ratios with exact thresholds (Gross Margin $\ge 40\%$, Free Cash Flow positive, Long Term Debt to Net Income $< 5\times$, ROE $> 15\%$).
  * **What to Say:**
    > *"In `calculate_buffett_score`, we took Warren Buffett's investment heuristics and translated them into clean Python rules. We evaluate 11 metrics across the income statement, balance sheet, and cash flow—checking for things like gross margins over 40%, positive free cash flow, and manageable long-term debt."*

---

#### File 5: Lookahead-Free Backtesting Engine ([`trading_backtest.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/trading_backtest.py))

* **Method 1: `backtest_moving_average()` & `backtest_rsi()` (Lines ~40–120)**
  * **What to Highlight:** 
    ```python
    df['signal'] = np.where(df['sma_fast'] > df['sma_slow'], 1.0, 0.0)
    df['position'] = df['signal'].shift(1)  # CRITICAL: Eliminate lookahead bias
    df['strategy_return'] = df['position'] * df['market_return']
    ```
  * **What to Say:**
    > *"In `trading_backtest.py`, look at line 58: `df['position'] = df['signal'].shift(1)`. This is the most important bug prevention in the backtester. If an algorithm generates a buy signal at Monday's market close, you can't buy at Monday's close—you can only trade on Tuesday morning. Shifting the signal forward by 1 bar strictly prevents lookahead bias, ensuring that our returns, Sharpe ratio, and drawdown represent realistic trading."*

* **Method 2: `calculate_metrics(strategy_returns, market_returns)` (Lines ~130–170)**
  * **What to Highlight:** Vectorized calculation of Annualized Sharpe Ratio ($\frac{\text{Mean Excess Return}}{\text{Std Dev}} \times \sqrt{252}$) and Maximum Drawdown calculation via cumulative peak tracking.
  * **What to Say:**
    > *"In `calculate_metrics`, we compute standard portfolio metrics: annualized volatility scaled by $\sqrt{252}$ trading days, Sharpe ratio assuming a risk-free benchmark, and maximum peak-to-trough drawdown."*

---

#### File 6: SEC EDGAR 13F-HR Institutional Ingestion ([`sec_filings.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/sec_filings.py))

* **Method 1: `fetch_berkshire_holdings()` (Lines ~25–90)**
  * **What to Highlight:** Direct HTTP queries to SEC EDGAR API with User-Agent compliance, querying CIK `0001067983` (Berkshire Hathaway Inc), and parsing the 13F-HR `informationTable` XML structure.
  * **What to Say:**
    > *"In `sec_filings.py`, we query the SEC EDGAR API using Berkshire Hathaway's CIK number (`0001067983`). We parse the 13F-HR XML document to extract stock names, share counts, and position values, comparing with the previous quarter to show portfolio changes in the UI."*

---

#### File 7: Asynchronous API Server & History Plumbing ([`server.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/server.py))

* **Method 1: `lifespan(app: FastAPI)` (Lines ~40–60)**
  * **What to Highlight:** Modern async lifespan context manager initializing singleton models in memory at startup rather than per-request.
  * **What to Say:**
    > *"In `server.py`, we use FastAPI's lifespan handler to load heavy models into memory once at startup rather than on every request, preventing cold-start delays for users."*

* **Method 2: `@app.post("/api/chat")` (Lines ~255–265)**
  * **What to Highlight:** 
    ```python
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history] if req.history else None
    return ask(req.question, history=history_dicts)
    ```
  * **What to Say:**
    > *"In `server.py` line 258, notice how we handle conversation history. The React frontend sends client-side history from `localStorage` as Pydantic models. We unpack them into clean Python dicts and pass them to `ask()`, connecting client history to our prompt window and 0ms query enrichment without needing a database for short sessions."*

---

#### File 8: Configuration & Dynamic Model Routing ([`config.py`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/config.py))

* **What to Highlight:** Line ~31:
  ```python
  GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
  ```
* **What to Say:**
  > *"In `config.py`, line 31 uses `os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')`. Reading configuration from environment variables rather than hardcoding model names allows us to swap models or update endpoints in `.env` without touching Python code."*

---

#### File 9: React UI Architecture ([`ui/src/pages/ChatPage.jsx`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/ui/src/pages/ChatPage.jsx))

* **What to Highlight:**
  1. `localStorage` chat session persistence (`activeId`, `conversations`).
  2. The `handleSubmit()` async flow and error catching block (lines ~270–285) that surfaces detailed backend error telemetry directly to the UI.
  3. Interactive citations accordion and stock scorecard component embedding.
* **What to Say:**
  > *"On the frontend in `ChatPage.jsx`, the UI is built with React 18, Vite, and Tailwind CSS. Chat sessions are saved to browser `localStorage` so chats persist across page reloads. Our catch block extracts detailed API error messages from the backend so users see meaningful feedback instead of a generic failure screen."*

---

### 15.3 Rapid Presentation Cheat-Sheet: 60-Second Elevators Per Module (SWE Friendly)

If an interviewer asks you to quickly summarize any specific part of the system, use these simple 60-second answers:

* **1. Data Ingestion & Preprocessing (`ingest.py`):**
  *"We ingested 1,010 pages of PDF shareholder letters and ~6,000 Q&A pairs. We cleaned character encoding issues, stripped boilerplate headers, and split text into 1,000-character chunks with a 200-character overlap to balance complete paragraphs with embedding token limits."*

* **2. Hybrid Search & Reranking (`retriever.py`):**
  *"Vector search is great for concepts but misses exact tickers and years, while keyword search misses semantic meaning. We run both together, combine their results by rank position so neither dominates, and run a fast reranker on the top 10 items to feed the 5 cleanest snippets into the prompt."*

* **3. Dual Models & Request Budgeting (`rag_chain.py`):**
  *"We use fast cloud models on Groq for sub-second generation, with an offline CPU model (`FLAN-T5`) as a zero-dependency fallback. To prevent the local model's 512-token limit from cutting off user questions, the backend dynamically trims context based on which model is active, and regex validates citation numbers."*

* **4. Conversational Context & Disambiguation (`rag_chain.py`, `server.py`):**
  *"The backend is stateless; the React UI passes conversation history from `localStorage`. For short follow-up questions under 8 words, we borrow context from the previous message in zero milliseconds, avoiding the 1 to 2 second delay of an extra AI prompt rewrite call."*

* **5. Financial Telemetry & Backtesting (`stock_analysis.py`, `trading_backtest.py`):**
  *"We detect equity tickers with regex, query Yahoo Finance for live balance sheets, and compute 11 Warren Buffett business ratios. Our backtester simulates SMA and RSI trading strategies with a strict 1-day lag (`shift(1)`) to eliminate lookahead bias and compute realistic returns and drawdowns."*
