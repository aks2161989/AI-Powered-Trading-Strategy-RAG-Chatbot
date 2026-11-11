# Warren Buffett Intelligence Hub — Resume Points & Interview Guide

This document contains tailored resume bullet points, technical metrics, and interview talking points for the **Warren Buffett Intelligence Hub (RAG & Algorithmic Trading Platform)** project.

---

## 1. Project Title & Tech Stack Options

### Option A: Standard / AI Engineer
```text
Warren Buffett Intelligence Hub – Financial RAG & Algorithmic Backtesting Platform
Tech Stack: Python, LangChain, FAISS, BM25, Sentence-Transformers, Groq (Llama 3.3-70B), FLAN-T5, FastAPI, yfinance, UMAP, React, Vite, Tailwind CSS
```

### Option B: Quantitative Finance / FinTech
```text
AI-Powered Equity Research & Algorithmic Trading Platform
Tech Stack: Python, FastAPI, yfinance, SEC EDGAR API, FAISS, BM25, Cross-Encoder, Groq (Llama 3.3), Pandas, NumPy, React, Recharts
```

---

## 2. Resume Bullet Sets

### Set 1: Balanced & Impact-Oriented (Recommended)
*Best for general Software Engineering, AI Engineering, and Data Science roles.*

* **Architected an end-to-end financial RAG platform** ingesting 1,010 pages of shareholder letters and ~6,000 curated Q&A pairs (~11,000 chunks) with metadata tagging, cross-file deduplication, and contextual multi-turn query enrichment.
* **Engineered a hybrid retrieval and reranking pipeline** combining FAISS dense semantic search and BM25 sparse keyword search via Reciprocal Rank Fusion (RRF, $k=60$), followed by a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to boost retrieval precision down to top-5 grounded contexts.
* **Integrated a dual-engine LLM backend with strict anti-hallucination constraints**, deploying Groq-hosted Llama 3.3-70B with automatic fallback to a local CPU `FLAN-T5` model, dynamically adapting prompt token budgets to prevent context truncation.
* **Built a quantitative stock analysis and backtesting engine** using `yfinance` to evaluate 11 fundamental Buffett ratios (Gross Margin, Debt/Equity, ROE) and execute lookahead-free algorithmic strategies (MA Crossover, RSI) computing Sharpe ratio and max drawdown.
* **Developed an automated evaluation benchmark** across a 15-question multi-tier test suite measuring Mean Reciprocal Rank (MRR) and Hit Rates, coupled with UMAP 2D projection to analyze embedding cluster separations.

---

### Set 2: AI / Machine Learning / NLP Focus
*Best for Applied ML, NLP Engineer, and GenAI Specialist roles.*

* **Designed a production-grade RAG pipeline** across an 11k-document corpus using `paraphrase-MiniLM-L6-v2` dense vectors and BM25Okapi, resolving bi-encoder retrieval limitations through cross-encoder reranking and RRF score fusion.
* **Implemented query classification and contextual rewriting** to detect financial tickers, disambiguate conversational follow-ups (<8 words) via dialogue history, and apply strict prompt grounding to eliminate domain hallucinations.
* **Benchmarked retrieval performance across 5 text-chunking configurations (500–2,000 chars)** and 3 retrieval paradigms, quantifying trade-offs in Source Hit Rate, Keyword Hit Rate, and MRR.
* **Visualized high-dimensional latent space representations** by deploying UMAP dimensionality reduction on 384-dimensional FAISS embeddings, mapping semantic clusters across 12 thematic financial categories.
* **Engineered a hybrid cloud-edge inference architecture**, leveraging Groq LPUs for sub-second 70B parameter inference while packaging a quantized local Seq2Seq model (`google/flan-t5-base`) for zero-dependency offline environments.

---

### Set 3: Quantitative Finance / FinTech Focus
*Best for Quant Developer, FinTech Engineer, and Equity Research roles.*

* **Developed an AI-driven equity research platform** synthesizing Berkshire Hathaway shareholder letters, SEC EDGAR 13F institutional filing feeds, and live Yahoo Finance market data into institutional-grade investment memos.
* **Programmed an automated fundamental scorecard** evaluating 11 Warren Buffett financial rules of thumb (Gross Margin ≥ 40%, CapEx < 25%, Adj. D/E < 0.8x) to compute economic moat strength (Wide/Narrow/None).
* **Constructed a vectorized backtesting framework** simulating 20/50 SMA Crossover and RSI (30/70) strategies on historical daily price action since 2018, computing equity curves, annualized returns, Sharpe ratio, and maximum drawdown.
* **Automated quarterly institutional portfolio tracking** by parsing SEC 13F-HR XML submissions for Berkshire Hathaway (CIK 0001067983), categorizing real-time portfolio buys, adds, trims, and exits.
* **Unified qualitative RAG retrieval with quantitative equity metrics**, dynamically injecting real-time balance sheet, income statement, and backtest results into LLM prompts for grounded valuation commentary.

---

### Set 4: Concise (3 Bullets for Space-Constrained Resumes)
*Best when you have limited vertical space on your resume.*

* **Engineered an end-to-end Financial RAG system** over an 11,000-chunk knowledge base using hybrid search (FAISS + BM25Okapi), Reciprocal Rank Fusion ($k=60$), and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) to maximize retrieval precision.
* **Built an algorithmic backtesting and fundamental analysis engine** via `yfinance` and SEC 13F XML feeds, automating 11 Warren Buffett financial ratio checks and historical strategy simulations (Sharpe, Drawdown, Win Rate).
* **Deployed an asynchronous FastAPI backend and React dark-mode dashboard** with Groq Llama 3.3-70B / local FLAN-T5 fallback, conversation memory, and an evaluation suite tracking MRR and UMAP embedding projections.

---

## 3. Comparison: Original Draft vs. Upgraded

| Aspect | Original Draft | Upgraded Version |
| :--- | :--- | :--- |
| **Scale & Numbers** | *"multi-source strategy data"* | **1,010 PDF pages, ~6,000 Q&A pairs, ~11,000 chunks, 11 Buffett ratios, 2 algorithmic strategies** |
| **Technical Specifics** | *"hybrid semantic/keyword search and cross-encoder reranking"* | **FAISS + BM25Okapi, Reciprocal Rank Fusion ($k=60$), `ms-marco-MiniLM-L-6-v2` down to top-5** |
| **LLM & Inference** | *"feeds grounded evidence to an LLM"* | **Groq Llama 3.3-70B LPU inference + local `FLAN-T5` CPU fallback with token budget handling** |
| **Trading & Quant Depth** | *Omitted entirely* | **Automated Buffett financial criteria, lookahead-free backtesting (Sharpe, Max Drawdown), SEC 13F parsing** |
| **Evaluation Rigor** | *"iterative evaluation and human review"* | **15-query multi-tier benchmark (MRR, Hit Rates), 5 chunk size tests, UMAP 2D latent space visualization** |

---

## 4. Key Metrics & Numbers Reference

* **Corpus Scale**: 1,010-page shareholder letter PDF + 7 domain CSVs = 5,992 unique Q&A pairs → ~10,000–12,000 embedded chunks.
* **Chunking Configuration**: 1,000 characters with 200-character overlap using `RecursiveCharacterTextSplitter`.
* **Embedding Model**: `sentence-transformers/paraphrase-MiniLM-L6-v2` (384 dimensions, L2-normalized for cosine similarity).
* **RRF Constant**: $k = 60$ fusing top-10 FAISS and top-10 BM25 results.
* **Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` filtering top-10 candidates to top-5.
* **Evaluation Suite**: 15 test questions across 4 difficulty tiers (Easy keyword, Medium paraphrased, Hard semantic, Cross-topic/adversarial); metrics: Source Hit Rate, Keyword Hit Rate, Mean Reciprocal Rank (MRR).
* **11 Buffett Ratios**:
  - *Income Statement*: Gross Margin (≥ 40%), SG&A Margin (≤ 30%), R&D Margin (≤ 30%), Depreciation Margin (≤ 10%), Interest Expense Margin (≤ 15%), Income Tax Rate (~21%), Net Profit Margin (≥ 20%), EPS Growth (> 1.0x YoY).
  - *Balance Sheet*: Cash > Debt Ratio (> 1.0x), Adjusted Debt-to-Equity (< 0.80x).
  - *Cash Flow*: CapEx Margin (< 25%).
* **Backtesting Parameters**:
  - *MA Crossover*: 20-day vs. 50-day SMA, 1-day execution lag (no lookahead bias).
  - *RSI*: 14-day window, 30 oversold / 70 overbought thresholds.
  - *Telemetry*: Total Return, Annualized Return, Volatility, Sharpe Ratio, Max Drawdown, Win Rate, Trade Logs.
* **SEC 13F XML Engine**: Berkshire Hathaway CIK `0001067983`, diffing quarterly 13F-HR filings.

---

## 5. Interview Cheat Sheet (Technical Talking Points)

> [!TIP]
> For the complete 31-question master interview handbook with deep architectural diagrams, quantitative derivations, and STAR-format behavioral answers, see the dedicated guide: [`docs/interview_prep.md`](file:///C:/Users/aksha/OneDrive/NEU-coursework/CSYE7380/Final%20project/CSYE7380_FinalProject/docs/interview_prep.md).

### Q: What steps did you take to develop this project from start to finish?
> **Answer**: I followed a 6-phase engineering lifecycle:
> 1. **Data Ingestion & Hygiene**: Ingested 1,010-page shareholder letters and ~6,000 Q&A pairs; sanitized UTF-8 mojibake, stripped synthetic conversational prefixes, filtered short/empty rows, and deduplicated records.
> 2. **Chunking & Indexing**: Applied `RecursiveCharacterTextSplitter` (1,000 chars, 200 overlap); embedded via `paraphrase-MiniLM-L6-v2` into FAISS (`IndexFlatIP`) and tokenized into BM25Okapi.
> 3. **Hybrid Retrieval & Reranking**: Combined FAISS and BM25 using Reciprocal Rank Fusion (RRF, $k=60$) to fetch top-10 candidates, then reranked down to top-5 via `cross-encoder/ms-marco-MiniLM-L-6-v2`.
> 4. **Dual-Tier Generation**: Configured Groq LPU (Llama 3.3-70B) for sub-second generation with local CPU fallback (`google/flan-t5-base`) with dynamic prompt token clamping to prevent context truncation.
> 5. **Quantitative Stock & Backtesting Engine**: Built ticker classification, automated 11 Warren Buffett financial ratios, and executed lookahead-free 20/50 SMA and RSI backtests with Sharpe and drawdown telemetry.
> 6. **Evaluation & Full-Stack UI**: Built a 15-query multi-tier benchmark suite (MRR, Hit Rates), projected embeddings via UMAP, and connected an async FastAPI backend to a React dark-mode dashboard.

### Q: Why did you choose a chunk size of 1,000 characters with 200 overlap?
> **Answer**: 
> 1. **Model Token Limit**: `paraphrase-MiniLM-L6-v2` has a sequence limit of 256 tokens (~1,000–1,200 chars). Chunks larger than 1,000 chars risk hard truncation or semantic dilution.
> 2. **Context Cohesion**: Shareholder letter arguments span 1–2 paragraphs. 500 characters split arguments in half, while 2,000 characters conflate multiple distinct topics.
> 3. **200 Overlap (20%)**: Acts as a sliding buffer preventing keywords and sentences from being split across chunk boundaries.
> 4. **Empirical Benchmarks**: Tested 500, 750, 1000, 1500, and 2000 chars in `evaluate.py`; 1,000 chars yielded the highest Mean Reciprocal Rank (MRR) and Keyword Hit Rate.

### Q: Why use Hybrid Search (FAISS + BM25) instead of just vector search?
> **Answer**: Dense embeddings excel at semantic similarity and paraphrasing (e.g., mapping "business moat" to "competitive advantage"), but often fail on exact keyword queries, proper nouns, ticker symbols, or specific years (e.g. "1977 operating earnings"). BM25 provides exact token-level matching. Combining both ensures high recall across both conceptual and entity-specific queries.

### Q: Why use Reciprocal Rank Fusion (RRF) instead of linear score combination?
> **Answer**: FAISS returns normalized cosine distances bounded in $[0, 1]$, whereas BM25 produces unbounded TF-IDF scores dependent on document length and term frequency. Normalizing and summing them creates score distortion and requires brittle manual weight tuning. RRF operates strictly on ordinal ranks:
> $$\text{RRF Score}(d) = \sum_{m \in \{\text{FAISS}, \text{BM25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
> This makes fusion scale-invariant, robust to query length, and parameter-free.

### Q: Why add a Cross-Encoder if you already have FAISS?
> **Answer**: FAISS uses a bi-encoder where query and documents are encoded independently ($O(1)$ search at query time) with no cross-attention between query and document tokens. Cross-encoders pass query and candidate text jointly through transformer attention layers, capturing intricate query-token and passage-token interactions. We get the speed of bi-encoders to filter ~11,000 docs to 10 in ~5ms, and the precision of cross-encoders to rank the top 10 down to 5 in ~25ms.

### Q: How did you mitigate LLM hallucinations?
> **Answer**: Three layers of defense:
> 1. **Prompt Grounding**: Enforced negative constraints ("use ONLY the retrieved context below", explicit refusal if evidence is insufficient).
> 2. **Algorithmic Citation Validation**: Validated `[1]`, `[2]` citation markers using regex (`validate_citations`), stripping fabricated source numbers.
> 3. **Dual Prompt Paths**: Separate prompt architectures for qualitative Buffett lore vs. quantitative stock analysis where actual financials and backtest results are pre-computed and injected directly into context.

### Q: How did you eliminate lookahead bias in backtesting?
> **Answer**: The trading signal generated at day $t$ (e.g., moving average crossover based on day $t$ close) is shifted by 1 trading day: `Position = Signal.shift(1)`. Strategy returns are calculated as `Position × Market_Return`, ensuring trades execute on day $t+1$ without future price awareness.

### Q: How did you optimize follow-up queries without adding LLM latency?
> **Answer**: Instead of calling an LLM rewriter (adding 1.5–2s latency), when a query is short (<8 words) and has conversation history, the system prepends the first 200 characters of the preceding assistant response to the retrieval query. This injects the antecedent entity/topic with zero latency overhead.

