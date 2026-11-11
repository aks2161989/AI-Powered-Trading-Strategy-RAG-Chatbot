flowchart TD
    A["PDF - 1010 pages of Shareholder Letters"] --> B["Chunk Text (1000 chars, 200 overlap)"]
    C["7 CSVs - 996 Q&A Pairs"] --> D["Deduplicate and Filter Rows"]
    D --> E["Format as Question/Answer pairs"]
    B --> F["Embed with MiniLM-L6-v2"]
    E --> F
    F --> G["FAISS Vector Store (index.faiss + index.pkl)"]