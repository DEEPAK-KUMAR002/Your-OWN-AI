# Your-OWN-AI — Java Vector Database and Local RAG

A lightweight vector database and browser demo implemented in **Java 17+**. It includes three nearest-neighbor search strategies, a REST API, a small web UI, and an optional document question-answering pipeline backed by a local Ollama instance.

The backend uses only JDK modules; no Maven or third-party Java libraries are required.

## Features

| Feature | Details |
|---|---|
| Search algorithms | HNSW approximate search, KD-Tree search, and exact brute-force search |
| Distance metrics | Cosine distance, Euclidean distance, and Manhattan distance |
| Demo data | 20 fixed 16-dimensional vectors across CS, Math, Food, and Sports |
| Document embeddings | Ollama `nomic-embed-text` embeddings; documents are split into overlapping 250-word chunks |
| RAG Q&A | Retrieves relevant chunks from the document index and asks Ollama `llama3.2` to generate an answer |
| REST API | Search, insert, delete, list, benchmark, HNSW graph info, document retrieval, and RAG routes |
| Browser UI | Search and compare algorithms, view a PCA scatter plot, add documents, and ask questions |

## How it works

```text
Document text ──> Ollama embedding model ──> 768D vectors ──> HNSW document index
                                                               │
Question ──> Ollama embedding model ──> nearest chunks ──> Ollama generation model ──> answer
```

The demo search index is separate from document embeddings: demo vectors are 16-dimensional, while Ollama document embeddings use the model's output dimension (768 for `nomic-embed-text`).

## Requirements

- JDK 17 or newer
- Git (to clone the repository)
- [Ollama](https://ollama.com/download) for document embedding and AI answers; demo vector search works without Ollama

Pull the models used by default:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Start Ollama if it is not already running:

```bash
ollama serve
```

## Run on Windows

```powershell
git clone https://github.com/DEEPAK-KUMAR002/Your-OWN-AI.git
cd Your-OWN-AI
.\run.bat
```

Open [http://localhost:8080](http://localhost:8080) in a browser. The batch file compiles the Java source into the ignored `out/` folder and starts the server.

## Run on macOS or Linux

```bash
git clone https://github.com/DEEPAK-KUMAR002/Your-OWN-AI.git
cd Your-OWN-AI
chmod +x run.sh
./run.sh
```

The server listens on port `8080` by default. Set `PORT` to use another port. Ollama settings can be changed with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama base URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_GENERATION_MODEL` | `llama3.2` | Text generation model |

## Use the application

### Search

Enter a concept such as `binary tree`, `sushi`, or `basketball`; select HNSW, KD-Tree, or brute force; then choose cosine, Euclidean, or Manhattan distance. The interface can run all three search strategies and compare their elapsed times. The chart projects the 16D demo vectors into 2D with PCA.

### Add documents

Enter a title and paste text in the Documents tab. The server divides longer text into overlapping 250-word chunks, requests an embedding for each chunk from Ollama, and stores the chunks in an HNSW index.

### Ask a question

Ask about the inserted documents. The server embeds the question, retrieves up to `k` similar chunks (default 3), and sends those chunks with the question to Ollama for answer generation. The response includes the retrieved chunks so the UI can show its sources.

## REST API

Base URL: `http://localhost:8080`

### Demo vector routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/search?v=...&k=5&metric=cosine&algo=hnsw` | Search a 16D query vector; `algo` accepts `hnsw`, `kdtree`, or `bruteforce` |
| `POST` | `/insert` | Add a vector: `{"metadata":"...","category":"...","embedding":[...]}` |
| `DELETE` | `/delete/{id}` | Delete a demo vector by ID |
| `GET` | `/items` | List demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare search times for all algorithms |
| `GET` | `/hnsw-info` | Get HNSW layers, nodes, and edges |
| `GET` | `/stats` | Get demo index counts, dimensions, algorithms, and metrics |

### Document and RAG routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/doc/insert` | Add and embed a document: `{"title":"...","text":"..."}` |
| `GET` | `/doc/list` | List indexed document chunks |
| `DELETE` | `/doc/delete/{id}` | Delete a document chunk by ID |
| `POST` | `/doc/search` | Retrieve similar chunks: `{"question":"...","k":3}` |
| `POST` | `/doc/ask` | Retrieve chunks and generate an answer: `{"question":"...","k":3}` |
| `GET` | `/status` | Report Ollama availability, models, and index sizes |

Example search request:

```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

Example RAG request:

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

## Project structure

```text
Your-OWN-AI/
├── src/main/java/vectordb/VectorDbServer.java  # Java server, indexes, Ollama client, JSON handling
├── index.html                                  # Browser UI
├── run.bat                                     # Windows compile and launch
├── run.sh                                      # macOS/Linux compile and launch
├── .gitignore
└── README.md
```

The Java source implements the HNSW graph index, KD-Tree, brute-force index, distance functions, in-memory demo and document stores, HTTP routes, and Ollama HTTP calls. The in-memory indexes reset when the server stops.

## Notes

- HNSW is approximate nearest-neighbor search; KD-Tree performance can degrade for high-dimensional vectors.
- Ollama is only needed for real text embeddings and generated answers. The preloaded demo search remains available when Ollama is offline.
- The built-in JSON codec and HTTP server use JDK APIs so the project can be compiled with `javac` alone.

## License

MIT (the original project README identifies its license as MIT).
