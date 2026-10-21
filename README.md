# Your-OWN-AI — Java Vector Database and Local RAG

A lightweight vector database and browser demo implemented in **Java 17+**. It includes three nearest-neighbor search strategies, a REST API, a small web UI, and document question-answering backed by Google Gemini or a reachable Ollama instance.

The backend uses only JDK modules; no Maven or third-party Java libraries are required.

## Features

| Feature | Details |
|---|---|
| Search algorithms | HNSW approximate search, KD-Tree search, and exact brute-force search |
| Distance metrics | Cosine distance, Euclidean distance, and Manhattan distance |
| Demo data | 20 fixed 16-dimensional vectors across CS, Math, Food, and Sports |
| Document embeddings | Gemini `gemini-embedding-2` (768D) or Ollama `nomic-embed-text`; documents are split into overlapping chunks |
| RAG Q&A | Retrieves relevant chunks and asks Gemini `gemini-2.5-flash` or Ollama `llama3.2` to generate an answer |
| REST API | Search, insert, delete, list, benchmark, HNSW graph info, document retrieval, and RAG routes |
| Browser UI | Search and compare algorithms, view a PCA scatter plot, add documents, and ask questions |

## How it works

```text
Document text ──> Gemini or Ollama embedding model ──> vectors ──> HNSW document index
                                                               │
Question ──> same embedding model ──> nearest chunks ──> Gemini or Ollama generation model ──> answer
```

The demo search index is separate from document embeddings: demo vectors are 16-dimensional. Gemini embeddings use 768 dimensions by default. Use the same embedding provider and model for documents and questions.

## Requirements

- JDK 17 or newer
- Git (to clone the repository)
- For hosted document RAG, a [Gemini API key](https://aistudio.google.com/apikey). Demo vector search works without a provider.
- For local inference, [Ollama](https://ollama.com/download) and the models below

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

## Deploy on Render

The included `Dockerfile` and `render.yaml` configure a single Render web service that serves both the frontend and Java API from the same origin. Set `GEMINI_API_KEY` in the service's private Environment settings using a key from [Google AI Studio](https://aistudio.google.com/apikey), then deploy. The key stays on the server and is never sent to the browser. Check the provider status in the Documents tab. Review Google's current API limits and pricing before use.

The server listens on port `8080` by default. Set `PORT` to use another port. Configure the provider with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | unset | Selects Gemini for document embeddings and answer generation |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-2` | Gemini embedding model; outputs 768D vectors |
| `GEMINI_GENERATION_MODEL` | `gemini-2.5-flash` | Gemini text generation model |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama base URL used when no Gemini key is configured |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_GENERATION_MODEL` | `llama3.2` | Ollama text generation model |

## Use the application

### Search

Enter a concept such as `binary tree`, `sushi`, or `basketball`; select HNSW, KD-Tree, or brute force; then choose cosine, Euclidean, or Manhattan distance. The interface can run all three search strategies and compare their elapsed times. The chart projects the 16D demo vectors into 2D with PCA.

### Add documents

Enter a title and paste text in the Documents tab. The server divides longer text into overlapping 250-word chunks, requests an embedding for each chunk from the configured provider, and stores the chunks in an HNSW index.

### Ask a question

Ask about the inserted documents. The server embeds the question, retrieves up to `k` similar chunks (default 3), and sends those chunks with the question to the configured provider for answer generation. The response includes the retrieved chunks so the UI can show its sources.

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
| `GET` | `/status` | Report provider availability, model names, and index sizes |

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
├── src/main/java/vectordb/VectorDbServer.java  # Java server, indexes, Gemini/Ollama clients, JSON handling
├── index.html                                  # Browser UI
├── run.bat                                     # Windows compile and launch
├── run.sh                                      # macOS/Linux compile and launch
├── .gitignore
└── README.md
```

The Java source implements the HNSW graph index, KD-Tree, brute-force index, distance functions, in-memory demo and document stores, HTTP routes, and Gemini/Ollama HTTP calls. The in-memory indexes reset when the server stops.

## Notes

- HNSW is approximate nearest-neighbor search; KD-Tree performance can degrade for high-dimensional vectors.
- Configure `GEMINI_API_KEY` for hosted embeddings and answers, or use a reachable Ollama instance. The preloaded demo search remains available when no provider is configured.
- The built-in JSON codec and HTTP server use JDK APIs so the project can be compiled with `javac` alone.

## License

MIT (the original project README identifies its license as MIT).
