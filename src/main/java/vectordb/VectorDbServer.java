package vectordb;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Java implementation of the VectorDB HTTP server and its in-memory indexes. */
public final class VectorDbServer {
    private static final int DEMO_DIMS = 16;
    private static final VectorStore DEMO_DB = new VectorStore(DEMO_DIMS);
    private static final DocumentStore DOCUMENT_DB = new DocumentStore();
    private static final OllamaClient OLLAMA = new OllamaClient();

    private VectorDbServer() { }

    public static void main(String[] args) throws IOException {
        loadDemoVectors();
        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "8080"));
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", port), 0);
        server.createContext("/", VectorDbServer::handle);
        server.setExecutor(Executors.newFixedThreadPool(Math.max(4, Runtime.getRuntime().availableProcessors())));
        server.start();

        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:" + port);
        System.out.println(DEMO_DB.size() + " demo vectors | " + DEMO_DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (OLLAMA.isAvailable() ? "ONLINE" : "OFFLINE (optional; see README)"));
        System.out.println("Press Ctrl+C to stop.");
    }

    private static void handle(HttpExchange exchange) throws IOException {
        String method = exchange.getRequestMethod();
        String path = exchange.getRequestURI().getPath();
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        if (method.equalsIgnoreCase("OPTIONS")) {
            exchange.sendResponseHeaders(204, -1);
            return;
        }

        try {
            Object result;
            int status = 200;
            Map<String, String> query = queryParams(exchange.getRequestURI());
            if (method.equals("GET") && path.equals("/")) {
                serveIndex(exchange);
                return;
            } else if (method.equals("GET") && path.equals("/search")) {
                List<Double> vector = parseVector(query.getOrDefault("v", ""));
                if (vector.size() != DEMO_DIMS) {
                    status = 400;
                    result = Map.of("error", "need " + DEMO_DIMS + "D vector");
                } else {
                    int k = positiveInt(query.get("k"), 5);
                    String metric = query.getOrDefault("metric", "cosine");
                    String algo = query.getOrDefault("algo", "hnsw");
                    SearchResult search = DEMO_DB.search(vector, k, metric, algo);
                    List<Object> hits = new ArrayList<>();
                    for (Hit hit : search.hits) hits.add(hit.toMap());
                    result = Map.of("results", hits, "latencyUs", search.latencyUs,
                            "algo", search.algo, "metric", search.metric);
                }
            } else if (method.equals("POST") && path.equals("/insert")) {
                Map<String, Object> body = readJsonBody(exchange);
                String metadata = stringField(body, "metadata");
                String category = stringField(body, "category");
                List<Double> embedding = numberList(body.get("embedding"));
                if (metadata.isBlank() || embedding.size() != DEMO_DIMS) {
                    status = 400;
                    result = Map.of("error", "invalid body: metadata and a 16D embedding are required");
                } else {
                    result = Map.of("id", DEMO_DB.insert(metadata, category, embedding));
                }
            } else if (method.equals("DELETE") && path.matches("/delete/\\d+")) {
                int id = Integer.parseInt(path.substring(path.lastIndexOf('/') + 1));
                result = Map.of("ok", DEMO_DB.remove(id));
            } else if (method.equals("GET") && path.equals("/items")) {
                result = DEMO_DB.all().stream().map(VectorItem::toMap).toList();
            } else if (method.equals("GET") && path.equals("/benchmark")) {
                List<Double> vector = parseVector(query.getOrDefault("v", ""));
                if (vector.size() != DEMO_DIMS) {
                    status = 400;
                    result = Map.of("error", "need " + DEMO_DIMS + "D vector");
                } else {
                    result = DEMO_DB.benchmark(vector, positiveInt(query.get("k"), 5),
                            query.getOrDefault("metric", "cosine"));
                }
            } else if (method.equals("GET") && path.equals("/hnsw-info")) {
                result = DEMO_DB.hnswInfo();
            } else if (method.equals("POST") && path.equals("/doc/insert")) {
                result = insertDocument(readJsonBody(exchange));
            } else if (method.equals("DELETE") && path.matches("/doc/delete/\\d+")) {
                int id = Integer.parseInt(path.substring(path.lastIndexOf('/') + 1));
                result = Map.of("ok", DOCUMENT_DB.remove(id));
            } else if (method.equals("GET") && path.equals("/doc/list")) {
                result = DOCUMENT_DB.listSummaries();
            } else if (method.equals("POST") && path.equals("/doc/search")) {
                result = searchDocuments(readJsonBody(exchange), false);
            } else if (method.equals("POST") && path.equals("/doc/ask")) {
                result = searchDocuments(readJsonBody(exchange), true);
            } else if (method.equals("GET") && path.equals("/status")) {
                result = Map.of("ollamaAvailable", OLLAMA.isAvailable(), "embedModel", OLLAMA.embedModel,
                        "genModel", OLLAMA.genModel, "docCount", DOCUMENT_DB.size(),
                        "docDims", DOCUMENT_DB.dimensions(), "demoDims", DEMO_DIMS,
                        "demoCount", DEMO_DB.size());
            } else if (method.equals("GET") && path.equals("/stats")) {
                result = Map.of("count", DEMO_DB.size(), "dims", DEMO_DIMS,
                        "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
                        "metrics", List.of("euclidean", "cosine", "manhattan"));
            } else {
                status = 404;
                result = Map.of("error", "not found");
            }
            sendJson(exchange, status, result);
        } catch (IllegalArgumentException ex) {
            sendJson(exchange, 400, Map.of("error", safeMessage(ex)));
        } catch (Exception ex) {
            ex.printStackTrace(System.err);
            sendJson(exchange, 500, Map.of("error", safeMessage(ex)));
        }
    }

    private static Object insertDocument(Map<String, Object> body) {
        String title = stringField(body, "title");
        String text = stringField(body, "text");
        if (title.isBlank() || text.isBlank()) throw new IllegalArgumentException("need title and text");
        List<String> chunks = chunkText(text, 250, 30);
        List<List<Double>> embeddings = new ArrayList<>();
        for (String chunk : chunks) {
            List<Double> embedding = OLLAMA.embed(chunk);
            if (embedding.isEmpty()) throw new IllegalArgumentException("Ollama unavailable or embedding failed. Install Ollama, run ollama serve, then pull nomic-embed-text.");
            embeddings.add(embedding);
        }
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            String chunkTitle = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
            ids.add(DOCUMENT_DB.insert(chunkTitle, chunks.get(i), embeddings.get(i)));
        }
        return Map.of("ids", ids, "chunks", chunks.size(), "dims", DOCUMENT_DB.dimensions());
    }

    private static Object searchDocuments(Map<String, Object> body, boolean generate) {
        String question = stringField(body, "question");
        int k = positiveInt(asString(body.get("k")), 3);
        if (question.isBlank()) throw new IllegalArgumentException("need question");
        List<Double> query = OLLAMA.embed(question);
        if (query.isEmpty()) throw new IllegalArgumentException("Ollama unavailable or embedding failed");
        List<DocHit> hits = DOCUMENT_DB.search(query, k);
        List<Object> contexts = new ArrayList<>();
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            DocHit hit = hits.get(i);
            if (generate) {
                contexts.add(Map.of("id", hit.item.id, "title", hit.item.title, "text", hit.item.text,
                        "distance", hit.distance));
                context.append('[').append(i + 1).append("] ").append(hit.item.title).append(":\n")
                        .append(hit.item.text).append("\n\n");
            } else {
                contexts.add(Map.of("id", hit.item.id, "title", hit.item.title, "distance", hit.distance));
            }
        }
        if (!generate) return Map.of("contexts", contexts);
        String prompt = "You are a helpful assistant. Answer the user's question directly. "
                + "Use the provided context if it contains relevant information. "
                + "If it doesn't, use your general knowledge. Do not mention the context unless useful.\n\n"
                + "Context:\n" + context + "Question: " + question + "\n\nAnswer:";
        String answer = OLLAMA.generate(prompt);
        return Map.of("answer", answer, "model", OLLAMA.genModel, "contexts", contexts,
                "docCount", DOCUMENT_DB.size());
    }

    private static void serveIndex(HttpExchange exchange) throws IOException {
        Path index = Path.of("index.html");
        if (!Files.isRegularFile(index)) {
            sendJson(exchange, 404, Map.of("error", "index.html not found; start the server from the project root"));
            return;
        }
        byte[] bytes = Files.readAllBytes(index);
        exchange.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
        exchange.sendResponseHeaders(200, bytes.length);
        try (var out = exchange.getResponseBody()) { out.write(bytes); }
    }

    private static Map<String, String> queryParams(URI uri) {
        Map<String, String> result = new HashMap<>();
        String raw = uri.getRawQuery();
        if (raw == null || raw.isEmpty()) return result;
        for (String pair : raw.split("&")) {
            String[] parts = pair.split("=", 2);
            result.put(URLDecoder.decode(parts[0], StandardCharsets.UTF_8),
                    URLDecoder.decode(parts.length == 2 ? parts[1] : "", StandardCharsets.UTF_8));
        }
        return result;
    }

    private static Map<String, Object> readJsonBody(HttpExchange exchange) throws IOException {
        byte[] bytes = exchange.getRequestBody().readNBytes(5 * 1024 * 1024 + 1);
        if (bytes.length > 5 * 1024 * 1024) throw new IllegalArgumentException("request body too large (5 MB limit)");
        Object value = Json.parse(new String(bytes, StandardCharsets.UTF_8));
        if (!(value instanceof Map<?, ?> map)) throw new IllegalArgumentException("JSON object expected");
        @SuppressWarnings("unchecked") Map<String, Object> body = (Map<String, Object>) map;
        return body;
    }

    private static void sendJson(HttpExchange exchange, int status, Object value) throws IOException {
        byte[] body = Json.stringify(value).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, body.length);
        try (var out = exchange.getResponseBody()) { out.write(body); }
    }

    private static String safeMessage(Exception ex) {
        return ex.getMessage() == null || ex.getMessage().isBlank() ? "request failed" : ex.getMessage();
    }

    private static int positiveInt(String raw, int fallback) {
        try { return Math.max(1, Math.min(1000, Integer.parseInt(raw))); }
        catch (RuntimeException ignored) { return fallback; }
    }

    private static String stringField(Map<String, Object> map, String key) {
        Object value = map.get(key);
        return value instanceof String ? (String) value : "";
    }

    private static String asString(Object value) { return value == null ? null : String.valueOf(value); }

    private static List<Double> numberList(Object value) {
        if (!(value instanceof List<?> list)) return List.of();
        List<Double> result = new ArrayList<>();
        for (Object item : list) {
            if (!(item instanceof Number number) || !Double.isFinite(number.doubleValue())) return List.of();
            result.add(number.doubleValue());
        }
        return result;
    }

    private static List<Double> parseVector(String raw) {
        if (raw.isBlank()) return List.of();
        try {
            List<Double> result = new ArrayList<>();
            for (String item : raw.split(",")) {
                double value = Double.parseDouble(item.trim());
                if (!Double.isFinite(value)) return List.of();
                result.add(value);
            }
            return result;
        } catch (NumberFormatException ignored) { return List.of(); }
    }

    private static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String trimmed = text.trim();
        if (trimmed.isEmpty()) return List.of();
        String[] words = trimmed.split("\\s+");
        if (words.length <= chunkWords) return List.of(trimmed);
        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int start = 0; start < words.length; start += step) {
            int end = Math.min(words.length, start + chunkWords);
            chunks.add(String.join(" ", Arrays.copyOfRange(words, start, end)));
            if (end == words.length) break;
        }
        return chunks;
    }

    private static void loadDemoVectors() {
        // The 16 dimensions are grouped by CS, Math, Food and Sports semantics.
        DEMO_DB.insert("Linked List: nodes connected by pointers", "cs", vec("0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06"));
        DEMO_DB.insert("Binary Search Tree: O(log n) search and insert", "cs", vec("0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10"));
        DEMO_DB.insert("Dynamic Programming: memoization overlapping subproblems", "cs", vec("0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07"));
        DEMO_DB.insert("Graph BFS and DFS: breadth and depth first traversal", "cs", vec("0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07"));
        DEMO_DB.insert("Hash Table: O(1) lookup with collision chaining", "cs", vec("0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09"));
        DEMO_DB.insert("Calculus: derivatives integrals and limits", "math", vec("0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10"));
        DEMO_DB.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math", vec("0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09"));
        DEMO_DB.insert("Probability: distributions random variables Bayes theorem", "math", vec("0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08"));
        DEMO_DB.insert("Number Theory: primes modular arithmetic RSA cryptography", "math", vec("0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06"));
        DEMO_DB.insert("Combinatorics: permutations combinations generating functions", "math", vec("0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07"));
        DEMO_DB.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", vec("0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07"));
        DEMO_DB.insert("Sushi: vinegared rice raw fish and nori rolls", "food", vec("0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08"));
        DEMO_DB.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", vec("0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06"));
        DEMO_DB.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food", vec("0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09"));
        DEMO_DB.insert("Croissant: laminated pastry with buttery flaky layers", "food", vec("0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06"));
        DEMO_DB.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports", vec("0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72"));
        DEMO_DB.insert("Football: tackles touchdowns field goals and strategy", "sports", vec("0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76"));
        DEMO_DB.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", vec("0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82"));
        DEMO_DB.insert("Chess: openings endgames tactics strategic board game", "sports", vec("0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90"));
        DEMO_DB.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports", vec("0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80"));
    }

    private static List<Double> vec(String csv) { return parseVector(csv); }

    private record VectorItem(int id, String metadata, String category, List<Double> embedding) {
        Map<String, Object> toMap() {
            return Map.of("id", id, "metadata", metadata, "category", category, "embedding", embedding);
        }
    }

    private record Hit(int id, String metadata, String category, double distance, List<Double> embedding) {
        Map<String, Object> toMap() {
            return Map.of("id", id, "metadata", metadata, "category", category,
                    "distance", distance, "embedding", embedding);
        }
    }

    private record SearchResult(List<Hit> hits, long latencyUs, String algo, String metric) { }
    private record Pair(double distance, int id) { }

    private interface Distance { double between(List<Double> a, List<Double> b); }

    private static Distance distanceFor(String metric) {
        return switch (metric.toLowerCase(Locale.ROOT)) {
            case "cosine" -> VectorDbServer::cosineDistance;
            case "manhattan" -> VectorDbServer::manhattanDistance;
            default -> VectorDbServer::euclideanDistance;
        };
    }

    private static double euclideanDistance(List<Double> a, List<Double> b) {
        requireSameDimensions(a, b);
        double sum = 0;
        for (int i = 0; i < a.size(); i++) { double d = a.get(i) - b.get(i); sum += d * d; }
        return Math.sqrt(sum);
    }

    private static double manhattanDistance(List<Double> a, List<Double> b) {
        requireSameDimensions(a, b);
        double sum = 0;
        for (int i = 0; i < a.size(); i++) sum += Math.abs(a.get(i) - b.get(i));
        return sum;
    }

    private static double cosineDistance(List<Double> a, List<Double> b) {
        requireSameDimensions(a, b);
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.size(); i++) {
            dot += a.get(i) * b.get(i); normA += a.get(i) * a.get(i); normB += b.get(i) * b.get(i);
        }
        if (normA < 1e-9 || normB < 1e-9) return 1;
        return 1 - dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private static void requireSameDimensions(List<Double> a, List<Double> b) {
        if (a.size() != b.size()) throw new IllegalArgumentException("vector dimensions do not match");
    }

    private static final class BruteForceIndex {
        private final Map<Integer, VectorItem> items = new LinkedHashMap<>();
        void insert(VectorItem item) { items.put(item.id, item); }
        void remove(int id) { items.remove(id); }
        List<Pair> search(List<Double> query, int k, Distance distance) {
            List<Pair> result = new ArrayList<>();
            for (VectorItem item : items.values()) result.add(new Pair(distance.between(query, item.embedding), item.id));
            result.sort(Comparator.comparingDouble(Pair::distance).thenComparingInt(Pair::id));
            return result.subList(0, Math.min(Math.max(k, 0), result.size()));
        }
        int size() { return items.size(); }
    }

    private static final class KdTreeIndex {
        private static final class Node { VectorItem item; Node left, right; Node(VectorItem item) { this.item = item; } }
        private Node root;
        private final int dimensions;
        private final Map<Integer, VectorItem> items = new LinkedHashMap<>();
        KdTreeIndex(int dimensions) { this.dimensions = dimensions; }
        void insert(VectorItem item) { items.put(item.id, item); root = insert(root, item, 0); }
        private Node insert(Node node, VectorItem item, int depth) {
            if (node == null) return new Node(item);
            int axis = depth % dimensions;
            if (item.embedding.get(axis) < node.item.embedding.get(axis)) node.left = insert(node.left, item, depth + 1);
            else node.right = insert(node.right, item, depth + 1);
            return node;
        }
        void remove(int id) { if (items.remove(id) != null) rebuild(); }
        private void rebuild() { root = null; for (VectorItem item : items.values()) root = insert(root, item, 0); }
        List<Pair> search(List<Double> query, int k, Distance distance) {
            if (k <= 0) return List.of();
            PriorityQueue<Pair> heap = new PriorityQueue<>(Comparator.comparingDouble(Pair::distance)
                    .thenComparingInt(Pair::id).reversed());
            search(root, query, k, 0, distance, heap);
            List<Pair> result = new ArrayList<>(heap);
            result.sort(Comparator.comparingDouble(Pair::distance).thenComparingInt(Pair::id));
            return result;
        }
        private void search(Node node, List<Double> query, int k, int depth, Distance distance, PriorityQueue<Pair> heap) {
            if (node == null) return;
            double d = distance.between(query, node.item.embedding);
            if (heap.size() < k || d < heap.peek().distance) {
                heap.add(new Pair(d, node.item.id)); if (heap.size() > k) heap.poll();
            }
            int axis = depth % dimensions;
            double delta = query.get(axis) - node.item.embedding.get(axis);
            Node near = delta < 0 ? node.left : node.right, far = delta < 0 ? node.right : node.left;
            search(near, query, k, depth + 1, distance, heap);
            if (heap.size() < k || Math.abs(delta) < heap.peek().distance) search(far, query, k, depth + 1, distance, heap);
        }
    }

    private static final class HnswIndex {
        private static final class Node { final VectorItem item; final int maxLevel; final List<List<Integer>> neighbors;
            Node(VectorItem item, int maxLevel) { this.item = item; this.maxLevel = maxLevel; this.neighbors = new ArrayList<>();
                for (int i = 0; i <= maxLevel; i++) neighbors.add(new ArrayList<>()); }
        }
        private record Edge(int src, int dst, int layer) { }
        private record GraphInfo(int topLayer, int nodeCount, List<Integer> nodesPerLayer,
                                 List<Integer> edgesPerLayer, List<Object> nodes, List<Object> edges) {
            Map<String, Object> toMap() { return Map.of("topLayer", topLayer, "nodeCount", nodeCount,
                    "nodesPerLayer", nodesPerLayer, "edgesPerLayer", edgesPerLayer, "nodes", nodes, "edges", edges); }
        }
        private final Map<Integer, Node> graph = new HashMap<>();
        private final int m, m0, efBuild;
        private final double mL;
        private final Random random = new Random(42);
        private int topLayer = -1, entryPoint = -1;
        HnswIndex(int m, int efBuild) { this.m = m; this.m0 = 2 * m; this.efBuild = efBuild; this.mL = 1 / Math.log(m); }
        int size() { return graph.size(); }
        private int randomLevel() { return (int) Math.floor(-Math.log(Math.max(random.nextDouble(), 1e-12)) * mL); }
        void insert(VectorItem item, Distance distance) {
            int level = randomLevel();
            Node added = new Node(item, level);
            graph.put(item.id, added);
            if (entryPoint == -1) { entryPoint = item.id; topLayer = level; return; }
            int ep = entryPoint;
            for (int layer = topLayer; layer > level; layer--) {
                if (layer < graph.get(ep).neighbors.size()) {
                    List<Pair> nearest = searchLayer(item.embedding, ep, 1, layer, distance);
                    if (!nearest.isEmpty()) ep = nearest.get(0).id;
                }
            }
            for (int layer = Math.min(topLayer, level); layer >= 0; layer--) {
                List<Pair> candidates = searchLayer(item.embedding, ep, efBuild, layer, distance);
                int maxM = layer == 0 ? m0 : m;
                List<Integer> selected = new ArrayList<>();
                for (int i = 0; i < Math.min(candidates.size(), maxM); i++) selected.add(candidates.get(i).id);
                added.neighbors.get(layer).addAll(selected);
                for (int neighborId : selected) {
                    Node neighbor = graph.get(neighborId);
                    if (neighbor == null) continue;
                    while (neighbor.neighbors.size() <= layer) neighbor.neighbors.add(new ArrayList<>());
                    List<Integer> connections = neighbor.neighbors.get(layer);
                    connections.add(item.id);
                    if (connections.size() > maxM) {
                        connections.sort(Comparator.comparingDouble(id -> distance.between(neighbor.item.embedding, graph.get(id).item.embedding)));
                        while (connections.size() > maxM) connections.remove(connections.size() - 1);
                    }
                }
                if (!candidates.isEmpty()) ep = candidates.get(0).id;
            }
            if (level > topLayer) { topLayer = level; entryPoint = item.id; }
        }
        List<Pair> search(List<Double> query, int k, int ef, Distance distance) {
            if (entryPoint == -1 || k <= 0) return List.of();
            int ep = entryPoint;
            for (int layer = topLayer; layer > 0; layer--) {
                if (layer < graph.get(ep).neighbors.size()) {
                    List<Pair> nearest = searchLayer(query, ep, 1, layer, distance);
                    if (!nearest.isEmpty()) ep = nearest.get(0).id;
                }
            }
            List<Pair> found = searchLayer(query, ep, Math.max(ef, k), 0, distance);
            return found.subList(0, Math.min(k, found.size()));
        }
        private List<Pair> searchLayer(List<Double> query, int ep, int ef, int layer, Distance distance) {
            Comparator<Pair> ascending = Comparator.comparingDouble(Pair::distance).thenComparingInt(Pair::id);
            PriorityQueue<Pair> candidates = new PriorityQueue<>(ascending);
            PriorityQueue<Pair> found = new PriorityQueue<>(ascending.reversed());
            Set<Integer> visited = new HashSet<>();
            double startDistance = distance.between(query, graph.get(ep).item.embedding);
            Pair start = new Pair(startDistance, ep); candidates.add(start); found.add(start); visited.add(ep);
            while (!candidates.isEmpty()) {
                Pair candidate = candidates.poll();
                if (found.size() >= ef && candidate.distance > found.peek().distance) break;
                Node node = graph.get(candidate.id);
                if (node == null || layer >= node.neighbors.size()) continue;
                for (int nextId : node.neighbors.get(layer)) {
                    if (!visited.add(nextId)) continue;
                    Node next = graph.get(nextId); if (next == null) continue;
                    Pair nextPair = new Pair(distance.between(query, next.item.embedding), nextId);
                    if (found.size() < ef || ascending.compare(nextPair, found.peek()) < 0) {
                        candidates.add(nextPair); found.add(nextPair); if (found.size() > ef) found.poll();
                    }
                }
            }
            List<Pair> result = new ArrayList<>(found); result.sort(ascending); return result;
        }
        void remove(int id) {
            Node removed = graph.remove(id); if (removed == null) return;
            for (Node node : graph.values()) for (List<Integer> layer : node.neighbors) layer.removeIf(value -> value == id);
            if (entryPoint == id) {
                entryPoint = -1; topLayer = -1;
                for (Map.Entry<Integer, Node> entry : graph.entrySet()) if (entry.getValue().maxLevel > topLayer) {
                    topLayer = entry.getValue().maxLevel; entryPoint = entry.getKey();
                }
            }
        }
        GraphInfo info() {
            int layers = Math.max(topLayer + 1, 1);
            List<Integer> nodesPerLayer = new ArrayList<>(java.util.Collections.nCopies(layers, 0));
            List<Integer> edgesPerLayer = new ArrayList<>(java.util.Collections.nCopies(layers, 0));
            List<Object> nodes = new ArrayList<>(), edges = new ArrayList<>();
            for (Map.Entry<Integer, Node> entry : graph.entrySet()) {
                int id = entry.getKey(); Node node = entry.getValue();
                nodes.add(Map.of("id", id, "metadata", node.item.metadata, "category", node.item.category, "maxLyr", node.maxLevel));
                for (int layer = 0; layer <= node.maxLevel && layer < layers; layer++) {
                    nodesPerLayer.set(layer, nodesPerLayer.get(layer) + 1);
                    if (layer < node.neighbors.size()) for (int neighbor : node.neighbors.get(layer)) if (id < neighbor) {
                        edgesPerLayer.set(layer, edgesPerLayer.get(layer) + 1);
                        edges.add(Map.of("src", id, "dst", neighbor, "lyr", layer));
                    }
                }
            }
            return new GraphInfo(topLayer, graph.size(), nodesPerLayer, edgesPerLayer, nodes, edges);
        }
    }

    private static final class VectorStore {
        private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
        private final BruteForceIndex brute = new BruteForceIndex();
        private final KdTreeIndex kdTree;
        private final HnswIndex hnsw = new HnswIndex(16, 200);
        private final int dimensions;
        private int nextId = 1;
        VectorStore(int dimensions) { this.dimensions = dimensions; this.kdTree = new KdTreeIndex(dimensions); }
        synchronized int insert(String metadata, String category, List<Double> embedding) {
            if (embedding.size() != dimensions) throw new IllegalArgumentException("embedding must have " + dimensions + " dimensions");
            int id = nextId++;
            VectorItem item = new VectorItem(id, metadata, category, List.copyOf(embedding));
            store.put(id, item); brute.insert(item); kdTree.insert(item); hnsw.insert(item, VectorDbServer::cosineDistance);
            return id;
        }
        synchronized boolean remove(int id) {
            if (store.remove(id) == null) return false;
            brute.remove(id); kdTree.remove(id); hnsw.remove(id); return true;
        }
        synchronized List<VectorItem> all() { return new ArrayList<>(store.values()); }
        synchronized int size() { return store.size(); }
        synchronized SearchResult search(List<Double> query, int k, String metric, String algo) {
            if (query.size() != dimensions) throw new IllegalArgumentException("query must have " + dimensions + " dimensions");
            Distance distance = distanceFor(metric);
            long start = System.nanoTime();
            List<Pair> raw = switch (algo.toLowerCase(Locale.ROOT)) {
                case "bruteforce", "brute-force" -> brute.search(query, k, distance);
                case "kdtree", "kd-tree" -> kdTree.search(query, k, distance);
                default -> hnsw.search(query, k, 50, distance);
            };
            long micros = (System.nanoTime() - start) / 1_000;
            List<Hit> hits = new ArrayList<>();
            for (Pair pair : raw) {
                VectorItem item = store.get(pair.id);
                if (item != null) hits.add(new Hit(item.id, item.metadata, item.category, pair.distance, item.embedding));
            }
            return new SearchResult(hits, micros, algo, metric);
        }
        synchronized Map<String, Object> benchmark(List<Double> query, int k, String metric) {
            if (query.size() != dimensions) throw new IllegalArgumentException("query must have " + dimensions + " dimensions");
            Distance distance = distanceFor(metric);
            long start = System.nanoTime(); brute.search(query, k, distance); long bruteUs = (System.nanoTime() - start) / 1_000;
            start = System.nanoTime(); kdTree.search(query, k, distance); long kdUs = (System.nanoTime() - start) / 1_000;
            start = System.nanoTime(); hnsw.search(query, k, 50, distance); long hnswUs = (System.nanoTime() - start) / 1_000;
            return Map.of("bruteforceUs", bruteUs, "kdtreeUs", kdUs, "hnswUs", hnswUs, "itemCount", store.size());
        }
        synchronized Map<String, Object> hnswInfo() { return hnsw.info().toMap(); }
    }

    private record DocItem(int id, String title, String text, List<Double> embedding) { }
    private record DocHit(double distance, DocItem item) { }

    private static final class DocumentStore {
        private final Map<Integer, DocItem> store = new LinkedHashMap<>();
        private final HnswIndex hnsw = new HnswIndex(16, 200);
        private final BruteForceIndex brute = new BruteForceIndex();
        private int nextId = 1;
        private int dimensions;
        synchronized int insert(String title, String text, List<Double> embedding) {
            if (embedding.isEmpty()) throw new IllegalArgumentException("embedding must not be empty");
            if (dimensions == 0) dimensions = embedding.size();
            if (embedding.size() != dimensions) throw new IllegalArgumentException("embedding dimension mismatch");
            int id = nextId++;
            List<Double> copy = List.copyOf(embedding);
            DocItem item = new DocItem(id, title, text, copy); store.put(id, item);
            VectorItem vector = new VectorItem(id, title, "doc", copy);
            hnsw.insert(vector, VectorDbServer::cosineDistance); brute.insert(vector);
            return id;
        }
        synchronized List<DocHit> search(List<Double> query, int k) {
            if (store.isEmpty()) return List.of();
            if (query.size() != dimensions) throw new IllegalArgumentException("query embedding dimension mismatch");
            List<Pair> raw = store.size() < 10 ? brute.search(query, k, VectorDbServer::cosineDistance)
                    : hnsw.search(query, k, 50, VectorDbServer::cosineDistance);
            List<DocHit> result = new ArrayList<>();
            for (Pair pair : raw) {
                DocItem item = store.get(pair.id);
                if (item != null && pair.distance <= 0.7) result.add(new DocHit(pair.distance, item));
            }
            return result;
        }
        synchronized boolean remove(int id) {
            if (store.remove(id) == null) return false;
            hnsw.remove(id); brute.remove(id); return true;
        }
        synchronized int size() { return store.size(); }
        synchronized int dimensions() { return dimensions; }
        synchronized List<Object> listSummaries() {
            List<Object> result = new ArrayList<>();
            for (DocItem item : store.values()) {
                String preview = item.text.length() > 120 ? item.text.substring(0, 120) + "…" : item.text;
                int words = item.text.isBlank() ? 0 : item.text.trim().split("\\s+").length;
                result.add(Map.of("id", item.id, "title", item.title, "preview", preview, "words", words));
            }
            return result;
        }
    }

    private static final class OllamaClient {
        private final String host = System.getenv().getOrDefault("OLLAMA_HOST", "http://127.0.0.1:11434").replaceAll("/$", "");
        private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
        private final String embedModel = System.getenv().getOrDefault("OLLAMA_EMBED_MODEL", "nomic-embed-text");
        private final String genModel = System.getenv().getOrDefault("OLLAMA_GENERATION_MODEL", "llama3.2");
        boolean isAvailable() {
            try {
                HttpRequest request = HttpRequest.newBuilder(URI.create(host + "/api/tags")).timeout(Duration.ofSeconds(4)).GET().build();
                return client.send(request, HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
            } catch (Exception ignored) { return false; }
        }
        List<Double> embed(String text) {
            try {
                String json = Json.stringify(Map.of("model", embedModel, "prompt", text));
                HttpRequest request = HttpRequest.newBuilder(URI.create(host + "/api/embeddings"))
                        .timeout(Duration.ofSeconds(90)).header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(json)).build();
                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() != 200) return List.of();
                Object parsed = Json.parse(response.body());
                if (!(parsed instanceof Map<?, ?> map)) return List.of();
                return numberList(map.get("embedding"));
            } catch (Exception ignored) { return List.of(); }
        }
        String generate(String prompt) {
            try {
                String json = Json.stringify(Map.of("model", genModel, "prompt", prompt, "stream", false));
                HttpRequest request = HttpRequest.newBuilder(URI.create(host + "/api/generate"))
                        .timeout(Duration.ofSeconds(180)).header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(json)).build();
                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() != 200) return "ERROR: Ollama returned HTTP " + response.statusCode();
                Object parsed = Json.parse(response.body());
                if (parsed instanceof Map<?, ?> map && map.get("response") instanceof String answer) return answer;
                return "ERROR: Ollama response did not include an answer";
            } catch (Exception ex) { return "ERROR: Ollama unavailable. Run: ollama serve"; }
        }
    }

    /** Small JSON reader/writer so the project builds with only a JDK. */
    private static final class Json {
        private Json() { }
        static Object parse(String text) { return new Parser(text).parse(); }
        static String stringify(Object value) { StringBuilder out = new StringBuilder(); write(value, out); return out.toString(); }
        private static void write(Object value, StringBuilder out) {
            if (value == null) { out.append("null"); return; }
            if (value instanceof String s) { quote(s, out); return; }
            if (value instanceof Number || value instanceof Boolean) { out.append(value); return; }
            if (value instanceof Map<?, ?> map) {
                out.append('{'); boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) out.append(','); first = false; quote(String.valueOf(entry.getKey()), out); out.append(':'); write(entry.getValue(), out);
                }
                out.append('}'); return;
            }
            if (value instanceof Collection<?> collection) {
                out.append('['); boolean first = true;
                for (Object item : collection) { if (!first) out.append(','); first = false; write(item, out); }
                out.append(']'); return;
            }
            throw new IllegalArgumentException("unsupported JSON value: " + value.getClass().getName());
        }
        private static void quote(String value, StringBuilder out) {
            out.append('"');
            for (int i = 0; i < value.length(); i++) {
                char ch = value.charAt(i);
                switch (ch) {
                    case '"' -> out.append("\\\""); case '\\' -> out.append("\\\\");
                    case '\b' -> out.append("\\b"); case '\f' -> out.append("\\f");
                    case '\n' -> out.append("\\n"); case '\r' -> out.append("\\r"); case '\t' -> out.append("\\t");
                    default -> { if (ch < 0x20) out.append(String.format("\\u%04x", (int) ch)); else out.append(ch); }
                }
            }
            out.append('"');
        }
        private static final class Parser {
            private final String source; private int at;
            Parser(String source) { this.source = source; }
            Object parse() { skip(); Object value = value(); skip(); if (at != source.length()) fail("trailing content"); return value; }
            private Object value() {
                skip(); if (at >= source.length()) return fail("unexpected end of JSON");
                return switch (source.charAt(at)) {
                    case '{' -> object(); case '[' -> array(); case '"' -> string();
                    case 't' -> literal("true", Boolean.TRUE); case 'f' -> literal("false", Boolean.FALSE); case 'n' -> literal("null", null);
                    default -> number();
                };
            }
            private Map<String, Object> object() {
                at++; Map<String, Object> result = new LinkedHashMap<>(); skip(); if (take('}')) return result;
                while (true) { skip(); if (at >= source.length() || source.charAt(at) != '"') fail("expected object key");
                    String key = string(); skip(); expect(':'); result.put(key, value()); skip(); if (take('}')) return result; expect(','); }
            }
            private List<Object> array() {
                at++; List<Object> result = new ArrayList<>(); skip(); if (take(']')) return result;
                while (true) { result.add(value()); skip(); if (take(']')) return result; expect(','); }
            }
            private String string() {
                expect('"'); StringBuilder out = new StringBuilder();
                while (at < source.length()) { char ch = source.charAt(at++); if (ch == '"') return out.toString();
                    if (ch == '\\') { if (at >= source.length()) fail("invalid escape"); char esc = source.charAt(at++);
                        switch (esc) { case '"', '\\', '/' -> out.append(esc); case 'b' -> out.append('\b'); case 'f' -> out.append('\f'); case 'n' -> out.append('\n'); case 'r' -> out.append('\r'); case 't' -> out.append('\t');
                            case 'u' -> { if (at + 4 > source.length()) fail("invalid unicode escape"); out.append((char) Integer.parseInt(source.substring(at, at + 4), 16)); at += 4; }
                            default -> fail("invalid escape sequence"); }
                    } else { if (ch < 0x20) fail("control character in string"); out.append(ch); }
                }
                return fail("unterminated string");
            }
            private Object number() {
                int start = at; if (take('-')) { }
                while (at < source.length() && Character.isDigit(source.charAt(at))) at++;
                if (take('.')) while (at < source.length() && Character.isDigit(source.charAt(at))) at++;
                if (at < source.length() && (source.charAt(at) == 'e' || source.charAt(at) == 'E')) { at++; if (at < source.length() && (source.charAt(at) == '+' || source.charAt(at) == '-')) at++; while (at < source.length() && Character.isDigit(source.charAt(at))) at++; }
                try { return Double.parseDouble(source.substring(start, at)); } catch (NumberFormatException ex) { return fail("invalid number"); }
            }
            private Object literal(String word, Object value) { if (!source.startsWith(word, at)) return fail("invalid value"); at += word.length(); return value; }
            private void skip() { while (at < source.length() && Character.isWhitespace(source.charAt(at))) at++; }
            private boolean take(char expected) { if (at < source.length() && source.charAt(at) == expected) { at++; return true; } return false; }
            private void expect(char expected) { if (!take(expected)) fail("expected '" + expected + "'"); }
            private <T> T fail(String message) { throw new IllegalArgumentException("invalid JSON at character " + at + ": " + message); }
        }
    }
}
