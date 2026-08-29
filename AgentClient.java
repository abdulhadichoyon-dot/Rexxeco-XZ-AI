import java.io.*;
import java.net.*;

/**
 * Java client for the Python-trained self-learning agent.
 *
 * This connects to agent_server.py over TCP, sends the current position,
 * and receives back the action the TRAINED agent chooses — the actual
 * learned intelligence lives entirely in Python; this class is a thin,
 * real network client, not a reimplementation or a mock.
 *
 * Protocol: newline-delimited JSON over a plain socket. No external JSON
 * library is used deliberately — this project should compile with nothing
 * but the JDK, so parsing is done with a small hand-rolled parser
 * (parseJsonInt/parseJsonBool below) sufficient for this flat, known
 * response shape. For a larger response schema, a real library
 * (Jackson/Gson) would replace this.
 */
public class AgentClient {

    private final String host;
    private final int port;
    private Socket socket;
    private BufferedReader in;
    private PrintWriter out;

    public AgentClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public void connect() throws IOException {
        socket = new Socket(host, port);
        in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        out = new PrintWriter(new OutputStreamWriter(socket.getOutputStream()), true);
        System.out.println("Connected to Python agent server at " + host + ":" + port);
    }

    public void close() throws IOException {
        if (socket != null) socket.close();
    }

    /** Sends a reset request and returns the starting state. */
    public StepResult reset() throws IOException {
        out.println("{\"reset\": true}");
        String response = in.readLine();
        if (response == null) {
            throw new IOException("Server closed the connection unexpectedly.");
        }
        return StepResult.fromJson(response);
    }

    /** Sends the current position and receives the trained agent's chosen next step. */
    public StepResult step(int row, int col) throws IOException {
        String request = String.format("{\"row\": %d, \"col\": %d}", row, col);
        out.println(request);
        String response = in.readLine();
        if (response == null) {
            throw new IOException("Server closed the connection unexpectedly.");
        }
        return StepResult.fromJson(response);
    }

    /**
     * Result of one step, parsed from the server's JSON response.
     * Hand-rolled parsing (see class-level note) — fine for this fixed,
     * flat schema; would not scale to arbitrary/nested JSON.
     */
    public static class StepResult {
        public int row, col;
        public double reward;
        public boolean done;
        public String actionName;
        public String event;

        static StepResult fromJson(String json) {
            StepResult r = new StepResult();
            r.row = parseJsonInt(json, "row");
            r.col = parseJsonInt(json, "col");
            r.reward = parseJsonDouble(json, "reward");
            r.done = parseJsonBool(json, "done");
            r.actionName = parseJsonString(json, "action_name");
            r.event = parseJsonString(json, "event");
            return r;
        }

        @Override
        public String toString() {
            return String.format(
                "pos=(%d,%d) action=%s reward=%.2f done=%b event=%s",
                row, col, actionName, reward, done, event
            );
        }
    }

    // --- Minimal hand-rolled JSON field extraction ---
    // Deliberately not a general JSON parser. Handles: "key": number,
    // "key": "string", "key": true/false/null. Sufficient for this
    // project's flat, fixed response shape; swap for Jackson/Gson if the
    // schema grows.

    private static int parseJsonInt(String json, String key) {
        String raw = extractRawValue(json, key);
        if (raw == null || raw.equals("null")) return 0;
        return Integer.parseInt(raw.trim());
    }

    private static double parseJsonDouble(String json, String key) {
        String raw = extractRawValue(json, key);
        if (raw == null || raw.equals("null")) return 0.0;
        return Double.parseDouble(raw.trim());
    }

    private static boolean parseJsonBool(String json, String key) {
        String raw = extractRawValue(json, key);
        return raw != null && raw.trim().equals("true");
    }

    private static String parseJsonString(String json, String key) {
        String pattern = "\"" + key + "\"";
        int keyIdx = json.indexOf(pattern);
        if (keyIdx == -1) return null;
        int colonIdx = json.indexOf(':', keyIdx);
        int firstQuote = json.indexOf('"', colonIdx + 1);
        if (firstQuote == -1) return null;
        int secondQuote = json.indexOf('"', firstQuote + 1);
        if (secondQuote == -1) return null;
        return json.substring(firstQuote + 1, secondQuote);
    }

    private static String extractRawValue(String json, String key) {
        String pattern = "\"" + key + "\"";
        int keyIdx = json.indexOf(pattern);
        if (keyIdx == -1) return null;
        int colonIdx = json.indexOf(':', keyIdx);
        if (colonIdx == -1) return null;

        int start = colonIdx + 1;
        while (start < json.length() && Character.isWhitespace(json.charAt(start))) start++;

        if (json.charAt(start) == '"') {
            return null; // caller should use parseJsonString for string fields
        }

        int end = start;
        while (end < json.length() && ",}\n".indexOf(json.charAt(end)) == -1) end++;
        return json.substring(start, end);
    }
}
