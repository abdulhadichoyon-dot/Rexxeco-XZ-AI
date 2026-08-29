import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Demonstrates the full pipeline: Java client connects to the Python agent
 * server, and the AI trained entirely in Python (via trial-and-error
 * reinforcement learning) drives navigation from the Java side.
 *
 * Run agent_server.py FIRST, then run this.
 */
public class Main {
    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "localhost";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 5555;

        AgentClient client = new AgentClient(host, port);

        try {
            client.connect();

            System.out.println("\nAsking the Python-trained agent to navigate from Java...\n");

            AgentClient.StepResult state = client.reset();
            System.out.println("Reset. Starting at (" + state.row + ", " + state.col + ")");

            List<int[]> path = new ArrayList<>();
            path.add(new int[]{state.row, state.col});

            int maxSteps = 30;
            int stepCount = 0;
            boolean done = false;

            while (!done && stepCount < maxSteps) {
                state = client.step(state.row, state.col);
                stepCount++;
                path.add(new int[]{state.row, state.col});

                System.out.printf(
                    "Step %2d: moved %-6s -> (%d, %d)  reward=%.2f%s%n",
                    stepCount, state.actionName, state.row, state.col, state.reward,
                    state.event != null && !state.event.equals("moved") ? "  [" + state.event + "]" : ""
                );

                done = state.done;
            }

            System.out.println();
            if ("reached_goal".equals(state.event)) {
                System.out.println("SUCCESS: Agent reached the goal in " + stepCount + " steps.");
            } else if ("hit_obstacle".equals(state.event)) {
                System.out.println("Agent hit an obstacle and the episode ended.");
            } else {
                System.out.println("Episode ended: " + state.event);
            }

            System.out.print("Path taken: ");
            for (int[] p : path) {
                System.out.print("(" + p[0] + "," + p[1] + ") ");
            }
            System.out.println();

        } catch (IOException e) {
            System.err.println("Connection error: " + e.getMessage());
            System.err.println("Make sure agent_server.py is running first:");
            System.err.println("  python3 agent_server.py " + port);
        } finally {
            try {
                client.close();
            } catch (IOException e) {
                // Ignore close errors on shutdown.
            }
        }
    }
}
