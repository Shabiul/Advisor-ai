/**
 * Trusted Advisor SDK — Full Server Example
 * ============================================
 * This single file replaces the entire backend/server.js.
 * It starts the full HTTP server with dashboard, video streaming,
 * and behavioral interpretation — all from the SDK.
 *
 * Run:
 *   node examples/full-server.js
 *
 * Then:
 *   1. Open http://localhost:3000 in your browser
 *   2. Start python pipeline: python python-core/main.py
 */

const { createServer } = require("../src/index");

const server = createServer({ port: 3000 });

// Listen to server-side events
server.on("analyze", ({ mode, sig, interpretation }) => {
  const face = sig ? sig.face_detected : false;
  const gestures = sig ? sig.gestures : 0;
  console.log(
    `[${new Date().toLocaleTimeString()}] ` +
    `Mode: ${mode} | Face: ${face} | Gestures: ${gestures} | ` +
    `Flags: ${interpretation.flags.length}`
  );
});

server.on("error", (err) => {
  console.error("Server error:", err.message);
});

// Start
server.start().then(() => {
  console.log("\n╔══════════════════════════════════════════════════╗");
  console.log("║   Trusted Advisor SDK — Full Server              ║");
  console.log("║   Dashboard: http://localhost:3000                ║");
  console.log("║   Analyze:   POST http://localhost:3000/analyze   ║");
  console.log("║   Data:      GET  http://localhost:3000/data      ║");
  console.log("║   Video:     GET  http://localhost:3000/video_feed║");
  console.log("║                                                   ║");
  console.log("║   Start Python: python python-core/main.py        ║");
  console.log("║   Press Ctrl+C to stop                            ║");
  console.log("╚══════════════════════════════════════════════════╝\n");
});

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("\nShutting down...");
  await server.stop();
  process.exit(0);
});
