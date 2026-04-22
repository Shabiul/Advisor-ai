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
 *   3. (Optional) Start audio: cd python-core && python -m aud.emo_service
 */

const { createServer } = require("../src/index");

const server = createServer({ port: 3000 });

// Listen to server-side events
server.on("analyze", ({ mode, sig, interpretation }) => {
  const face = sig ? sig.face_detected : false;
  const gestures = sig ? sig.gestures : 0;
  const modality = interpretation.modality || "vision_only";
  const fusedEmo = interpretation.multimodal ? interpretation.multimodal.fused_emotion : "—";
  const congruence = interpretation.multimodal ? interpretation.multimodal.congruence_level : "—";

  console.log(
    `[${new Date().toLocaleTimeString()}] ` +
    `${modality === "multimodal" ? "🔀" : "👁️"} ` +
    `Mode: ${mode} | Face: ${face} | Gestures: ${gestures} | ` +
    `Flags: ${interpretation.flags.length} | ` +
    `Emotion: ${fusedEmo} | Congruence: ${congruence}`
  );
});

server.on("audio_emotion", (data) => {
  const ae = data.audio_emotion || {};
  if (ae.status !== "silence" && ae.label) {
    console.log(
      `[${new Date().toLocaleTimeString()}] ` +
      `🎤 Voice: ${ae.label} (${Math.round((ae.confidence || 0) * 100)}%) | ` +
      `VAD: ${ae.vad_quadrant || "—"}`
    );
  }
});

server.on("error", (err) => {
  console.error("Server error:", err.message);
});

// Start
server.start().then(() => {
  console.log("\n╔══════════════════════════════════════════════════════╗");
  console.log("║   Trusted Advisor SDK — Full Server (Multimodal)     ║");
  console.log("║   Dashboard:      http://localhost:3000               ║");
  console.log("║   Analyze:        POST http://localhost:3000/analyze  ║");
  console.log("║   Audio Emotion:  POST http://localhost:3000/audio_emotion ║");
  console.log("║   Data:           GET  http://localhost:3000/data     ║");
  console.log("║   Video:          GET  http://localhost:3000/video_feed║");
  console.log("║                                                       ║");
  console.log("║   Terminal 1: node sdk/examples/full-server.js        ║");
  console.log("║   Terminal 2: cd python-core && python main.py        ║");
  console.log("║   Terminal 3: cd python-core && python -m aud.emo_service            ║");
  console.log("║   Press Ctrl+C to stop                                ║");
  console.log("╚══════════════════════════════════════════════════════╝\n");
});

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("\nShutting down...");
  await server.stop();
  process.exit(0);
});
