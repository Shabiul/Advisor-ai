/**
 * Trusted Advisor SDK — Standalone Modules Example
 * ===================================================
 * Shows how to use individual SDK modules without the Session
 * orchestrator, for cases where you want fine-grained control.
 */

const {
  ApiClient,
  AttentionTracker,
  AwayTracker,
  GestureCounter,
  Interpreter,
  SignalStore,
} = require("../src/index");

// ── 1. API Client (standalone) ─────────────────────────────────────

async function apiExample() {
  const api = new ApiClient("http://localhost:3000");

  // Health check
  const alive = await api.healthCheck();
  console.log(`Backend alive: ${alive}`);

  // Fetch latest data
  const data = await api.getData();
  console.log("Latest data status:", data.status);

  return data;
}

// ── 2. Attention Tracker (standalone) ──────────────────────────────

function attentionExample() {
  const tracker = new AttentionTracker({ maxHistory: 100 });

  tracker.on("focus_change", ({ from, to }) => {
    console.log(`Focus: ${from} → ${to}`);
  });

  // Simulate attention scores
  const scores = [0.9, 0.85, 0.7, 0.6, 0.3, 0.2, 0.5, 0.8, 0.95];
  for (const score of scores) {
    tracker.push(score);
  }

  console.log("Average attention:", (tracker.getAverage() * 100).toFixed(1) + "%");
  console.log("Current level:", tracker.getCurrentLevel());
  console.log("Timeline:", tracker.getRecentScores(5));
}

// ── 3. Away Tracker (standalone) ───────────────────────────────────

function awayExample() {
  const away = new AwayTracker({ minAwayDuration: 2 });

  away.on("away_end", (log) => {
    console.log(`Away log: ${log.durationSec}s`);
  });

  // Simulate presence changes
  away.update(true);   // present
  away.update(false);  // away
  // ... after 5 seconds ...
  setTimeout(() => {
    away.update(true);  // returned
    console.log("Away summary:", away.getSummary());
  }, 5000);
}

// ── 4. Gesture Counter (standalone) ────────────────────────────────

function gestureExample() {
  const counter = new GestureCounter();

  counter.on("gesture", ({ total }) => {
    console.log(`Gesture #${total}`);
  });

  // Simulate hand detection states (rising edge = new gesture)
  counter.update(false);  // no hands
  counter.update(true);   // hands appear → gesture #1
  counter.update(true);   // still there → no increment
  counter.update(false);  // hands gone
  counter.update(true);   // hands appear again → gesture #2

  console.log("Total gestures:", counter.getTotal());
}

// ── 5. Interpreter (offline) ──────────────────────────────────────

function interpreterExample() {
  const interp = new Interpreter("PROCTORING");

  const mockReport = {
    summary: { attention_score: 0.35, focus_level: "LOW" },
    metrics: {
      off_screen_time: "30%",
      head_down_time: "20%",
      face_missing_time: "15%",
    },
    alerts: [
      { type: "LOOK_AWAY" },
      { type: "FACE_MISSING" },
      { type: "SLOUCHED_POSTURE" },
    ],
    body_analysis: { posture: "SLOUCHED", neck: "FORWARD_HEAD" },
  };

  const result = interp.interpret(mockReport);
  console.log("\nInterpretation Result:");
  console.log("  Suspicion:", result.suspicion_level);
  console.log("  Flags:", result.flags.length);
  for (const flag of result.flags) {
    console.log(`    [${flag.severity}] ${flag.message}`);
  }
  console.log("  Risk Score:", interp.getRiskScore(mockReport));
}

// ── Run All ───────────────────────────────────────────────────────

(async () => {
  console.log("═══ API Client ═══");
  try {
    await apiExample();
  } catch (e) {
    console.log("  (Backend not running, skipping)");
  }

  console.log("\n═══ Attention Tracker ═══");
  attentionExample();

  console.log("\n═══ Gesture Counter ═══");
  gestureExample();

  console.log("\n═══ Interpreter ═══");
  interpreterExample();

  console.log("\n✅ All standalone module examples completed.");
})();
