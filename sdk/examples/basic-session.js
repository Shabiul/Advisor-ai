/**
 * Trusted Advisor SDK — Basic Session Example
 * ==============================================
 * Demonstrates how to use the SDK to connect to the backend,
 * track attention, detect away intervals, count gestures,
 * and receive real-time behavioral alerts.
 *
 * Prerequisites:
 *   1. Node.js backend running: `node backend/server.js`
 *   2. Python pipeline running: `python python-core/main.py`
 *
 * Run:
 *   node examples/basic-session.js
 */

const { createSession } = require("../src/index");
const { AwayTracker } = require("../src/away");

// ── Create Session ───────────────────────────────────────────────────

const session = createSession({
  backendUrl: "http://localhost:3000",
  mode: "PROCTORING",
  pollInterval: 500,       // Fetch data every 500ms
  minAwayDuration: 3,      // Only log away intervals ≥ 3 seconds
});

// ── Event Listeners ──────────────────────────────────────────────────

session.on("update", (data) => {
  const sig = data.sig;
  const metrics = session.signals.getMetrics();
  process.stdout.write(
    `\r[LIVE] Engagement: ${metrics.engagement}/10 | ` +
    `Eye Contact: ${Math.round(metrics.eyeContact * 100)}% | ` +
    `Gaze: ${sig.gaze || "-"} | ` +
    `Gestures: ${session.gestures.getTotal()}     `
  );
});

session.on("focus_change", ({ from, to }) => {
  console.log(`\n⚡ Focus changed: ${from} → ${to}`);
});

session.on("away_start", ({ startTime }) => {
  console.log(`\n🔴 User went AWAY at ${startTime.toLocaleTimeString()}`);
});

session.on("away_end", ({ start, end, durationSec }) => {
  console.log(
    `\n✅ User returned at ${end.toLocaleTimeString()} ` +
    `(away for ${durationSec}s, since ${start.toLocaleTimeString()})`
  );
});

session.on("gesture", ({ total }) => {
  console.log(`\n🖐️  Gesture detected! Total this session: ${total}`);
});

session.on("alert", (flags) => {
  for (const flag of flags) {
    console.log(`\n🚨 [${flag.severity}] ${flag.message}`);
  }
});

session.on("emotion", ({ emotion, confidence }) => {
  if (confidence > 50) {
    console.log(`\n🎭 Emotion: ${emotion} (${Math.round(confidence)}%)`);
  }
});

session.on("error", (err) => {
  console.error(`\n❌ Error: ${err.message}`);
});

session.on("tick", ({ sessionSec, awaySec }) => {
  // Update terminal title with timers every second
  process.title = `Session: ${AwayTracker.formatTime(sessionSec)} | Away: ${AwayTracker.formatTime(awaySec)}`;
});

// ── Start Session ────────────────────────────────────────────────────

console.log("╔══════════════════════════════════════════════════╗");
console.log("║   Trusted Advisor SDK — Live Session Monitor     ║");
console.log("║   Mode: PROCTORING                               ║");
console.log("║   Press Ctrl+C to stop and view summary          ║");
console.log("╚══════════════════════════════════════════════════╝\n");

session.start();

// ── Graceful Shutdown ────────────────────────────────────────────────

process.on("SIGINT", () => {
  console.log("\n\n═══════════════ SESSION SUMMARY ═══════════════\n");
  
  const summary = session.getSummary();
  
  console.log(`Mode:              ${summary.mode}`);
  console.log(`Duration:          ${AwayTracker.formatTime(summary.sessionDurationSec)}`);
  console.log(`Attention Avg:     ${summary.attention.average}%`);
  console.log(`Focus Level:       ${summary.attention.current || "N/A"}`);
  console.log(`Focus Streak:      ${summary.attention.streak.type} for ${summary.attention.streak.durationSec}s`);
  console.log(`Total Away Time:   ${AwayTracker.formatTime(summary.presence.awaySec)} (${summary.presence.awayPercent}%)`);
  console.log(`Away Intervals:    ${summary.awayLogs.length}`);
  console.log(`Total Gestures:    ${summary.gestures.total} (${summary.gestures.perMinute}/min)`);
  console.log(`Engagement:        ${summary.metrics.engagement}/10`);
  console.log(`Tension:           ${summary.metrics.tension}/10`);
  console.log(`Risk Score:        ${summary.riskScore}/100`);
  
  if (summary.interpretation) {
    console.log(`Suspicion Level:   ${summary.interpretation.suspicion_level || "N/A"}`);
    console.log(`Flags:             ${summary.interpretation.flags.length}`);
  }

  if (summary.awayLogs.length > 0) {
    console.log("\n── Away Interval Log ──");
    for (const log of summary.awayLogs.slice(0, 10)) {
      console.log(
        `  ${log.start.toLocaleTimeString()} → ${log.end.toLocaleTimeString()} (${log.durationSec}s)`
      );
    }
  }

  console.log("\n══════════════════════════════════════════════════\n");

  session.stop();
  process.exit(0);
});
