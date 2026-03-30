/**
 * Trusted Advisor SDK — Unit Tests
 * ==================================
 * Lightweight test suite — zero dependencies.
 * Run: node test/run.js
 */

let passed = 0;
let failed = 0;

function assert(condition, label) {
  if (condition) {
    passed++;
    console.log(`  ✅ ${label}`);
  } else {
    failed++;
    console.error(`  ❌ ${label}`);
  }
}

function section(name) {
  console.log(`\n═══ ${name} ═══`);
}

// ── EventEmitter ─────────────────────────────────────────────────────

section("EventEmitter");

const { EventEmitter } = require("../src/events");

(() => {
  const ee = new EventEmitter();
  let called = false;
  ee.on("test", () => { called = true; });
  ee.emit("test");
  assert(called, "on() + emit() works");
})();

(() => {
  const ee = new EventEmitter();
  let count = 0;
  ee.once("test", () => { count++; });
  ee.emit("test");
  ee.emit("test");
  assert(count === 1, "once() fires only once");
})();

(() => {
  const ee = new EventEmitter();
  let received = null;
  ee.on("*", (event, data) => { received = { event, data }; });
  ee.emit("hello", 42);
  assert(received && received.event === "hello" && received.data === 42, "wildcard listener receives all events");
})();

(() => {
  const ee = new EventEmitter();
  let count = 0;
  const handler = () => { count++; };
  ee.on("test", handler);
  ee.off("test", handler);
  ee.emit("test");
  assert(count === 0, "off() removes handler");
})();

(() => {
  const ee = new EventEmitter();
  ee.on("a", () => {});
  ee.on("a", () => {});
  ee.once("a", () => {});
  assert(ee.listenerCount("a") === 3, "listenerCount() returns correct count");
  ee.removeAllListeners("a");
  assert(ee.listenerCount("a") === 0, "removeAllListeners(event) clears event");
})();

// ── AttentionTracker ─────────────────────────────────────────────────

section("AttentionTracker");

const { AttentionTracker } = require("../src/attention");

(() => {
  const tracker = new AttentionTracker();
  tracker.push(0.9);
  tracker.push(0.8);
  tracker.push(0.7);
  assert(tracker.getCurrentLevel() === "HIGH", "classifies HIGH correctly");
  assert(tracker.getTimeline().length === 3, "timeline has 3 entries");
  assert(tracker.getRecentScores(2).length === 2, "getRecentScores(2) returns 2");
})();

(() => {
  const tracker = new AttentionTracker();
  tracker.push(0.3);
  assert(tracker.getCurrentLevel() === "LOW", "classifies LOW correctly");
})();

(() => {
  const tracker = new AttentionTracker();
  let changed = false;
  tracker.on("focus_change", () => { changed = true; });
  tracker.push(0.9); // HIGH
  tracker.push(0.2); // LOW → triggers focus_change
  assert(changed, "emits focus_change on transition");
})();

(() => {
  const tracker = new AttentionTracker();
  tracker.push(0.5);
  tracker.push(0.6);
  tracker.push(0.4);
  const avg = tracker.getAverage();
  assert(Math.abs(avg - 0.5) < 0.01, "getAverage() computes correctly");
})();

(() => {
  const tracker = new AttentionTracker();
  tracker.push(0.8);
  tracker.reset();
  assert(tracker.getTimeline().length === 0, "reset() clears history");
  assert(tracker.getCurrentLevel() === null, "reset() clears level");
})();

// ── AwayTracker ──────────────────────────────────────────────────────

section("AwayTracker");

const { AwayTracker } = require("../src/away");

(() => {
  const away = new AwayTracker({ minAwayDuration: 0 });
  assert(!away.isCurrentlyAway(), "starts as present");
  away.update(false);
  assert(away.isCurrentlyAway(), "transitions to away");
  away.update(true);
  assert(!away.isCurrentlyAway(), "transitions back to present");
})();

(() => {
  const away = new AwayTracker({ minAwayDuration: 0 });
  let logReceived = false;
  away.on("away_end", () => { logReceived = true; });
  away.update(false); // Go away
  away.update(true);  // Come back
  assert(logReceived, "emits away_end on return");
  assert(away.getAwayLogs().length === 1, "logs the interval");
})();

(() => {
  const away = new AwayTracker({ minAwayDuration: 10 });
  away.update(false);
  away.update(true);
  assert(away.getAwayLogs().length === 0, "skips intervals below minAwayDuration");
})();

(() => {
  assert(AwayTracker.formatTime(3661) === "01:01:01", "formatTime() works correctly");
  assert(AwayTracker.formatTime(0) === "00:00:00", "formatTime(0) returns 00:00:00");
})();

(() => {
  const summary = new AwayTracker().getSummary();
  assert(summary.sessionSec === 0, "initial summary has 0 session seconds");
  assert(summary.awayPercent === 0, "initial away percent is 0");
})();

// ── GestureCounter ───────────────────────────────────────────────────

section("GestureCounter");

const { GestureCounter } = require("../src/gestures");

(() => {
  const counter = new GestureCounter();
  counter.update(false);
  counter.update(true);  // rising edge → +1
  counter.update(true);  // no change
  counter.update(false);
  counter.update(true);  // rising edge → +1
  assert(counter.getTotal() === 2, "counts rising edges correctly (2)");
})();

(() => {
  const counter = new GestureCounter();
  let eventCount = 0;
  counter.on("gesture", () => { eventCount++; });
  counter.update(true);
  counter.update(false);
  counter.update(true);
  assert(eventCount === 2, "emits gesture event on each rising edge");
})();

(() => {
  const counter = new GestureCounter();
  counter.update(true);
  counter.reset();
  assert(counter.getTotal() === 0, "reset() clears count");
})();

(() => {
  const counter = new GestureCounter();
  counter.update(2);  // number > 0 = true
  counter.update(0);  // number 0 = false
  counter.update(1);  // number > 0 = true (rising edge)
  assert(counter.getTotal() === 2, "handles numeric input correctly");
})();

// ── Interpreter ──────────────────────────────────────────────────────

section("Interpreter");

const { Interpreter, THRESHOLDS } = require("../src/interpreter");

(() => {
  const interp = new Interpreter("PROCTORING");
  const result = interp.interpret({
    summary: { attention_score: 0.9 },
    metrics: {},
    alerts: [],
    body_analysis: {},
  });
  assert(result.suspicion_level === "LOW", "clean report = LOW suspicion");
  assert(result.flags.length === 0, "clean report = no flags");
})();

(() => {
  const interp = new Interpreter("PROCTORING");
  const result = interp.interpret({
    summary: { attention_score: 0.2 },
    metrics: {
      off_screen_time: "50%",
      head_down_time: "30%",
      face_missing_time: "20%",
    },
    alerts: [1, 2, 3],
    body_analysis: { posture: "SLOUCHED", neck: "FORWARD_HEAD" },
  });
  assert(result.suspicion_level === "HIGH", "bad report = HIGH suspicion");
  assert(result.flags.length >= 4, "multiple flags generated");
})();

(() => {
  const interp = new Interpreter("MEETING");
  const result = interp.interpret({
    summary: { attention_score: 0.8 },
    metrics: {},
    alerts: [],
    body_analysis: {},
  });
  assert(result.engagement_level === "HIGHLY_ENGAGED", "high attention = HIGHLY_ENGAGED");
  assert(!result.suspicion_level, "MEETING mode has no suspicion_level");
})();

(() => {
  const interp = new Interpreter("PROCTORING");
  const risk = interp.getRiskScore({
    summary: { attention_score: 0.9 },
    metrics: {},
    alerts: [],
    body_analysis: {},
  });
  assert(risk === 0, "clean report risk score = 0");
})();

(() => {
  try {
    new Interpreter("INVALID");
    assert(false, "should throw for invalid mode");
  } catch (e) {
    assert(e.message.includes("Unknown mode"), "throws for invalid mode");
  }
})();

// ── SignalStore ──────────────────────────────────────────────────────

section("SignalStore");

const { SignalStore } = require("../src/signals");

(() => {
  const store = new SignalStore();
  assert(store.getSignals() === null, "starts empty");
  store.ingest({ sig: { gaze: "CENTER", face_detected: true }, data: {} });
  assert(store.getSignals() !== null, "stores ingested signal");
  assert(store.get("gaze") === "CENTER", "get() retrieves signal value");
})();

(() => {
  const store = new SignalStore();
  let faceLost = false;
  store.on("face_lost", () => { faceLost = true; });
  store.ingest({ sig: { face_detected: true }, data: {} });
  store.ingest({ sig: { face_detected: false }, data: {} });
  assert(faceLost, "emits face_lost on true→false transition");
})();

(() => {
  const store = new SignalStore();
  store.ingest({
    sig: {
      engagement_score: 7,
      micro_tension_score: 3,
      eye_contact_score: 0.85,
      blinks_per_minute: 15,
    },
    data: {},
  });
  const metrics = store.getMetrics();
  assert(metrics.engagement === 7, "getMetrics().engagement correct");
  assert(metrics.eyeContact === 0.85, "getMetrics().eyeContact correct");
})();

(() => {
  const store = new SignalStore({ maxHistory: 3 });
  store.ingest({ sig: { a: 1 }, data: {} });
  store.ingest({ sig: { a: 2 }, data: {} });
  store.ingest({ sig: { a: 3 }, data: {} });
  store.ingest({ sig: { a: 4 }, data: {} });
  assert(store.getHistory().length === 3, "respects maxHistory cap");
})();

// ── Session ──────────────────────────────────────────────────────────

section("Session (unit, no backend)");

const { Session } = require("../src/session");

(() => {
  const session = new Session({ mode: "PROCTORING" });
  assert(session.mode === "PROCTORING", "mode is set correctly");
  assert(session.attention instanceof AttentionTracker, "has AttentionTracker");
  assert(session.away instanceof AwayTracker, "has AwayTracker");
  assert(session.gestures instanceof GestureCounter, "has GestureCounter");
})();

(() => {
  const session = new Session();
  const mockPayload = {
    mode: "PROCTORING",
    data: {
      summary: { attention_score: 0.75, focus_level: "HIGH" },
      metrics: {},
      alerts: [],
    },
    sig: {
      face_detected: true,
      engagement_score: 8,
      gestures: 0,
      gaze: "CENTER",
    },
  };

  let updateReceived = false;
  session.on("update", () => { updateReceived = true; });
  session.ingest(mockPayload);

  assert(updateReceived, "ingest() triggers update event");
  assert(session.signals.isFaceDetected(), "face_detected properly tracked");
  assert(session.attention.getCurrentLevel() === "HIGH", "attention tracked from ingest");
})();

(() => {
  const session = new Session();
  const summary = session.getSummary();
  assert(summary.mode === "PROCTORING", "getSummary() includes mode");
  assert(typeof summary.attention === "object", "getSummary() has attention block");
  assert(typeof summary.presence === "object", "getSummary() has presence block");
  assert(typeof summary.gestures === "object", "getSummary() has gestures block");
})();

// ── Final Report ─────────────────────────────────────────────────────

console.log(`\n${"═".repeat(50)}`);
console.log(`  Results: ${passed} passed, ${failed} failed`);
console.log(`${"═".repeat(50)}\n`);

process.exit(failed > 0 ? 1 : 0);
