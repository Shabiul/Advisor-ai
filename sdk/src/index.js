/**
 * Trusted Advisor SDK — Entry Point (v2.0)
 * ==========================================
 * Comprehensive Node.js SDK for the Trusted Advisor AI
 * Behavioral Intelligence Platform.
 *
 * This SDK IS the full project. It contains:
 *   - The complete HTTP backend server (POST /analyze, GET /data, video streaming)
 *   - The full Twitch-style dashboard (HTML + CSS + JS embedded in memory)
 *   - Session management, attention tracking, away detection, gesture counting
 *   - Mode-based behavioral interpretation (PROCTORING / MEETING)
 *
 * Quick Start (full server):
 * ──────────────────────────
 *   const { createServer } = require("trusted-advisor-sdk");
 *   const server = createServer({ port: 3000 });
 *   server.start();
 *   // Dashboard live at http://localhost:3000
 *   // Python pipeline posts to http://localhost:3000/analyze
 *
 * Quick Start (client session):
 * ─────────────────────────────
 *   const { createSession } = require("trusted-advisor-sdk");
 *   const session = createSession({ backendUrl: "http://localhost:3000" });
 *   session.on("update", console.log);
 *   session.start();
 */

// Server (the full backend)
const { AdvisorServer, createServer } = require("./server");
const { getDashboardHTML, getDashboardCSS, getDashboardJS } = require("./dashboard");

// Client Session
const { createSession, Session } = require("./session");
const { ApiClient } = require("./api");
const { SignalStore } = require("./signals");
const { AttentionTracker } = require("./attention");
const { AwayTracker } = require("./away");
const { GestureCounter } = require("./gestures");
const { Interpreter, THRESHOLDS } = require("./interpreter");
const { EventEmitter } = require("./events");

module.exports = {
  // ── Server (full project backend) ──────────
  createServer,
  AdvisorServer,
  getDashboardHTML,
  getDashboardCSS,
  getDashboardJS,

  // ── Client Session ─────────────────────────
  createSession,
  Session,

  // ── Sub-modules ────────────────────────────
  ApiClient,
  SignalStore,
  AttentionTracker,
  AwayTracker,
  GestureCounter,
  Interpreter,
  THRESHOLDS,
  EventEmitter,
};
