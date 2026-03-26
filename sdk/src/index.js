/**
 * Trusted Advisor SDK — Entry Point
 * ===================================
 * Re-exports all public API for consumers.
 *
 * Usage:
 *   const { createSession } = require("trusted-advisor-sdk");
 */

const { createSession, Session } = require("./session");
const { ApiClient } = require("./api");
const { EventEmitter } = require("./events");

module.exports = {
  createSession,
  Session,
  ApiClient,
  EventEmitter,
};
