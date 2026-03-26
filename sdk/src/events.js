/**
 * Trusted Advisor SDK — Events
 * =============================
 * Lightweight event emitter for SDK internal use.
 */

class EventEmitter {
  constructor() {
    this._listeners = {};
  }

  on(event, callback) {
    if (!this._listeners[event]) {
      this._listeners[event] = [];
    }
    this._listeners[event].push(callback);
    return this;
  }

  off(event, callback) {
    if (!this._listeners[event]) return this;
    this._listeners[event] = this._listeners[event].filter((cb) => cb !== callback);
    return this;
  }

  emit(event, data) {
    const cbs = this._listeners[event] || [];
    for (const cb of cbs) {
      try {
        cb(data);
      } catch (err) {
        console.error(`[EventEmitter] Error in '${event}' handler:`, err);
      }
    }
  }

  removeAllListeners(event) {
    if (event) {
      delete this._listeners[event];
    } else {
      this._listeners = {};
    }
    return this;
  }
}

module.exports = { EventEmitter };
