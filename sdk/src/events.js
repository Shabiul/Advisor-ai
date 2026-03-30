/**
 * Trusted Advisor SDK — EventEmitter
 * ====================================
 * Lightweight pub/sub event system with wildcard and once() support.
 * Used internally by all SDK modules for decoupled communication.
 */

class EventEmitter {
  constructor() {
    this._listeners = {};
    this._onceListeners = {};
  }

  /**
   * Subscribe to an event.
   * @param {string} event - Event name
   * @param {Function} callback - Handler function
   * @returns {EventEmitter} this (for chaining)
   */
  on(event, callback) {
    if (!this._listeners[event]) {
      this._listeners[event] = [];
    }
    this._listeners[event].push(callback);
    return this;
  }

  /**
   * Subscribe to an event once — auto-removes after first fire.
   * @param {string} event - Event name
   * @param {Function} callback - Handler function
   * @returns {EventEmitter} this
   */
  once(event, callback) {
    if (!this._onceListeners[event]) {
      this._onceListeners[event] = [];
    }
    this._onceListeners[event].push(callback);
    return this;
  }

  /**
   * Unsubscribe a specific callback from an event.
   * @param {string} event - Event name
   * @param {Function} callback - Handler to remove
   * @returns {EventEmitter} this
   */
  off(event, callback) {
    if (this._listeners[event]) {
      this._listeners[event] = this._listeners[event].filter(cb => cb !== callback);
    }
    if (this._onceListeners[event]) {
      this._onceListeners[event] = this._onceListeners[event].filter(cb => cb !== callback);
    }
    return this;
  }

  /**
   * Emit an event with data.
   * @param {string} event - Event name
   * @param {...*} args - Arguments to pass to handlers
   */
  emit(event, ...args) {
    // Regular listeners
    const cbs = this._listeners[event] || [];
    for (const cb of cbs) {
      try { cb(...args); } catch (err) {
        console.error(`[EventEmitter] Error in '${event}' handler:`, err);
      }
    }

    // Once listeners — fire and remove
    const onceCbs = this._onceListeners[event] || [];
    for (const cb of onceCbs) {
      try { cb(...args); } catch (err) {
        console.error(`[EventEmitter] Error in once '${event}' handler:`, err);
      }
    }
    if (onceCbs.length > 0) {
      this._onceListeners[event] = [];
    }

    // Wildcard listeners (subscribe to '*' to hear everything)
    const wildcardCbs = this._listeners["*"] || [];
    for (const cb of wildcardCbs) {
      try { cb(event, ...args); } catch (err) {
        console.error(`[EventEmitter] Error in wildcard handler:`, err);
      }
    }
  }

  /**
   * Remove all listeners, optionally for a specific event.
   * @param {string} [event] - If provided, only clears that event
   * @returns {EventEmitter} this
   */
  removeAllListeners(event) {
    if (event) {
      delete this._listeners[event];
      delete this._onceListeners[event];
    } else {
      this._listeners = {};
      this._onceListeners = {};
    }
    return this;
  }

  /**
   * Get count of listeners for an event.
   * @param {string} event
   * @returns {number}
   */
  listenerCount(event) {
    return (this._listeners[event] || []).length + (this._onceListeners[event] || []).length;
  }
}

module.exports = { EventEmitter };
