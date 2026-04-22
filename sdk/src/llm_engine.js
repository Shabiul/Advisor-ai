/**
 * Trusted Advisor SDK — LLM Engine (Event-Driven)
 * =================================================
 * Uses Ollama (local LLM) to generate behavioral insights.
 *
 * KEY DESIGN: Not called continuously — triggered only when:
 *   1. Significant emotional change occurs
 *
 * Rate-limited to ~1 request per 5 seconds.
 * Falls back to rule-based insights if LLM is unavailable.
 */

const http = require("http");

class LLMEngine {
  /**
   * @param {Object} [options]
   * @param {string} [options.baseUrl="http://localhost:11434"] - Ollama API URL
   * @param {string} [options.model="qwen2:7b"] - Model to use
   * @param {number} [options.timeout=15000] - Request timeout in ms
   * @param {number} [options.cooldownMs=5000] - Min time between requests
   */
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || "http://localhost:11434";
    this.model = options.model || "qwen2:7b";
    this.timeout = options.timeout || 15000;
    this.cooldownMs = options.cooldownMs || 5000;

    this._lastRequestTime = 0;
    this._available = null; // unknown until first check
    this._pendingRequest = null;
  }

  /**
   * Generate behavioral insight from multimodal data.
   * Only runs if LLM is available and cooldown has elapsed.
   *
   * @param {Object} multimodalData - Output from interpreter.interpretMultimodal()
   * @returns {Promise<Object>} { insight, suggestions, risk_note, source }
   */
  async generateInsight(multimodalData) {
    const now = Date.now();

    // Rate limit check
    if (now - this._lastRequestTime < this.cooldownMs) {
      return this._fallbackInsight(multimodalData, "rate_limited");
    }

    // Check availability
    if (this._available === false) {
      return this._fallbackInsight(multimodalData, "llm_unavailable");
    }

    this._lastRequestTime = now;

    try {
      // Build prompt from multimodal data
      const prompt = this._buildPrompt(multimodalData);

      // Call Ollama
      const response = await this._callOllama(prompt);

      if (response && response.response) {
        const parsed = this._parseResponse(response.response);
        return {
          ...parsed,
          source: "llm",
          model: this.model,
          processing_time_ms: Date.now() - now,
        };
      }

      return this._fallbackInsight(multimodalData, "empty_response");
    } catch (err) {
      if (err.code === "ECONNREFUSED") {
        this._available = false;
        console.log("[LLM] Ollama not running. Using rule-based insights.");
      }
      return this._fallbackInsight(multimodalData, "error");
    }
  }

  /**
   * Check if Ollama is running and responsive.
   * @returns {Promise<boolean>}
   */
  async checkAvailability() {
    try {
      const result = await this._httpGet(`${this.baseUrl}/api/tags`);
      this._available = true;

      // Check if requested model is available
      const models = result.models || [];
      const hasModel = models.some(
        (m) => m.name === this.model || m.name.startsWith(this.model)
      );

      if (!hasModel) {
        console.log(`[LLM] Model '${this.model}' not found. Available: ${models.map((m) => m.name).join(", ")}`);
        console.log(`[LLM] Pull with: ollama pull ${this.model}`);
      }

      return true;
    } catch (err) {
      this._available = false;
      return false;
    }
  }

  /**
   * Get current status.
   */
  getStatus() {
    return {
      available: this._available,
      model: this.model,
      baseUrl: this.baseUrl,
      cooldownMs: this.cooldownMs,
    };
  }

  // ── PROMPT BUILDING ──────────────────────────────────────────────

  _buildPrompt(data) {
    const signals = data || {};
    const faceEmotion = signals.emotion || "unknown";
    const engagement = signals.engagement_score || 0;

    const sections = [];

    sections.push("You are a behavioral intelligence analyst. Analyze the following multimodal signals and provide a brief behavioral insight.");
    sections.push("");
    sections.push("VISUAL SIGNALS:");
    sections.push(`- Face emotion: ${faceEmotion}`);
    sections.push(`- Engagement score: ${engagement}/10`);

    if (signals.micro_tension_score) sections.push(`- Facial tension: ${signals.micro_tension_score}/10`);
    if (signals.smile_label) sections.push(`- Smile: ${signals.smile_label} (genuine: ${signals.smile_genuine || false})`);
    if (signals.brow_label) sections.push(`- Brow: ${signals.brow_label}`);
    if (signals.gaze) sections.push(`- Gaze: ${signals.gaze}`);
    if (signals.head_pose) sections.push(`- Head pose: ${signals.head_pose}`);

    // Audio signals (from multimodal fusion)
    if (signals._audioData) {
      const ae = signals._audioData.audio_emotion || {};
      if (ae.label && ae.status !== "silence") {
        sections.push("");
        sections.push("VOCAL SIGNALS:");
        sections.push(`- Vocal emotion: ${ae.label} (confidence: ${Math.round((ae.confidence || 0) * 100)}%)`);
        if (ae.valence !== undefined) sections.push(`- Valence: ${ae.valence.toFixed(3)}`);
        if (ae.arousal !== undefined) sections.push(`- Arousal: ${ae.arousal.toFixed(3)}`);
        if (ae.dominance !== undefined) sections.push(`- Dominance: ${ae.dominance.toFixed(3)}`);
        if (ae.vad_quadrant) sections.push(`- VAD quadrant: ${ae.vad_quadrant}`);
      }
    }

    // Multimodal fusion results (from interpreter)
    if (signals._multimodal) {
      const mm = signals._multimodal;
      sections.push("");
      sections.push("MULTIMODAL FUSION:");
      sections.push(`- Vision emotion: ${mm.vision_emotion || "?"}`);
      sections.push(`- Audio emotion: ${mm.audio_emotion || "?"}`);
      sections.push(`- Fused emotion: ${mm.fused_emotion || "?"}`);
      sections.push(`- Congruence: ${mm.congruence_level || "?"} (${Math.round((mm.congruence || 0) * 100)}%)`);
      if (mm.behavioral_insights && mm.behavioral_insights.length > 0) {
        sections.push(`- Insights: ${mm.behavioral_insights.join("; ")}`);
      }
    }

    sections.push("");
    sections.push("Respond with EXACTLY this JSON format, nothing else:");
    sections.push('{"insight": "brief observation (1-2 sentences)", "suggestion": "brief actionable advice (1 sentence)", "risk_level": "LOW|MEDIUM|HIGH"}');

    return sections.join("\n");
  }

  _parseResponse(text) {
    // Try to extract JSON from the response
    try {
      const jsonMatch = text.match(/\{[^}]+\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          insight: parsed.insight || "No specific insight available.",
          suggestion: parsed.suggestion || "",
          risk_level: parsed.risk_level || "LOW",
        };
      }
    } catch (e) {
      // Fall through to text parsing
    }

    // If JSON parsing fails, use the raw text as insight
    return {
      insight: text.slice(0, 200).trim(),
      suggestion: "",
      risk_level: "LOW",
    };
  }

  // ── RULE-BASED FALLBACK ──────────────────────────────────────────

  _fallbackInsight(data, reason) {
    let insight = "";
    let suggestion = "";
    let risk_level = "LOW";

    const faceEmotion = (data.emotion || "neutral").toLowerCase();
    const audioEmotion = (data._audioData && data._audioData.audio_emotion)
      ? (data._audioData.audio_emotion.label || "").toLowerCase()
      : "";
    const multimodal = data._multimodal || null;

    // Multimodal insights take priority
    if (multimodal && multimodal.congruence_level) {
      if (multimodal.congruence_level === "LOW") {
        insight = `Emotional mismatch detected — face shows ${multimodal.vision_emotion}, voice shows ${multimodal.audio_emotion}. Client may be masking emotions.`;
        suggestion = "Gently check in on how they're really feeling.";
        risk_level = "HIGH";
      } else if (multimodal.congruence_level === "MEDIUM") {
        insight = `Mixed emotional signals — face (${multimodal.vision_emotion}) and voice (${multimodal.audio_emotion}) partially misaligned.`;
        suggestion = "Monitor for developing emotional patterns.";
        risk_level = "MEDIUM";
      } else if (multimodal.fused_emotion === "happy" || multimodal.fused_emotion === "calm") {
        insight = `Client appears genuinely ${multimodal.fused_emotion} — face and voice in agreement.`;
        suggestion = "Good rapport. Continue current approach.";
        risk_level = "LOW";
      } else {
        insight = `Fused emotional state: ${multimodal.fused_emotion}. Both modalities are congruent.`;
        suggestion = "";
        risk_level = ["angry", "fear", "disgust"].includes(multimodal.fused_emotion) ? "MEDIUM" : "LOW";
      }
    }
    // Audio-only fallback
    else if (audioEmotion && audioEmotion !== "neutral") {
      if (["angry", "fear", "disgust"].includes(audioEmotion)) {
        insight = `Vocal signals indicate ${audioEmotion}. Visual signals show ${faceEmotion}.`;
        suggestion = "Consider pausing or changing approach.";
        risk_level = "MEDIUM";
      } else if (audioEmotion === "sad") {
        insight = "Voice shows signs of sadness.";
        suggestion = "Prioritize emotional support.";
        risk_level = "MEDIUM";
      } else if (audioEmotion === "happy") {
        insight = "Positive vocal tone detected.";
        suggestion = "";
        risk_level = "LOW";
      }
    }
    // Vision-only fallback
    else if (faceEmotion === "sad" || faceEmotion === "fear") {
      insight = "User is displaying signs of sadness or fear.";
      suggestion = "Prioritize emotional support.";
      risk_level = "HIGH";
    } else if (faceEmotion === "angry") {
      insight = "User appears frustrated or angry.";
      suggestion = "Consider pausing challenging activities.";
      risk_level = "MEDIUM";
    } else {
      insight = "No significant behavioral anomalies detected.";
      suggestion = "";
      risk_level = "LOW";
    }

    return {
      insight,
      suggestion,
      risk_level,
      source: `rule_based_${reason}`,
      model: null,
      processing_time_ms: 0,
    };
  }

  // ── HTTP HELPERS ─────────────────────────────────────────────────

  _callOllama(prompt) {
    return new Promise((resolve, reject) => {
      const url = new URL(`${this.baseUrl}/api/generate`);
      const payload = JSON.stringify({
        model: this.model,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.3,
          num_predict: 256,
        },
      });

      const options = {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
        timeout: this.timeout,
      };

      const req = http.request(options, (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(new Error("Invalid JSON from Ollama"));
          }
        });
      });

      req.on("error", reject);
      req.on("timeout", () => {
        req.destroy();
        reject(new Error("Ollama request timeout"));
      });

      req.write(payload);
      req.end();
    });
  }

  _httpGet(url) {
    return new Promise((resolve, reject) => {
      const parsed = new URL(url);
      const req = http.get(
        { hostname: parsed.hostname, port: parsed.port, path: parsed.pathname, timeout: 3000 },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => {
            try {
              resolve(JSON.parse(data));
            } catch (e) {
              resolve({});
            }
          });
        }
      );
      req.on("error", reject);
      req.on("timeout", () => {
        req.destroy();
        reject(new Error("timeout"));
      });
    });
  }
}

module.exports = { LLMEngine };
