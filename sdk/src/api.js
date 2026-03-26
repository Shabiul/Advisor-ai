/**
 * Trusted Advisor SDK — API Client
 * ==================================
 * Handles HTTP communication with the Trusted Advisor AI backend.
 */

class ApiClient {
  constructor(baseUrl = "http://localhost:3000") {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  async analyze(mode, report) {
    const url = `${this.baseUrl}/analyze`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, data: report }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`[ApiClient] ${response.status}: ${text}`);
    }

    return response.json();
  }
}

module.exports = { ApiClient };
