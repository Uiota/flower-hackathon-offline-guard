class MCPModelContext {
  constructor(nodeId, sharedSecret, baseUrl = "http://127.0.0.1:9090/api/v1") {
    this.nodeId = nodeId;
    this.secret = sharedSecret;
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  setConfig({ nodeId, sharedSecret, baseUrl }) {
    if (nodeId !== undefined) this.nodeId = nodeId;
    if (sharedSecret !== undefined) this.secret = sharedSecret;
    if (baseUrl !== undefined) this.baseUrl = String(baseUrl).replace(/\/$/, "");
  }

  async getEnv() {
    this.#ensureConfig();
    const url = `${this.baseUrl}/env/${encodeURIComponent(this.nodeId)}`;
    const res = await fetch(url, {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${this.secret}`,
        "Accept": "application/json"
      }
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`GET ${url} failed: ${res.status} ${res.statusText} ${text}`.trim());
    }
    return res.json();
  }

  async reportStatus({ metrics, epistemicState }) {
    this.#ensureConfig();
    const url = `${this.baseUrl}/status`;
    const payload = {
      node_id: this.nodeId,
      model_metrics: metrics || {},
      epistemic_state: epistemicState || {}
    };
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.secret}`,
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`POST ${url} failed: ${res.status} ${res.statusText} ${text}`.trim());
    }
    return res.json().catch(() => ({}));
  }

  #ensureConfig() {
    if (!this.nodeId) throw new Error("nodeId is required");
    if (!this.secret) throw new Error("sharedSecret is required");
    if (!this.baseUrl) throw new Error("baseUrl is required");
  }
}

// Minimal helper for the demo UI
window.MCPModelContext = MCPModelContext;

