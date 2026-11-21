/* SignalChurn frontend: renders model metadata and scores the form via /predict. */

const NUMERIC_FIELDS = new Set([
  "tenure",
  "MonthlyCharges",
  "TotalCharges",
  "SeniorCitizen",
]);

const form = document.getElementById("churn-form");
const submitBtn = document.getElementById("submit-btn");
const errorBox = document.getElementById("form-error");
const resultCard = document.getElementById("result-card");
const resultEmpty = document.getElementById("result-empty");
const resultBody = document.getElementById("result-body");
const resultSkeleton = document.getElementById("result-skeleton");
const driversList = document.getElementById("drivers");

let lastResult = null;
let meta = null;

const EXAMPLES = {
  low: {
    tenure: 68,
    Contract: "Two year",
    PaperlessBilling: "No",
    PaymentMethod: "Bank transfer (automatic)",
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "Yes",
    Dependents: "Yes",
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "No",
    StreamingMovies: "No",
    MonthlyCharges: 24.35,
    TotalCharges: 1653.5,
  },
  high: {
    tenure: 2,
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    MonthlyCharges: 95.5,
    TotalCharges: 191.0,
  },
};

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function renderModelInfo(info) {
  meta = info;
  const chip = document.getElementById("model-chip");
  document.getElementById("model-chip-name").textContent = info.model_name;
  chip.hidden = false;

  const metrics = info.metrics || {};
  document.getElementById("meta-auc").textContent =
    metrics.roc_auc != null ? metrics.roc_auc.toFixed(3) : "—";
  document.getElementById("meta-f1").textContent =
    metrics.f1 != null ? metrics.f1.toFixed(3) : "—";
  document.getElementById("meta-threshold").textContent =
    info.threshold != null ? info.threshold.toFixed(2) : "—";
  document.getElementById("meta-rows").textContent =
    info.train_rows != null ? info.train_rows.toLocaleString() : "—";
  document.getElementById("meta-row").hidden = false;

  renderDrivers(info.top_features || []);
}

function renderDrivers(features) {
  if (!features.length) {
    driversList.innerHTML = '<li class="drivers-loading">No importance data available.</li>';
    return;
  }
  const max = Math.max(...features.map((f) => f.importance)) || 1;
  driversList.innerHTML = features
    .slice(0, 7)
    .map((f) => {
      const width = Math.max(4, (f.importance / max) * 100);
      const dir = f.direction
        ? `<span class="driver-dir">${f.direction === "increases churn" ? "↑ churn" : "↓ churn"}</span>`
        : "";
      return `
        <li class="driver">
          <div class="driver-head">
            <span class="driver-name" title="${escapeHtml(f.feature)}">${escapeHtml(f.feature)}</span>
            ${dir}
          </div>
          <div class="driver-bar"><span style="width:${width}%"></span></div>
        </li>`;
    })
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderResult(result) {
  lastResult = result;
  resultEmpty.hidden = true;
  resultSkeleton.hidden = true;
  resultBody.hidden = false;
  resultCard.dataset.risk = result.risk_level;

  const probabilityPct = result.churn_probability * 100;
  document.getElementById("risk-badge").textContent = result.risk_label;
  document.getElementById("prob-value").textContent = probabilityPct.toFixed(1);
  document.getElementById("meter-fill").style.width = `${Math.min(100, Math.max(0, probabilityPct))}%`;

  const thresholdMark = document.getElementById("meter-threshold");
  thresholdMark.style.left = `calc(${Math.min(100, result.threshold * 100)}% - 1px)`;
  thresholdMark.title = `decision threshold ${result.threshold.toFixed(2)}`;

  const verdict = document.getElementById("verdict");
  verdict.textContent = result.churn_prediction
    ? "Predicted to churn: above the decision threshold."
    : "Predicted to stay: below the decision threshold.";

  document.getElementById("threshold-note").textContent =
    `Threshold ${result.threshold.toFixed(2)} tuned on out-of-fold CV to maximise F1. ` +
    `Model: ${result.model_name}.`;

  if (result.risk_level === "high") {
    verdict.textContent += " Prioritise retention outreach.";
  } else if (result.risk_level === "low") {
    verdict.textContent += " No immediate retention action needed.";
  }
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.querySelector(".btn-label").textContent = isLoading ? "Scoring…" : "Score churn risk";
  if (isLoading) {
    resultEmpty.hidden = true;
    resultBody.hidden = true;
    resultSkeleton.hidden = false;
  }
}

function collectPayload() {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    payload[key] = NUMERIC_FIELDS.has(key) ? Number(value) : value;
  }
  return payload;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  if (!form.checkValidity()) {
    showError("Please fill in every field with a valid value.");
    form.reportValidity();
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const body = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = body.detail;
      const message = Array.isArray(detail)
        ? detail.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join(" · ")
        : detail || `Request failed with status ${response.status}.`;
      throw new Error(message);
    }
    renderResult(body);
  } catch (error) {
    resultSkeleton.hidden = true;
    resultEmpty.hidden = false;
    resultCard.dataset.risk = "none";
    showError(error.message || "Something went wrong while scoring this customer.");
  } finally {
    setLoading(false);
  }
});

function fillExample(kind) {
  const example = EXAMPLES[kind];
  for (const [key, value] of Object.entries(example)) {
    const field = form.elements.namedItem(key);
    if (field) field.value = value;
  }
  clearError();
  form.querySelector("input, select")?.focus();
}

document.getElementById("fill-low").addEventListener("click", () => fillExample("low"));
document.getElementById("fill-high").addEventListener("click", () => fillExample("high"));

document.getElementById("copy-result").addEventListener("click", async () => {
  if (!lastResult) return;
  const payload = { ...lastResult, input: collectPayload() };
  try {
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    const btn = document.getElementById("copy-result");
    btn.textContent = "Copied ✓";
    setTimeout(() => (btn.textContent = "Copy result as JSON"), 1600);
  } catch {
    showError("Clipboard is not available in this browser context.");
  }
});

async function init() {
  try {
    const response = await fetch("/model-info");
    if (!response.ok) throw new Error("model info unavailable");
    renderModelInfo(await response.json());
  } catch {
    document.getElementById("model-chip-name").textContent = "Model unavailable";
    document.getElementById("model-chip").hidden = false;
    driversList.innerHTML = '<li class="drivers-loading">Model metrics unavailable.</li>';
  }
}

init();
