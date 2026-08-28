// Browser side: upload PDFs, ask questions, show answers with their sources.

const $ = (id) => document.getElementById(id);

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Something went wrong.");
  return data;
}

async function checkHealth() {
  const status = $("status");
  const data = await api("/api/health").catch(() => ({ ok: false, error: "Server not reachable." }));
  status.className = "status " + (data.ok ? "ok" : "err");
  status.textContent = data.ok ? `Ready · ${data.chat_model} + ${data.embed_model}` : data.error;
}

async function loadDocuments() {
  const docs = await api("/api/documents");
  $("documents").innerHTML = docs.length
    ? docs.map((d) => `<li><span>${escapeHtml(d.source)}<br><small>${d.chunks} chunks</small></span>
        <button title="Remove" data-source="${escapeHtml(d.source)}">×</button></li>`).join("")
    : "<li><small>No documents yet.</small></li>";
}

$("documents").addEventListener("click", async (event) => {
  const source = event.target.dataset.source;
  if (!source) return;
  await api("/api/documents/" + encodeURIComponent(source), { method: "DELETE" });
  loadDocuments();
});

$("file").addEventListener("change", async () => {
  const file = $("file").files[0];
  if (!file) return;
  const label = $("upload-label");
  label.textContent = `Indexing ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const info = await api("/api/upload", { method: "POST", body: form });
    label.textContent = `Added ${info.source} (${info.pages} pages)`;
    loadDocuments();
  } catch (error) {
    label.textContent = error.message;
  }
  $("file").value = "";
});

function addMessage(className, html) {
  const div = document.createElement("div");
  div.className = className;
  div.innerHTML = html;
  $("messages").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
  return div;
}

function renderAnswer(result) {
  // turn [1] into a highlighted citation marker
  const answer = escapeHtml(result.answer).replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
  const badge = result.grounded
    ? '<span class="badge ok">Grounded in your PDF</span>'
    : '<span class="badge warn">Not found in your PDF</span>';
  const sources = result.sources.map((s) => `
    <div class="source ${result.cited.includes(s.n) ? "cited" : ""}">
      <b>[${s.n}] ${escapeHtml(s.source)} · page ${s.page} · similarity ${s.similarity}</b>
      ${escapeHtml(s.text)}
    </div>`).join("");
  return `<p>${answer}</p>${badge}` +
    (sources ? `<details><summary>Sources (${result.sources.length})</summary>${sources}</details>` : "");
}

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  $("messages").querySelector(".hint")?.remove();
  addMessage("question", escapeHtml(question));
  const reply = addMessage("answer pending", "<p>Thinking…</p>");
  const button = $("ask-form").querySelector("button");
  button.disabled = true;
  try {
    const result = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    reply.className = "answer";
    reply.innerHTML = renderAnswer(result);
  } catch (error) {
    reply.className = "answer";
    reply.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
  }
  button.disabled = false;
});

checkHealth();
loadDocuments();
