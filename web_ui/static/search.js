(function () {
  "use strict";

  const METRICS = [
    { id: "rank", label: "Rank" },
    { id: "score", label: "Model score" },
    { id: "doc_id", label: "Document ID" },
    { id: "doc_tokens", label: "Doc length (tokens)" },
    { id: "length_ratio", label: "Tokens / avg doc length" },
    { id: "term_coverage", label: "Query term coverage" },
    { id: "matched_terms", label: "Matched query terms" },
    { id: "query_terms", label: "Query terms (count)" },
    { id: "relevant", label: "Relevant (qrels)" },
    { id: "precision_at_rank", label: "P@rank" },
    { id: "recall_at_rank", label: "Recall@rank" },
    { id: "dcg_increment", label: "DCG increment" },
  ];

  const LS_THEME = "ir_ui_theme";
  const LS_METRICS = "ir_ui_metrics";
  const LS_MODEL = "ir_ui_model";

  const els = {
    form: document.getElementById("searchForm"),
    q: document.getElementById("q"),
    queryId: document.getElementById("queryId"),
    statusBar: document.getElementById("statusBar"),
    resultsWrap: document.getElementById("resultsWrap"),
    resultsMeta: document.getElementById("resultsMeta"),
    resultsList: document.getElementById("resultsList"),
    pager: document.getElementById("pager"),
    prevPage: document.getElementById("prevPage"),
    nextPage: document.getElementById("nextPage"),
    pageInfo: document.getElementById("pageInfo"),
    metricToggles: document.getElementById("metricToggles"),
    modelList: document.getElementById("modelList"),
    themeToggle: document.getElementById("themeToggle"),
    landingHint: document.getElementById("landingHint"),
    submitBtn: document.getElementById("submitBtn"),
    articleRoot: document.getElementById("articleRoot"),
    articleBackdrop: document.getElementById("articleBackdrop"),
    articleClose: document.getElementById("articleClose"),
    articleTitle: document.getElementById("articleTitle"),
    articleMeta: document.getElementById("articleMeta"),
    articleBody: document.getElementById("articleBody"),
    articleDocId: document.getElementById("articleDocId"),
  };

  let models = [];
  let selectedModel = "bm25";
  let page = 1;
  let lastPayload = null;
  let articleOpen = false;
  let articlePrevFocus = null;
  let articleCloseTimer = null;

  function loadTheme() {
    const t = localStorage.getItem(LS_THEME) || "light";
    document.documentElement.setAttribute("data-theme", t);
    const icon = els.themeToggle.querySelector(".theme-fab__icon");
    icon.textContent = t === "dark" ? "☾" : "☀";
  }

  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    const next = cur === "light" ? "dark" : "light";
    localStorage.setItem(LS_THEME, next);
    loadTheme();
  }

  function loadMetricState() {
    try {
      const raw = localStorage.getItem(LS_METRICS);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  function saveMetricState(enabledSet) {
    localStorage.setItem(LS_METRICS, JSON.stringify([...enabledSet]));
  }

  function applyMetricClasses(enabledSet) {
    const body = document.body;
    METRICS.forEach((m) => {
      body.classList.toggle(`hide-m-${m.id}`, !enabledSet.has(m.id));
    });
  }

  function buildMetricToggles() {
    const saved = loadMetricState();
    const enabled = new Set(
      saved && saved.length
        ? saved
        : METRICS.map((m) => m.id)
    );

    els.metricToggles.innerHTML = "";
    METRICS.forEach((m) => {
      const label = document.createElement("label");
      label.className = "toggle-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = enabled.has(m.id);
      cb.dataset.metric = m.id;
      cb.addEventListener("change", () => {
        if (cb.checked) enabled.add(m.id);
        else enabled.delete(m.id);
        saveMetricState(enabled);
        applyMetricClasses(enabled);
      });
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" " + m.label));
      els.metricToggles.appendChild(label);
    });
    applyMetricClasses(enabled);
  }

  function setModel(m) {
    selectedModel = m;
    localStorage.setItem(LS_MODEL, m);
    document.querySelectorAll(".model-card").forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.model === m);
    });
  }

  function renderModels() {
    els.modelList.innerHTML = "";
    const stored = localStorage.getItem(LS_MODEL);
    let initial = models.find((x) => x.id === stored)?.available
      ? stored
      : null;
    if (!initial) {
      const def = models.find((x) => x.id === "bm25");
      initial = def && def.available ? "bm25" : models.find((x) => x.available)?.id;
    }
    models.forEach((mod, idx) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "model-card";
      card.dataset.model = mod.id;
      card.style.animationDelay = `${idx * 0.05}s`;
      if (!mod.available) card.classList.add("is-disabled");
      card.innerHTML =
        `<div class="model-card__label">${escapeHtml(mod.label)}</div>` +
        `<div class="model-card__id">${escapeHtml(mod.id)}</div>`;
      card.addEventListener("click", () => {
        if (!mod.available) {
          setStatus("Bill BM25 needs bill_index.pkl — run python bill_index_builder.py", true);
          return;
        }
        setModel(mod.id);
      });
      els.modelList.appendChild(card);
    });
    if (models.some((x) => x.id === initial && x.available)) {
      setModel(initial);
    } else {
      setModel(models.find((x) => x.available).id);
    }
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function fmt(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (typeof v === "number" && !Number.isFinite(v)) return "—";
    return String(v);
  }

  function setStatus(msg, isError) {
    if (!msg) {
      els.statusBar.hidden = true;
      els.statusBar.textContent = "";
      return;
    }
    els.statusBar.hidden = false;
    els.statusBar.textContent = msg;
  }

  function metricRows(m) {
    return METRICS.map(
      (def) =>
        `<div class="metric-row" data-key="${def.id}">` +
        `<span class="metric-row__k">${escapeHtml(def.label)}</span>` +
        `<span class="metric-row__v">${escapeHtml(fmt(m[def.id]))}</span>` +
        `</div>`
    ).join("");
  }

  function fillArticleParagraphs(text) {
    const body = els.articleBody;
    body.innerHTML = "";
    let chunks = text
      .split(/\n\n+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (chunks.length <= 1) {
      chunks = text
        .split(/\n/)
        .map((s) => s.trim())
        .filter(Boolean);
    }
    if (chunks.length === 0) {
      const p = document.createElement("p");
      p.className = "article-para";
      p.textContent = "(Empty document.)";
      body.appendChild(p);
      return;
    }
    chunks.forEach((chunk) => {
      const p = document.createElement("p");
      p.className = "article-para";
      p.textContent = chunk;
      body.appendChild(p);
    });
  }

  function closeArticle() {
    if (!articleOpen) return;
    articleOpen = false;
    els.articleRoot.classList.remove("is-open");
    document.body.classList.remove("article-modal-open");
    const t = els.articleRoot;
    if (articleCloseTimer) window.clearTimeout(articleCloseTimer);
    articleCloseTimer = window.setTimeout(() => {
      articleCloseTimer = null;
      if (!articleOpen) {
        t.hidden = true;
        t.setAttribute("aria-hidden", "true");
      }
    }, 420);
    if (articlePrevFocus && typeof articlePrevFocus.focus === "function") {
      articlePrevFocus.focus({ preventScroll: true });
    }
    articlePrevFocus = null;
  }

  async function openArticle(docId) {
    if (articleCloseTimer) {
      window.clearTimeout(articleCloseTimer);
      articleCloseTimer = null;
    }
    articlePrevFocus = document.activeElement;
    els.articleRoot.hidden = false;
    els.articleRoot.setAttribute("aria-hidden", "false");
    document.body.classList.add("article-modal-open");
    els.articleTitle.textContent = "Opening…";
    els.articleTitle.classList.add("is-loading");
    els.articleMeta.textContent = "";
    els.articleBody.innerHTML = "";
    els.articleDocId.textContent = docId;
    requestAnimationFrame(() => {
      els.articleRoot.classList.add("is-open");
    });
    els.articleClose.focus({ preventScroll: true });
    articleOpen = true;

    try {
      const res = await fetch(
        "/api/document/" + encodeURIComponent(docId)
      );
      const data = await res.json();
      if (!res.ok) {
        els.articleTitle.classList.remove("is-loading");
        els.articleTitle.textContent = "Document not found";
        els.articleBody.innerHTML = "";
        const p = document.createElement("p");
        p.className = "article-para";
        p.textContent = "This document id is not in the corpus.";
        els.articleBody.appendChild(p);
        return;
      }
      els.articleTitle.classList.remove("is-loading");
      els.articleTitle.textContent = data.title || data.doc_id;
      const wc = (data.char_count || 0).toLocaleString();
      els.articleMeta.textContent =
        `${wc} characters · id ${data.doc_id}`;
      fillArticleParagraphs(data.text || "");
    } catch {
      els.articleTitle.classList.remove("is-loading");
      els.articleTitle.textContent = "Could not load";
      els.articleBody.innerHTML = "";
      const p = document.createElement("p");
      p.className = "article-para";
      p.textContent = "Network error while loading the document.";
      els.articleBody.appendChild(p);
    }
  }

  function renderResults(data) {
    lastPayload = data;
    els.resultsList.innerHTML = "";
    data.results.forEach((r, i) => {
      const li = document.createElement("li");
      li.className = "hit";
      li.style.animationDelay = `${i * 0.04}s`;
      const a = document.createElement("a");
      a.className = "hit__link";
      a.href = "#";
      a.textContent = r.title;
      a.dataset.docId = r.doc_id;
      a.addEventListener("click", (e) => {
        e.preventDefault();
        openArticle(r.doc_id);
      });
      const url = document.createElement("button");
      url.type = "button";
      url.className = "hit__url hit__url--btn";
      url.textContent = r.doc_id;
      url.title = "Open full document";
      url.addEventListener("click", () => openArticle(r.doc_id));
      const sn = document.createElement("div");
      sn.className = "hit__snippet";
      sn.textContent = r.snippet;
      const main = document.createElement("div");
      main.className = "hit__main";
      main.appendChild(a);
      main.appendChild(url);
      main.appendChild(sn);
      const metrics = document.createElement("div");
      metrics.className = "hit__metrics";
      metrics.innerHTML = metricRows(r.metrics);
      li.appendChild(main);
      li.appendChild(metrics);
      els.resultsList.appendChild(li);
    });

    const qrelsNote = data.qrels_active
      ? ` · qrels: ${data.total_relevant_in_qrels} relevant`
      : "";
    els.resultsMeta.textContent =
      `About ${data.depth} ranked · page ${data.page} (${data.returned} shown)${qrelsNote}`;

    els.pager.hidden = data.page <= 1 && !data.has_more;
    els.prevPage.disabled = data.page <= 1;
    els.nextPage.disabled = !data.has_more;
    els.pageInfo.textContent = `Page ${data.page} · ${data.page_size} per page`;
  }

  async function fetchModels() {
    const res = await fetch("/api/models");
    const data = await res.json();
    models = data.models || [];
    renderModels();
  }

  async function doSearch(resetPage) {
    if (resetPage) page = 1;
    const q = els.q.value.trim();
    if (!q) {
      setStatus("Enter a search query.", true);
      return;
    }
    setStatus("");
    els.submitBtn.disabled = true;
    try {
      const body = {
        q,
        model: selectedModel,
        page,
        page_size: 5,
      };
      const qid = els.queryId.value.trim();
      if (qid) body.query_id = qid;

      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.error === "bill_index_missing") {
          setStatus("Bill index missing. Run: python bill_index_builder.py", true);
        } else {
          setStatus(data.error || "Search failed.", true);
        }
        return;
      }
      els.landingHint.hidden = true;
      els.resultsWrap.hidden = false;
      renderResults(data);
    } catch (e) {
      setStatus("Network error — is the server running?", true);
    } finally {
      els.submitBtn.disabled = false;
    }
  }

  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    doSearch(true);
  });

  els.prevPage.addEventListener("click", () => {
    if (page > 1) {
      page -= 1;
      doSearch(false);
    }
  });

  els.nextPage.addEventListener("click", () => {
    if (lastPayload && lastPayload.has_more) {
      page += 1;
      doSearch(false);
    }
  });

  els.themeToggle.addEventListener("click", toggleTheme);

  els.articleBackdrop.addEventListener("click", closeArticle);
  els.articleClose.addEventListener("click", closeArticle);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && articleOpen) closeArticle();
  });

  loadTheme();
  buildMetricToggles();
  fetchModels().catch(() => {
    setStatus("Could not load models.", true);
  });
})();
