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
  const PAGE_SIZE = 5;
  const PAGER_MAX_PAGES = 5;
  const SUGGEST_DEBOUNCE_MS = 220;
  const SUGGEST_MIN_CHARS = 2;

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
    pageNums: document.getElementById("pageNums"),
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
    suggestDropdown: document.getElementById("suggestDropdown"),
    explainRoot: document.getElementById("explainRoot"),
    explainBackdrop: document.getElementById("explainBackdrop"),
    explainClose: document.getElementById("explainClose"),
    explainTitle: document.getElementById("explainTitle"),
    explainBody: document.getElementById("explainBody"),
  };

  let closeMobileDrawers = () => {};

  let models = [];
  let selectedModel = "bm25";
  let page = 1;
  let lastPayload = null;
  let articleOpen = false;
  let articlePrevFocus = null;
  let articleCloseTimer = null;

  let suggestTimer = null;
  let suggestAbort = null;
  let suggestBlurTimer = null;
  let suggestHighlight = -1;

  let explainOpen = false;
  let explainCloseTimer = null;

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
          setStatus("That ranking model is not available.", true);
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

  function renderExplainFormulaLatex(mountEl, latex) {
    if (!latex || !mountEl) return;
    try {
      if (typeof katex !== "undefined") {
        katex.render(latex, mountEl, {
          displayMode: true,
          throwOnError: false,
          strict: "ignore",
          trust: false,
        });
        return;
      }
    } catch {
      /* fall through */
    }
    mountEl.classList.add("explain-formula-latex-fallback");
    mountEl.textContent = latex;
  }

  function fmt(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (typeof v === "number" && !Number.isFinite(v)) return "—";
    return String(v);
  }

  function fmtExplainNum(x) {
    if (x === null || x === undefined) return "—";
    if (typeof x !== "number" || !Number.isFinite(x)) return escapeHtml(String(x));
    const ax = Math.abs(x);
    const decimals = ax >= 100 ? 2 : ax >= 1 ? 4 : 6;
    return escapeHtml(x.toFixed(decimals));
  }

  function closeExplain() {
    if (!explainOpen) return;
    explainOpen = false;
    els.explainRoot.classList.remove("is-open");
    document.body.classList.remove("explain-modal-open");
    if (explainCloseTimer) window.clearTimeout(explainCloseTimer);
    explainCloseTimer = window.setTimeout(() => {
      explainCloseTimer = null;
      if (!explainOpen) {
        els.explainRoot.hidden = true;
        els.explainRoot.setAttribute("aria-hidden", "true");
        els.explainBody.innerHTML = "";
      }
    }, 380);
  }

  function renderExplainKv(card, row) {
    const skip = new Set(["term", "status", "note"]);
    const dl = document.createElement("dl");
    dl.className = "explain-kv";
    Object.keys(row).forEach((k) => {
      if (skip.has(k)) return;
      const v = row[k];
      if (v === undefined) return;
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.innerHTML =
        typeof v === "number" ? fmtExplainNum(v) : escapeHtml(String(v));
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
    if (row.note) {
      const dt = document.createElement("dt");
      dt.textContent = "note";
      const dd = document.createElement("dd");
      dd.textContent = row.note;
      dl.appendChild(dt);
      dl.appendChild(dd);
    }
    card.appendChild(dl);
  }

  function appendExplainVisualization(container, data, bumpAnim) {
    const qSeg = data.query_segments;
    const dSeg = data.document_segments;
    if ((!qSeg || !qSeg.length) && (!dSeg || !dSeg.length)) return;

    const ql = document.createElement("div");
    ql.className = "explain-section-label explain-animate-in";
    ql.style.setProperty("--explain-i", String(bumpAnim()));
    ql.textContent = "Your query";
    container.appendChild(ql);

    const qh = document.createElement("p");
    qh.className = "explain-visual-hint explain-animate-in";
    qh.style.setProperty("--explain-i", String(bumpAnim()));
    qh.textContent =
      "Full query string as you typed it: every token is emphasized. Stronger styling marks stems that occur in this document and feed the score.";
    container.appendChild(qh);

    const qVis = document.createElement("div");
    qVis.className = "explain-query-visual explain-animate-in";
    qVis.style.setProperty("--explain-i", String(bumpAnim()));
    (qSeg || []).forEach((seg) => {
      if (!seg.word) {
        qVis.appendChild(document.createTextNode(seg.text));
        return;
      }
      const sp = document.createElement("span");
      sp.className = "explain-query-word explain-query-word--lit";
      if (seg.contributing) sp.classList.add("explain-query-word--contributing");
      if (!seg.indexed) sp.classList.add("explain-query-word--nostem");
      sp.textContent = seg.text;
      const stemHint = seg.stem ? `Stem: ${seg.stem}` : "Stopped word — not indexed";
      sp.title = seg.contributing ? `${stemHint} · contributes here` : stemHint;
      qVis.appendChild(sp);
    });
    container.appendChild(qVis);

    const dl = document.createElement("div");
    dl.className = "explain-section-label explain-animate-in";
    dl.style.setProperty("--explain-i", String(bumpAnim()));
    dl.textContent =
      data.document_truncated === true
        ? "Indexed document text (truncated for speed)"
        : "Indexed document text";
    container.appendChild(dl);

    const dh = document.createElement("p");
    dh.className = "explain-visual-hint explain-animate-in";
    dh.style.setProperty("--explain-i", String(bumpAnim()));
    dh.textContent =
      "Yellow highlights mark tokens whose stems match contributing query terms for this ranker.";
    container.appendChild(dh);

    const shell = document.createElement("div");
    shell.className = "explain-document-shell explain-animate-in";
    shell.style.setProperty("--explain-i", String(bumpAnim()));
    const inner = document.createElement("div");
    inner.className = "explain-document-body";
    let docTok = 0;
    (dSeg || []).forEach((seg) => {
      if (!seg.word) {
        inner.appendChild(document.createTextNode(seg.text));
        return;
      }
      const sp = document.createElement("span");
      sp.className = "explain-doc-word";
      const idx = docTok++;
      sp.style.setProperty("--doc-i", String(Math.min(idx, 48)));
      if (seg.hit) sp.classList.add("explain-doc-word--hit");
      sp.textContent = seg.text;
      inner.appendChild(sp);
    });
    shell.appendChild(inner);
    container.appendChild(shell);
  }

  function renderExplainPayload(data) {
    const body = els.explainBody;
    body.innerHTML = "";
    let animIdx = 0;
    const bumpAnim = () => animIdx++;

    if (data.error && data.error !== undefined) {
      els.explainTitle.textContent = "Explanation unavailable";
      const b = document.createElement("div");
      b.className = "explain-banner explain-animate-in";
      b.style.setProperty("--explain-i", String(bumpAnim()));
      b.textContent =
        data.error === "unknown_doc"
          ? "Document is not in the indexed corpus."
          : String(data.error);
      body.appendChild(b);
      return;
    }

    if (data.unsupported_detail) {
      els.explainTitle.textContent = data.title || "Explanation";
      const b = document.createElement("div");
      b.className = "explain-banner explain-animate-in";
      b.style.setProperty("--explain-i", String(bumpAnim()));
      b.textContent = data.message || "";
      body.appendChild(b);
      appendExplainVisualization(body, data, bumpAnim);
      return;
    }

    els.explainTitle.textContent = data.title || data.model || "Ranking explanation";

    const scoreRow = document.createElement("div");
    scoreRow.className = "explain-score-row explain-animate-in";
    scoreRow.style.setProperty("--explain-i", String(bumpAnim()));
    const sid = document.createElement("span");
    sid.className = "explain-doc-id";
    sid.textContent = data.doc_id || "";
    const sc = document.createElement("span");
    sc.className = "explain-score-val explain-score-pulse";
    sc.innerHTML = fmtExplainNum(data.total_score);
    scoreRow.appendChild(sid);
    scoreRow.appendChild(sc);
    body.appendChild(scoreRow);

    appendExplainVisualization(body, data, bumpAnim);

    const formula = data.formula || {};
    if (formula.name) {
      const fb = document.createElement("div");
      fb.className = "explain-formula-block explain-animate-in";
      fb.style.setProperty("--explain-i", String(bumpAnim()));
      const fname = document.createElement("div");
      fname.className = "explain-formula-name";
      fname.textContent = formula.name;
      fb.appendChild(fname);
      if (formula.latex) {
        const mount = document.createElement("div");
        mount.className = "explain-formula-katex";
        mount.setAttribute("role", "math");
        fb.appendChild(mount);
        renderExplainFormulaLatex(mount, formula.latex);
      }
      body.appendChild(fb);
      if (formula.steps && formula.steps.length) {
        const ul = document.createElement("ul");
        ul.className = "explain-steps explain-animate-in";
        ul.style.setProperty("--explain-i", String(bumpAnim()));
        formula.steps.forEach((s) => {
          const li = document.createElement("li");
          li.textContent = s;
          ul.appendChild(li);
        });
        body.appendChild(ul);
      }
    }

    const constants = data.constants;
    if (constants && Object.keys(constants).length) {
      const lbl = document.createElement("div");
      lbl.className = "explain-section-label explain-animate-in";
      lbl.style.setProperty("--explain-i", String(bumpAnim()));
      lbl.textContent = "Parameters & corpus stats";
      body.appendChild(lbl);
      const pre = document.createElement("pre");
      pre.className = "explain-json explain-animate-in";
      pre.style.setProperty("--explain-i", String(bumpAnim()));
      try {
        pre.textContent = JSON.stringify(constants, null, 2);
      } catch {
        pre.textContent = "";
      }
      body.appendChild(pre);
    }

    if (data.fusion && data.fusion.length) {
      const lbl = document.createElement("div");
      lbl.className = "explain-section-label explain-animate-in";
      lbl.style.setProperty("--explain-i", String(bumpAnim()));
      lbl.textContent = "RRF contributions";
      body.appendChild(lbl);
      const wrap = document.createElement("div");
      wrap.className = "explain-animate-in";
      wrap.style.setProperty("--explain-i", String(bumpAnim()));
      const table = document.createElement("table");
      table.className = "explain-fusion-table";
      table.innerHTML =
        "<thead><tr><th>Ranker</th><th>Rank</th><th>Raw score</th><th>1/(k+r)</th></tr></thead>";
      const tb = document.createElement("tbody");
      data.fusion.forEach((row) => {
        const tr = document.createElement("tr");
        const rk =
          row.rank === null || row.rank === undefined ? "—" : String(row.rank);
        tr.innerHTML =
          `<td>${escapeHtml(row.ranker)}</td>` +
          `<td>${escapeHtml(rk)}</td>` +
          `<td>${fmtExplainNum(row.raw_score)}</td>` +
          `<td>${fmtExplainNum(row.rrf_component)}</td>`;
        tb.appendChild(tr);
      });
      table.appendChild(tb);
      wrap.appendChild(table);
      body.appendChild(wrap);
    }

    if (data.feedback_docs && data.feedback_docs.length) {
      const lbl = document.createElement("div");
      lbl.className = "explain-section-label explain-animate-in";
      lbl.style.setProperty("--explain-i", String(bumpAnim()));
      lbl.textContent = "RM3 feedback documents";
      body.appendChild(lbl);
      const pre = document.createElement("pre");
      pre.className = "explain-json explain-animate-in";
      pre.style.setProperty("--explain-i", String(bumpAnim()));
      pre.textContent = JSON.stringify(data.feedback_docs, null, 2);
      body.appendChild(pre);
    }

    if (data.expanded_terms_preview && data.expanded_terms_preview.length) {
      const lbl = document.createElement("div");
      lbl.className = "explain-section-label explain-animate-in";
      lbl.style.setProperty("--explain-i", String(bumpAnim()));
      lbl.textContent = "Rocchio expansion preview";
      body.appendChild(lbl);
      const pre = document.createElement("pre");
      pre.className = "explain-json explain-animate-in";
      pre.style.setProperty("--explain-i", String(bumpAnim()));
      pre.textContent = JSON.stringify(data.expanded_terms_preview, null, 2);
      body.appendChild(pre);
    }

    const terms = data.terms || [];
    if (terms.length && data.model !== "hybrid_rrf") {
      const lbl = document.createElement("div");
      lbl.className = "explain-section-label explain-animate-in";
      lbl.style.setProperty("--explain-i", String(bumpAnim()));
      lbl.textContent = "Per-term breakdown";
      body.appendChild(lbl);
      const grid = document.createElement("div");
      grid.className = "explain-term-grid";
      terms.forEach((row) => {
        const card = document.createElement("article");
        card.className =
          "explain-term-card explain-animate-in" +
          (row.status === "matched" ? " explain-term-card--matched" : "");
        card.style.setProperty("--explain-i", String(bumpAnim()));
        const head = document.createElement("div");
        head.className = "explain-term-head";
        const stem = document.createElement("span");
        stem.className = "explain-term-stem";
        stem.textContent = row.term || "(?)";
        const st = document.createElement("span");
        st.className =
          "explain-status " +
          (row.status === "matched"
            ? "explain-status--matched"
            : "explain-status--miss");
        st.textContent = (row.status || "").replace(/_/g, " ");
        head.appendChild(stem);
        head.appendChild(st);
        card.appendChild(head);
        renderExplainKv(card, row);
        grid.appendChild(card);
      });
      body.appendChild(grid);
    }
  }

  async function openExplain(docId) {
    const q = els.q.value.trim();
    if (!q) {
      setStatus("Enter a query before opening an explanation.", true);
      return;
    }
    if (!els.explainRoot || !els.explainBody) return;

    if (explainCloseTimer) {
      window.clearTimeout(explainCloseTimer);
      explainCloseTimer = null;
    }

    closeMobileDrawers();
    explainOpen = true;
    els.explainRoot.hidden = false;
    els.explainRoot.setAttribute("aria-hidden", "false");
    document.body.classList.add("explain-modal-open");
    els.explainTitle.textContent = "Loading…";
    els.explainBody.innerHTML = "";
    requestAnimationFrame(() => els.explainRoot.classList.add("is-open"));

    try {
      const res = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          q,
          doc_id: docId,
          model: selectedModel,
        }),
      });
      const ct = res.headers.get("content-type") || "";
      if (!ct.includes("application/json")) {
        els.explainTitle.textContent = "Invalid response";
        const b = document.createElement("div");
        b.className = "explain-banner";
        b.textContent = "Server returned non-JSON.";
        els.explainBody.appendChild(b);
        return;
      }
      const data = await res.json();
      if (!res.ok) {
        els.explainTitle.textContent = "Explanation unavailable";
        const b = document.createElement("div");
        b.className = "explain-banner";
        b.textContent = escapeHtml(data.error || "Request failed.");
        els.explainBody.appendChild(b);
        return;
      }
      renderExplainPayload(data);
    } catch {
      els.explainTitle.textContent = "Network error";
      const b = document.createElement("div");
      b.className = "explain-banner";
      b.textContent = "Could not reach /api/explain.";
      els.explainBody.appendChild(b);
    }

    els.explainClose.focus({ preventScroll: true });
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
    closeMobileDrawers();
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

  function renderPageNums(data) {
    if (!els.pageNums) return;
    els.pageNums.innerHTML = "";
    const total = Math.max(1, data.pager_total_pages || 1);
    const cur = data.page || 1;
    for (let p = 1; p <= total; p++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pager__num" + (p === cur ? " is-current" : "");
      btn.textContent = String(p);
      btn.setAttribute("aria-label", `Page ${p}`);
      if (p === cur) {
        btn.disabled = true;
        btn.setAttribute("aria-current", "page");
      } else {
        btn.addEventListener("click", () => {
          page = p;
          doSearch(false);
        });
      }
      els.pageNums.appendChild(btn);
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
      const explainBtn = document.createElement("button");
      explainBtn.type = "button";
      explainBtn.className = "hit__explain-btn";
      explainBtn.textContent = "Explain ranking";
      explainBtn.title =
        "Show formulas and token contributions for this hit with the selected model";
      explainBtn.addEventListener("click", () => openExplain(r.doc_id));
      const titleRow = document.createElement("div");
      titleRow.className = "hit__title-row";
      titleRow.appendChild(a);
      titleRow.appendChild(explainBtn);
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
      main.appendChild(titleRow);
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
    const tp = Math.max(1, data.pager_total_pages || 1);
    els.resultsMeta.textContent =
      `About ${data.depth} ranked · page ${data.page} of ${tp} (up to ${PAGER_MAX_PAGES}×${PAGE_SIZE}=${PAGER_MAX_PAGES * PAGE_SIZE} results)${qrelsNote}`;

    renderPageNums(data);

    els.pager.hidden = tp <= 1;
    els.prevPage.disabled = data.page <= 1;
    els.nextPage.disabled = data.page >= tp || !data.has_more;
    els.pageInfo.textContent =
      tp <= 1
        ? `${data.returned} result${data.returned === 1 ? "" : "s"}`
        : `${data.returned} on this page · ${PAGE_SIZE} per page`;
  }

  function hideSuggest() {
    if (!els.suggestDropdown) return;
    if (suggestBlurTimer) {
      window.clearTimeout(suggestBlurTimer);
      suggestBlurTimer = null;
    }
    els.suggestDropdown.hidden = true;
    els.suggestDropdown.innerHTML = "";
    els.q.setAttribute("aria-expanded", "false");
    suggestHighlight = -1;
  }

  function updateSuggestHighlight() {
    if (!els.suggestDropdown) return;
    const opts = els.suggestDropdown.querySelectorAll(".suggest-option");
    opts.forEach((el, i) => {
      el.classList.toggle("is-active", i === suggestHighlight);
      el.setAttribute(
        "aria-selected",
        i === suggestHighlight ? "true" : "false"
      );
    });
  }

  function renderSuggest(items) {
    if (!els.suggestDropdown) return;
    els.suggestDropdown.innerHTML = "";
    if (!items.length) {
      hideSuggest();
      return;
    }
    items.forEach((item, idx) => {
      const opt = document.createElement("button");
      opt.type = "button";
      opt.className = "suggest-option";
      opt.setAttribute("role", "option");
      opt.id = `suggest-opt-${idx}`;
      opt.dataset.index = String(idx);
      opt.textContent = item.text;
      opt.addEventListener("mousedown", (e) => {
        e.preventDefault();
      });
      opt.addEventListener("click", () => {
        els.q.value = item.text;
        hideSuggest();
        els.q.focus();
      });
      els.suggestDropdown.appendChild(opt);
    });
    els.suggestDropdown.hidden = false;
    els.q.setAttribute("aria-expanded", "true");
    suggestHighlight = -1;
    updateSuggestHighlight();
  }

  async function runSuggestFetch() {
    if (!els.suggestDropdown) return;
    const q = els.q.value.trim();
    if (q.length < SUGGEST_MIN_CHARS) {
      hideSuggest();
      return;
    }
    if (suggestAbort) suggestAbort.abort();
    suggestAbort = new AbortController();
    try {
      const res = await fetch(
        `/api/suggest?q=${encodeURIComponent(q)}&limit=8`,
        { signal: suggestAbort.signal }
      );
      const ct = res.headers.get("content-type") || "";
      if (!ct.includes("application/json")) {
        hideSuggest();
        return;
      }
      const data = await res.json();
      if (!res.ok) {
        hideSuggest();
        return;
      }
      if (els.q.value.trim() !== q) return;
      renderSuggest(data.suggestions || []);
    } catch (e) {
      if (e.name === "AbortError") return;
      hideSuggest();
    }
  }

  function scheduleSuggest() {
    if (suggestTimer) window.clearTimeout(suggestTimer);
    suggestTimer = window.setTimeout(() => {
      suggestTimer = null;
      runSuggestFetch();
    }, SUGGEST_DEBOUNCE_MS);
  }

  async function fetchModels() {
    const res = await fetch("/api/models");
    const data = await res.json();
    models = data.models || [];
    renderModels();
  }

  async function doSearch(resetPage) {
    if (resetPage) page = 1;
    hideSuggest();
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
        page_size: PAGE_SIZE,
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
        setStatus(data.error || "Search failed.", true);
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

  els.q.addEventListener("input", () => {
    scheduleSuggest();
    if (els.q.value.trim().length < SUGGEST_MIN_CHARS) hideSuggest();
  });

  els.q.addEventListener("focus", () => {
    if (els.q.value.trim().length >= SUGGEST_MIN_CHARS) scheduleSuggest();
  });

  els.q.addEventListener("blur", () => {
    suggestBlurTimer = window.setTimeout(() => hideSuggest(), 180);
  });

  if (els.suggestDropdown) {
    els.suggestDropdown.addEventListener("mouseenter", () => {
      if (suggestBlurTimer) {
        window.clearTimeout(suggestBlurTimer);
        suggestBlurTimer = null;
      }
    });
  }

  els.q.addEventListener("keydown", (e) => {
    if (!els.suggestDropdown || els.suggestDropdown.hidden) return;
    const opts = els.suggestDropdown.querySelectorAll(".suggest-option");
    const n = opts.length;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (n === 0) return;
      suggestHighlight =
        suggestHighlight < 0 ? 0 : Math.min(suggestHighlight + 1, n - 1);
      updateSuggestHighlight();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (n === 0) return;
      if (suggestHighlight <= 0) suggestHighlight = -1;
      else suggestHighlight -= 1;
      updateSuggestHighlight();
    } else if (e.key === "Enter" && suggestHighlight >= 0) {
      e.preventDefault();
      opts[suggestHighlight].click();
    } else if (e.key === "Escape") {
      hideSuggest();
    }
  });

  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    hideSuggest();
    doSearch(true);
  });

  els.prevPage.addEventListener("click", () => {
    if (page > 1) {
      page -= 1;
      doSearch(false);
    }
  });

  els.nextPage.addEventListener("click", () => {
    const tp = lastPayload?.pager_total_pages ?? PAGER_MAX_PAGES;
    if (lastPayload && page < tp && lastPayload.has_more !== false) {
      page += 1;
      doSearch(false);
    }
  });

  els.themeToggle.addEventListener("click", toggleTheme);

  els.articleBackdrop.addEventListener("click", closeArticle);
  els.articleClose.addEventListener("click", closeArticle);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const drawersOpen =
        document.body.classList.contains("layout--drawer-left-open") ||
        document.body.classList.contains("layout--drawer-right-open");
      if (drawersOpen && !explainOpen && !articleOpen) {
        e.preventDefault();
        closeMobileDrawers();
        return;
      }
      if (explainOpen) closeExplain();
      else if (articleOpen) closeArticle();
    }
  });

  if (els.explainBackdrop) {
    els.explainBackdrop.addEventListener("click", closeExplain);
  }
  if (els.explainClose) {
    els.explainClose.addEventListener("click", closeExplain);
  }

  function initAmbientCursor() {
    const root = document.documentElement;
    const main = document.querySelector(".main");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    let targetX = 50;
    let targetY = 42;
    let smoothX = targetX;
    let smoothY = targetY;
    let targetGlowScale = 1;
    let smoothGlowScale = 1;
    let raf = 0;

    function ambientHitLooksClickable(el) {
      if (!el || el.nodeType !== 1 || el.closest(".atmosphere")) return false;
      if (
        el.closest(
          [
            'a[href]',
            "button",
            'input:not([type="hidden"])',
            "select",
            "textarea",
            "label",
            "summary",
            '[role="button"]',
            '[role="link"]',
            '[role="tab"]',
            '[role="menuitem"]',
            '[role="switch"]',
            '[role="checkbox"]',
            '[role="radio"]',
          ].join(", ")
        )
      )
        return true;
      const cs = window.getComputedStyle(el);
      const cur = cs.cursor;
      if (
        (cur === "pointer" || cur === "grab" || cur === "zoom-in") &&
        cs.pointerEvents !== "none"
      )
        return true;
      return false;
    }

    function syncCss() {
      root.style.setProperty("--cursor-x", `${smoothX}%`);
      root.style.setProperty("--cursor-y", `${smoothY}%`);
      if (!reduceMotion.matches) {
        root.style.setProperty("--cursor-glow-scale", smoothGlowScale.toFixed(4));
      }
      const nx = (smoothX / 100 - 0.5) * 2;
      const ny = (smoothY / 100 - 0.5) * 2;
      root.style.setProperty("--cursor-nx", nx.toFixed(4));
      root.style.setProperty("--cursor-ny", ny.toFixed(4));

      if (!main) return;
      if (reduceMotion.matches) {
        main.style.removeProperty("transform");
        return;
      }
      const rx = (targetX / 100 - 0.5) * 2;
      const ry = (targetY / 100 - 0.5) * 2;
      main.style.transform = `perspective(1400px) rotateX(${(-ry * 0.52).toFixed(3)}deg) rotateY(${(rx * 0.52).toFixed(3)}deg)`;
    }

    function tick() {
      raf = 0;
      smoothX += (targetX - smoothX) * 0.14;
      smoothY += (targetY - smoothY) * 0.14;
      smoothGlowScale += (targetGlowScale - smoothGlowScale) * 0.14;
      syncCss();

      const dx = Math.abs(targetX - smoothX);
      const dy = Math.abs(targetY - smoothY);
      const dg = Math.abs(targetGlowScale - smoothGlowScale);
      if (
        (dx > 0.06 || dy > 0.06 || dg > 0.004) &&
        !reduceMotion.matches
      ) {
        raf = window.requestAnimationFrame(tick);
      }
    }

    function onMove(e) {
      if (reduceMotion.matches) return;
      targetX = (e.clientX / Math.max(window.innerWidth, 1)) * 100;
      targetY = (e.clientY / Math.max(window.innerHeight, 1)) * 100;
      let hit = null;
      try {
        hit = document.elementFromPoint(e.clientX, e.clientY);
      } catch {
        hit = null;
      }
      targetGlowScale = ambientHitLooksClickable(hit) ? 0.34 : 1;
      if (!raf) raf = window.requestAnimationFrame(tick);
    }

    document.addEventListener("mousemove", onMove, { passive: true });
    document.documentElement.addEventListener("mouseleave", () => {
      targetGlowScale = 1;
      if (!reduceMotion.matches && !raf)
        raf = window.requestAnimationFrame(tick);
    });

    reduceMotion.addEventListener("change", () => {
      if (reduceMotion.matches && raf) {
        window.cancelAnimationFrame(raf);
        raf = 0;
      }
      root.style.removeProperty("--cursor-glow-scale");
      syncCss();
    });

    syncCss();
  }

  function initMobileDrawers() {
    const scrim = document.getElementById("drawerScrim");
    const leftSb = document.getElementById("leftSidebar");
    const rightSb = document.getElementById("rightSidebar");
    const metricsBtn = document.getElementById("metricsDrawerToggle");
    const modelBtn = document.getElementById("modelDrawerToggle");
    const leftClose = document.getElementById("leftSidebarClose");
    const rightClose = document.getElementById("rightSidebarClose");

    if (!scrim || !leftSb || !rightSb) return;

    const mq = window.matchMedia("(max-width: 960px)");

    function isDrawerMode() {
      return mq.matches;
    }

    function setScrim(on) {
      scrim.classList.toggle("is-visible", on);
      scrim.setAttribute("aria-hidden", on ? "false" : "true");
      scrim.tabIndex = on ? 0 : -1;
    }

    function syncDrawerState() {
      if (!isDrawerMode()) {
        document.body.classList.remove(
          "layout--drawer-open",
          "layout--drawer-left-open",
          "layout--drawer-right-open"
        );
        setScrim(false);
        metricsBtn?.setAttribute("aria-expanded", "false");
        modelBtn?.setAttribute("aria-expanded", "false");
        return;
      }
      const leftOn = document.body.classList.contains(
        "layout--drawer-left-open"
      );
      const rightOn = document.body.classList.contains(
        "layout--drawer-right-open"
      );
      const any = leftOn || rightOn;
      document.body.classList.toggle("layout--drawer-open", any);
      setScrim(any);
    }

    function closeDrawers() {
      document.body.classList.remove(
        "layout--drawer-left-open",
        "layout--drawer-right-open",
        "layout--drawer-open"
      );
      metricsBtn?.setAttribute("aria-expanded", "false");
      modelBtn?.setAttribute("aria-expanded", "false");
      setScrim(false);
    }

    closeMobileDrawers = closeDrawers;

    function openLeft() {
      document.body.classList.remove("layout--drawer-right-open");
      document.body.classList.add("layout--drawer-left-open");
      modelBtn?.setAttribute("aria-expanded", "false");
      metricsBtn?.setAttribute("aria-expanded", "true");
      syncDrawerState();
    }

    function openRight() {
      document.body.classList.remove("layout--drawer-left-open");
      document.body.classList.add("layout--drawer-right-open");
      metricsBtn?.setAttribute("aria-expanded", "false");
      modelBtn?.setAttribute("aria-expanded", "true");
      syncDrawerState();
    }

    function toggleLeft(ev) {
      if (!isDrawerMode()) return;
      if (ev) ev.preventDefault();
      if (document.body.classList.contains("layout--drawer-left-open")) {
        closeDrawers();
        return;
      }
      openLeft();
    }

    function toggleRight(ev) {
      if (!isDrawerMode()) return;
      if (ev) ev.preventDefault();
      if (document.body.classList.contains("layout--drawer-right-open")) {
        closeDrawers();
        return;
      }
      openRight();
    }

    function closeLeftOnly() {
      document.body.classList.remove("layout--drawer-left-open");
      metricsBtn?.setAttribute("aria-expanded", "false");
      syncDrawerState();
    }

    function closeRightOnly() {
      document.body.classList.remove("layout--drawer-right-open");
      modelBtn?.setAttribute("aria-expanded", "false");
      syncDrawerState();
    }

    metricsBtn?.addEventListener("click", toggleLeft);
    modelBtn?.addEventListener("click", toggleRight);
    scrim.addEventListener("click", closeDrawers);
    leftClose?.addEventListener("click", closeLeftOnly);
    rightClose?.addEventListener("click", closeRightOnly);

    mq.addEventListener("change", closeDrawers);
    window.addEventListener("orientationchange", () => {
      window.setTimeout(closeDrawers, 300);
    });
  }

  loadTheme();
  buildMetricToggles();
  initMobileDrawers();
  initAmbientCursor();
  fetchModels().catch(() => {
    setStatus("Could not load models.", true);
  });
})();
