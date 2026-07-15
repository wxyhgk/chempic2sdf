const state = {
  runs: [],
  run: null,
  filtered: [],
  selectedId: null,
  status: "all",
  difficulty: "all",
  errorType: "all",
  query: "",
  page: 1,
  pageSize: 60,
  view: "image",
  sdfCache: new Map(),
};

const statusLabels = {
  exact: "完全一致",
  similar: "原子/键数一致，结构有差异",
  mismatch: "原子或键数量不同",
  invalid: "无效 SDF",
};

const difficultyLabels = {
  standard_bond_length: "标准键长",
  long_bond: "长键",
  extreme_long_bond: "极端长键",
};

const errorLabels = {
  exact: "图结构一致",
  atom_detection: "顶点检测或定位",
  missing_atom: "漏原子",
  extra_atom: "多原子",
  ocr: "OCR 字符",
  atom_property: "原子属性",
  aromaticity: "芳香性标记",
  missing_bond: "漏键",
  extra_bond: "多余键",
  bond_rewire: "错连键",
  crossing_extra_bond: "交叉多余键",
  extra_bond_through_atom: "多余键穿过原子",
  bond_type: "键型",
  stereo: "立体化学",
  invalid_disconnected: "无效：断图",
  invalid_kekulize: "无效：芳香环",
  invalid_valence: "无效：价态",
  invalid_other: "无效：其他",
  other: "其他",
};

const elements = {
  runSelect: document.querySelector("#runSelect"),
  runMeta: document.querySelector("#runMeta"),
  metrics: document.querySelector("#metrics"),
  searchInput: document.querySelector("#searchInput"),
  statusFilter: document.querySelector("#statusFilter"),
  difficultyFilter: document.querySelector("#difficultyFilter"),
  errorFilter: document.querySelector("#errorFilter"),
  resultCount: document.querySelector("#resultCount"),
  sampleList: document.querySelector("#sampleList"),
  previousPage: document.querySelector("#previousPage"),
  nextPage: document.querySelector("#nextPage"),
  pageLabel: document.querySelector("#pageLabel"),
  detail: document.querySelector("#detail"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatSeconds(value) {
  if (value === null || value === undefined) return "-";
  return `${Number(value).toFixed(3)} s`;
}

async function requestJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function initialize() {
  bindControls();
  try {
    const payload = await requestJson("/api/runs");
    state.runs = payload.runs;
    if (!state.runs.length) {
      elements.runMeta.textContent = "没有找到评测记录";
      elements.detail.innerHTML = `
        <div class="empty-state">
          <strong>没有 benchmark 结果</strong>
          <span>评测结果目录：outputs/evaluations/</span>
        </div>`;
      return;
    }
    elements.runSelect.innerHTML = state.runs
      .map(
        (run) =>
          `<option value="${escapeHtml(run.id)}">${escapeHtml(run.id)} · ${run.samples} 张</option>`,
      )
      .join("");
    await loadRun(state.runs[0].id);
  } catch (error) {
    showFatal(error);
  }
}

function bindControls() {
  elements.runSelect.addEventListener("change", () => loadRun(elements.runSelect.value));
  elements.searchInput.addEventListener("input", () => {
    state.query = elements.searchInput.value.trim().toLowerCase();
    state.page = 1;
    applyFilters(true);
  });
  elements.difficultyFilter.addEventListener("change", () => {
    state.difficulty = elements.difficultyFilter.value;
    state.page = 1;
    applyFilters(true);
  });
  elements.errorFilter.addEventListener("change", () => {
    state.errorType = elements.errorFilter.value;
    state.page = 1;
    applyFilters(true);
  });
  elements.statusFilter.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-status]");
    if (!button) return;
    state.status = button.dataset.status;
    state.page = 1;
    for (const item of elements.statusFilter.querySelectorAll("button")) {
      item.classList.toggle("active", item === button);
    }
    applyFilters(true);
  });
  elements.previousPage.addEventListener("click", () => {
    state.page = Math.max(1, state.page - 1);
    renderSampleList();
  });
  elements.nextPage.addEventListener("click", () => {
    const pages = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
    state.page = Math.min(pages, state.page + 1);
    renderSampleList();
  });
}

async function loadRun(runId) {
  elements.detail.innerHTML = '<div class="loading-state">正在读取评测结果</div>';
  state.selectedId = null;
  state.sdfCache.clear();
  try {
    state.run = await requestJson(`/api/runs/${encodeURIComponent(runId)}`);
    elements.runSelect.value = runId;
    const checkpoint = state.run.config.checkpoint?.split("/").slice(-2).join("/") ?? "-";
    elements.runMeta.textContent = `${state.run.samples.length} 个样本 · ${checkpoint} · ${state.run.updated_at}`;
    renderErrorFilter();
    renderMetrics();
    applyFilters(true);
  } catch (error) {
    showFatal(error);
  }
}

function renderErrorFilter() {
  const available = new Set();
  for (const sample of state.run.samples) {
    if (sample.primary_error) available.add(sample.primary_error);
    for (const flag of sample.error_flags || []) available.add(flag);
  }
  const options = [...available].sort((left, right) =>
    (errorLabels[left] || left).localeCompare(errorLabels[right] || right, "zh-CN"),
  );
  elements.errorFilter.innerHTML = [
    '<option value="all">全部错误类型</option>',
    ...options.map(
      (value) =>
        `<option value="${escapeHtml(value)}">${escapeHtml(errorLabels[value] || value)}</option>`,
    ),
  ].join("");
  if (!available.has(state.errorType)) state.errorType = "all";
  elements.errorFilter.value = state.errorType;
}

function renderMetrics() {
  const samples = state.run.samples;
  const count = (status) => samples.filter((sample) => sample.status === status).length;
  const valid = samples.filter((sample) => sample.sdf_valid).length;
  const averageSeconds =
    samples.reduce((total, sample) => total + Number(sample.seconds || 0), 0) /
    Math.max(samples.length, 1);
  const similarities = samples
    .map((sample) => sample.fingerprint_similarity)
    .filter((value) => value !== null && value !== undefined)
    .map(Number);
  const averageSimilarity =
    similarities.reduce((total, value) => total + value, 0) /
    Math.max(similarities.length, 1);
  const metrics = [
    ["样本总数", samples.length.toLocaleString("zh-CN")],
    ["有效 SDF", `${valid.toLocaleString("zh-CN")} · ${formatPercent(valid / samples.length)}`],
    ["完全一致", `${count("exact").toLocaleString("zh-CN")} · ${formatPercent(count("exact") / samples.length)}`],
    ["原子/键数一致", count("similar").toLocaleString("zh-CN")],
    ["无效 SDF", count("invalid").toLocaleString("zh-CN")],
    ["平均推理", formatSeconds(averageSeconds)],
  ];
  if (similarities.length) {
    metrics[3] = ["平均结构相似度", formatPercent(averageSimilarity)];
  }
  elements.metrics.innerHTML = metrics
    .map(
      ([label, value]) =>
        `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`,
    )
    .join("");
}

function applyFilters(selectFirst = false) {
  const query = state.query;
  state.filtered = state.run.samples.filter((sample) => {
    if (state.status !== "all" && sample.status !== state.status) return false;
    if (state.difficulty !== "all" && sample.difficulty !== state.difficulty) return false;
    if (
      state.errorType !== "all" &&
      sample.primary_error !== state.errorType &&
      !(sample.error_flags || []).includes(state.errorType)
    ) return false;
    if (!query) return true;
    const haystack = [
      sample.molecule_id,
      sample.expected_formula,
      sample.actual_formula,
      sample.failure_category,
      sample.primary_error,
      ...(sample.error_flags || []),
      sample.error,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
  const pages = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
  state.page = Math.min(state.page, pages);
  elements.resultCount.textContent = `${state.filtered.length.toLocaleString("zh-CN")} 个样本`;
  renderSampleList();
  if (selectFirst) {
    if (state.filtered.length) {
      selectSample(state.filtered[0].molecule_id);
    } else {
      state.selectedId = null;
      elements.detail.innerHTML = `
        <div class="empty-state">
          <strong>没有符合条件的样本</strong>
          <span>调整状态、难度或搜索条件</span>
        </div>`;
    }
  }
}

function renderSampleList() {
  const pages = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
  const start = (state.page - 1) * state.pageSize;
  const pageItems = state.filtered.slice(start, start + state.pageSize);
  elements.sampleList.innerHTML = pageItems
    .map((sample) => {
      const score = sample.fingerprint_similarity;
      const scoreText = sample.status === "exact" ? "100%" : formatPercent(score, 0);
      return `
        <button class="sample-row ${sample.molecule_id === state.selectedId ? "selected" : ""}"
                type="button" data-id="${escapeHtml(sample.molecule_id)}">
          <span class="status-dot ${escapeHtml(sample.status)}"></span>
          <span class="sample-main">
            <span class="sample-id">${escapeHtml(sample.molecule_id)}</span>
            <span class="sample-formula">${escapeHtml(sample.expected_formula || "-")} → ${escapeHtml(sample.actual_formula || "无有效结构")}</span>
            <span class="sample-error ${escapeHtml(sample.primary_error || "")}">${escapeHtml(errorLabels[sample.primary_error] || sample.primary_error || "未诊断")}</span>
          </span>
          <span class="sample-score">${escapeHtml(scoreText)}</span>
        </button>`;
    })
    .join("");
  for (const row of elements.sampleList.querySelectorAll(".sample-row")) {
    row.addEventListener("click", () => selectSample(row.dataset.id));
  }
  elements.pageLabel.textContent = `${state.page} / ${pages}`;
  elements.previousPage.disabled = state.page <= 1;
  elements.nextPage.disabled = state.page >= pages;
}

async function selectSample(moleculeId) {
  state.selectedId = moleculeId;
  renderSampleList();
  elements.detail.innerHTML = '<div class="loading-state">正在比较两份 SDF</div>';
  try {
    const detail = await requestJson(
      `/api/runs/${encodeURIComponent(state.run.id)}/samples/${encodeURIComponent(moleculeId)}`,
    );
    renderDetail(detail);
  } catch (error) {
    elements.detail.innerHTML = `
      <div class="empty-state">
        <strong>样本读取失败</strong>
        <span>${escapeHtml(error.message)}</span>
      </div>`;
  }
}

function artifactLink(url, label, download = false) {
  if (!url) return "";
  return `<a class="download-link" href="${escapeHtml(url)}" ${download ? "download" : "target=\"_blank\""}>${escapeHtml(label)}</a>`;
}

function renderDetail(detail) {
  const { record, manifest, artifacts, comparison, status, prediction_kind: predictionKind } = detail;
  const predictionAvailable = Boolean(artifacts.prediction_sdf);
  const similarity = comparison?.fingerprint_similarity ?? record.fingerprint_similarity;
  const mcsAtoms = comparison
    ? `${comparison.mcs_atoms} / ${record.expected_atoms}`
    : "-";
  const mcsBonds = comparison
    ? `${comparison.mcs_bonds} / ${record.expected_bonds}`
    : "-";
  const links = [
    artifactLink(artifacts.ground_truth_sdf, "真值 SDF", true),
    artifactLink(artifacts.prediction_sdf, "预测 SDF", true),
    artifactLink(artifacts.raw_chemsvg, "原始 ChemSVG"),
    artifactLink(artifacts.repaired_chemsvg, "修复后 ChemSVG"),
  ].join("");
  const diagnostics = record.diagnostics || {};
  const diagnosticFlags = (record.error_flags || [])
    .map((flag) => errorLabels[flag] || flag)
    .join(" · ");

  elements.detail.innerHTML = `
    <div class="detail-header">
      <div class="detail-title">
        <h2>${escapeHtml(record.molecule_id)}</h2>
        <div class="detail-subtitle">${escapeHtml(difficultyLabels[record.difficulty] || record.difficulty || "-")} · 最长键/中位键 ${escapeHtml(Number(record.maximum_to_median_bond_ratio || 0).toFixed(2))}× · ${escapeHtml(formatSeconds(record.seconds))}</div>
      </div>
      <span class="status-badge ${escapeHtml(status)}">${escapeHtml(statusLabels[status])}</span>
    </div>

    <div class="view-toolbar">
      <div class="segmented" id="detailView">
        <button type="button" data-view="image" class="active">结构图</button>
        <button type="button" data-view="sdf">SDF 文本</button>
      </div>
      <div class="artifact-links">${links}</div>
    </div>

    <div class="comparison-grid" data-view-content="image">
      <section class="structure-panel">
        <div class="panel-heading">
          <h3>真值 SDF</h3>
          <span>${escapeHtml(manifest.formula)}</span>
        </div>
        <div class="molecule-frame">
          <img src="${escapeHtml(artifacts.ground_truth_image)}" alt="${escapeHtml(record.molecule_id)} 真值结构">
        </div>
      </section>
      <section class="structure-panel">
        <div class="panel-heading">
          <h3>${predictionKind === "invalid" ? "模型诊断 SDF" : "模型预测 SDF"}</h3>
          <span>${escapeHtml(record.actual_formula || "-")}</span>
        </div>
        <div class="molecule-frame">
          ${
            predictionAvailable
              ? `<img src="${escapeHtml(artifacts.prediction_image)}" alt="${escapeHtml(record.molecule_id)} 预测结构">`
              : `<div class="missing-prediction"><strong>没有有效的预测 SDF</strong><span>${escapeHtml(record.error || "该历史运行未保存预测 SDF")}</span></div>`
          }
        </div>
      </section>
    </div>

    <div class="comparison-grid hidden" data-view-content="sdf">
      <section class="structure-panel">
        <div class="panel-heading"><h3>真值 SDF</h3></div>
        <pre class="sdf-view" id="truthSdf">正在读取</pre>
      </section>
      <section class="structure-panel">
        <div class="panel-heading"><h3>模型预测 SDF</h3></div>
        <pre class="sdf-view" id="predictionSdf">${predictionAvailable ? "正在读取" : "没有有效的预测 SDF"}</pre>
      </section>
    </div>

    <div class="comparison-metrics">
      ${comparisonMetric("结构相似度", formatPercent(similarity))}
      ${comparisonMetric("共同子结构原子", mcsAtoms)}
      ${comparisonMetric("共同子结构键", mcsBonds)}
      ${comparisonMetric("原子数 预测/真值", `${record.predicted_atoms} / ${record.expected_atoms}`)}
      ${comparisonMetric("键数 预测/真值", `${record.predicted_bonds} / ${record.expected_bonds}`)}
      ${comparisonMetric("分子式", record.formula_match ? "一致" : "有差异")}
      ${comparisonMetric("原始键 → 修复后", formatStrokeChange(record))}
      ${comparisonMetric("主要错误", errorLabels[record.primary_error] || record.primary_error || "未诊断")}
      ${comparisonMetric("OCR 错字", diagnostics.element_mismatches ?? "-")}
      ${comparisonMetric("漏/多原子", formatPair(diagnostics.missing_atoms, diagnostics.extra_atoms))}
      ${comparisonMetric("漏/多键", formatPair(diagnostics.missing_bonds, diagnostics.extra_bonds))}
      ${comparisonMetric("交叉多余键", diagnostics.crossing_extra_bonds ?? "-")}
      ${comparisonMetric("多余键穿过原子", diagnostics.extra_bonds_through_atom ?? "-")}
      ${comparisonMetric("键型错误", diagnostics.bond_type_mismatches ?? "-")}
    </div>

    ${diagnosticFlags ? `<div class="diagnostic-block"><strong>并发错误</strong><span>${escapeHtml(diagnosticFlags)}</span></div>` : ""}
    ${record.error ? `<div class="error-block">${escapeHtml(record.error)}</div>` : ""}
  `;

  const viewControl = elements.detail.querySelector("#detailView");
  viewControl.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-view]");
    if (!button) return;
    switchDetailView(button.dataset.view, detail);
  });
  const predictionImage = elements.detail.querySelector(
    `img[src="${CSS.escape(artifacts.prediction_image || "")}"]`,
  );
  predictionImage?.addEventListener("error", () => {
    predictionImage.parentElement.innerHTML = `
      <div class="missing-prediction">
        <strong>诊断 SDF 无法渲染</strong>
        <span>可切换到 SDF 文本或查看原始 ChemSVG</span>
      </div>`;
  });
}

function comparisonMetric(label, value) {
  return `<div class="comparison-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function formatPair(first, second) {
  if (first === undefined || second === undefined) return "-";
  return `${first} / ${second}`;
}

function formatStrokeChange(record) {
  if (record.raw_strokes === undefined || record.repaired_strokes === undefined) {
    return "-";
  }
  return `${record.raw_strokes} → ${record.repaired_strokes}`;
}

async function switchDetailView(view, detail) {
  state.view = view;
  for (const button of elements.detail.querySelectorAll("#detailView button")) {
    button.classList.toggle("active", button.dataset.view === view);
  }
  for (const content of elements.detail.querySelectorAll("[data-view-content]")) {
    content.classList.toggle("hidden", content.dataset.viewContent !== view);
  }
  if (view !== "sdf") return;
  const truth = elements.detail.querySelector("#truthSdf");
  const prediction = elements.detail.querySelector("#predictionSdf");
  truth.textContent = await fetchText(detail.artifacts.ground_truth_sdf);
  if (detail.artifacts.prediction_sdf) {
    prediction.textContent = await fetchText(detail.artifacts.prediction_sdf);
  }
}

async function fetchText(url) {
  if (state.sdfCache.has(url)) return state.sdfCache.get(url);
  const response = await fetch(url);
  const text = response.ok ? await response.text() : `读取失败：${response.status}`;
  state.sdfCache.set(url, text);
  return text;
}

function showFatal(error) {
  elements.runMeta.textContent = "前端读取失败";
  elements.detail.innerHTML = `
    <div class="empty-state">
      <strong>无法读取 benchmark</strong>
      <span>${escapeHtml(error.message)}</span>
    </div>`;
}

initialize();
