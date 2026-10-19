const auditState = {
  summary: null,
  records: [],
  annotations: {},
  filtered: [],
  selectedId: null,
  query: "",
  cohort: "all",
  reviewed: "all",
  bucket: "all",
  difficulty: "all",
};

const bucketLabels = {
  atom_count_mixed: "原子同时多检/漏检",
  atom_over_detection: "原子多检",
  atom_under_detection: "原子漏检",
  atom_property: "原子属性",
  bond_over_detection: "键多检",
  bond_rewire: "键重连",
  bond_type: "键型",
  bond_under_detection: "键漏检",
  control_exact: "正确对照",
  crossing_extra_bond: "交叉多余键",
  ocr_transcription: "OCR 转写",
  invalid_compilation: "编译/价态无效",
  unresolved: "未解析",
};

const causeLabels = {
  text_stroke_as_structure: "文字笔画被当作键或顶点",
  bond_as_text: "化学键被当作文字",
  implicit_carbon_missing: "隐式碳漏检",
  implicit_carbon_duplicate: "隐式碳重复",
  semantic_bond_extra: "语义键多检",
  semantic_bond_missing: "语义键漏检",
  semantic_bond_order: "单/双/三/芳香键型错误",
  topology_given_counts: "数量正确但拓扑错误",
  ocr_transcription: "文字区域正确但转写错误",
  atom_property_or_stereo: "电荷、同位素、芳香性或立体错误",
  compiler_or_valence: "编译或价态错误",
  ground_truth_or_nonstandard: "真值或非标准画法问题",
  unresolved: "暂时无法确定",
};

const tagLabels = {
  decorated_text: "修饰文字",
  decorated_bond: "修饰键",
  bond_crossing: "键交叉",
  large_font: "大字号",
  long_bond: "长键",
  thick_bond: "粗键",
  thin_bond: "细键",
  long_functional_group: "长官能团文字",
  dense_layout: "拥挤布局",
  colored_atom: "彩色原子",
  blurry: "模糊",
};

const el = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status}`);
  return payload;
}

async function initializeAudit() {
  bindAuditControls();
  try {
    const payload = await requestJson("/api/audit");
    auditState.summary = payload.summary;
    auditState.records = payload.records;
    auditState.annotations = payload.annotations || {};
    el("#auditMeta").textContent = `${payload.summary.model_version} · commitment ${payload.summary.selection_sha256.slice(0, 12)}`;
    const buckets = [...new Set(payload.records.map((row) => row.automatic_bucket))].sort();
    el("#bucketFilter").innerHTML = '<option value="all">全部自动首因</option>' + buckets
      .map((bucket) => `<option value="${escapeHtml(bucket)}">${escapeHtml(bucketLabels[bucket] || bucket)}</option>`).join("");
    renderAuditSummary();
    applyAuditFilters();
    selectAuditRecord(auditState.filtered[0]?.molecule_id);
  } catch (error) {
    el("#auditDetail").innerHTML = `<div class="audit-empty"><strong>审计集读取失败</strong><span>${escapeHtml(error.message)}</span></div>`;
  }
}

function bindAuditControls() {
  el("#auditSearch").addEventListener("input", (event) => { auditState.query = event.target.value.trim().toLowerCase(); applyAuditFilters(); });
  el("#cohortFilter").addEventListener("change", (event) => { auditState.cohort = event.target.value; applyAuditFilters(); });
  el("#reviewFilter").addEventListener("change", (event) => { auditState.reviewed = event.target.value; applyAuditFilters(); });
  el("#bucketFilter").addEventListener("change", (event) => { auditState.bucket = event.target.value; applyAuditFilters(); });
  el("#auditDifficulty").addEventListener("change", (event) => { auditState.difficulty = event.target.value; applyAuditFilters(); });
  el("#exportAnnotations").addEventListener("click", exportAuditAnnotations);
}

function renderAuditSummary() {
  const summary = auditState.summary;
  const reviewed = Object.keys(auditState.annotations).length;
  el("#reviewProgress").textContent = `${reviewed} / ${summary.samples}`;
  const universe = summary.universe || {};
  const metrics = [
    ["固定审计集", summary.samples],
    ["错误 / 对照", `${summary.errors} / ${summary.controls}`],
    ["全量错误", `${universe.errors ?? "-"} / ${universe.samples ?? "-"}`],
    ["已人工复核", reviewed],
    ["自动 OCR 桶", summary.automatic_bucket_counts.ocr_transcription || 0],
    ["自动对象计数桶", (summary.automatic_bucket_counts.atom_count_mixed || 0) + (summary.automatic_bucket_counts.atom_over_detection || 0) + (summary.automatic_bucket_counts.atom_under_detection || 0)],
  ];
  el("#auditMetrics").innerHTML = metrics.map(([label, value]) => `<div class="audit-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

function applyAuditFilters() {
  auditState.filtered = auditState.records.filter((row) => {
    const annotation = auditState.annotations[row.molecule_id];
    const reviewed = Boolean(annotation);
    const text = `${row.molecule_id} ${row.automatic_bucket} ${row.primary_error}`.toLowerCase();
    return (!auditState.query || text.includes(auditState.query))
      && (auditState.cohort === "all" || row.cohort === auditState.cohort)
      && (auditState.reviewed === "all" || (auditState.reviewed === "reviewed") === reviewed)
      && (auditState.bucket === "all" || row.automatic_bucket === auditState.bucket)
      && (auditState.difficulty === "all" || row.difficulty === auditState.difficulty);
  });
  el("#auditCount").textContent = `${auditState.filtered.length} 个样本`;
  renderAuditList();
  if (!auditState.filtered.some((row) => row.molecule_id === auditState.selectedId)) {
    selectAuditRecord(auditState.filtered[0]?.molecule_id);
  }
}

function renderAuditList() {
  el("#auditList").innerHTML = auditState.filtered.map((row) => {
    const reviewed = Boolean(auditState.annotations[row.molecule_id]);
    return `<button class="audit-row ${row.molecule_id === auditState.selectedId ? "selected" : ""}" data-id="${escapeHtml(row.molecule_id)}">
      <i class="audit-dot ${row.cohort === "control" ? "control" : ""} ${reviewed ? "reviewed" : ""}"></i>
      <span><strong>${escapeHtml(row.molecule_id)}</strong><span>${escapeHtml(bucketLabels[row.automatic_bucket] || row.automatic_bucket)}</span></span>
      <small>${row.predicted_atoms}/${row.expected_atoms} · ${row.predicted_bonds}/${row.expected_bonds}</small>
    </button>`;
  }).join("");
  document.querySelectorAll(".audit-row").forEach((button) => button.addEventListener("click", () => selectAuditRecord(button.dataset.id)));
}

function artifactUrl(row, key) {
  return `/api/audit/artifacts/${encodeURIComponent(row.molecule_id)}/${encodeURIComponent(key)}`;
}

function selectAuditRecord(moleculeId) {
  auditState.selectedId = moleculeId || null;
  renderAuditList();
  if (!moleculeId) {
    el("#auditDetail").innerHTML = '<div class="audit-empty"><strong>没有符合筛选条件的样本</strong></div>';
    return;
  }
  const row = auditState.records.find((item) => item.molecule_id === moleculeId);
  const annotation = auditState.annotations[moleculeId] || {};
  const position = auditState.filtered.findIndex((item) => item.molecule_id === moleculeId);
  el("#auditDetail").innerHTML = `
    <div class="audit-detail-header">
      <div><h2>${escapeHtml(row.molecule_id)}</h2><p>${escapeHtml(bucketLabels[row.automatic_bucket] || row.automatic_bucket)} · ${escapeHtml(row.difficulty)} · ${row.sdf_valid ? "SDF 有效" : "SDF 无效"}</p></div>
      <div class="audit-nav"><button id="auditPrev" type="button" title="上一张">←</button><button id="auditNext" type="button" title="下一张">→</button></div>
    </div>
    <div class="audit-images">
      ${imagePanel("输入图片", artifactUrl(row, "input"))}
      ${imagePanel("GT / 预测对象叠加", artifactUrl(row, "overlay"), '<div class="overlay-legend"><b>绿色：真值</b><b>洋红：预测</b></div>')}
      ${imagePanel("真值 SDF 渲染", artifactUrl(row, "expected_image"))}
      ${imagePanel(row.sdf_valid ? "预测 SDF 渲染" : "无效预测诊断 SDF", artifactUrl(row, "prediction_image"))}
    </div>
    <div class="audit-facts">
      ${fact("原子 预测 / 真值", `${row.predicted_atoms} / ${row.expected_atoms}`)}
      ${fact("键 预测 / 真值", `${row.predicted_bonds} / ${row.expected_bonds}`)}
      ${fact("漏 / 多原子", `${row.diagnostics.missing_atoms} / ${row.diagnostics.extra_atoms}`)}
      ${fact("漏 / 多键", `${row.diagnostics.missing_bonds} / ${row.diagnostics.extra_bonds}`)}
      ${fact("OCR / 键型错", `${row.diagnostics.element_mismatches} / ${row.diagnostics.bond_type_mismatches}`)}
    </div>
    <form id="reviewForm" class="review-form">
      <h3>人工首因复核</h3>
      <div class="review-grid">
        <label>主要原因<select id="primaryCause" required><option value="">请选择</option>${Object.entries(causeLabels).map(([value, label]) => `<option value="${value}" ${annotation.primary_cause === value ? "selected" : ""}>${escapeHtml(label)}</option>`).join("")}</select></label>
        <label>判断置信度<select id="reviewConfidence" required><option value="high" ${annotation.confidence === "high" ? "selected" : ""}>高</option><option value="medium" ${!annotation.confidence || annotation.confidence === "medium" ? "selected" : ""}>中</option><option value="low" ${annotation.confidence === "low" ? "selected" : ""}>低</option></select></label>
      </div>
      <fieldset class="tag-field"><legend>视觉困难标签</legend><div class="tag-grid">${Object.entries(tagLabels).map(([value, label]) => `<label><input type="checkbox" name="visualTag" value="${value}" ${(annotation.visual_tags || []).includes(value) ? "checked" : ""}>${escapeHtml(label)}</label>`).join("")}</div></fieldset>
      <label>备注<textarea id="reviewNote" maxlength="2000" placeholder="记录具体位置、错连方式或无法判断的原因">${escapeHtml(annotation.note || "")}</textarea></label>
      <div class="review-actions"><div class="artifact-links"><a href="${artifactUrl(row, "expected_sdf")}" target="_blank">真值 SDF</a><a href="${artifactUrl(row, "prediction_sdf")}" target="_blank">预测 SDF</a></div><span id="saveStatus">${annotation.reviewed_at ? `已保存 ${escapeHtml(annotation.reviewed_at.slice(0, 19))}` : "尚未复核"}</span><button type="submit">保存并下一张</button></div>
    </form>`;
  el("#auditPrev").addEventListener("click", () => selectAuditRecord(auditState.filtered[Math.max(position - 1, 0)]?.molecule_id));
  el("#auditNext").addEventListener("click", () => selectAuditRecord(auditState.filtered[Math.min(position + 1, auditState.filtered.length - 1)]?.molecule_id));
  el("#reviewForm").addEventListener("submit", (event) => saveAuditReview(event, row, position));
}

const imagePanel = (title, src, footer = "") => `<section class="audit-panel"><h3>${escapeHtml(title)}</h3><div class="audit-frame"><img src="${src}" alt="${escapeHtml(title)}"></div>${footer}</section>`;
const fact = (label, value) => `<div class="audit-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;

async function saveAuditReview(event, row, position) {
  event.preventDefault();
  const payload = {
    primary_cause: el("#primaryCause").value,
    confidence: el("#reviewConfidence").value,
    visual_tags: [...document.querySelectorAll('input[name="visualTag"]:checked')].map((input) => input.value),
    note: el("#reviewNote").value,
  };
  el("#saveStatus").textContent = "正在保存";
  try {
    const result = await requestJson(`/api/audit/annotations/${encodeURIComponent(row.molecule_id)}`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    auditState.annotations[row.molecule_id] = result.annotation;
    renderAuditSummary();
    applyAuditFilters();
    const next = auditState.filtered[Math.min(position + 1, auditState.filtered.length - 1)];
    selectAuditRecord(next?.molecule_id || row.molecule_id);
  } catch (error) {
    el("#saveStatus").textContent = `保存失败：${error.message}`;
  }
}

function exportAuditAnnotations() {
  const payload = {schema_version: "mr-tadf-causal-audit-export-0.1", selection_sha256: auditState.summary.selection_sha256, annotations: auditState.annotations};
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = "mr-tadf-causal-audit-annotations.json"; link.click();
  URL.revokeObjectURL(url);
}

initializeAudit();
