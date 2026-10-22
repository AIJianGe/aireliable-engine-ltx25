/* ============================================================
   LTX-2.5 Video WebUI 逻辑（原生 JS，直连引擎 API）
   五模式：t2v / i2v / 首尾帧 / 多关键帧 / 多宫格
   模式按钮从 /capabilities 动态生成（capabilities 复数列表）
   多图卡片通用上传区（单图/首尾帧/多图 统一管理）
   四档方案联动（F16 / I8 / G8 / G4）
   distilled 硬约束（UI 层固化，禁用编辑）：8 步 · CFG=1 · 分辨率被 32 整除 · 帧数 8k+1
   ============================================================ */
"use strict";

const API = "";  // 同源直连（引擎自身挂载 /ui）

// capability 名（manifest.yaml）→ task_type（引擎 inputs.task_type）
const CAPABILITY_TO_MODE = {
  "text-to-video": "t2v",
  "image-to-video": "i2v",
  "first-last-frame-to-video": "first_last_frame_v",
  "multi-image-to-video": "multi_image_v",
  "multi-reference-to-video": "multi_reference_v",
};

// task_type → 详细配置（标签 / 提示 / 图片需求）
const MODE_INFO = {
  t2v: {
    label: "文生视频",
    hint: "仅根据文本描述生成视频（distilled 8 步蒸馏版）",
    min_imgs: 0, max_imgs: 0, roles: [],
  },
  i2v: {
    label: "图生视频",
    hint: "上传 1 张图片作为首帧，模型生成后续帧（首帧一致性）",
    min_imgs: 1, max_imgs: 1, roles: ["reference"],
  },
  first_last_frame_v: {
    label: "首尾帧插值",
    hint: "上传 2 张图：首帧 + 尾帧，模型生成中间平滑过渡",
    min_imgs: 2, max_imgs: 2, roles: ["first_frame", "last_frame"],
  },
  multi_image_v: {
    label: "多关键帧插值",
    hint: "上传 2~4 张关键帧，模型按时间顺序均匀分布生成视频",
    min_imgs: 2, max_imgs: 4, roles: ["reference"],
  },
  multi_reference_v: {
    label: "多宫格参考",
    hint: "上传 2~4 张参考图，模型融合多图特征生成视频",
    min_imgs: 2, max_imgs: 4, roles: ["reference"],
  },
};

// ---------- 状态 ----------
const state = {
  mode: "t2v",           // 当前 task_type（t2v / i2v / first_last_frame_v / multi_image_v / multi_reference_v）
  images: [],            // 多图列表：[{ localUrl, role, frame_idx, strength, name, serverPath }]
  numImages: 2,          // 多图模式下的图片数量（2~4）
  availableModes: ["t2v", "i2v"],  // 引擎实际支持的模式（从 /capabilities 动态填充）
  width: 1024,
  height: 576,
  variants: [],
  schemes: [],
  defaultSchemeId: "",
  loadedSchemeId: "",
  generating: false,
  loading: false,
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

// SSE 阶段 → 中文（stage 定义见 progress_reporter.py VideoStage）
const STAGE_TEXT = {
  "video.start": "开始生成",
  "video.model_ready": "模型就绪",
  "video.prompt_processing": "处理提示词",
  "video.generating": "扩散去噪",
  "video.vae_decoding": "VAE 解码视频",
  "video.saving": "保存产物",
  "video.mux_warning": "合成警告",
  "video.completed": "完成",
};

// 方案联动表（scheme YAML 的 offload_level 默认值回显）
const SCHEME_LINKAGE = {
  F16: { offload: "L1" },
  I8: { offload: "L1" },
  G8: { offload: "L1" },
  G4: { offload: "L1" },
};

// ============================================================
// 初始化
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  bindSliders();
  bindSizePresets();
  bindButtons();
  refreshAll();
  setInterval(refreshStatus, 15000);
});

// ============================================================
// 设置区：模式按钮（从 /capabilities 动态渲染）
// ============================================================
function renderModeButtons(engineCaps) {
  // engineCaps 是 ["text-to-video", "image-to-video", ...]（manifest capabilities）
  // 或 engineCaps.capabilities（响应体里可能嵌在对象中）
  let caps = Array.isArray(engineCaps) ? engineCaps : (engineCaps?.capabilities || engineCaps?.variants ? [] : []);
  // fallback：从单数 capability 推导
  if (!caps && engineCaps?.capability) {
    caps = [engineCaps.capability];
  }

  // 映射到 task_type 列表
  const modeList = [];
  for (const cap of caps) {
    const m = CAPABILITY_TO_MODE[cap];
    if (m && MODE_INFO[m]) modeList.push(m);
  }
  // 兜底：至少有 t2v
  if (!modeList.includes("t2v")) modeList.unshift("t2v");

  state.availableModes = modeList;

  // 渲染按钮
  const seg = $("mode-seg");
  seg.innerHTML = "";
  modeList.forEach((mode) => {
    const btn = document.createElement("button");
    btn.className = "seg-btn";
    btn.dataset.mode = mode;
    const info = MODE_INFO[mode];
    btn.textContent = info.label;
    btn.title = info.hint;
    if (mode === state.mode) btn.classList.add("active");
    btn.addEventListener("click", () => switchMode(mode));
    seg.appendChild(btn);
  });

  // 初始化图片区显隐 + 卡片
  applyModeUI();
}

function switchMode(newMode) {
  state.mode = newMode;
  document.querySelectorAll("#mode-seg .seg-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === newMode);
  });
  applyModeUI();
}

function applyModeUI() {
  const info = MODE_INFO[state.mode];
  // 更新模式提示
  $("mode-hint").textContent = info.hint;

  // 图片区显隐
  const imgField = $("images-field");
  if (info.max_imgs > 0) {
    imgField.style.display = "";
    $("images-req-label").textContent = `（${info.min_imgs === info.max_imgs ? info.min_imgs : info.min_imgs + "~" + info.max_imgs} 张必填）`;
    $("images-hint").textContent = info.hint;
    // 多图模式（2~4）显示数量选择
    if (info.max_imgs > info.min_imgs) {
      $("image-cards").dataset.flexible = "1";
      renderImageCountPicker(info.min_imgs, info.max_imgs);
    } else {
      $("image-cards").dataset.flexible = "0";
      state.numImages = info.min_imgs;
    }
    renderImageCards();
  } else {
    imgField.style.display = "none";
    state.images = [];
  }
}

function renderImageCountPicker(minN, maxN) {
  const container = $("images-hint");
  // 替换 hint 为：数量选择器 + hint
  container.innerHTML = "";
  const picker = document.createElement("div");
  picker.className = "image-count-picker";
  picker.style.marginBottom = "6px";
  const label = document.createElement("span");
  label.textContent = "图片数量：";
  label.style.fontSize = "12px";
  label.style.color = "var(--text-dim)";
  picker.appendChild(label);
  for (let n = minN; n <= maxN; n++) {
    const b = document.createElement("button");
    b.className = "count-btn";
    b.textContent = n + " 张";
    if (n === state.numImages) b.classList.add("active");
    b.addEventListener("click", () => {
      state.numImages = n;
      // 超过当前已有图片数时，新增空卡；少了时截断
      if (state.images.length > n) state.images.length = n;
      renderImageCountPicker(minN, maxN);
      renderImageCards();
    });
    picker.appendChild(b);
  }
  container.appendChild(picker);
  const hint = document.createElement("div");
  hint.textContent = MODE_INFO[state.mode].hint;
  hint.style.fontSize = "12px";
  hint.style.color = "var(--text-dim)";
  container.appendChild(hint);
}

// ============================================================
// 参数区：多图卡片管理
// ============================================================
function renderImageCards() {
  const info = MODE_INFO[state.mode];
  const container = $("image-cards");
  container.innerHTML = "";

  // 确定每张图的 role
  const fixedRoles = info.roles.length > 1 ? info.roles : null;  // 只有首尾帧有固定 role
  const total = state.numImages;

  // 确保 state.images 有 total 个元素
  while (state.images.length < total) {
    const idx = state.images.length;
    state.images.push({
      localUrl: null,
      role: fixedRoles ? fixedRoles[idx] : info.roles[0] || "reference",
      frame_idx: 0,       // 引擎端 role_to_frame_idx 会覆盖
      strength: 1.0,
      name: "",
      serverPath: "",
    });
  }

  for (let i = 0; i < total; i++) {
    const card = state.images[i];
    const el = document.createElement("div");
    el.className = "image-card";
    el.dataset.index = i;

    // header：角色标签 + 位置
    const header = document.createElement("div");
    header.className = "card-header";
    const roleSpan = document.createElement("span");
    const roleClass = card.role === "first_frame" ? "role-first" : card.role === "last_frame" ? "role-last" : "role-ref";
    roleSpan.className = "card-role " + roleClass;
    roleSpan.textContent = roleLabel(card.role);
    header.appendChild(roleSpan);
    el.appendChild(header);

    // 上传区 / 缩略图
    if (card.localUrl) {
      const img = document.createElement("img");
      img.className = "card-thumb";
      img.src = card.localUrl;
      img.alt = card.name || "条件图 " + (i + 1);
      img.title = card.name;
      el.appendChild(img);
      // 点击替换
      img.addEventListener("click", () => card.querySelector(".card-drop")?.click());
    } else {
      const drop = document.createElement("div");
      drop.className = "card-drop";
      drop.innerHTML = `<p>📁 拖拽或点击上传</p><p style="font-size:11px;margin-top:4px">第 ${i + 1} 张（${roleLabel(card.role)}）</p>`;
      el.appendChild(drop);
      setupCardDropzone(drop, i);
    }

    container.appendChild(el);
  }
}

function roleLabel(role) {
  return { first_frame: "首帧", last_frame: "尾帧", reference: "参考" }[role] || role;
}

function setupCardDropzone(dropEl, imgIdx) {
  // 点击上传
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.style.display = "none";
  fileInput.addEventListener("change", (e) => {
    if (e.target.files?.[0]) uploadToCard(imgIdx, e.target.files[0]);
  });
  document.body.appendChild(fileInput);

  dropEl.addEventListener("click", () => fileInput.click());

  // 拖拽
  ["dragover", "dragenter"].forEach((ev) => {
    dropEl.addEventListener(ev, (e) => {
      e.preventDefault();
      dropEl.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropEl.addEventListener(ev, (e) => {
      e.preventDefault();
      dropEl.classList.remove("dragover");
    });
  });
  dropEl.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) uploadToCard(imgIdx, file);
  });
}

async function uploadToCard(imgIdx, file) {
  if (imgIdx >= state.images.length) return;
  // 本地预览
  const localUrl = URL.createObjectURL(file);
  state.images[imgIdx].localUrl = localUrl;
  state.images[imgIdx].name = file.name;

  // 上传到引擎（POST /upload）
  try {
    const form = new FormData();
    form.append("file", file);
    const r = await fetch(`${API}/upload`, { method: "POST", body: form });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    state.images[imgIdx].serverPath = data.path || "";
  } catch (err) {
    console.warn("图片上传失败（将尝试 base64 内联）:", err);
    // base64 fallback
    state.images[imgIdx].serverPath = await fileToBase64(file);
  }

  renderImageCards();
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ============================================================
// 参数区：滑块联动 + 帧率提示
// ============================================================
function bindSliders() {
  $("duration-range").addEventListener("input", (e) => {
    $("duration-val").textContent = e.target.value + "s";
  });
  // 自动时长开关：开启后时长由 DurationHead 预测，滑块禁用
  $("auto-duration-check").addEventListener("change", (e) => {
    const on = e.target.checked;
    $("duration-range").disabled = on;
    $("duration-val").textContent = on ? "自动" : $("duration-range").value + "s";
  });
  $("fps-select").addEventListener("change", (e) => {
    const fps = parseInt(e.target.value, 10);
    $("fps-hint").textContent = fps >= 48
      ? "高帧率 48/50 时长上限 10s（已锁定）"
      : "高帧率 48/50 时长上限 10s";
  });
}

// ============================================================
// 参数区：分辨率预设按钮组 + 自定义输入（64 整除对齐）
// 硬约束（T3.5）：两阶段分辨率需被 64 整除（一阶段半分辨率 32 整除）。
// 预设值全部满足约束；自定义输入 change 时自动吸附到最近的 64 倍数。
// ============================================================
const SIZE_ALIGN = 64;
const SIZE_MIN = 256;   // 最小可用尺寸（两侧各 4 个 latent token）
const SIZE_MAX = 2048;

function bindSizePresets() {
  // 预设按钮：点击写入 state + 自定义输入框，切换 active
  document.querySelectorAll("#size-presets .preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#size-presets .preset").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.width = parseInt(btn.dataset.w, 10);
      state.height = parseInt(btn.dataset.h, 10);
      $("width-input").value = state.width;
      $("height-input").value = state.height;
    });
  });

  // 自定义输入：失焦/回车时对齐到 64 的倍数（clamp 到 [SIZE_MIN, SIZE_MAX]）
  const alignInput = (input) => {
    let v = parseInt(input.value, 10);
    if (!Number.isFinite(v)) v = SIZE_ALIGN;
    v = Math.min(SIZE_MAX, Math.max(SIZE_MIN, v));
    v = Math.round(v / SIZE_ALIGN) * SIZE_ALIGN;
    input.value = v;
    return v;
  };
  const onCustomSize = () => {
    state.width = alignInput($("width-input"));
    state.height = alignInput($("height-input"));
    // 自定义值不再匹配任何预设时清除高亮；匹配则回亮
    const matched = document.querySelector(
      `#size-presets .preset[data-w="${state.width}"][data-h="${state.height}"]`
    );
    document.querySelectorAll("#size-presets .preset").forEach((b) => b.classList.remove("active"));
    if (matched) matched.classList.add("active");
  };
  for (const id of ["width-input", "height-input"]) {
    $(id).addEventListener("change", onCustomSize);
    $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $(id).blur(); } });
  }

  // 宽高互换（竖屏/横屏一键切换）
  $("btn-swap-size").addEventListener("click", () => {
    const w = state.width;
    state.width = state.height;
    state.height = w;
    $("width-input").value = state.width;
    $("height-input").value = state.height;
    document.querySelectorAll("#size-presets .preset").forEach((b) => b.classList.remove("active"));
    const matched = document.querySelector(
      `#size-presets .preset[data-w="${state.width}"][data-h="${state.height}"]`
    );
    if (matched) matched.classList.add("active");
  });
}

// ============================================================
// 设置区：方案管理 + 变体过滤
// ============================================================
async function refreshAll() {
  await refreshStatus();
  await refreshCapabilities();
  await refreshSchemes();
}

async function refreshCapabilities() {
  try {
    const r = await fetch(`${API}/capabilities`);
    const cap = await r.json();
    state.variants = cap.variants || [];
    refreshVariantSelect();
    // 动态渲染模式按钮（从 capabilities 复数列表，旧引擎 fallback 到单数 capability）
    renderModeButtons(cap.capabilities?.length ? cap.capabilities : cap);
  } catch {
    $("mode-hint").textContent = "capabilities 获取失败（引擎未启动？）";
  }
}

// 变体过滤：当前方案的 applicable_variants 白名单 ∩ available
function filterVariants() {
  const sid = $("scheme-select").value;
  let applicable = new Set();
  if (sid) {
    const scheme = state.schemes.find((s) => s.id === sid);
    if (scheme?.applicable_variants) applicable = new Set(scheme.applicable_variants);
  }
  return state.variants.filter((v) => {
    if (!v.available) return false;
    if (applicable.size && !applicable.has(v.name)) return false;
    return true;
  });
}

function refreshVariantSelect() {
  const sel = $("variant-select");
  const cur = sel.value;
  const filtered = filterVariants();
  sel.innerHTML = "";
  if (!filtered.length) {
    sel.innerHTML = `<option value="">该方案暂无可用模型，请到引擎详情页下载</option>`;
    return;
  }
  for (const v of filtered) {
    const opt = document.createElement("option");
    opt.value = v.name;
    opt.textContent = `${v.name}（${v.precision || "?"}）`;
    sel.appendChild(opt);
  }
  if (cur && filtered.some((v) => v.name === cur)) sel.value = cur;
  else sel.value = filtered[0].name;
}

async function refreshSchemes() {
  try {
    const r = await fetch(`${API}/schemes`);
    const data = await r.json();
    state.schemes = data.schemes || [];
    state.defaultSchemeId = data.default_scheme_id || "";
    state.loadedSchemeId = data.loaded_scheme_id || "";
    const sel = $("scheme-select");
    sel.innerHTML = "";
    for (const s of state.schemes) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.id} - ${s.name}（${s.precision}）`;
      sel.appendChild(opt);
    }
    // 默认选中：已加载方案 > 默认方案 > 第一个
    const selected = state.loadedSchemeId || state.defaultSchemeId || (state.schemes[0]?.id ?? "");
    if (selected) sel.value = selected;
    updateSchemeDesc();
    // 已加载方案的 offload 默认值回显
    if (state.loadedSchemeId) applySchemeLinkage(state.loadedSchemeId, false);
  } catch {
    $("scheme-desc").textContent = "方案列表获取失败（引擎未启动？）";
  }
}

function updateSchemeDesc() {
  const sid = $("scheme-select").value;
  const s = state.schemes.find((x) => x.id === sid);
  if (!s) { $("scheme-desc").textContent = "—"; return; }
  $("scheme-desc").textContent =
    `${s.name} · ${s.description} · 精度 ${s.precision} · 显存 ${s.vram_required} · offload ${s.offload_level}` +
    (state.loadedSchemeId === sid ? ` ·【当前已加载】` : "");
}

// 方案联动：过滤变体 + offload 默认值
function applySchemeLinkage(sid, setVariant = true) {
  const linkage = SCHEME_LINKAGE[sid?.toUpperCase()];
  if (!linkage) return;
  $("offload-select").value = linkage.offload;
  if (setVariant) refreshVariantSelect();
}

async function saveDefaultScheme() {
  const sid = $("scheme-select").value;
  if (!sid) { showError("请先选择方案"); return; }
  try {
    const r = await fetch(`${API}/schemes/default`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scheme_id: sid }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    state.defaultSchemeId = sid;
    $("scheme-desc").textContent = `✓ ${data.message || `默认方案已设置为 ${sid}`}`;
  } catch (err) {
    showError(`保存默认方案失败：${err.message}`);
  }
}

// ============================================================
// 设置区：加载 / 卸载
// ============================================================
function bindButtons() {
  $("btn-refresh-status").addEventListener("click", refreshAll);
  $("btn-scheme-save").addEventListener("click", saveDefaultScheme);
  $("scheme-select").addEventListener("change", () => {
    updateSchemeDesc();
    applySchemeLinkage($("scheme-select").value);
  });
  $("btn-load").addEventListener("click", loadModel);
  $("btn-unload").addEventListener("click", unloadModel);
  $("btn-generate").addEventListener("click", generate);
}

// /load 请求体（offload 级别随加载固定）
function buildLoadBody() {
  return {
    variant: $("variant-select").value || null,
    inputs: { cpu_offload_level: $("offload-select").value },
  };
}

async function loadModel() {
  if (state.loading) return;
  const variant = $("variant-select").value;
  if (!variant) { showError("请先选择模型变体"); return; }
  state.loading = true;
  $("btn-load").disabled = true;
  $("btn-load").textContent = "加载中…";
  try {
    await postLoad(buildLoadBody());
    await waitLoadComplete();  // /load fire-and-forget，轮询直到 ready
  } catch (err) {
    showError(`模型加载失败：${err.message}`);
  } finally {
    state.loading = false;
    $("btn-load").disabled = false;
    $("btn-load").textContent = "加载模型";
    refreshStatus();
    refreshSchemes();
  }
}

async function unloadModel() {
  try {
    const r = await fetch(`${API}/unload`, { method: "POST" });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
  } catch (err) {
    showError(`模型卸载失败：${err.message}`);
  } finally {
    refreshStatus();
    refreshSchemes();
  }
}

// ============================================================
// 状态轮询
// ============================================================
async function refreshStatus() {
  if (state.generating) return null;
  try {
    const r = await fetch(`${API}/status`);
    const st = await r.json();
    const badge = $("state-badge");
    badge.className = "badge " + (st.state || "");
    badge.textContent = {
      idle: "空闲", loading: "加载中", ready: "就绪",
      inferring: "推理中", error: "异常",
    }[st.state] || st.state;
    const loaded = st.loaded_variant || "";
    const vram = st.extra?.vram_used || st.vram_used || "—";
    const offload = st.extra?.offload_level || "—";
    const dev = st.extra?.device_name || st.extra?.device || "—";
    const scheme = st.extra?.scheme_id || "";
    $("state-detail").textContent =
      `${loaded || "未加载"}${scheme ? " · " + scheme : ""} · offload ${offload} · 显存 ${vram} · ${dev}`;
    return st;
  } catch {
    $("state-badge").className = "badge";
    $("state-badge").textContent = "未连接";
    $("state-detail").textContent = "引擎不可达（127.0.0.1:7008）";
    return null;
  }
}

// ============================================================
// 结果区：生成（auto-load + SSE 流）
// ============================================================
async function generate() {
  if (state.generating) return;

  // ----- 1. 校验 -----
  const prompt = $("prompt-input").value.trim();
  if (!prompt) { showError("请输入正向提示词"); return; }

  // 图片需求校验
  const info = MODE_INFO[state.mode];
  const needImgs = info.max_imgs > 0;
  if (needImgs) {
    const uploaded = state.images.filter((img) => img.serverPath).length;
    if (uploaded < info.min_imgs) {
      showError(`${info.label} 模式需要至少 ${info.min_imgs} 张已上传的图片，当前 ${uploaded} 张`);
      return;
    }
  }

  const variant = $("variant-select").value;
  if (!variant) { showError("请先选择模型变体"); return; }

  // ----- 2. auto-load（未加载自动 /load，variant 不匹配自动切换） -----
  const st = await refreshStatus();
  if (st) {
    if (st.state === "error") {
      showError("引擎处于错误状态，请到引擎详情页查看错误详情或重启引擎后重试");
      return;
    }
    const loaded = st.loaded_variant || "";
    try {
      if (st.state !== "ready" || !loaded) {
        setProgress("正在加载模型…（22B 模型较大，可能需要数分钟）", 0);
        await postLoad(buildLoadBody());
        await waitLoadComplete();
      } else if (loaded !== variant) {
        setProgress(`正在切换模型 ${loaded} → ${variant}…`, 0);
        await fetch(`${API}/unload`, { method: "POST" });
        await postLoad(buildLoadBody());
        await waitLoadComplete();
      }
    } catch (err) {
      showError(`模型加载/切换失败：${err.message}`);
      return;
    }
  }

  // ----- 3. 构造请求体（distilled 约束固化在 UI：8 步 / CFG=1 / 分辨率 64 整除） -----
  const inputs = {
    prompt,
    negative_prompt: $("negative-input").value.trim(),
    task_type: state.mode,
    duration: parseInt($("duration-range").value, 10),
    auto_duration: $("auto-duration-check").checked,
    fps: parseInt($("fps-select").value, 10),
    width: state.width,
    height: state.height,
    num_inference_steps: parseInt($("steps-input").value, 10) || 8,
    guidance_scale: parseFloat($("guidance-input").value) || 1.0,
    seed: parseInt($("seed-input").value, 10) || -1,
    cpu_offload_level: $("offload-select").value,
  };

  // 多图条件（所有需要图片的模式统一构建）
  const images = [];
  for (const img of state.images) {
    if (img.serverPath) {
      images.push({
        path: img.serverPath,
        role: img.role,
        strength: img.strength ?? 1.0,
        // frame_idx 由引擎端 role_to_frame_idx 自动计算，前端传 role 即可
      });
    }
  }

  // 不携带 task_id：直连链路契约（引擎以 request.task_id 是否非空区分
  // gateway 转发/直连）。引擎会自动生成 task_id 上报任务中心全生命周期。
  const body = { variant, inputs, images };

  // ----- 4. SSE 流式推理 -----
  state.generating = true;
  $("btn-generate").disabled = true;
  $("btn-generate").textContent = "生成中…";
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = "0%";
  $("progress-text").textContent = "连接引擎…";
  $("error-box").style.display = "none";
  $("state-badge").className = "badge inferring";
  $("state-badge").textContent = "推理中";

  try {
    const resp = await fetch(`${API}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finalResult = null;
    let finalTiming = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();  // 末行可能不完整，留在缓冲
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        let frame;
        try { frame = JSON.parse(line.slice(5)); } catch { continue; }
        const r = handleFrame(frame);
        if (r) { finalResult = r.result; finalTiming = r.timing; }
      }
    }

    if (finalResult) {
      showResult(finalResult, finalTiming);
    } else {
      throw new Error("SSE 流结束但未收到结果帧");
    }
  } catch (err) {
    showError(`生成失败：${err.message}`);
  } finally {
    state.generating = false;
    $("btn-generate").disabled = false;
    $("btn-generate").textContent = "生成视频";
    $("progress-wrap").style.display = "none";
    refreshStatus();
  }
}

async function postLoad(body) {
  const r = await fetch(`${API}/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

// /load fire-and-forget 轮询：2s 间隔，600s 超时（22B 模型加载慢）
async function waitLoadComplete() {
  const maxWait = 600;
  const pollInterval = 2000;
  let waited = 0;
  while (waited < maxWait) {
    await new Promise((res) => setTimeout(res, pollInterval));
    waited += pollInterval / 1000;
    let st;
    try {
      const r = await fetch(`${API}/status`);
      st = await r.json();
    } catch { continue; }  // 轮询失败继续重试
    if (st.state === "ready") {
      setProgress(`模型加载完成（${waited}s），开始推理…`, 0);
      return;
    }
    if (st.state === "error") {
      throw new Error(`模型加载失败（${waited}s）：${st.error || st.error_message || "未知错误"}`);
    }
    if (st.state === "idle" || st.state === "stopped") {
      throw new Error(`模型加载被中断（${waited}s，state=${st.state}）`);
    }
    setProgress(`正在加载模型… ${waited}s`, 0);
  }
  throw new Error(`模型加载超时（${maxWait}s）`);
}

function setProgress(text, pct) {
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = `${pct}%`;
  $("progress-text").textContent = text;
}

// 处理一帧，返回最终结果（若有）
function handleFrame(f) {
  if (f.stage === "error" || f.stage === "video.error") {
    throw new Error(f.error?.message || "引擎错误");
  }
  const pct = Math.round((f.progress || 0) * 100);
  $("progress-fill").style.width = `${pct}%`;
  let stageText = STAGE_TEXT[f.stage] || f.stage;
  // 扩散去噪阶段带步数（video.generating 含 step/total_steps）
  if (f.stage === "video.generating" && f.total_steps) {
    stageText += ` ${f.step || 0}/${f.total_steps}`;
  }
  // 心跳帧带 elapsed
  if (f.stage === "video.generating" && f.elapsed) {
    stageText += `（${f.elapsed}s）`;
  }
  $("progress-text").textContent = `${stageText} ${pct}%`;
  if ((f.stage === "completed" || f.stage === "video.completed") && f.result) {
    return { result: f.result, timing: f.timing };
  }
  return null;
}

function showResult(result, timing) {
  const view = $("result-view");
  const t = Date.now();  // 防缓存
  // result: {video_url, video_only_url, audio_url, duration, mode, scheme_id, offload_level, seed, is_placeholder}
  const videoUrl = result.video_url ? result.video_url + "?t=" + t : "";
  const videoOnlyUrl = result.video_only_url ? result.video_only_url + "?t=" + t : "";

  let html = "";
  if (videoUrl) {
    html += `
      <div class="video-result">
        <span class="media-label">生成的视频（${MODE_CN[result.mode] || result.mode} 模式，方案 ${result.scheme_id || "—"}）</span>
        <video src="${videoUrl}" controls autoplay></video>
      </div>`;
  }
  view.innerHTML = html || `<p class="placeholder">未收到视频产物</p>`;

  const chips = [
    `<span class="meta-chip">模式 ${MODE_CN[result.mode] || result.mode}</span>`,
    `<span class="meta-chip">方案 ${result.scheme_id || "—"}</span>`,
    `<span class="meta-chip">时长 ${result.duration?.toFixed?.(1) ?? result.duration ?? "—"}s</span>`,
    `<span class="meta-chip">fps ${result.fps ?? "—"}</span>`,
    `<span class="meta-chip">offload ${result.offload_level || $("offload-select").value}</span>`,
    `<span class="meta-chip">seed ${result.seed ?? $("seed-input").value}</span>`,
    `<span class="meta-chip">分辨率 ${state.width}×${state.height}</span>`,
    `<span class="meta-chip">总耗时 ${timing?.total_seconds?.toFixed(1) ?? "—"}s</span>`,
  ];
  if (result.is_placeholder) chips.push(`<span class="meta-chip" style="color:var(--warning)">⚠️ placeholder 实现</span>`);
  if (videoUrl) chips.push(`<span class="meta-chip"><a href="${videoUrl}" download>下载视频</a></span>`);
  if (videoOnlyUrl) chips.push(`<span class="meta-chip"><a href="${videoOnlyUrl}" download>下载纯视频</a></span>`);
  $("result-meta").innerHTML = chips.join("");
}

// ---------- 错误提示（不自动消失，手动关闭） ----------
function showError(msg) {
  $("error-text").textContent = msg;
  $("error-box").style.display = "block";
}
