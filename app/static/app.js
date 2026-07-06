let currentResult = null;
let selectedWindowIndex = null;
let welcomeAnimationFrame = null;
const chartState = {
  points: [],
  fullStart: 0,
  fullEnd: 1,
  viewStart: 0,
  viewEnd: 1,
  yMin: 0,
  yMax: 1,
  yAuto: true,
  drag: null,
};

const el = (id) => document.getElementById(id);
const appShell = el('appShell');

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || 'request failed');
  }
  return data;
}

function setStatus(text) {
  el('statusText').textContent = text;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function renderMarkdown(content) {
  const lines = String(content || '').split(/\r?\n/);
  const html = [];
  let listMode = null;

  const closeList = () => {
    if (listMode) {
      html.push(`</${listMode}>`);
      listMode = null;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      continue;
    }

    const heading = line.match(/^#{1,4}\s+(.+)$/);
    if (heading) {
      closeList();
      html.push(`<div class="md-h">${inlineMarkdown(heading[1])}</div>`);
      continue;
    }

    const ordered = line.match(/^\d+[.)]\s+(.+)$/);
    if (ordered) {
      if (listMode !== 'ol') {
        closeList();
        html.push('<ol>');
        listMode = 'ol';
      }
      html.push(`<li>${inlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (listMode !== 'ul') {
        closeList();
        html.push('<ul>');
        listMode = 'ul';
      }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p class="md-p">${inlineMarkdown(line)}</p>`);
  }
  closeList();
  return html.join('');
}

function formatValue(value) {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return value ?? '-';
}

function renderKeyValues(container, entries) {
  container.innerHTML = '';
  entries.forEach(([label, value]) => {
    const item = document.createElement('div');
    item.className = container.id === 'metricsGrid' ? 'metric' : 'summary-item';
    item.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(formatValue(value))}</strong>`;
    container.appendChild(item);
  });
}

function renderResult(result) {
  currentResult = result;
  selectedWindowIndex = null;
  el('resultMeta').textContent = `${result.summary.dataset} · ${result.summary.total_points} points`;
  el('windowCount').textContent = `${result.windows.length}`;
  el('selectedWindowLabel').textContent = '全局';

  renderKeyValues(el('summaryGrid'), [
    ['异常点', result.summary.anomaly_points],
    ['异常比例', result.summary.anomaly_point_ratio],
    ['异常窗口', result.summary.anomaly_window_count],
    ['阈值', result.summary.threshold],
    ['最高分数', result.summary.score_max],
    ['平均分数', result.summary.score_mean],
  ]);

  renderKeyValues(el('metricsGrid'), [
    ['F1-PA', result.metrics.f1_adjust],
    ['Precision-PA', result.metrics.pc_adjust],
    ['Recall-PA', result.metrics.rc_adjust],
    ['AUC-ROC', result.metrics.auc_roc],
    ['AUC-PR', result.metrics.auc_pr],
    ['耗时(s)', result.metrics.tst],
  ]);

  renderWindows(result.windows);
  resetChartView(result.chart);
}

function renderWindows(windows) {
  const list = el('windowsList');
  list.innerHTML = '';
  windows.slice(0, 120).forEach((window) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'window-item';
    const variables = (window.top_variables || [])
      .slice(0, 3)
      .map((v) => v.name)
      .join(', ');
    item.innerHTML = `
      <header><span>#${window.window_index}</span><span>${window.start}-${window.end}</span></header>
      <p>duration ${window.duration}, peak ${formatValue(window.peak_score)}, mean ${formatValue(window.mean_score)}</p>
      <p>${escapeHtml(variables || 'no variable evidence')}</p>
    `;
    item.addEventListener('click', () => {
      selectedWindowIndex = window.window_index;
      el('selectedWindowLabel').textContent = `窗口 #${selectedWindowIndex}`;
      document.querySelectorAll('.window-item').forEach((node) => node.classList.remove('selected'));
      item.classList.add('selected');
    });
    list.appendChild(item);
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function minMaxBy(items, readValue) {
  let min = Infinity;
  let max = -Infinity;
  items.forEach((item) => {
    const value = Number(readValue(item));
    if (!Number.isFinite(value)) return;
    if (value < min) min = value;
    if (value > max) max = value;
  });
  return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
}

function chartLayout(width, height) {
  const pad = { left: 58, right: 18, top: 20, bottom: 38 };
  const plotLeft = pad.left;
  const plotTop = pad.top;
  const plotRight = Math.max(plotLeft + 1, width - pad.right);
  const plotBottom = Math.max(plotTop + 1, height - pad.bottom);
  return {
    plotLeft,
    plotTop,
    plotRight,
    plotBottom,
    plotWidth: plotRight - plotLeft,
    plotHeight: plotBottom - plotTop,
  };
}

function visibleChartPoints() {
  return (chartState.points || []).filter(
    (point) => point.index >= chartState.viewStart && point.index <= chartState.viewEnd,
  );
}

function syncRangeInputs() {
  const startInput = el('rangeStartInput');
  const endInput = el('rangeEndInput');
  startInput.min = Math.floor(chartState.fullStart);
  startInput.max = Math.ceil(chartState.fullEnd);
  endInput.min = Math.floor(chartState.fullStart);
  endInput.max = Math.ceil(chartState.fullEnd);
  startInput.value = String(Math.round(chartState.viewStart));
  endInput.value = String(Math.round(chartState.viewEnd));
}

function setAutoYRange() {
  const renderPoints = visibleChartPoints();
  const source = renderPoints.length ? renderPoints : chartState.points;
  const rangeInfo = minMaxBy(source || [], (point) => point.score);
  if (!rangeInfo) {
    chartState.yMin = 0;
    chartState.yMax = 1;
    return;
  }

  const range = rangeInfo.max - rangeInfo.min || Math.max(1, Math.abs(rangeInfo.max) * 0.2);
  const pad = range * 0.08;
  chartState.yMin = rangeInfo.min - pad;
  chartState.yMax = rangeInfo.max + pad;
}

function setXRange(start, end, { autoY = false, syncInputs = true } = {}) {
  if (!chartState.points.length) return;

  const fullStart = chartState.fullStart;
  const fullEnd = chartState.fullEnd;
  const fullSpan = Math.max(1, fullEnd - fullStart);
  let nextStart = Number(start);
  let nextEnd = Number(end);

  if (!Number.isFinite(nextStart) || !Number.isFinite(nextEnd)) return;
  if (nextEnd < nextStart) {
    [nextStart, nextEnd] = [nextEnd, nextStart];
  }

  let span = Math.max(1, nextEnd - nextStart);
  if (span >= fullSpan) {
    nextStart = fullStart;
    nextEnd = fullEnd;
  } else {
    if (nextStart < fullStart) {
      nextEnd += fullStart - nextStart;
      nextStart = fullStart;
    }
    if (nextEnd > fullEnd) {
      nextStart -= nextEnd - fullEnd;
      nextEnd = fullEnd;
    }
    nextStart = clamp(nextStart, fullStart, fullEnd - span);
    nextEnd = nextStart + span;
  }

  chartState.viewStart = nextStart;
  chartState.viewEnd = nextEnd;
  if (autoY || chartState.yAuto) {
    chartState.yAuto = true;
    setAutoYRange();
  }
  if (syncInputs) syncRangeInputs();
  requestAnimationFrame(() => drawChart());
}

function resetChartView(points = chartState.points) {
  chartState.points = points || [];
  const rangeInfo = minMaxBy(chartState.points, (point) => point.index);
  chartState.fullStart = rangeInfo ? rangeInfo.min : 0;
  chartState.fullEnd = rangeInfo ? rangeInfo.max : 1;
  chartState.viewStart = chartState.fullStart;
  chartState.viewEnd = chartState.fullEnd;
  chartState.yAuto = true;
  chartState.drag = null;
  setAutoYRange();
  syncRangeInputs();
  requestAnimationFrame(() => drawChart());
}

function applyChartRange() {
  const start = Number(el('rangeStartInput').value);
  const end = Number(el('rangeEndInput').value);
  setXRange(start, end, { autoY: true });
}

function canvasPoint(event) {
  const canvas = el('scoreCanvas');
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
    width: rect.width,
    height: rect.height,
  };
}

function chartAxisAt(event) {
  const point = canvasPoint(event);
  const layout = chartLayout(point.width, point.height);
  if (point.y >= layout.plotBottom && point.x >= layout.plotLeft && point.x <= layout.plotRight) {
    return 'x';
  }
  if (point.x <= layout.plotLeft && point.y >= layout.plotTop && point.y <= layout.plotBottom) {
    return 'y';
  }
  return null;
}

function startChartAxisDrag(event) {
  const axis = chartAxisAt(event);
  if (!axis) return;

  event.preventDefault();
  const canvas = el('scoreCanvas');
  const rect = canvas.getBoundingClientRect();
  const layout = chartLayout(rect.width, rect.height);
  chartState.drag = {
    axis,
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    viewStart: chartState.viewStart,
    viewEnd: chartState.viewEnd,
    yMin: chartState.yMin,
    yMax: chartState.yMax,
    plotWidth: layout.plotWidth,
    plotHeight: layout.plotHeight,
  };
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add('dragging');
}

function moveChartAxisDrag(event) {
  const canvas = el('scoreCanvas');
  const drag = chartState.drag;
  if (!drag) {
    const axis = chartAxisAt(event);
    canvas.style.cursor = axis === 'x' ? 'ew-resize' : axis === 'y' ? 'ns-resize' : 'default';
    return;
  }

  event.preventDefault();
  if (drag.axis === 'x') {
    const span = drag.viewEnd - drag.viewStart;
    const delta = -((event.clientX - drag.startX) / Math.max(1, drag.plotWidth)) * span;
    setXRange(drag.viewStart + delta, drag.viewEnd + delta, { syncInputs: true });
    return;
  }

  const range = drag.yMax - drag.yMin || 1;
  const delta = ((event.clientY - drag.startY) / Math.max(1, drag.plotHeight)) * range;
  chartState.yMin = drag.yMin + delta;
  chartState.yMax = drag.yMax + delta;
  chartState.yAuto = false;
  requestAnimationFrame(() => drawChart());
}

function endChartAxisDrag(event) {
  const canvas = el('scoreCanvas');
  if (chartState.drag && canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  chartState.drag = null;
  canvas.classList.remove('dragging');
}

function drawChart(points = chartState.points) {
  if (points) {
    chartState.points = points;
  }

  const canvas = el('scoreCanvas');
  const viewport = el('chartViewport');
  const width = Math.max(1, viewport.clientWidth);
  const height = Math.max(1, viewport.clientHeight);
  const ratio = window.devicePixelRatio || 1;

  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const layout = chartLayout(width, height);
  const renderPoints = visibleChartPoints();

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, layout.plotTop, layout.plotLeft, layout.plotHeight);
  ctx.fillRect(layout.plotLeft, layout.plotBottom, layout.plotWidth, height - layout.plotBottom);

  if (!chartState.points.length) {
    ctx.fillStyle = '#9ba7b4';
    ctx.font = '13px system-ui';
    ctx.fillText('暂无数据', layout.plotLeft, layout.plotTop + 18);
    return;
  }

  const yMin = chartState.yMin;
  const yMax = chartState.yMax;
  const yRange = yMax - yMin || 1;
  const xRange = chartState.viewEnd - chartState.viewStart || 1;
  const x = (index) => layout.plotLeft + ((index - chartState.viewStart) / xRange) * layout.plotWidth;
  const y = (score) => layout.plotBottom - ((score - yMin) / yRange) * layout.plotHeight;

  ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const yy = layout.plotTop + (layout.plotHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(layout.plotLeft, yy);
    ctx.lineTo(layout.plotRight, yy);
    ctx.stroke();
  }

  renderPoints.forEach((point) => {
    if (point.pred === 1) {
      ctx.fillStyle = 'rgba(249, 115, 22, 0.11)';
      const px = x(point.index);
      ctx.fillRect(px, layout.plotTop, Math.max(1.2, layout.plotWidth / Math.max(1, renderPoints.length)), layout.plotHeight);
    }
  });

  if (renderPoints.length) {
    const gradient = ctx.createLinearGradient(layout.plotLeft, 0, layout.plotRight, 0);
    gradient.addColorStop(0, '#2563eb');
    gradient.addColorStop(0.45, '#10b981');
    gradient.addColorStop(1, '#ec4899');
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.beginPath();
    renderPoints.forEach((point, index) => {
      const px = x(point.index);
      const py = y(point.score);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    ctx.shadowColor = 'rgba(59, 130, 246, 0.22)';
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  ctx.strokeStyle = '#d7dee9';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(layout.plotLeft, layout.plotTop);
  ctx.lineTo(layout.plotLeft, layout.plotBottom);
  ctx.lineTo(layout.plotRight, layout.plotBottom);
  ctx.stroke();

  ctx.fillStyle = '#737b86';
  ctx.font = '12px system-ui';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i += 1) {
    const value = yMax - (yRange / 4) * i;
    const yy = layout.plotTop + (layout.plotHeight / 4) * i;
    ctx.fillText(value.toFixed(3), layout.plotLeft - 8, yy);
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i += 1) {
    const value = chartState.viewStart + (xRange / 4) * i;
    const xx = layout.plotLeft + (layout.plotWidth / 4) * i;
    ctx.fillText(String(Math.round(value)), xx, layout.plotBottom + 9);
  }
}

async function runDetection() {
  if (window.location.protocol === 'file:') {
    setStatus('静态预览模式：请通过 http://localhost:8000 打开页面后运行检测');
    return;
  }

  const button = el('runDetectBtn');
  button.disabled = true;
  setStatus('检测运行中');
  try {
    const result = await api('/api/detect', {
      method: 'POST',
      body: JSON.stringify({
        detector_method: el('detectorSelect').value,
        dataset: el('datasetSelect').value,
        threshold_setting: el('thresholdSelect').value,
        anomaly_ratio: Number(el('anomalyRatioInput').value),
        win_size: Number(el('winSizeInput').value),
        batch_size: Number(el('batchSizeInput').value),
        max_chart_points: 100000,
      }),
    });
    renderResult(result);
    setStatus(`检测完成：${result.id}`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    button.disabled = false;
  }
}

function addMessage(role, content) {
  const node = document.createElement('div');
  node.className = `message ${role}`;
  if (role === 'assistant') {
    node.innerHTML = renderMarkdown(content);
  } else {
    node.textContent = content;
  }
  el('chatMessages').appendChild(node);
  el('chatMessages').scrollTop = el('chatMessages').scrollHeight;
}

async function sendChat(event) {
  event.preventDefault();
  if (window.location.protocol === 'file:') {
    addMessage('assistant', '当前是静态预览模式。请启动 FastAPI 后端，并通过 http://localhost:8000 打开页面再使用 LLM 分析。');
    return;
  }

  const input = el('chatInput');
  const message = input.value.trim();
  if (!message || !currentResult) return;

  input.value = '';
  addMessage('user', message);
  try {
    const response = await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        result_id: currentResult.id,
        message,
        window_index: selectedWindowIndex,
        model: el('llmModelSelect').value,
      }),
    });
    addMessage('assistant', response.answer);
  } catch (error) {
    addMessage('assistant', error.message);
  }
}

async function loadDatasets() {
  if (window.location.protocol === 'file:') {
    setStatus('静态预览模式：后端启动后可运行检测');
    return;
  }

  try {
    const data = await api('/api/datasets');
    const select = el('datasetSelect');
    select.innerHTML = '';
    data.datasets.forEach((dataset) => {
      const option = document.createElement('option');
      option.value = dataset.name;
      option.textContent = dataset.name;
      select.appendChild(option);
    });
  } catch (error) {
    setStatus(error.message);
  }
}

function showApp() {
  el('welcomeScreen').classList.add('hidden');
  appShell.classList.remove('hidden');
  document.body.style.overflow = '';
  cancelAnimationFrame(welcomeAnimationFrame);
  requestAnimationFrame(() => currentResult && drawChart());
}

function setSidebarVisible(visible) {
  appShell.classList.toggle('sidebar-collapsed', !visible);
  clampChatPanelWidth();
  requestAnimationFrame(() => currentResult && drawChart());
}

function setChatVisible(visible) {
  appShell.classList.toggle('chat-collapsed', !visible);
  if (visible) clampChatPanelWidth();
  requestAnimationFrame(() => currentResult && drawChart());
}

function chatWidthBounds() {
  if (window.innerWidth <= 1180) {
    return { min: 300, max: 720 };
  }

  const hasSidebar = !appShell.classList.contains('sidebar-collapsed');
  const shellPadding = 28;
  const resizerWidth = 8;
  const gap = 14;
  const sidebarWidth = hasSidebar ? 292 : 0;
  const gapTotal = hasSidebar ? gap * 3 : gap * 2;
  const minWorkspace = hasSidebar ? 560 : 680;
  const layoutMax = window.innerWidth - shellPadding - sidebarWidth - resizerWidth - gapTotal - minWorkspace;
  return {
    min: 300,
    max: Math.max(300, Math.min(720, Math.floor(window.innerWidth * 0.55), Math.floor(layoutMax))),
  };
}

function setChatPanelWidth(width, { persist = true } = {}) {
  const bounds = chatWidthBounds();
  const next = clamp(Number(width) || 380, bounds.min, bounds.max);
  appShell.style.setProperty('--chat-width', `${next}px`);
  if (persist) {
    localStorage.setItem('tsad-chat-width', String(next));
  }
  return next;
}

function clampChatPanelWidth() {
  const saved = Number(localStorage.getItem('tsad-chat-width'));
  const current = saved || Number.parseFloat(getComputedStyle(appShell).getPropertyValue('--chat-width')) || 380;
  setChatPanelWidth(current);
}

function setupChatResizer() {
  const resizer = el('chatResizer');
  let dragging = false;

  const setWidth = (clientX) => {
    setChatPanelWidth(window.innerWidth - clientX - 14);
    requestAnimationFrame(() => currentResult && drawChart());
  };

  const saved = Number(localStorage.getItem('tsad-chat-width'));
  setChatPanelWidth(saved || 380, { persist: Boolean(saved) });

  resizer.addEventListener('pointerdown', (event) => {
    dragging = true;
    resizer.classList.add('dragging');
    resizer.setPointerCapture(event.pointerId);
  });

  resizer.addEventListener('pointermove', (event) => {
    if (dragging) setWidth(event.clientX);
  });

  resizer.addEventListener('pointerup', () => {
    dragging = false;
    resizer.classList.remove('dragging');
  });
}

function animateWelcome() {
  const canvas = el('welcomeCanvas');
  const ctx = canvas.getContext('2d');
  const previewCanvas = el('previewChartCanvas');
  const previewCtx = previewCanvas.getContext('2d');
  const ratio = window.devicePixelRatio || 1;

  const resizeCanvas = (target, targetCtx) => {
    const rect = target.getBoundingClientRect();
    target.width = Math.max(1, Math.floor(rect.width * ratio));
    target.height = Math.max(1, Math.floor(rect.height * ratio));
    targetCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  };

  const resize = () => {
    resizeCanvas(canvas, ctx);
    resizeCanvas(previewCanvas, previewCtx);
  };
  resize();

  const traces = Array.from({ length: 52 }, (_, index) => ({
    x: (index / 52) * window.innerWidth,
    y: Math.random() * window.innerHeight,
    speed: 0.18 + Math.random() * 0.44,
    amp: 18 + Math.random() * 70,
  }));

  const drawPreviewChart = (time) => {
    const width = previewCanvas.clientWidth;
    const height = previewCanvas.clientHeight;
    const mode = Math.floor(time / 2600) % 3;
    const phase = (time % 2600) / 2600;
    previewCtx.clearRect(0, 0, width, height);

    const gradient = previewCtx.createLinearGradient(22, 0, width - 22, 0);
    gradient.addColorStop(0, '#2563eb');
    gradient.addColorStop(0.45, '#10b981');
    gradient.addColorStop(0.72, '#8b5cf6');
    gradient.addColorStop(1, '#ec4899');

    if (mode === 0) {
      const bars = 9;
      for (let i = 0; i < bars; i += 1) {
        const x = 34 + i * ((width - 68) / (bars - 1));
        const base = 0.22 + ((i * 17) % 5) * 0.1;
        const wave = 0.18 + Math.abs(Math.sin(time * 0.0022 + i * 1.4)) * 0.54;
        const barHeight = height * Math.min(0.86, base + wave);
        previewCtx.strokeStyle = i % 3 === 1 ? '#ec4899' : i % 3 === 2 ? '#10b981' : '#3b82f6';
        previewCtx.lineWidth = 3;
        previewCtx.lineCap = 'round';
        previewCtx.beginPath();
        previewCtx.moveTo(x, height - 34);
        previewCtx.lineTo(x, height - 34 - barHeight * 0.62);
        previewCtx.stroke();
      }
      return;
    }

    if (mode === 1) {
      previewCtx.strokeStyle = gradient;
      previewCtx.lineWidth = 2.4;
      previewCtx.lineCap = 'round';
      previewCtx.beginPath();
      for (let i = 0; i < 92; i += 1) {
        const t = i / 91;
        const x = 24 + t * (width - 48);
        const spike = [0.18, 0.43, 0.69].some((center) => Math.abs(t - center) < 0.018) ? 40 : 0;
        const y = height * 0.62
          + Math.sin(t * 12 + time * 0.002) * 16
          + Math.sin(t * 38 + time * 0.004) * 7
          - spike;
        if (i === 0) previewCtx.moveTo(x, y);
        else previewCtx.lineTo(x, y);
      }
      previewCtx.stroke();
      return;
    }

    for (let i = 0; i < 34; i += 1) {
      const x = 18 + i * ((width - 36) / 34);
      const alpha = 0.04 + Math.abs(Math.sin(i * 0.7 + phase * Math.PI * 2)) * 0.14;
      previewCtx.fillStyle = `rgba(249, 115, 22, ${alpha})`;
      previewCtx.fillRect(x, 26, Math.max(2, width / 120), height - 58);
    }
    previewCtx.strokeStyle = gradient;
    previewCtx.lineWidth = 2;
    previewCtx.beginPath();
    for (let i = 0; i < 100; i += 1) {
      const t = i / 99;
      const x = 24 + t * (width - 48);
      const y = height * 0.72 - Math.sin(t * 10 + time * 0.002) * 14 - Math.max(0, Math.sin(t * 24 - phase * 4)) * 34;
      if (i === 0) previewCtx.moveTo(x, y);
      else previewCtx.lineTo(x, y);
    }
    previewCtx.stroke();
  };

  const draw = (time) => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    const palette = ['rgba(37, 99, 235, 0.18)', 'rgba(16, 185, 129, 0.18)', 'rgba(236, 72, 153, 0.16)', 'rgba(249, 115, 22, 0.16)'];
    ctx.lineWidth = 1;

    traces.forEach((trace, index) => {
      ctx.strokeStyle = palette[index % palette.length];
      trace.x += trace.speed;
      if (trace.x > width + 80) trace.x = -80;
      const y = trace.y + Math.sin(time * 0.001 + index) * trace.amp;
      ctx.beginPath();
      ctx.moveTo(trace.x - 90, y);
      ctx.lineTo(trace.x, y + Math.sin(index) * 16);
      ctx.stroke();
    });

    drawPreviewChart(time);
    welcomeAnimationFrame = requestAnimationFrame(draw);
  };

  window.addEventListener('resize', resize);
  welcomeAnimationFrame = requestAnimationFrame(draw);
}

el('startAppBtn').addEventListener('click', showApp);
el('runDetectBtn').addEventListener('click', runDetection);
el('chatForm').addEventListener('submit', sendChat);
el('hideSidebarBtn').addEventListener('click', () => setSidebarVisible(false));
el('toggleSidebarBtn').addEventListener('click', () => setSidebarVisible(appShell.classList.contains('sidebar-collapsed')));
el('hideChatBtn').addEventListener('click', () => setChatVisible(false));
el('toggleChatBtn').addEventListener('click', () => setChatVisible(appShell.classList.contains('chat-collapsed')));
el('applyRangeBtn').addEventListener('click', applyChartRange);
el('resetRangeBtn').addEventListener('click', () => resetChartView());
['rangeStartInput', 'rangeEndInput'].forEach((id) => {
  el(id).addEventListener('keydown', (event) => {
    if (event.key === 'Enter') applyChartRange();
  });
});
el('scoreCanvas').addEventListener('pointerdown', startChartAxisDrag);
el('scoreCanvas').addEventListener('pointermove', moveChartAxisDrag);
el('scoreCanvas').addEventListener('pointerup', endChartAxisDrag);
el('scoreCanvas').addEventListener('pointercancel', endChartAxisDrag);
el('scoreCanvas').addEventListener('pointerleave', (event) => {
  if (!chartState.drag) event.currentTarget.style.cursor = 'default';
});
window.addEventListener('resize', () => {
  clampChatPanelWidth();
  if (currentResult) drawChart();
});

setupChatResizer();
animateWelcome();
loadDatasets();
