const colors = {
  observed: "#182a35",
  gcn_lstm: "#126cbe",
  gcn_gru: "#7357b3",
  gat_lstm: "#c85c2c",
  gat_gru: "#118b78",
  persistence: "#788995",
  window_mean: "#ad6a62",
};

const ui = Object.fromEntries([
  "group", "graph", "horizon", "time-range", "sensor-filter", "sensor", "selection-note",
  "fact-sensor", "fact-sensor-detail", "fact-node", "fact-node-detail",
  "fact-horizon", "fact-horizon-detail", "fact-observed", "fact-observed-detail",
  "period-tag", "chart", "chart-subtitle", "chart-caption", "tooltip", "legend",
  "scores", "leader-tag", "metric-note", "download-png", "download-csv",
  "config-picker", "chosen-count", "chosen-models", "ranking-progress", "ranking-note",
  "ranking-mse", "ranking-mae", "ranking-prev", "ranking-next", "ranking-page",
  "ranking-phase", "filter-architecture", "filter-model", "filter-alpha", "filter-layers", "filter-hidden",
  "show-chart", "comparison-architecture", "comparison-criterion", "comparison-chart", "comparison-rows", "comparison-note",
].map((id) => [id, document.getElementById(id)]));

const state = { meta: null, sensors: [], selectedId: null, payload: null, series: [],
  visible: new Map(), sensorRequest: 0, seriesRequest: 0, rankRequest: 0, compareRequest: 0,
  cursor: -1, plot: null, view: "curves", selectedConfigs: [], chosenMeta: new Map(), rankOffset: 0 };
const dateFormatter = new Intl.DateTimeFormat("es-ES", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });

function niceDate(value) {
  return dateFormatter.format(new Date(value));
}

function setStatus(message, error = false) {
  ui["selection-note"].textContent = message;
  ui["selection-note"].dataset.tone = error ? "error" : "normal";
}

async function fetchJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Error HTTP ${response.status}`);
  return data;
}

function addOptions(select, options) {
  const fragment = document.createDocumentFragment();
  for (const entry of options) {
    const option = document.createElement("option");
    option.value = entry.key;
    option.textContent = entry.label;
    fragment.append(option);
  }
  select.replaceChildren(fragment);
}

function normalize(text) {
  return String(text).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("es");
}

function sensorOptions() {
  const query = normalize(ui["sensor-filter"].value.trim());
  const matching = state.sensors.filter((row) => !query || normalize(`${row.id} ${row.name}`).includes(query));
  const shown = matching.slice(0, 120);
  const selected = state.sensors.find((row) => row.id === state.selectedId);
  if (selected && !shown.some((row) => row.id === selected.id)) shown.unshift(selected);
  addOptions(ui.sensor, shown.map((row) => ({ key: String(row.id), label: `${row.id} · ${row.name || "Nombre no disponible"}` })));
  if (selected) ui.sensor.value = String(selected.id);
  if (!matching.length) setStatus("Ningún sensor coincide con el filtro. Se mantiene el sensor actual.");
  else setStatus(`Sensores en esta agrupación: ${state.sensors.length.toLocaleString("es-ES")}. ${query ? `${matching.length} coinciden;` : ""} mostrando hasta 120. Selecciona un sensor para actualizar la gráfica.`);
}

function configLabel(row) {
  return `${state.meta.models.find((item) => item.key === row.model)?.label ?? row.model} · ${row.architecture === "pfm" ? "PFM" : "GitHub"} · ${row.graph === "correlation" ? "Correlación" : "Cercanía"} · α=${row.time_alpha} · ${row.layers}C / ${row.hidden_feats}D`;
}

function renderChosen() {
  if (!state.meta?.configurations) return;
  ui["chosen-count"].textContent = `${state.selectedConfigs.length} / 4`;
  const container = document.createDocumentFragment();
  for (const id of state.selectedConfigs) {
    const row = state.chosenMeta.get(id);
    const chip = document.createElement("span");
    chip.className = "chosen-chip";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "×";
    button.setAttribute("aria-label", `Quitar ${row ? configLabel(row) : id}`);
    button.addEventListener("click", () => {
      state.selectedConfigs = state.selectedConfigs.filter((key) => key !== id);
      renderChosen();
      loadSeries();
      loadRankings();
    });
    chip.append(document.createTextNode(row ? configLabel(row) : id), button);
    container.append(chip);
  }
  ui["chosen-models"].replaceChildren(container);
}

function renderRankingRows(key, rows, offset) {
  const fragment = document.createDocumentFragment();
  for (const [index, row] of rows.entries()) {
    state.chosenMeta.set(row.id, row);
    const tr = document.createElement("tr");
    const values = [String(offset + index + 1), configLabel(row), formatScore(row.mse), formatScore(row.mae)];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      tr.append(cell);
    }
    const cell = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "action-button";
    button.disabled = !row.curves_ready;
    button.textContent = !row.curves_ready ? "Pendiente" : state.selectedConfigs.includes(row.id) ? "Quitar" : "Añadir";
    button.addEventListener("click", () => {
      if (state.selectedConfigs.includes(row.id)) state.selectedConfigs = state.selectedConfigs.filter((id) => id !== row.id);
      else if (state.selectedConfigs.length < 4) state.selectedConfigs.push(row.id);
      else { ui["ranking-note"].textContent = "Máximo cuatro curvas. Quita una antes de añadir otra."; return; }
      renderChosen();
      renderRankingRows("mse", state.rankingData.mse, state.rankOffset);
      renderRankingRows("mae", state.rankingData.mae, state.rankOffset);
      loadSeries();
    });
    cell.append(button);
    tr.append(cell);
    fragment.append(tr);
  }
  if (!rows.length) {
    const tr = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "Sin resultados para estos filtros o aún pendientes de evaluación.";
    tr.append(cell);
    fragment.append(tr);
  }
  ui[`ranking-${key}`].replaceChildren(fragment);
}

async function loadRankings() {
  if (!state.meta?.configurations) return;
  const token = ++state.rankRequest;
  const params = new URLSearchParams({ group: ui.group.value, graph: ui.graph.value, phase: ui["ranking-phase"].value,
    horizon: ui.horizon.value, offset: String(state.rankOffset), limit: "25" });
  for (const [id, key] of [["filter-architecture", "architecture"], ["filter-model", "model"],
                           ["filter-alpha", "time_alpha"], ["filter-layers", "layers"], ["filter-hidden", "hidden_feats"]]) {
    if (ui[id].value) params.set(key, ui[id].value);
  }
  try {
    const data = await fetchJSON(`/api/rankings?${params}`);
    if (token !== state.rankRequest) return;
    state.rankingData = data;
    renderRankingRows("mse", data.mse, data.offset);
    renderRankingRows("mae", data.mae, data.offset);
    renderChosen();
    ui["ranking-progress"].textContent = `${data.precompute.completed}/${data.precompute.total} evaluados`;
    ui["ranking-page"].textContent = `${Math.floor(data.offset / data.limit) + 1} / ${Math.max(1, Math.ceil((data.phase === "test" ? data.evaluated : data.total) / data.limit))}`;
    ui["ranking-prev"].disabled = data.offset === 0;
    ui["ranking-next"].disabled = data.offset + data.limit >= (data.phase === "test" ? data.evaluated : data.total);
    ui["ranking-note"].textContent = `${data.evaluated}/${data.total} configuraciones en ${data.phase === "test" ? "prueba (ranking descriptivo)" : "validación (elección original)"}. MSE y MAE incluyen solo observaciones reales; horizonte +${15 * Number(data.horizon)} min.`;
  } catch (error) {
    if (token === state.rankRequest) ui["ranking-note"].textContent = `No se pudo cargar el ranking: ${error.message}`;
  }
}

function drawComparison(data) {
  const canvas = ui["comparison-chart"];
  const width = Math.max(570, canvas.clientWidth);
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const bounds = { left: 60, right: width - 18, top: 25, bottom: height - 88 };
  const maximum = Math.max(1.15, ...data.rows.map((row) => row.ratio ?? 0)) * 1.12;
  const yAt = (value) => bounds.bottom - value / maximum * (bounds.bottom - bounds.top);
  ctx.font = "12px sans-serif";
  ctx.textAlign = "right";
  for (let step = 0; step <= 4; step++) {
    const value = maximum * step / 4;
    const y = yAt(value);
    ctx.strokeStyle = "#e2e9e9";
    ctx.beginPath();
    ctx.moveTo(bounds.left, y);
    ctx.lineTo(bounds.right, y);
    ctx.stroke();
    ctx.fillStyle = "#526770";
    ctx.fillText(value.toFixed(2), bounds.left - 10, y + 4);
  }
  ctx.setLineDash([7, 5]);
  ctx.strokeStyle = "#1d3036";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(bounds.left, yAt(1));
  ctx.lineTo(bounds.right, yAt(1));
  ctx.stroke();
  ctx.setLineDash([]);
  const step = (bounds.right - bounds.left) / data.rows.length;
  const bar = Math.min(step * .68, 110);
  data.rows.forEach((row, index) => {
    const center = bounds.left + step * (index + .5);
    if (row.ratio !== null) {
      ctx.fillStyle = row.graph === "correlation" ? "#ef7836" : "#4477aa";
      ctx.fillRect(center - bar / 2, yAt(row.ratio), bar, bounds.bottom - yAt(row.ratio));
      ctx.fillStyle = "#253840";
      ctx.textAlign = "center";
      ctx.fillText(row.ratio.toFixed(3), center, yAt(row.ratio) - 9);
    } else {
      ctx.fillStyle = "#8b999d";
      ctx.textAlign = "center";
      ctx.fillText("Pendiente", center, yAt(.28));
    }
    ctx.save();
    ctx.translate(center, bounds.bottom + 14);
    ctx.rotate(-Math.PI / 5);
    ctx.textAlign = "right";
    ctx.fillStyle = "#3c535a";
    ctx.fillText(`${row.graph === "correlation" ? "Correlación" : "Cercanía"} · ${state.meta.models.find((item) => item.key === row.model)?.label}`, 0, 0);
    ctx.restore();
  });
}

async function loadComparison() {
  if (!state.meta?.configurations) return;
  const token = ++state.compareRequest;
  const params = new URLSearchParams({ group: ui.group.value, graph: ui.graph.value, horizon: ui.horizon.value,
    architecture: ui["comparison-architecture"].value, criterion: ui["comparison-criterion"].value });
  try {
    const data = await fetchJSON(`/api/comparison?${params}`);
    if (token !== state.compareRequest) return;
    state.comparisonData = data;
    if (state.view === "comparison") drawComparison(data);
    const fragment = document.createDocumentFragment();
    for (const row of data.rows) {
      const item = document.createElement("span");
      const model = state.meta.models.find((entry) => entry.key === row.model)?.label ?? row.model;
      item.textContent = `${model} · ${row.graph === "correlation" ? "Correlación" : "Cercanía"} · α=${row.time_alpha} · ${row.layers}C/${row.hidden_feats}D · ${row.ratio?.toFixed(3) ?? "pendiente"}`;
      fragment.append(item);
    }
    ui["comparison-rows"].replaceChildren(fragment);
    ui["comparison-note"].textContent = `${data.description} +${15 * Number(data.horizon)} min. Barras azules: cercanía; naranjas: correlación.`;
  } catch (error) {
    if (token === state.compareRequest) ui["comparison-note"].textContent = `No se pudo cargar la comparación: ${error.message}`;
  }
}

function setView(view) {
  state.view = view;
  for (const button of document.querySelectorAll(".view-tabs [role=tab]")) {
    const active = button.dataset.view === view;
    button.setAttribute("aria-selected", String(active));
    document.getElementById(`${button.dataset.view === "curves" ? "curve" : button.dataset.view}-view`).hidden = !active;
  }
  if (view === "comparison" && state.comparisonData) drawComparison(state.comparisonData);
  if (view === "comparison") loadComparison();
  if (view === "rankings") loadRankings();
}

async function loadGroup(preferredSensor = null) {
  const token = ++state.sensorRequest;
  ++state.seriesRequest;
  ui.sensor.disabled = true;
  ui["sensor-filter"].disabled = true;
  ui["download-png"].disabled = true;
  ui["download-csv"].disabled = true;
  setStatus("Leyendo los sensores de esta agrupación...");
  try {
    const data = await fetchJSON(`/api/sensors?group=${encodeURIComponent(ui.group.value)}`);
    if (token !== state.sensorRequest) return;
    state.sensors = data.sensors;
    if (!state.sensors.length) throw new Error("Esta agrupación no contiene sensores seleccionables.");
    state.selectedId = state.sensors.some((row) => row.id === preferredSensor) ? preferredSensor : state.sensors[0].id;
    ui["sensor-filter"].value = "";
    sensorOptions();
    ui.sensor.disabled = false;
    ui["sensor-filter"].disabled = false;
    await loadSeries();
    if (state.meta.configurations) {
      await loadRankings();
      await loadComparison();
    }
  } catch (error) {
    if (token === state.sensorRequest) setStatus(`No se pudieron cargar los sensores: ${error.message}`, true);
  }
}

async function loadSeries() {
  if (state.selectedId === null) return;
  const token = ++state.seriesRequest;
  ui["download-png"].disabled = true;
  ui["download-csv"].disabled = true;
  state.payload = null;
  state.plot = null;
  state.cursor = -1;
  ui.tooltip.hidden = true;
  ui["leader-tag"].textContent = "Cargando comparación...";
  const ctx = ui.chart.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ui.chart.width, ui.chart.height);
  ctx.font = "15px -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif";
  ctx.fillStyle = "#647680";
  ctx.textAlign = "center";
  ctx.fillText("Cargando curvas del sensor...", ui.chart.width / 2, ui.chart.height / 2);
  const loadingRow = document.createElement("tr");
  const loadingCell = document.createElement("td");
  loadingCell.colSpan = 4;
  loadingCell.textContent = "Calculando errores del sensor...";
  loadingRow.append(loadingCell);
  ui.scores.replaceChildren(loadingRow);
  setStatus("Preparando las curvas y los errores de este sensor...");
  const selected = state.meta.configurations && state.selectedConfigs.length
    ? `&configs=${encodeURIComponent(state.selectedConfigs.join(","))}` : "";
  const url = `/api/series?group=${encodeURIComponent(ui.group.value)}&graph=${encodeURIComponent(ui.graph.value)}&sensor=${state.selectedId}&range=${ui["time-range"].value}&horizon=${ui.horizon.value}${selected}`;
  try {
    const payload = await fetchJSON(url);
    if (token !== state.seriesRequest) return;
    state.payload = payload;
    state.cursor = -1;
    ui.tooltip.hidden = true;
    state.series = buildSeries(payload);
    if (state.meta.configurations && !state.selectedConfigs.length) {
      state.selectedConfigs = payload.scores.slice(0, 4).map((row) => row.id.replace(/^final__/, ""));
      renderChosen();
    }
    renderLegend();
    renderDetails(payload);
    renderScores(payload);
    drawChart();
    ui["download-png"].disabled = false;
    ui["download-csv"].disabled = false;
    const group = state.meta.groups.find((item) => item.key === payload.group);
    const graph = state.meta.graphs.find((item) => item.key === payload.adjacency);
    setStatus(`${group.label} · ${graph.label}. Sensor ${payload.sensor.id}; ${payload.observed_count} fechas con medición de las ${payload.points} mostradas.`);
    const current = new URL(window.location.href);
    current.searchParams.set("group", payload.group);
    current.searchParams.set("graph", payload.adjacency);
    current.searchParams.set("sensor", String(payload.sensor.id));
    current.searchParams.set("range", payload.range);
    current.searchParams.set("horizon", String(ui.horizon.value));
    window.history.replaceState(null, "", current);
  } catch (error) {
    if (token === state.seriesRequest) setStatus(`No se puede mostrar esta comparación: ${error.message}`, true);
  }
}

function buildSeries(payload) {
  const output = [{ key: "observed", label: "Índice PFM observado", compact: "Observado", values: payload.target,
    color: colors.observed, width: 2.7, dash: [], fixed: true }];
  for (const row of payload.scores) {
    const graph = payload.adjacency === "both" ? (row.graph === "correlation" ? " (correlación)" : " (cercanía)") : "";
    output.push({ key: row.id, label: row.label, compact: row.architecture ? row.label.replace("GitHub bidireccional", "GitHub").replace("Correlación", "Corr.").replace("Cercanía", "Cerc.") : `${state.meta.models.find((item) => item.key === row.model).label}${graph}`,
      model: row.model, graph: row.graph, values: payload.prediction[row.id], color: colors[row.model], width: 1.9,
      dash: row.graph === "correlation" ? [7, 4] : [] });
  }
  output.push({ key: "persistence", label: "Repetir el último valor", compact: "Persistencia", values: payload.persistence,
    color: colors.persistence, width: 1.8, dash: [3, 5] });
  output.push({ key: "window_mean", label: "Media de los últimos 12", compact: "Media reciente", values: payload.window_mean,
    color: colors.window_mean, width: 1.6, dash: [2, 5] });
  for (const row of output) if (!state.visible.has(row.key)) state.visible.set(row.key, row.key !== "window_mean");
  return output;
}

function renderLegend() {
  const fragment = document.createDocumentFragment();
  for (const row of state.series) {
    const item = document.createElement(row.fixed ? "span" : "button");
    item.className = row.fixed ? "legend-static" : "";
    if (!row.fixed) {
      item.type = "button";
      item.setAttribute("aria-pressed", String(state.visible.get(row.key)));
      item.classList.toggle("is-muted", !state.visible.get(row.key));
      item.title = "Mostrar u ocultar esta curva";
      item.addEventListener("click", () => {
        state.visible.set(row.key, !state.visible.get(row.key));
        renderLegend();
        drawChart();
        if (state.cursor >= 0) showTooltip(state.cursor);
      });
    }
    const swatch = document.createElement("span");
    swatch.className = `legend-swatch ${row.model || row.key} ${row.graph === "correlation" ? "correlation" : ""}`;
    item.append(swatch, document.createTextNode(row.label));
    fragment.append(item);
  }
  ui.legend.replaceChildren(fragment);
}

function renderDetails(data) {
  ui["fact-sensor"].textContent = String(data.sensor.id);
  ui["fact-sensor-detail"].textContent = data.sensor.name || "Nombre no disponible";
  ui["fact-node"].textContent = data.aggregated ? `Grupo ${data.sensor.node_label}` : "Individual";
  ui["fact-node-detail"].textContent = data.aggregated ? `Media de ${data.sensor.members} sensor${data.sensor.members === 1 ? "" : "es"}` : "Índice de este sensor";
  ui["fact-horizon"].textContent = `+${data.ahead_minutes} min`;
  ui["fact-horizon-detail"].textContent = data.range === "all" ? "Periodo de prueba completo" : `Últimas ${data.range.replace("h", " horas")}`;
  ui["fact-observed"].textContent = `${data.observed_count.toLocaleString("es-ES")} / ${data.points.toLocaleString("es-ES")}`;
  ui["fact-observed-detail"].textContent = "Fechas con valor real";
  ui["chart-subtitle"].textContent = `${data.scores.length} predicciones de modelo en las mismas fechas. ${data.aggregated ? "El valor observado es la media del grupo, no el dato aislado del sensor." : "El valor observado corresponde a este sensor."}`;
  ui["chart-caption"].textContent = "Negro: índice observado; colores: modelos; gris: último valor disponible. Los huecos sin medición no se dibujan como datos reales. El índice no es una probabilidad. Puedes ocultar curvas en la leyenda y explorar los valores con el cursor o las flechas del teclado.";
}

function formatScore(value) {
  return value === null || !Number.isFinite(value) ? "Sin dato" : value.toFixed(6);
}

function renderScores(data) {
  const fragment = document.createDocumentFragment();
  const baseline = data.reference_scores.persistence;
  const ordered = [...data.scores].sort((a, b) => (a.mse ?? Infinity) - (b.mse ?? Infinity));
  for (const row of ordered) {
    const tr = document.createElement("tr");
    if (row.id === data.test_leader) tr.className = "best-row";
    const comparison = document.createElement("td");
    if (row.mse === null) comparison.textContent = "Sin medidas";
    else if (baseline.mse === 0) comparison.textContent = row.mse === 0 ? "Empate" : "La referencia no falla";
    else {
      const delta = 100 * (row.mse / baseline.mse - 1);
      comparison.textContent = `${Math.abs(delta).toFixed(1)} % ${delta < 0 ? "menos" : delta > 0 ? "más" : "igual"} MSE`;
      comparison.className = delta < 0 ? "score-better" : delta > 0 ? "score-worse" : "";
    }
    for (const value of [row.label, formatScore(row.mse), formatScore(row.mae)]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    tr.append(comparison);
    fragment.append(tr);
  }
  for (const [label, score] of [["Repetir el último valor", data.reference_scores.persistence], ["Media reciente", data.reference_scores.window_mean]]) {
    const tr = document.createElement("tr");
    tr.className = "reference-row";
    for (const value of [label, formatScore(score.mse), formatScore(score.mae), label.startsWith("Repetir") ? "Referencia" : "Comparación adicional"]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    fragment.append(tr);
  }
  ui.scores.replaceChildren(fragment);
  ui["leader-tag"].textContent = `Mejor global en validación: ${data.validation_leader.label}`;
  ui["metric-note"].textContent = data.observed_count
    ? `Errores sobre ${data.observed_count.toLocaleString("es-ES")} fechas observadas del periodo mostrado. La fila verde es el menor MSE de ESTE sensor en prueba: solo es descriptiva, no se utilizó para entrenar ni elegir el modelo.`
    : "Este sensor no tiene mediciones reales en el periodo mostrado. Las predicciones se pueden ver, pero no podemos calcular sus errores: elige otro sensor o amplía el periodo.";
}

function layoutLegend(ctx, series, width) {
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif";
  const rows = [];
  let line = [];
  let used = 0;
  const available = width - 94;
  for (const item of series) {
    const size = ctx.measureText(item.compact).width + 32;
    if (line.length && used + size > available) {
      rows.push(line);
      line = [];
      used = 0;
    }
    line.push({ item, size });
    used += size;
  }
  if (line.length) rows.push(line);
  return rows;
}

function drawChart() {
  if (!state.payload) return;
  const canvas = ui.chart;
  const width = Math.max(320, canvas.clientWidth);
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const active = state.series.filter((row) => row.fixed || state.visible.get(row.key));
  const legend = layoutLegend(ctx, active, width);
  const area = { left: 63, right: width - 20, top: 35, bottom: height - 57 - legend.length * 19 };
  const times = state.payload.time.map((value) => Date.parse(value));
  const first = times[0], last = times[times.length - 1];
  const allValues = active.flatMap((row) => row.values.filter((value) => value !== null));
  const low = Math.min(0, ...allValues);
  const high = Math.max(1, ...allValues);
  const padding = Math.max(.04, (high - low) * .055);
  const lowY = low - padding, highY = high + padding;
  const xAt = (time) => area.left + (time - first) / (last - first || 1) * (area.right - area.left);
  const yAt = (value) => area.bottom - (value - lowY) / (highY - lowY) * (area.bottom - area.top);
  state.plot = { area, times, xAt, yAt };
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif";
  ctx.textBaseline = "middle";
  ctx.strokeStyle = "#e4ebed";
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  for (let tick = 0; tick <= 5; tick++) {
    const value = lowY + (highY - lowY) * tick / 5;
    const y = yAt(value);
    ctx.beginPath();
    ctx.moveTo(area.left, y);
    ctx.lineTo(area.right, y);
    ctx.stroke();
    ctx.fillStyle = "#536970";
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(2), area.left - 10, y);
  }
  const dateTicks = width < 520 ? 2 : width < 800 ? 3 : 5;
  for (let tick = 0; tick <= dateTicks; tick++) {
    const timestamp = first + (last - first) * tick / dateTicks;
    const x = xAt(timestamp);
    ctx.fillStyle = "#536970";
    ctx.textAlign = tick === 0 ? "left" : tick === dateTicks ? "right" : "center";
    ctx.fillText(niceDate(new Date(timestamp)), x, area.bottom + 20);
  }
  ctx.textAlign = "center";
  ctx.fillStyle = "#50646b";
  ctx.fillText("Fecha del valor que se intenta predecir", (area.left + area.right) / 2, area.bottom + 45);
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, area.right - area.left, area.bottom - area.top);
  ctx.clip();
  const typicalStep = times.length > 1 ? times[1] - times[0] : 900000;
  for (const row of active) {
    ctx.strokeStyle = row.color;
    ctx.lineWidth = row.width;
    ctx.setLineDash(row.dash);
    ctx.beginPath();
    let previous = -1;
    for (let i = 0; i < times.length; i++) {
      const value = row.values[i];
      if (value === null) { previous = -1; continue; }
      const x = xAt(times[i]), y = yAt(value);
      if (previous < 0 || i - previous > 1 || times[i] - times[previous] > typicalStep * 1.5) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      previous = i;
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
  if (state.cursor >= 0) {
    const x = xAt(times[state.cursor]);
    ctx.strokeStyle = "#83959b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, area.top);
    ctx.lineTo(x, area.bottom);
    ctx.stroke();
    for (const row of active) {
      const value = row.values[state.cursor];
      if (value === null) continue;
      ctx.fillStyle = row.color;
      ctx.beginPath();
      ctx.arc(x, yAt(value), row.key === "observed" ? 4 : 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.restore();
  ctx.setLineDash([]);
  ctx.textAlign = "left";
  for (let i = 0; i < legend.length; i++) {
    let left = area.left;
    const y = height - legend.length * 19 + i * 19 - 3;
    for (const { item, size } of legend[i]) {
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash(item.dash);
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(left + 15, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#425962";
      ctx.fillText(item.compact, left + 21, y);
      left += size;
    }
  }
}

function showTooltip(index) {
  if (!state.payload || index < 0) return;
  const fragment = document.createDocumentFragment();
  const title = document.createElement("strong");
  title.textContent = niceDate(state.payload.time[index]);
  fragment.append(title);
  const issued = document.createElement("div");
  issued.textContent = `Pronóstico desde ${niceDate(state.payload.origin_time[index])}`;
  fragment.append(issued);
  for (const row of state.series) {
    if (!row.fixed && !state.visible.get(row.key)) continue;
    const line = document.createElement("div");
    line.className = "tooltip-row";
    const name = document.createElement("span");
    name.textContent = row.compact;
    const value = document.createElement("span");
    value.textContent = row.values[index] === null ? "sin medida" : row.values[index].toFixed(3);
    line.append(name, value);
    fragment.append(line);
  }
  ui.tooltip.replaceChildren(fragment);
  ui.tooltip.hidden = false;
}

function moveCursor(event) {
  if (!state.payload || !state.plot) return;
  const { area, times, xAt } = state.plot;
  const rect = ui.chart.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  if (x < area.left || x > area.right || y < area.top || y > area.bottom) {
    ui.tooltip.hidden = true;
    if (state.cursor !== -1) { state.cursor = -1; drawChart(); }
    return;
  }
  let left = 0, right = times.length - 1;
  while (left < right) {
    const middle = (left + right) >> 1;
    if (xAt(times[middle]) < x) left = middle + 1;
    else right = middle;
  }
  const nearest = left > 0 && Math.abs(xAt(times[left - 1]) - x) < Math.abs(xAt(times[left]) - x) ? left - 1 : left;
  if (nearest !== state.cursor) {
    state.cursor = nearest;
    drawChart();
    showTooltip(nearest);
  }
}

function download(blob, name) {
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadCSV() {
  const data = state.payload;
  if (!data) return;
  const keys = data.scores.map((row) => row.id);
  const rows = [["fecha_emision", "fecha_objetivo", "horizonte_pasos", "medicion_disponible", "indice_pfm_observado", ...keys, "persistencia", "media_reciente"]];
  for (let i = 0; i < data.points; i++) rows.push([data.origin_time[i], data.time[i], data.horizon ?? 1, data.target[i] === null ? 0 : 1,
    data.target[i] ?? "", ...keys.map((key) => data.prediction[key][i]), data.persistence[i], data.window_mean[i]]);
  const csv = rows.map((row) => row.join(",")).join("\n") + "\n";
  download(new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" }), `pfm_${data.group}_${data.adjacency}_sensor_${data.sensor.id}_h${data.horizon ?? 1}_${data.range}.csv`);
}

function bindEvents() {
  ui.group.addEventListener("change", () => {
    state.selectedConfigs = [];
    state.chosenMeta.clear();
    state.rankOffset = 0;
    loadGroup(state.selectedId);
  });
  ui.graph.addEventListener("change", () => {
    if (ui.graph.value !== "both") state.selectedConfigs = state.selectedConfigs.filter((id) => id.includes(`__${ui.graph.value}__`));
    state.rankOffset = 0;
    renderChosen();
    loadSeries();
    loadRankings();
    loadComparison();
  });
  ui.horizon.addEventListener("change", () => { loadSeries(); loadRankings(); loadComparison(); });
  ui["time-range"].addEventListener("change", loadSeries);
  for (const button of document.querySelectorAll(".view-tabs [role=tab]")) {
    button.addEventListener("click", () => setView(button.dataset.view));
  }
  for (const id of ["ranking-phase", "filter-architecture", "filter-model", "filter-alpha", "filter-layers", "filter-hidden"]) {
    ui[id].addEventListener("change", () => { state.rankOffset = 0; loadRankings(); });
  }
  ui["ranking-prev"].addEventListener("click", () => { state.rankOffset = Math.max(0, state.rankOffset - 25); loadRankings(); });
  ui["ranking-next"].addEventListener("click", () => { state.rankOffset += 25; loadRankings(); });
  ui["show-chart"].addEventListener("click", () => { setView("curves"); loadSeries(); });
  for (const id of ["comparison-architecture", "comparison-criterion"]) ui[id].addEventListener("change", loadComparison);
  ui.sensor.addEventListener("change", () => {
    state.selectedId = Number(ui.sensor.value);
    loadSeries();
  });
  let timer;
  ui["sensor-filter"].addEventListener("input", () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(sensorOptions, 110);
  });
  ui.chart.addEventListener("pointermove", moveCursor);
  ui.chart.addEventListener("pointerleave", () => {
    if (state.cursor !== -1) { state.cursor = -1; drawChart(); }
    ui.tooltip.hidden = true;
  });
  ui.chart.addEventListener("keydown", (event) => {
    if (!state.payload) return;
    if (event.key === "Escape") { state.cursor = -1; ui.tooltip.hidden = true; drawChart(); return; }
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    state.cursor = Math.max(0, Math.min(state.payload.points - 1, state.cursor + (event.key === "ArrowRight" ? 1 : -1)));
    drawChart();
    showTooltip(state.cursor);
  });
  ui["download-png"].addEventListener("click", () => {
    if (!state.payload) return;
    const name = `pfm_${state.payload.group}_${state.payload.adjacency}_sensor_${state.payload.sensor.id}_h${state.payload.horizon ?? 1}_${state.payload.range}.png`;
    ui.chart.toBlob((blob) => { if (blob) download(blob, name); }, "image/png");
  });
  ui["download-csv"].addEventListener("click", downloadCSV);
  new ResizeObserver(() => { if (state.payload) drawChart(); }).observe(ui.chart);
  new ResizeObserver(() => { if (state.comparisonData && state.view === "comparison") drawComparison(state.comparisonData); }).observe(ui["comparison-chart"]);
}

async function start() {
  bindEvents();
  try {
    state.meta = await fetchJSON("/api/meta");
    document.querySelector(".view-tabs").hidden = !state.meta.configurations;
    ui["config-picker"].hidden = !state.meta.configurations;
    if (!state.meta.configurations) {
      document.querySelector(".intro h1").innerHTML = "Cuatro modelos.<br><span>Un mismo sensor.</span>";
      document.querySelector(".intro > p:last-child").textContent = "Compara las predicciones con lo que realmente se midió. Cambia la agrupación o las conexiones sin perder de vista la referencia más sencilla.";
    }
    addOptions(ui.group, state.meta.groups);
    addOptions(ui.graph, state.meta.graphs);
    addOptions(ui.horizon, Array.from({ length: state.meta.horizons ?? 1 }, (_, i) => ({ key: String(i + 1), label: `+${15 * (i + 1)} min` })));
    ui.group.disabled = false;
    ui.graph.disabled = false;
    ui.horizon.disabled = (state.meta.horizons ?? 1) === 1;
    ui["period-tag"].textContent = `Prueba histórica · ${niceDate(state.meta.period.start)} – ${niceDate(state.meta.period.end)}`;
    const query = new URLSearchParams(window.location.search);
    ui.group.value = state.meta.groups.some((entry) => entry.key === query.get("group")) ? query.get("group") : "base";
    ui.graph.value = state.meta.graphs.some((entry) => entry.key === query.get("graph")) ? query.get("graph") : "closeness";
    ui.horizon.value = Number(query.get("horizon")) >= 1 && Number(query.get("horizon")) <= (state.meta.horizons ?? 1) ? query.get("horizon") : "1";
    ui["time-range"].value = ["all", "48h", "24h"].includes(query.get("range")) ? query.get("range") : "all";
    const sensor = query.get("sensor");
    await loadGroup(sensor && /^\d+$/.test(sensor) ? Number(sensor) : null);
    if (state.meta.configurations) window.setInterval(async () => {
      try {
        const progress = await fetchJSON("/api/progress");
        ui["ranking-progress"].textContent = `${progress.completed}/${progress.total} evaluados`;
        if (state.view === "rankings") loadRankings();
        if (state.view === "comparison") loadComparison();
      } catch (error) {
        ui["ranking-progress"].textContent = `Progreso no disponible: ${error.message}`;
      }
    }, 30000);
  } catch (error) {
    setStatus(`No se pudo iniciar el comparador: ${error.message}`, true);
  }
}

start();
