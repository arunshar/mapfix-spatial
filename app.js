const samples = {
  bay: {
    label: "Bay shoreline",
    points: [
      [-122.52, 37.71],
      [-122.48, 37.74],
      [-122.43, 37.78],
      [-122.39, 37.81],
      [-122.35, 37.83],
      [-122.3, 37.82],
      [-122.26, 37.79],
      [-122.22, 37.75],
      [-122.19, 37.7],
      [-122.24, 37.66],
      [-122.31, 37.64],
      [-122.39, 37.66],
    ],
  },
  transit: {
    label: "Transit alignment",
    points: [
      [-122.49, 37.62],
      [-122.45, 37.65],
      [-122.41, 37.68],
      [-122.36, 37.71],
      [-122.31, 37.74],
      [-122.27, 37.77],
      [-122.23, 37.8],
      [-122.18, 37.83],
      [-122.14, 37.86],
    ],
  },
  sensors: {
    label: "Sensor grid",
    points: Array.from({ length: 24 }, (_, index) => {
      const col = index % 6;
      const row = Math.floor(index / 6);
      return [-122.49 + col * 0.07, 37.62 + row * 0.07];
    }),
  },
  quake: {
    label: "Quake cluster",
    points: [
      [-122.51, 37.82],
      [-122.47, 37.78],
      [-122.44, 37.84],
      [-122.39, 37.76],
      [-122.36, 37.81],
      [-122.33, 37.71],
      [-122.29, 37.79],
      [-122.25, 37.69],
      [-122.21, 37.75],
      [-122.17, 37.66],
      [-122.14, 37.73],
      [-122.11, 37.63],
    ],
  },
};

const projectionTuning = {
  mercator: { curve: 0.74, shear: 0.42, label: "Mercator repair" },
  equalEarth: { curve: 0.48, shear: 0.28, label: "Equal Earth" },
  albers: { curve: 0.62, shear: -0.22, label: "Albers local" },
  tangent: { curve: 0.28, shear: 0.12, label: "Local tangent" },
};

const canvas = document.querySelector("#mapCanvas");
const context = canvas.getContext("2d");
const sampleSelect = document.querySelector("#sampleSelect");
const projectionSelect = document.querySelector("#projectionSelect");
const strengthRange = document.querySelector("#strengthRange");
const noiseRange = document.querySelector("#noiseRange");
const strengthValue = document.querySelector("#strengthValue");
const noiseValue = document.querySelector("#noiseValue");
const runButton = document.querySelector("#runButton");
const shuffleButton = document.querySelector("#shuffleButton");
const aiRenderButton = document.querySelector("#aiRenderButton");
const exportButton = document.querySelector("#exportButton");
const coordBody = document.querySelector("#coordBody");
const pointCount = document.querySelector("#pointCount");
const runStatus = document.querySelector("#runStatus");
const meanError = document.querySelector("#meanError");
const skewMetric = document.querySelector("#skewMetric");
const areaMetric = document.querySelector("#areaMetric");
const confidenceMetric = document.querySelector("#confidenceMetric");
const pipelineTrace = document.querySelector("#pipelineTrace");
const aiSummary = document.querySelector("#aiSummary");
const backendStatus = document.querySelector("#backendStatus");
const backendModels = document.querySelector("#backendModels");
const aiRenderPreview = document.querySelector("#aiRenderPreview");
const aiMapImage = document.querySelector("#aiMapImage");

let driftSeed = 0.38;
let animationFrame = 0;
let lastRender = null;
let backendState = {
  checked: false,
  apiAvailable: false,
  openaiConfigured: false,
  sdkAvailable: false,
  textModel: "",
  imageModel: "",
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function boundsFor(points) {
  const lonValues = points.map(([lon]) => lon);
  const latValues = points.map(([, lat]) => lat);
  const minLon = Math.min(...lonValues);
  const maxLon = Math.max(...lonValues);
  const minLat = Math.min(...latValues);
  const maxLat = Math.max(...latValues);
  const lonPad = Math.max((maxLon - minLon) * 0.28, 0.04);
  const latPad = Math.max((maxLat - minLat) * 0.28, 0.04);

  return {
    minLon: minLon - lonPad,
    maxLon: maxLon + lonPad,
    minLat: minLat - latPad,
    maxLat: maxLat + latPad,
  };
}

function normalizePoint(point, bounds) {
  const [lon, lat] = point;
  return {
    x: (lon - bounds.minLon) / (bounds.maxLon - bounds.minLon),
    y: 1 - (lat - bounds.minLat) / (bounds.maxLat - bounds.minLat),
  };
}

function denormalizePoint(point, bounds) {
  return [
    bounds.minLon + point.x * (bounds.maxLon - bounds.minLon),
    bounds.minLat + (1 - point.y) * (bounds.maxLat - bounds.minLat),
  ];
}

function distortionVector(point, noise, projection) {
  const centeredX = point.x - 0.5;
  const centeredY = point.y - 0.5;
  const waveX = Math.sin((point.y * 2.8 + driftSeed) * Math.PI);
  const waveY = Math.cos((point.x * 2.2 - driftSeed) * Math.PI);
  const radial = centeredX * centeredX + centeredY * centeredY;

  return {
    x:
      noise *
      (projection.shear * centeredY * 0.18 +
        projection.curve * waveX * 0.055 +
        centeredX * radial * 0.28),
    y:
      noise *
      (-projection.shear * centeredX * 0.14 +
        projection.curve * waveY * 0.045 -
        centeredY * radial * 0.22),
  };
}

function applyDistortion(point, noise, projection) {
  const vector = distortionVector(point, noise, projection);
  return {
    x: clamp(point.x + vector.x, -0.1, 1.1),
    y: clamp(point.y + vector.y, -0.1, 1.1),
  };
}

function correctPoint(rawPoint, noise, projection, strength) {
  let estimate = { ...rawPoint };
  for (let index = 0; index < 5; index += 1) {
    const vector = distortionVector(estimate, noise, projection);
    estimate = {
      x: rawPoint.x - vector.x * strength,
      y: rawPoint.y - vector.y * strength,
    };
  }
  return {
    x: clamp(estimate.x, -0.08, 1.08),
    y: clamp(estimate.y, -0.08, 1.08),
  };
}

function gradientAt(point, noise, projection) {
  const epsilon = 0.004;
  const base = distortionVector(point, noise, projection);
  const dx = distortionVector({ x: point.x + epsilon, y: point.y }, noise, projection);
  const dy = distortionVector({ x: point.x, y: point.y + epsilon }, noise, projection);

  return {
    x: (dx.x - base.x + dy.x - base.x) / epsilon,
    y: (dx.y - base.y + dy.y - base.y) / epsilon,
  };
}

function fitToCanvas(point, width, height) {
  const gutterX = Math.max(42, width * 0.06);
  const gutterY = Math.max(40, height * 0.08);
  return {
    x: gutterX + point.x * (width - gutterX * 2),
    y: gutterY + point.y * (height - gutterY * 2),
  };
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

function drawGrid(ctx, width, height, noise, projection) {
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(100, 113, 116, 0.22)";

  for (let step = 0; step <= 10; step += 1) {
    const value = step / 10;

    ctx.beginPath();
    for (let index = 0; index <= 80; index += 1) {
      const point = applyDistortion({ x: index / 80, y: value }, noise, projection);
      const screen = fitToCanvas(point, width, height);
      if (index === 0) ctx.moveTo(screen.x, screen.y);
      else ctx.lineTo(screen.x, screen.y);
    }
    ctx.stroke();

    ctx.beginPath();
    for (let index = 0; index <= 80; index += 1) {
      const point = applyDistortion({ x: value, y: index / 80 }, noise, projection);
      const screen = fitToCanvas(point, width, height);
      if (index === 0) ctx.moveTo(screen.x, screen.y);
      else ctx.lineTo(screen.x, screen.y);
    }
    ctx.stroke();
  }

  ctx.restore();
}

function drawGradientField(ctx, width, height, noise, projection) {
  ctx.save();
  ctx.strokeStyle = "rgba(194, 145, 43, 0.78)";
  ctx.fillStyle = "rgba(194, 145, 43, 0.88)";
  ctx.lineWidth = 1.5;

  for (let x = 0.12; x <= 0.9; x += 0.13) {
    for (let y = 0.14; y <= 0.9; y += 0.13) {
      const point = { x, y };
      const gradient = gradientAt(point, noise, projection);
      const start = fitToCanvas(applyDistortion(point, noise, projection), width, height);
      const end = {
        x: start.x + gradient.x * width * 0.045,
        y: start.y + gradient.y * height * 0.045,
      };
      const angle = Math.atan2(end.y - start.y, end.x - start.x);

      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(end.x, end.y);
      ctx.lineTo(end.x - Math.cos(angle - 0.6) * 7, end.y - Math.sin(angle - 0.6) * 7);
      ctx.lineTo(end.x - Math.cos(angle + 0.6) * 7, end.y - Math.sin(angle + 0.6) * 7);
      ctx.closePath();
      ctx.fill();
    }
  }

  ctx.restore();
}

function drawPath(ctx, points, width, height, color, lineWidth, dash = []) {
  if (!points.length) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash(dash);
  ctx.beginPath();

  points.forEach((point, index) => {
    const screen = fitToCanvas(point, width, height);
    if (index === 0) ctx.moveTo(screen.x, screen.y);
    else ctx.lineTo(screen.x, screen.y);
  });

  ctx.stroke();
  ctx.setLineDash([]);

  points.forEach((point, index) => {
    const screen = fitToCanvas(point, width, height);
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, index === 0 ? 5.8 : 4.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = index === 0 ? "#1f2527" : color;
  });

  ctx.restore();
}

function drawCanvas(renderData) {
  const pixelRatio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  const width = Math.max(620, Math.round(box.width * pixelRatio));
  const height = Math.max(420, Math.round(box.height * pixelRatio));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  context.clearRect(0, 0, width, height);
  context.save();
  context.fillStyle = "#fbfaf6";
  context.fillRect(0, 0, width, height);

  drawRoundedRect(context, width * 0.02, height * 0.03, width * 0.96, height * 0.91, 18);
  context.fillStyle = "rgba(236, 230, 217, 0.46)";
  context.fill();

  drawGrid(context, width, height, renderData.noise, renderData.projection);
  drawGradientField(context, width, height, renderData.noise, renderData.projection);
  drawPath(context, renderData.rawPoints, width, height, "#df6b54", 3.5, [10, 9]);
  drawPath(context, renderData.correctedPoints, width, height, "#147c72", 4.5);

  context.font = `${14 * pixelRatio}px Inter, system-ui, sans-serif`;
  context.fillStyle = "rgba(31, 37, 39, 0.78)";
  context.fillText(renderData.projection.label, width * 0.055, height * 0.09);
  context.restore();
}

function metricsFor(normalized, rawPoints, correctedPoints, strength, noise) {
  const errors = correctedPoints.map((point, index) => {
    const target = normalized[index];
    return Math.hypot(point.x - target.x, point.y - target.y) * 1000;
  });
  const rawErrors = rawPoints.map((point, index) => {
    const target = normalized[index];
    return Math.hypot(point.x - target.x, point.y - target.y) * 1000;
  });
  const mean = errors.reduce((sum, value) => sum + value, 0) / errors.length;
  const rawMean = rawErrors.reduce((sum, value) => sum + value, 0) / rawErrors.length;
  const recovered = rawMean === 0 ? 1 : clamp(1 - mean / rawMean, 0, 1);
  const skew = clamp(noise * 64 * (1 - strength * 0.68), 0, 42);
  const confidence = clamp(0.58 + recovered * 0.35 - noise * 0.15, 0.42, 0.98);

  return { mean, recovered, skew, confidence };
}

function coordinateRowsForLocal(originalPoints, rawPoints, correctedPoints, bounds) {
  return originalPoints.slice(0, 12).map((original, index) => {
    const rawGeo = denormalizePoint(rawPoints[index], bounds);
    const fixedGeo = denormalizePoint(correctedPoints[index], bounds);
    const target = normalizePoint(original, bounds);
    const residual =
      Math.hypot(correctedPoints[index].x - target.x, correctedPoints[index].y - target.y) * 1000;

    return {
      id: `P${String(index + 1).padStart(2, "0")}`,
      rawLon: rawGeo[0],
      rawLat: rawGeo[1],
      fixedLon: fixedGeo[0],
      fixedLat: fixedGeo[1],
      residual,
    };
  });
}

function updateCoordinateRows(rows, totalPoints) {
  coordBody.innerHTML = "";

  rows.forEach((item) => {
    const row = document.createElement("tr");
    const values = [
      item.id,
      item.rawLon.toFixed(5),
      item.rawLat.toFixed(5),
      item.fixedLon.toFixed(5),
      item.fixedLat.toFixed(5),
      item.residual.toFixed(2),
    ];

    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    coordBody.append(row);
  });

  pointCount.textContent = `${totalPoints} points`;
}

function updateTrace(active = true) {
  [...pipelineTrace.children].forEach((item, index) => {
    item.classList.toggle("done", active && index <= 2);
  });
}

function setMetrics(metrics) {
  meanError.textContent = `${metrics.mean.toFixed(2)} px`;
  skewMetric.textContent = `${metrics.skew.toFixed(1)} deg`;
  areaMetric.textContent = `${Math.round(metrics.recovered * 100)}%`;
  confidenceMetric.textContent = metrics.confidence.toFixed(2);
}

function buildLocalRenderData() {
  const sample = samples[sampleSelect.value];
  const projection = projectionTuning[projectionSelect.value];
  const strength = Number(strengthRange.value) / 100;
  const noise = Number(noiseRange.value) / 100;
  const bounds = boundsFor(sample.points);
  const normalized = sample.points.map((point) => normalizePoint(point, bounds));
  const rawPoints = normalized.map((point) => applyDistortion(point, noise, projection));
  const correctedPoints = rawPoints.map((point) => correctPoint(point, noise, projection, strength));
  const metrics = metricsFor(normalized, rawPoints, correctedPoints, strength, noise);

  return {
    normalized,
    rawPoints,
    correctedPoints,
    bounds,
    strength,
    noise,
    projection,
    sample,
    metrics,
    coordinates: coordinateRowsForLocal(sample.points, rawPoints, correctedPoints, bounds),
    pointCount: sample.points.length,
    mode: "local",
    ai: {
      correctionSummary: "Local deterministic correction is ready.",
      gradientSummary: "Gradient field calculated in the browser.",
    },
  };
}

function payloadForCurrentState() {
  const sample = samples[sampleSelect.value];
  return {
    sampleLabel: sample.label,
    points: sample.points,
    projectionKey: projectionSelect.value,
    projectionLabel: projectionTuning[projectionSelect.value].label,
    strength: Number(strengthRange.value) / 100,
    noise: Number(noiseRange.value) / 100,
    seed: driftSeed,
  };
}

function applyRenderData(renderData, statusText = "Solved") {
  const strength = renderData.strength ?? Number(strengthRange.value) / 100;
  const noise = renderData.noise ?? Number(noiseRange.value) / 100;
  strengthValue.textContent = `${Math.round(strength * 100)}%`;
  noiseValue.textContent = `${Math.round(noise * 100)}%`;
  setMetrics(renderData.metrics);
  runStatus.textContent = statusText;
  updateTrace(true);

  lastRender = renderData;
  drawCanvas(lastRender);
  updateCoordinateRows(renderData.coordinates, renderData.pointCount);
  aiSummary.textContent =
    renderData.ai?.correctionSummary ||
    renderData.ai?.gradientSummary ||
    "Correction completed.";
}

function render() {
  applyRenderData(buildLocalRenderData());
}

function scheduleRender() {
  cancelAnimationFrame(animationFrame);
  animationFrame = requestAnimationFrame(render);
}

function mapServerCorrection(response) {
  return {
    normalized: response.normalized,
    rawPoints: response.rawPoints,
    correctedPoints: response.correctedPoints,
    bounds: response.bounds,
    strength: response.strength,
    noise: response.noise,
    projection: {
      ...projectionTuning[response.projectionKey],
      ...response.projection,
    },
    sample: {
      label: response.sampleLabel,
      points: samples[sampleSelect.value].points,
    },
    metrics: response.metrics,
    coordinates: response.coordinates,
    pointCount: response.pointCount,
    mode: response.mode,
    ai: response.ai,
    model: response.model,
  };
}

function updateBackendDisplay() {
  if (!backendState.apiAvailable) {
    backendStatus.textContent = "Static demo mode";
    backendModels.textContent = "Run python server.py to enable backend APIs.";
    return;
  }

  if (!backendState.sdkAvailable) {
    backendStatus.textContent = "Backend missing SDK";
    backendModels.textContent = "Install the openai package in this environment.";
    return;
  }

  if (!backendState.openaiConfigured) {
    backendStatus.textContent = "Backend ready, key missing";
    backendModels.textContent = "Set OPENAI_API_KEY for live GPT and Image API calls.";
    return;
  }

  backendStatus.textContent = "OpenAI backend live";
  backendModels.textContent = `${backendState.textModel} + ${backendState.imageModel}`;
}

async function checkBackend() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("No API health route");
    backendState = {
      ...(await response.json()),
      checked: true,
      apiAvailable: true,
    };
  } catch {
    backendState = {
      checked: true,
      apiAvailable: false,
      openaiConfigured: false,
      sdkAvailable: false,
      textModel: "",
      imageModel: "",
    };
  }
  updateBackendDisplay();
}

async function runWithPulse() {
  runStatus.textContent = "Running";
  updateTrace(false);
  aiRenderPreview.hidden = true;

  if (!backendState.apiAvailable || !backendState.openaiConfigured) {
    window.setTimeout(scheduleRender, 220);
    return;
  }

  try {
    const response = await fetch("/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadForCurrentState()),
    });
    if (!response.ok) throw new Error(`Backend returned ${response.status}`);

    const result = await response.json();
    const renderData = mapServerCorrection(result);
    const statusText = result.mode === "openai" ? "AI solved" : "Solved";
    applyRenderData(renderData, statusText);
    backendStatus.textContent = result.mode === "openai" ? "GPT analysis live" : "Backend fallback";
    backendModels.textContent =
      result.mode === "openai" ? `${backendState.textModel} + ${backendState.imageModel}` : "Local correction returned from server.";
  } catch (error) {
    backendStatus.textContent = "Backend call failed";
    backendModels.textContent = error.message;
    window.setTimeout(scheduleRender, 120);
  }
}

function exportCanvas() {
  if (!lastRender) render();
  const link = document.createElement("a");
  link.download = "mapfix-corrected-projection.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
}

async function generateAIMap() {
  if (!lastRender) render();

  if (!backendState.apiAvailable) {
    aiSummary.textContent = "Start the Python backend with python server.py to enable Image API rendering.";
    return;
  }

  if (!backendState.openaiConfigured) {
    aiSummary.textContent = "Set OPENAI_API_KEY on the backend to enable Image API rendering.";
    return;
  }

  aiRenderButton.disabled = true;
  aiRenderButton.textContent = "Generating...";
  runStatus.textContent = "Rendering";

  try {
    const response = await fetch("/api/render-map", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sampleLabel: lastRender.sample.label,
        projectionLabel: lastRender.projection.label,
        metrics: lastRender.metrics,
        coordinates: lastRender.coordinates,
      }),
    });
    if (!response.ok) throw new Error(`Image API returned ${response.status}`);

    const result = await response.json();
    if (!result.available) {
      aiSummary.textContent = result.message || "Image API rendering is unavailable.";
      return;
    }

    aiMapImage.src = result.imageDataUrl;
    aiRenderPreview.hidden = false;
    backendStatus.textContent = "Image API render ready";
    backendModels.textContent = result.model || backendState.imageModel;
    runStatus.textContent = "Rendered";
  } catch (error) {
    aiSummary.textContent = error.message;
    runStatus.textContent = "Solved";
  } finally {
    aiRenderButton.disabled = false;
    aiRenderButton.textContent = "Generate AI map";
  }
}

function clearGeneratedImage() {
  aiRenderPreview.hidden = true;
  aiMapImage.removeAttribute("src");
}

sampleSelect.addEventListener("change", () => {
  clearGeneratedImage();
  scheduleRender();
});
projectionSelect.addEventListener("change", () => {
  clearGeneratedImage();
  scheduleRender();
});
strengthRange.addEventListener("input", () => {
  clearGeneratedImage();
  scheduleRender();
});
noiseRange.addEventListener("input", () => {
  clearGeneratedImage();
  scheduleRender();
});
runButton.addEventListener("click", runWithPulse);
shuffleButton.addEventListener("click", () => {
  driftSeed = Math.random();
  clearGeneratedImage();
  runWithPulse();
});
aiRenderButton.addEventListener("click", generateAIMap);
exportButton.addEventListener("click", exportCanvas);
window.addEventListener("resize", scheduleRender);

scheduleRender();
checkBackend();
