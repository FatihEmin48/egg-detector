const MODEL_URL = "model/egg_yolo.onnx";
const INPUT_SIZE = 640;
// The underlying model's output has 2 class columns, but only "egg" is used —
// everything is reported and labeled as a plain egg count.
const NUM_CLASSES = 2;
const SCORE_THRESHOLD = 0.35;
const IOU_THRESHOLD = 0.45;

const statusEl = document.getElementById("status");
const countsEl = document.getElementById("counts");
const eggCountEl = document.getElementById("egg-count");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const video = document.getElementById("video");
const fpsEl = document.getElementById("fps");

const modeImageBtn = document.getElementById("mode-image");
const modeWebcamBtn = document.getElementById("mode-webcam");
const imageControls = document.getElementById("image-controls");
const webcamControls = document.getElementById("webcam-controls");
const imageInput = document.getElementById("image-input");
const webcamStartBtn = document.getElementById("webcam-start");
const webcamStopBtn = document.getElementById("webcam-stop");

let session = null;
let webcamStream = null;
let webcamLoopActive = false;

const BOX_COLOR = "#4f8cff";

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function loadModel() {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency ? Math.min(4, navigator.hardwareConcurrency) : 1;
  setStatus("Model indiriliyor (ilk seferde ~85MB, sonrasında önbellekten anında yüklenir)...");
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  setStatus("Hazır. Bir fotoğraf yükleyin ya da kamerayı başlatın.");
}

// Draws `source` (image/video) letterboxed into an INPUT_SIZE x INPUT_SIZE canvas
// and returns the input tensor plus the scale/padding needed to map boxes back.
function preprocess(source, srcWidth, srcHeight) {
  const scale = Math.min(INPUT_SIZE / srcWidth, INPUT_SIZE / srcHeight);
  const newW = Math.round(srcWidth * scale);
  const newH = Math.round(srcHeight * scale);
  const padX = Math.floor((INPUT_SIZE - newW) / 2);
  const padY = Math.floor((INPUT_SIZE - newH) / 2);

  const off = document.createElement("canvas");
  off.width = INPUT_SIZE;
  off.height = INPUT_SIZE;
  const offCtx = off.getContext("2d");
  offCtx.fillStyle = "rgb(114,114,114)";
  offCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  offCtx.drawImage(source, 0, 0, srcWidth, srcHeight, padX, padY, newW, newH);

  const imageData = offCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const area = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < area; i++) {
    chw[i] = imageData[i * 4] / 255;
    chw[area + i] = imageData[i * 4 + 1] / 255;
    chw[2 * area + i] = imageData[i * 4 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", chw, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale,
    padX,
    padY,
  };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter = interW * interH;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function nms(boxes) {
  const byClass = new Map();
  for (const b of boxes) {
    if (!byClass.has(b.classId)) byClass.set(b.classId, []);
    byClass.get(b.classId).push(b);
  }
  const kept = [];
  for (const group of byClass.values()) {
    group.sort((a, b) => b.score - a.score);
    const active = [...group];
    while (active.length) {
      const best = active.shift();
      kept.push(best);
      for (let i = active.length - 1; i >= 0; i--) {
        if (iou(best, active[i]) > IOU_THRESHOLD) active.splice(i, 1);
      }
    }
  }
  return kept;
}

// The detection output has shape [1, 4 + NUM_CLASSES + 32, numAnchors]: the
// trailing 32 rows are segmentation-mask coefficients from the underlying
// YOLO segmentation model, which this box-only demo ignores.
function postprocess(output, scale, padX, padY) {
  const data = output.data;
  const [, , numAnchors] = output.dims;
  const boxes = [];

  for (let i = 0; i < numAnchors; i++) {
    let bestClass = -1;
    let bestScore = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = data[(4 + c) * numAnchors + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestScore < SCORE_THRESHOLD) continue;

    const cx = data[0 * numAnchors + i];
    const cy = data[1 * numAnchors + i];
    const w = data[2 * numAnchors + i];
    const h = data[3 * numAnchors + i];

    const x1 = (cx - w / 2 - padX) / scale;
    const y1 = (cy - h / 2 - padY) / scale;
    const x2 = (cx + w / 2 - padX) / scale;
    const y2 = (cy + h / 2 - padY) / scale;

    boxes.push({ x1, y1, x2, y2, score: bestScore, classId: bestClass });
  }

  return nms(boxes);
}

function drawDetections(boxes) {
  ctx.lineWidth = 3;
  ctx.font = "16px sans-serif";
  ctx.textBaseline = "top";
  for (const b of boxes) {
    ctx.strokeStyle = BOX_COLOR;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const label = `Yumurta ${(b.score * 100).toFixed(0)}%`;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = BOX_COLOR;
    ctx.fillRect(b.x1, b.y1 - 20, textWidth + 8, 20);
    ctx.fillStyle = "#0b0d11";
    ctx.fillText(label, b.x1 + 4, b.y1 - 18);
  }
}

function updateCounts(boxes) {
  eggCountEl.textContent = String(boxes.length);
  countsEl.hidden = false;
}

async function runOnSource(source, width, height) {
  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(source, 0, 0, width, height);

  const { tensor, scale, padX, padY } = preprocess(source, width, height);
  const outputs = await session.run({ images: tensor });
  const output = outputs["output0"] ?? outputs[Object.keys(outputs)[0]];
  const boxes = postprocess(output, scale, padX, padY);
  drawDetections(boxes);
  updateCounts(boxes);
  return boxes.length;
}

async function runOnImageFile(file) {
  const img = new Image();
  const url = URL.createObjectURL(file);
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = url;
  });
  setStatus("Analiz ediliyor...");
  const count = await runOnSource(img, img.naturalWidth, img.naturalHeight);
  setStatus(`${count} nesne tespit edildi.`);
  URL.revokeObjectURL(url);
}

function setMode(mode) {
  const isImage = mode === "image";
  modeImageBtn.classList.toggle("active", isImage);
  modeWebcamBtn.classList.toggle("active", !isImage);
  imageControls.hidden = !isImage;
  webcamControls.hidden = isImage;
  fpsEl.hidden = isImage;
  if (isImage) {
    stopWebcam();
  }
}

modeImageBtn.addEventListener("click", () => setMode("image"));
modeWebcamBtn.addEventListener("click", () => setMode("webcam"));

imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) runOnImageFile(file);
});

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
  } catch (err) {
    setStatus("Kameraya erişilemedi: " + err.message);
    return;
  }
  video.srcObject = webcamStream;
  await video.play();
  webcamStartBtn.disabled = true;
  webcamStopBtn.disabled = false;
  webcamLoopActive = true;
  webcamLoop();
}

function stopWebcam() {
  webcamLoopActive = false;
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  webcamStartBtn.disabled = false;
  webcamStopBtn.disabled = true;
}

async function webcamLoop() {
  let lastTime = performance.now();
  while (webcamLoopActive) {
    if (video.videoWidth === 0) {
      await new Promise((r) => requestAnimationFrame(r));
      continue;
    }
    const count = await runOnSource(video, video.videoWidth, video.videoHeight);
    const now = performance.now();
    const fps = 1000 / (now - lastTime);
    lastTime = now;
    fpsEl.textContent = `${fps.toFixed(1)} FPS · ${count} nesne`;
    await new Promise((r) => requestAnimationFrame(r));
  }
}

webcamStartBtn.addEventListener("click", startWebcam);
webcamStopBtn.addEventListener("click", stopWebcam);

loadModel();
