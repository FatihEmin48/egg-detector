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

const modeImageBtn = document.getElementById("mode-image");
const modeWebcamBtn = document.getElementById("mode-webcam");
const imageControls = document.getElementById("image-controls");
const webcamControls = document.getElementById("webcam-controls");
const imageInput = document.getElementById("image-input");
const captureBtn = document.getElementById("capture-btn");
const resetBtn = document.getElementById("reset-btn");
const saveBtn = document.getElementById("save-btn");

let session = null;
let webcamStream = null;

const BOX_COLOR = "#4f8cff";

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function loadModel() {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency ? Math.min(4, navigator.hardwareConcurrency) : 1;
  setStatus("Model indiriliyor (ilk seferde ~85MB, sonrasında önbellekten anında yüklenir)...");
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  setStatus("Hazır. Bir fotoğraf yükleyin ya da Kamera sekmesine geçin.");
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

function drawSummary(count) {
  const label = `${count} Yumurta Tespit Edildi`;
  ctx.font = "bold 20px sans-serif";
  ctx.textBaseline = "middle";
  const paddingX = 14;
  const textWidth = ctx.measureText(label).width;
  const boxWidth = textWidth + paddingX * 2;
  const boxHeight = 36;
  const x = Math.max(8, (canvas.width - boxWidth) / 2);
  const y = 14;
  ctx.fillStyle = "rgba(11, 13, 17, 0.75)";
  ctx.fillRect(x, y, boxWidth, boxHeight);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, x + paddingX, y + boxHeight / 2 + 1);
}

function updateCounts(boxes) {
  eggCountEl.textContent = String(boxes.length);
  countsEl.hidden = false;
}

async function runOnSource(source, width, height) {
  if (source !== canvas) {
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(source, 0, 0, width, height);
  }

  const { tensor, scale, padX, padY } = preprocess(source, width, height);
  const outputs = await session.run({ images: tensor });
  const output = outputs["output0"] ?? outputs[Object.keys(outputs)[0]];
  const boxes = postprocess(output, scale, padX, padY);
  drawDetections(boxes);
  drawSummary(boxes.length);
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
  if (isImage) {
    stopWebcamStream();
    video.hidden = true;
  } else {
    startWebcamPreview();
  }
}

modeImageBtn.addEventListener("click", () => setMode("image"));
modeWebcamBtn.addEventListener("click", () => setMode("webcam"));

imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) runOnImageFile(file);
});

async function startWebcamPreview() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
  } catch (err) {
    setStatus("Kameraya erişilemedi: " + err.message);
    return;
  }
  video.srcObject = webcamStream;
  await video.play();
  video.hidden = false;
  canvas.hidden = true;
  captureBtn.hidden = false;
  resetBtn.hidden = true;
  countsEl.hidden = true;
  setStatus("Hazır. Fotoğraf çekmek için butona basın.");
}

function stopWebcamStream() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
}

async function captureAndDetect() {
  if (!video.videoWidth) return;
  if (!session) {
    setStatus("Model henüz yüklenmedi, lütfen bekleyin.");
    return;
  }
  // Grab the frame onto the canvas and stop the camera stream immediately,
  // before the (slower) inference runs, so the live feed doesn't keep
  // playing underneath while the user waits.
  const width = video.videoWidth;
  const height = video.videoHeight;
  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(video, 0, 0, width, height);
  stopWebcamStream();
  video.hidden = true;
  canvas.hidden = false;
  captureBtn.hidden = true;
  setStatus("Analiz ediliyor...");
  const count = await runOnSource(canvas, width, height);
  resetBtn.hidden = false;
  setStatus(`${count} yumurta tespit edildi.`);
}

function saveToGallery() {
  canvas.toBlob(async (blob) => {
    if (!blob) return;
    const file = new File([blob], `yumurta-tespiti-${Date.now()}.png`, { type: "image/png" });

    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: "Yumurta Tespiti" });
        return;
      } catch (err) {
        if (err.name === "AbortError") return;
      }
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, "image/png");
}

captureBtn.addEventListener("click", captureAndDetect);
resetBtn.addEventListener("click", startWebcamPreview);
saveBtn.addEventListener("click", saveToGallery);

loadModel();
