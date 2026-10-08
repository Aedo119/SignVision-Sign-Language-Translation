/**
 * SignVision — Real-Time Sign Language Tracking & Translation
 * Uses @mediapipe/tasks-vision (HandLandmarker) for in-browser landmark detection.
 */

// ── MediaPipe Tasks Vision CDN (ES module) ────────────────────────────────────
const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

// ── Hand connections (21-point skeleton) ─────────────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

// ── UI Elements ───────────────────────────────────────────────────────────────
const systemStatusPill = document.getElementById('system-status-pill');
const systemStatusText = document.getElementById('system-status-text');
const modeBadge        = document.getElementById('mode-badge');

const tabWebcam   = document.getElementById('tab-webcam');
const tabUpload   = document.getElementById('tab-upload');
const webcamPanel = document.getElementById('webcam-panel');
const uploadPanel = document.getElementById('upload-panel');

const webcamVideo       = document.getElementById('webcam-video');
const overlayCanvas     = document.getElementById('overlay-canvas');
const hiddenCanvas      = document.getElementById('hidden-canvas');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const liveIndicator     = document.getElementById('live-indicator');
const landmarksStatus   = document.getElementById('landmarks-status');

const startBtn        = document.getElementById('start-btn');
const stopBtn         = document.getElementById('stop-btn');
const toggleSkeleton  = document.getElementById('toggle-skeleton');
const toggleMirror    = document.getElementById('toggle-mirror');
const speakBtn        = document.getElementById('speak-btn');
const clearHistoryBtn = document.getElementById('clear-history-btn');
const holdBadge       = document.getElementById('hold-badge');
const backspaceBtn    = document.getElementById('backspace-btn');
const copyBtn         = document.getElementById('copy-btn');

const glossWebcam    = document.getElementById('gloss-webcam');
const textWebcam     = document.getElementById('text-webcam');
const confidenceVal  = document.getElementById('confidence-val');
const confidenceFill = document.getElementById('confidence-fill');
const tokensStream   = document.getElementById('tokens-stream');

const dropZone          = document.getElementById('drop-zone');
const fileInput         = document.getElementById('file-input');
const browseBtn         = document.getElementById('browse-btn');
const previewContainer  = document.getElementById('preview-container');
const previewVideo      = document.getElementById('preview-video');
const selectedFileName  = document.getElementById('selected-file-name');
const removeFileBtn     = document.getElementById('remove-file-btn');
const processBtn        = document.getElementById('process-btn');
const uploadProgressCard = document.getElementById('upload-progress-card');
const progressStatus    = document.getElementById('progress-status');

const glossUpload      = document.getElementById('gloss-upload');
const textUpload       = document.getElementById('text-upload');
const metricFrames     = document.getElementById('metric-frames');
const metricLatency    = document.getElementById('metric-latency');
const metricConfidence = document.getElementById('metric-confidence');

// ── State ────────────────────────────────────────────────────────────────────
let mediaStream      = null;
let websocket        = null;
let isStreaming      = false;
let handLandmarker   = null;
let drawingUtils     = null;
let animFrameId      = null;
let fallbackInterval = null;
let lastVideoTime    = -1;

let recognizedHistory       = [];
let currentCandidate        = null;
let currentCandidateText    = null;
let candidateCount          = 0;
let lastCommittedSign       = null;
let lastCommitTime          = 0;
let lastValidPredictionTime = 0;
let holdBadgeTimeout        = null;
let lastSendTime            = 0;
let predictionBuffer        = [];
let currentVocabMode        = 'combined';

const REQUIRED_HOLD_FRAMES  = 3;   // 3 stable frames (~400-500ms) for responsive hold-to-commit
const SEND_THROTTLE_MS      = 100; // ~10 fps to backend

// POSE indices we forward (must match backend pipeline)
const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16];

// ── MediaPipe HandLandmarker initialiser ──────────────────────────────────────
async function initHandLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(MP_CDN);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU'
      },
      runningMode:           'VIDEO',
      numHands:              2,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence:  0.35,
      minTrackingConfidence:      0.35
    });
    drawingUtils = new DrawingUtils(overlayCanvas.getContext('2d'));
    console.log('[SignVision] HandLandmarker initialised (GPU).');
    return true;
  } catch (gpuErr) {
    console.warn('[SignVision] GPU delegate failed, falling back to CPU:', gpuErr);
    try {
      const vision = await FilesetResolver.forVisionTasks(MP_CDN);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'CPU'
        },
        runningMode:           'VIDEO',
        numHands:              2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence:  0.35,
        minTrackingConfidence:      0.35
      });
      drawingUtils = new DrawingUtils(overlayCanvas.getContext('2d'));
      console.log('[SignVision] HandLandmarker initialised (CPU).');
      return true;
    } catch (err) {
      console.error('[SignVision] HandLandmarker init failed:', err);
      return false;
    }
  }
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkServerHealth() {
  try {
    const res  = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      if (data.model_mode === 'transformer') {
        modeBadge.textContent = 'PyTorch Transformer: Online';
        modeBadge.style.borderColor = 'rgba(16,185,129,0.4)';
      } else if (data.model_mode === 'baseline_classifier') {
        modeBadge.textContent = 'Sign Classifier (98.6% Acc): Online';
        modeBadge.style.borderColor = 'rgba(6,182,212,0.4)';
      } else {
        modeBadge.textContent = 'Demo Mode (No Checkpoint)';
        modeBadge.style.borderColor = 'rgba(245,158,11,0.4)';
      }
    }
  } catch {
    systemStatusPill.style.background = 'rgba(244,63,94,0.15)';
    systemStatusPill.style.color = '#f43f5e';
    systemStatusText.textContent = 'Server Offline';
  }
}
checkServerHealth();

// ── Tab switcher ──────────────────────────────────────────────────────────────
tabWebcam.addEventListener('click', () => {
  tabWebcam.classList.add('active');
  tabUpload.classList.remove('active');
  webcamPanel.classList.add('active');
  uploadPanel.classList.remove('active');
});
tabUpload.addEventListener('click', () => {
  tabUpload.classList.add('active');
  tabWebcam.classList.remove('active');
  uploadPanel.classList.add('active');
  webcamPanel.classList.remove('active');
});

// ── Mirror toggle ─────────────────────────────────────────────────────────────
toggleMirror.addEventListener('change', (e) => {
  webcamVideo.classList.toggle('mirror', e.target.checked);
});

// ── Vocabulary Mode Switcher ──────────────────────────────────────────────────
const vocabPills = document.querySelectorAll('.vocab-pill');
const vocabModeHint = document.getElementById('vocab-mode-hint');

vocabPills.forEach(pill => {
  pill.addEventListener('click', () => {
    vocabPills.forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    currentVocabMode = pill.getAttribute('data-mode') || 'combined';

    if (vocabModeHint) {
      if (currentVocabMode === 'words') {
        vocabModeHint.textContent = 'Translating to ASL Words (Hello, Yes, Love...)';
      } else if (currentVocabMode === 'numbers') {
        vocabModeHint.textContent = 'Translating to Number Words (Zero–Nine)';
      } else {
        vocabModeHint.textContent = 'All Signs & English Words';
      }
    }

    // Reset hold state on mode switch
    currentCandidate = null;
    currentCandidateText = null;
    candidateCount = 0;
    predictionBuffer = [];
    if (holdBadge) {
      holdBadge.textContent = 'Hold to commit';
      holdBadge.className = 'hold-badge';
    }
  });
});

// ── Start Camera ──────────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  systemStatusText.textContent = 'Loading MediaPipe...';

  // Init handLandmarker on first use
  if (!handLandmarker) {
    const ok = await initHandLandmarker();
    if (!ok) {
      alert('MediaPipe HandLandmarker failed to load. Will use backend-side detection.');
    }
  }

  try {
    systemStatusText.textContent = 'Starting Camera...';
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false
    });

    webcamVideo.srcObject = mediaStream;
    await new Promise((resolve) => {
      webcamVideo.onloadedmetadata = () => { webcamVideo.play(); resolve(); };
    });

    const vw = webcamVideo.videoWidth  || 640;
    const vh = webcamVideo.videoHeight || 480;
    overlayCanvas.width  = vw;
    overlayCanvas.height = vh;

    const vc = document.querySelector('.video-container');
    if (vc && vw && vh) vc.style.aspectRatio = `${vw} / ${vh}`;

    cameraPlaceholder.classList.add('hidden');
    liveIndicator.classList.remove('hidden');
    stopBtn.disabled = false;

    // WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    websocket = new WebSocket(`${protocol}//${window.location.host}/ws`);

    websocket.onopen = () => {
      isStreaming = true;
      systemStatusText.textContent = 'Live Tracking Active';
      systemStatusPill.className = 'status-pill status-active';
      if (handLandmarker) {
        startLandmarkLoop();
      } else {
        startFallbackLoop();
      }
    };
    websocket.onmessage = (e) => handleSocketMessage(e.data);
    websocket.onclose   = () => { if (isStreaming) stopCamera(); };
    websocket.onerror   = (err) => { console.error('WS error', err); systemStatusText.textContent = 'Socket Error'; };

  } catch (err) {
    console.error('Camera startup error:', err);
    alert('Could not start camera. Check browser permissions.');
    startBtn.disabled = false;
    systemStatusPill.className = 'status-pill status-ready';
    systemStatusText.textContent = 'Camera Blocked';
  }
});

stopBtn.addEventListener('click', stopCamera);

function stopCamera() {
  isStreaming = false;

  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  if (fallbackInterval) { clearInterval(fallbackInterval); fallbackInterval = null; }

  if (websocket && websocket.readyState === WebSocket.OPEN) websocket.close();
  websocket = null;

  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  webcamVideo.srcObject = null;

  cameraPlaceholder.classList.remove('hidden');
  liveIndicator.classList.add('hidden');
  landmarksStatus.classList.add('hidden');
  startBtn.disabled = false;
  stopBtn.disabled  = true;

  systemStatusPill.className = 'status-pill status-ready';
  systemStatusText.textContent = 'System Ready';

  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// ── Main landmark loop (MediaPipe Tasks Vision) ───────────────────────────────
function startLandmarkLoop() {
  const ctx = overlayCanvas.getContext('2d');
  lastVideoTime = -1;

  function loop() {
    if (!isStreaming) return;
    animFrameId = requestAnimationFrame(loop);

    if (webcamVideo.readyState < 2) return;
    if (webcamVideo.currentTime === lastVideoTime) return;
    lastVideoTime = webcamVideo.currentTime;

    // Detect hands
    const nowMs = performance.now();
    let result;
    try {
      result = handLandmarker.detectForVideo(webcamVideo, nowMs);
    } catch (e) {
      console.warn('Landmark detection error:', e);
      return;
    }

    const w = overlayCanvas.width;
    const h = overlayCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const hasHands = result.landmarks && result.landmarks.length > 0;
    landmarksStatus.classList.toggle('hidden', !hasHands);

    // Draw skeleton mesh
    if (toggleSkeleton.checked && hasHands) {
      ctx.save();
      if (toggleMirror.checked) { ctx.translate(w, 0); ctx.scale(-1, 1); }

      for (let hi = 0; hi < result.landmarks.length; hi++) {
        const lms = result.landmarks[hi];
        const handedness = result.handednesses[hi]?.[0]?.categoryName || 'Right';
        const isLeft = handedness === 'Left';

        // Draw connections
        const connColor = isLeft ? 'rgba(16,185,129,0.85)' : 'rgba(139,92,246,0.85)';
        ctx.strokeStyle = connColor;
        ctx.lineWidth   = 2.5;
        for (const [a, b] of HAND_CONNECTIONS) {
          const pa = lms[a], pb = lms[b];
          ctx.beginPath();
          ctx.moveTo(pa.x * w, pa.y * h);
          ctx.lineTo(pb.x * w, pb.y * h);
          ctx.stroke();
        }

        // Draw landmarks
        const dotColor   = isLeft ? '#34d399' : '#c084fc';
        const fillColor  = isLeft ? '#059669' : '#7c3aed';
        for (const lm of lms) {
          ctx.beginPath();
          ctx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
          ctx.fillStyle   = fillColor;
          ctx.fill();
          ctx.strokeStyle = dotColor;
          ctx.lineWidth   = 1.5;
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    // Send keypoints to backend at ~10 fps
    if (nowMs - lastSendTime > SEND_THROTTLE_MS) {
      lastSendTime = nowMs;
      sendKeypointsToBackend(result, hasHands);
    }
  }

  animFrameId = requestAnimationFrame(loop);
}

// ── Build 49-point vector from HandLandmarker result & send via WebSocket ─────
function sendKeypointsToBackend(result, hasHands) {
  if (!websocket || websocket.readyState !== WebSocket.OPEN) return;

  // 7 pose placeholders (NaN — server skips them; hand signals carry the sign)
  const kp49 = Array.from({ length: 7 }, () => [NaN, NaN, NaN]);

  // 21 left + 21 right hand points
  const leftHand  = Array.from({ length: 21 }, () => [NaN, NaN, NaN]);
  const rightHand = Array.from({ length: 21 }, () => [NaN, NaN, NaN]);

  if (result.landmarks) {
    for (let hi = 0; hi < result.landmarks.length; hi++) {
      const lms    = result.landmarks[hi];
      const handed = result.handednesses[hi]?.[0]?.categoryName || 'Right';
      const arr    = handed === 'Left' ? leftHand : rightHand;
      for (let i = 0; i < Math.min(lms.length, 21); i++) {
        arr[i] = [lms[i].x, lms[i].y, lms[i].z || 0];
      }
    }
  }

  const payload = kp49.concat(leftHand, rightHand); // 7+21+21 = 49

  websocket.send(JSON.stringify({
    type: 'keypoints',
    keypoints: payload,
    has_hands: hasHands,
    mode: currentVocabMode
  }));
}

// ── Fallback: send raw JPEG frames when HandLandmarker unavailable ────────────
function startFallbackLoop() {
  console.log('[SignVision] Using backend-side frame detection fallback.');
  const hCtx = hiddenCanvas.getContext('2d');
  hiddenCanvas.width  = 480;
  hiddenCanvas.height = 360;

  fallbackInterval = setInterval(() => {
    if (!isStreaming || !websocket || websocket.readyState !== WebSocket.OPEN) return;
    hCtx.drawImage(webcamVideo, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.65);
    websocket.send(JSON.stringify({ type: 'frame', frame: dataUrl, mode: currentVocabMode }));
  }, 120);
}

// ── WebSocket message handler ─────────────────────────────────────────────────
function handleSocketMessage(rawJson) {
  try {
    const data = JSON.parse(rawJson);

    if (data.type === 'prediction' && data.gloss && data.gloss !== '...') {
      applyPrediction(data);
    }

    if (data.type === 'frame_result') {
      landmarksStatus.classList.toggle('hidden', !data.has_landmarks);

      // Draw server-side skeleton in fallback mode
      if (toggleSkeleton.checked && data.landmarks && !handLandmarker) {
        drawFallbackSkeleton(data.landmarks);
      }

      if (data.prediction && data.prediction.gloss && data.prediction.gloss !== '...') {
        applyPrediction(data.prediction);
      }
    }
  } catch (err) {
    console.error('WS parse error:', err);
  }
}

// ── Draw server-supplied skeleton (fallback only) ─────────────────────────────
function drawFallbackSkeleton(landmarks) {
  const ctx = overlayCanvas.getContext('2d');
  const w   = overlayCanvas.width;
  const h   = overlayCanvas.height;
  ctx.clearRect(0, 0, w, h);

  function drawPoints(pts, color) {
    if (!pts || pts.length === 0) return;
    ctx.fillStyle = color;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  drawPoints(landmarks.left_hand,  '#34d399');
  drawPoints(landmarks.right_hand, '#c084fc');
}

// ── HTML Escaper Helper ───────────────────────────────────────────────────────
function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = String(str ?? '');
  return d.innerHTML;
}

// ── Smart Natural English Sentence Formatter ─────────────────────────────────
function buildEnglishSentence(tokens) {
  if (!tokens || tokens.length === 0) return 'Awaiting gestures...';
  const words = tokens.map(t => String(t).trim()).filter(Boolean);
  if (words.length === 0) return 'Awaiting gestures...';

  const lower = words.map(w => w.toLowerCase());
  const numWords = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
  const allNumbers = lower.every(w => numWords.includes(w));

  // Sequence of number words: "One, two, three."
  if (allNumbers) {
    return words.map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(', ') + '.';
  }

  // Single word
  if (words.length === 1) {
    const w = words[0];
    if (w.toLowerCase() === 'hello') return 'Hello!';
    return w.charAt(0).toUpperCase() + w.slice(1) + '.';
  }

  // Check if sentence starts with or contains questions
  const isQuestion = lower.some(w => ['who', 'what', 'where', 'when', 'why', 'how', 'how are you'].includes(w));

  let sentence = '';
  if (lower[0] === 'hello' || lower[0] === 'hi') {
    sentence = 'Hello';
    if (words.length > 1) {
      sentence += ', ' + words.slice(1).map(w => (w.toLowerCase() === 'i' ? 'I' : w.toLowerCase())).join(' ');
    }
  } else {
    sentence = words.map((w, idx) => {
      if (idx === 0) return w.charAt(0).toUpperCase() + w.slice(1);
      if (w.toLowerCase() === 'i') return 'I';
      return w.toLowerCase();
    }).join(' ');
  }

  if (!/[.!?]$/.test(sentence)) {
    sentence += isQuestion ? '?' : '.';
  }
  return sentence;
}

// ── Prediction stabilisation & commit ────────────────────────────────────────
function applyPrediction(pred) {
  const gloss = (pred.gloss || '').trim();
  const text  = (pred.text || gloss).trim();
  const digit = (pred.digit !== undefined && pred.digit !== null && pred.digit !== '') ? String(pred.digit) : null;
  const conf  = Math.round((pred.confidence || 0.85) * 100);

  // If no hand detected / waiting sign
  if (!gloss || gloss === '...') {
    if (currentCandidate && (performance.now() - lastValidPredictionTime > 550)) {
      currentCandidate     = null;
      currentCandidateText = null;
      candidateCount       = 0;
      predictionBuffer     = [];
      lastCommittedSign    = null; // Re-arm so user can sign the same gesture again
      updateRunningSentence(null);
      if (holdBadge) {
        holdBadge.textContent = 'Hold to commit';
        holdBadge.className   = 'hold-badge';
      }
    }
    return;
  }

  lastValidPredictionTime = performance.now();

  // 1. Display Current Sign / Gloss in English words
  let displayHtml = escapeHtml(gloss.toUpperCase());
  if (digit !== null) {
    displayHtml += ` <span class="digit-subbadge">#${escapeHtml(digit)}</span>`;
  }
  glossWebcam.innerHTML      = displayHtml;
  confidenceVal.textContent  = `${conf}%`;
  confidenceFill.style.width = `${conf}%`;

  // 2. Add to sliding majority buffer
  predictionBuffer.push({ gloss, text, digit, conf, time: performance.now() });
  if (predictionBuffer.length > 5) predictionBuffer.shift();

  // Majority vote over recent frames to eliminate camera jitter
  const counts = {};
  const textMap = {};
  for (const item of predictionBuffer) {
    counts[item.gloss] = (counts[item.gloss] || 0) + 1;
    textMap[item.gloss] = item.text;
  }

  let dominantGloss = null;
  let maxCount = 0;
  for (const [g, count] of Object.entries(counts)) {
    if (count > maxCount) {
      maxCount = count;
      dominantGloss = g;
    }
  }

  // Stable detection
  if (dominantGloss && maxCount >= 2 && conf >= 45) {
    const dominantText = textMap[dominantGloss] || dominantGloss;
    if (dominantGloss === currentCandidate) {
      candidateCount++;
    } else {
      currentCandidate     = dominantGloss;
      currentCandidateText = dominantText;
      candidateCount       = 1;
    }

    // LIVE SENTENCE PREVIEW: instantly show the candidate word forming in the sentence
    updateRunningSentence(currentCandidateText);

    if (candidateCount < REQUIRED_HOLD_FRAMES) {
      if (holdBadge) {
        holdBadge.textContent = `Holding (${candidateCount}/${REQUIRED_HOLD_FRAMES})...`;
        holdBadge.className   = 'hold-badge holding';
      }
    } else {
      // Hold threshold reached! Commit sign to the sentence
      const now = performance.now();
      if (currentCandidate !== lastCommittedSign || (now - lastCommitTime) > 1100) {
        commitSign(currentCandidateText);
        lastCommittedSign = currentCandidate;
        lastCommitTime    = now;
        if (holdBadge) {
          holdBadge.textContent = `Committed: ${currentCandidateText} ✓`;
          holdBadge.className   = 'hold-badge committed';
          clearTimeout(holdBadgeTimeout);
          holdBadgeTimeout = setTimeout(() => {
            if (holdBadge) {
              holdBadge.textContent = 'Hold to commit';
              holdBadge.className   = 'hold-badge';
            }
          }, 850);
        }
      }
      candidateCount = 0;
    }
  } else {
    candidateCount = Math.max(0, candidateCount - 1);
  }
}

function commitSign(sign) {
  recognizedHistory.push(sign);
  renderTokensStream();
  updateRunningSentence(null);
}

function renderTokensStream() {
  tokensStream.innerHTML = '';
  if (recognizedHistory.length === 0) {
    tokensStream.innerHTML = '<span class="empty-tokens">No signs recognized yet</span>';
    return;
  }
  recognizedHistory.forEach((token, index) => {
    const chip = document.createElement('span');
    chip.className  = 'token-chip';
    chip.innerHTML = `
      <span>${escapeHtml(token)}</span>
      <button class="token-chip-del" type="button" title="Remove ${escapeHtml(token)}">×</button>
    `;
    chip.querySelector('.token-chip-del').addEventListener('click', (e) => {
      e.stopPropagation();
      removeTokenAtIndex(index);
    });
    tokensStream.appendChild(chip);
  });
  tokensStream.scrollTop = tokensStream.scrollHeight;
}

function removeTokenAtIndex(idx) {
  if (idx >= 0 && idx < recognizedHistory.length) {
    recognizedHistory.splice(idx, 1);
    lastCommittedSign = null;
    renderTokensStream();
    updateRunningSentence(null);
  }
}

function updateRunningSentence(previewWord = null) {
  if (recognizedHistory.length === 0) {
    if (previewWord) {
      textWebcam.innerHTML = `<span class="sentence-preview">${escapeHtml(previewWord)}...</span>`;
    } else {
      textWebcam.textContent = 'Awaiting gestures...';
    }
    return;
  }

  const formatted = buildEnglishSentence(recognizedHistory);
  if (previewWord) {
    textWebcam.innerHTML = `${escapeHtml(formatted)} <span class="sentence-preview">${escapeHtml(previewWord.toLowerCase())}...</span>`;
  } else {
    textWebcam.textContent = formatted;
  }
}

// ── Sentence actions ─────────────────────────────────────────────────────────
backspaceBtn?.addEventListener('click', () => {
  if (recognizedHistory.length > 0) {
    recognizedHistory.pop();
    lastCommittedSign = null;
    renderTokensStream();
    updateRunningSentence(null);
  }
});

copyBtn?.addEventListener('click', async () => {
  const text = textWebcam.textContent;
  if (!text || text === 'Awaiting gestures...') return;
  try {
    await navigator.clipboard.writeText(text);
    const orig = copyBtn.textContent;
    copyBtn.textContent = '✓ Copied!';
    setTimeout(() => { copyBtn.textContent = orig; }, 1200);
  } catch (e) { console.warn('Clipboard error:', e); }
});

clearHistoryBtn?.addEventListener('click', () => {
  recognizedHistory       = [];
  currentCandidate        = null;
  currentCandidateText    = null;
  candidateCount          = 0;
  lastCommittedSign       = null;
  predictionBuffer        = [];
  renderTokensStream();
  textWebcam.textContent     = 'Awaiting gestures...';
  glossWebcam.textContent    = '—';
  confidenceVal.textContent  = '0%';
  confidenceFill.style.width = '0%';
  if (holdBadge) {
    holdBadge.textContent = 'Hold to commit';
    holdBadge.className   = 'hold-badge';
  }
});

speakBtn?.addEventListener('click', () => {
  const text = textWebcam.textContent;
  if (!text || text === 'Awaiting gestures...') return;
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 0.95;
    window.speechSynthesis.speak(utt);
  }
});

// ── Video file upload ─────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(ev => {
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.add('dragover'); });
});
['dragleave', 'drop'].forEach(ev => {
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('dragover'); });
});

dropZone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length > 0) handleSelectedFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleSelectedFile(fileInput.files[0]);
});

let currentUploadFile = null;

function handleSelectedFile(file) {
  currentUploadFile = file;
  selectedFileName.textContent = file.name;
  previewVideo.src = URL.createObjectURL(file);
  dropZone.classList.add('hidden');
  previewContainer.classList.remove('hidden');
  uploadProgressCard.classList.add('hidden');
}

removeFileBtn.addEventListener('click', () => {
  currentUploadFile = null;
  fileInput.value   = '';
  previewVideo.pause();
  previewVideo.src  = '';
  previewContainer.classList.add('hidden');
  dropZone.classList.remove('hidden');
  uploadProgressCard.classList.add('hidden');
});

processBtn.addEventListener('click', async () => {
  if (!currentUploadFile) return;
  processBtn.disabled = true;
  uploadProgressCard.classList.remove('hidden');
  progressStatus.textContent = 'Extracting Keypoints & Translating...';

  const fd = new FormData();
  fd.append('file', currentUploadFile);
  fd.append('mode', currentVocabMode);

  try {
    const res    = await fetch('/api/upload', { method: 'POST', body: fd });
    const result = await res.json();

    if (!res.ok || result.error) throw new Error(result.error || 'Failed to process video');

    const glossText = (result.gloss || '—').toUpperCase();
    if (result.digit !== null && result.digit !== undefined) {
      glossUpload.innerHTML = `${escapeHtml(glossText)} <span class="digit-subbadge">#${escapeHtml(result.digit)}</span>`;
    } else {
      glossUpload.textContent = glossText;
    }
    textUpload.textContent       = result.text || '—';
    metricFrames.textContent     = result.total_frames || '0';
    metricLatency.textContent    = `${result.processing_time_sec}s`;
    metricConfidence.textContent = `${Math.round((result.confidence || 0.9) * 100)}%`;

    progressStatus.textContent = 'Translation Complete!';
    setTimeout(() => { uploadProgressCard.classList.add('hidden'); processBtn.disabled = false; }, 1200);
  } catch (err) {
    console.error('Upload error:', err);
    alert(`Error: ${err.message}`);
    uploadProgressCard.classList.add('hidden');
    processBtn.disabled = false;
  }
});
