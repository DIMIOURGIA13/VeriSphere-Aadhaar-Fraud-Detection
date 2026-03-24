"""
py flask_app.py
Flask web server for Aadhaar Fraud Detection
Serves the Stitch HTML frontend and exposes /analyze API endpoint.
"""
import os
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

from aadhaar_pipeline.pipeline import run_pipeline
from aadhaar_pipeline.detector import load_model
from aadhaar_pipeline.tampering import load_tampering_model

app = Flask(__name__, static_folder="Templates/static")

# cache models at startup
_yolo = None
_resnet = None

def get_models():
    global _yolo, _resnet
    if _yolo is None:
        _yolo = load_model("aadhaar_best.pt")
    if _resnet is None:
        _resnet = load_tampering_model(None, device="cpu")
    return _yolo, _resnet


@app.route("/")
def index():
    html = Path("Templates/stitch 1/code.html").read_text(encoding="utf-8")
    # inject the JS glue code before </body>
    js = _get_glue_js()
    html = html.replace("</body>", js + "\n</body>")
    return html


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    tmp = Path("temp_upload.jpg")
    file.save(str(tmp))

    try:
        yolo, resnet = get_models()
        result = run_pipeline(
            image_path=str(tmp),
            yolo_weights="aadhaar_best.pt",
            device="cpu",
            verbose=False,
            yolo_model=yolo,
            resnet_tuple=resnet,
        )

        # build annotated image as base64 for the frontend
        image = cv2.imread(str(tmp))
        annotated = _annotate(image, result["detections"])
        _, buf = cv2.imencode(".jpg", annotated)
        annotated_b64 = base64.b64encode(buf).decode()

        tmp.unlink(missing_ok=True)
        result["annotated_image"] = annotated_b64
        # make result JSON-serialisable
        result.pop("qr_fields", None)
        return jsonify(result)

    except Exception as e:
        tmp.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


def _annotate(image, detections):
    colors = {
        "aadhaar_number": (99, 102, 241),
        "address":        (251, 191, 36),
        "dob":            (52, 211, 153),
        "face":           (248, 113, 113),
        "name":           (167, 139, 250),
        "qr_code":        (56, 189, 248),
    }
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        col = colors.get(det["class_name"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        lbl = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x1, max(y1 - 18, 0)), (x1 + tw + 6, max(y1, 18)), col, -1)
        cv2.putText(out, lbl, (x1 + 3, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 1)
    return out


def _get_glue_js():
    """JavaScript that wires the upload zone + button to /analyze and renders results."""
    return """
<script>
(function () {
  // ── state ──────────────────────────────────────────────────────────────────
  let selectedFile = null;

  // ── grab elements ──────────────────────────────────────────────────────────
  const uploadZone  = document.querySelector('.border-dashed');
  const analyzeBtn  = document.querySelector('button[disabled]') ||
                      [...document.querySelectorAll('button')].find(b => b.textContent.includes('Analyze'));
  const previewWrap = document.querySelector('.aspect-\\\\[1\\\\.58\\\\/1\\\\]') ||
                      document.querySelector('[class*="aspect"]');

  // hidden file input
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/jpeg,image/png,image/webp';
  fileInput.style.display = 'none';
  document.body.appendChild(fileInput);

  // ── upload zone click ──────────────────────────────────────────────────────
  if (uploadZone) uploadZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (!fileInput.files.length) return;
    selectedFile = fileInput.files[0];
    // show preview
    const url = URL.createObjectURL(selectedFile);
    if (previewWrap) {
      previewWrap.innerHTML = `<img src="${url}" class="w-full h-full object-cover rounded-xl" />`;
    }
    // enable button — swap to gradient style
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed',
        'bg-surface-container-highest', 'text-on-surface-variant', 'border', 'border-white/5');
      analyzeBtn.classList.add('bg-gradient-to-r', 'from-[#8083ff]', 'to-[#571bc1]',
        'text-white', 'shadow-[0_0_20px_rgba(128,131,255,0.35)]', 'cursor-pointer',
        'hover:opacity-90', 'transition-all');
    }
  });

  // drag-and-drop
  if (uploadZone) {
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('border-[#8083ff]'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('border-[#8083ff]'));
    uploadZone.addEventListener('drop', e => {
      e.preventDefault();
      uploadZone.classList.remove('border-[#8083ff]');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event('change'));
      }
    });
  }

  // ── analyze button ─────────────────────────────────────────────────────────
  const loadingQuotes = [
    '🔍 Detecting card regions...',
    '📖 Reading text fields...',
    '🔢 Validating Aadhaar number...',
    '📷 Decoding QR code...',
    '🧪 Checking for tampering...',
    '📊 Computing fraud score...',
    '🛡️ Running forensics...',
    '✅ Finalising verdict...',
  ];

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      // start rotating quotes
      let qi = 0;
      analyzeBtn.innerHTML = `<span class="material-symbols-outlined animate-spin" style="animation:spin 1s linear infinite">progress_activity</span>&nbsp;${loadingQuotes[0]}`;
      analyzeBtn.disabled = true;
      const quoteTimer = setInterval(() => {
        qi = (qi + 1) % loadingQuotes.length;
        analyzeBtn.innerHTML = `<span class="material-symbols-outlined" style="display:inline-block;animation:spin 1s linear infinite">progress_activity</span>&nbsp;${loadingQuotes[qi]}`;
      }, 1400);

      const fd = new FormData();
      fd.append('image', selectedFile);

      try {
        const res  = await fetch('/analyze', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error) { alert('Error: ' + data.error); return; }
        renderResults(data);
      } catch (err) {
        alert('Request failed: ' + err);
      } finally {
        clearInterval(quoteTimer);
        analyzeBtn.innerHTML = '<span class="material-symbols-outlined">search</span>&nbsp;Analyze Card';
        analyzeBtn.disabled = false;
      }
    });
  }

  // inject spin keyframe once
  if (!document.getElementById('_spin_style')) {
    const s = document.createElement('style');
    s.id = '_spin_style';
    s.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
    document.head.appendChild(s);
  }

  // ── render results ─────────────────────────────────────────────────────────
  function renderResults(d) {
    // load the output page HTML and inject data
    const verdict     = d.verdict;       // "Genuine" | "Suspicious" | "Fake"
    const fraudScore  = Math.round(d.fraud_score * 100);
    const confidence  = Math.round(d.confidence * 100);
    const verhoeff    = d.verhoeff || {};
    const consistency = d.consistency || {};
    const tampering   = d.tampering || {};
    const reasons     = d.reasons || [];
    const annotated   = d.annotated_image;

    const verdictColor = { Genuine: '#4ade80', Suspicious: '#facc15', Fake: '#f87171' }[verdict] || '#8083ff';
    const verdictIcon  = { Genuine: '✓', Suspicious: '⚠', Fake: '✗' }[verdict] || '?';
    const verdictBorder= { Genuine: 'border-emerald-500/40', Suspicious: 'border-yellow-500/40', Fake: 'border-red-500/40' }[verdict] || '';
    const verdictGlow  = { Genuine: 'glow-green', Suspicious: '', Fake: '' }[verdict] || '';

    const qrAvail  = consistency.qr_available;
    const qrMatch  = Math.round(consistency.qr_match_score || 0);
    const qrFmt    = consistency.qr_format || 'unknown';
    const qrBadge  = !qrAvail ? '—' : qrMatch >= 75 ? '✓' : qrMatch >= 50 ? '⚠' : '✗';
    const qrColor  = !qrAvail ? '#9ca3af' : qrMatch >= 75 ? '#4ade80' : qrMatch >= 50 ? '#facc15' : '#f87171';
    const qrLabel  = !qrAvail ? 'Not decoded' : `${qrMatch}% match — ${qrFmt}`;

    const vValid   = verhoeff.valid;
    const vColor   = vValid ? '#4ade80' : '#f87171';
    const vBadge   = vValid ? '✓' : '✗';
    const vLabel   = vValid ? 'Valid checksum — OK' : `Failed — ${verhoeff.reason || ''}`;

    const tLabel   = tampering.label || 'unknown';
    const tConf    = Math.round((tampering.confidence || 0) * 100);
    const tColor   = tLabel === 'real' ? '#4ade80' : tLabel === 'fake' ? '#f87171' : '#facc15';
    const tBadge   = tLabel === 'real' ? '✓' : tLabel === 'fake' ? '✗' : '⚠';
    const tText    = tLabel === 'real' ? `Clean (${tConf}%)` : tLabel === 'fake' ? `Tampered (${tConf}%)` : `Suspicious (${tConf}%)`;

    const forensics = tampering.forensics || {};
    const elaVal    = (forensics.ela_mean || 0).toFixed(2);
    const elaFlag   = forensics.ela_flagged;
    const noiseVal  = (forensics.noise_cv || 0).toFixed(3);
    const noiseFlag = forensics.noise_flagged;
    const sharpVal  = (forensics.sharpness_cv || 0).toFixed(3);
    const sharpFlag = forensics.sharpness_flagged;
    const flags     = forensics.flags_triggered || 0;
    const flagColor = flags >= 2 ? '#f87171' : flags === 1 ? '#facc15' : '#4ade80';

    const findingsHtml = reasons.map(r =>
      `<li class="flex items-start space-x-3">
        <div class="mt-1 w-2 h-2 rounded-full flex-shrink-0" style="background:${verdictColor};box-shadow:0 0 8px ${verdictColor}"></div>
        <span class="text-sm text-on-surface-variant">${r}</span>
      </li>`
    ).join('');

    const annotatedHtml = annotated
      ? `<img src="data:image/jpeg;base64,${annotated}" class="w-full h-full object-cover" />`
      : '<div class="text-on-surface-variant text-xs text-center p-4">No annotated image</div>';

    // build the right-column results HTML
    const resultsHtml = `
      <div class="border-2 ${verdictBorder} ${verdictGlow} rounded-xl p-6 relative overflow-hidden" style="background:#1c1f29">
        <div class="absolute top-0 right-0 p-4">
          <div class="w-12 h-12 rounded-lg flex items-center justify-center" style="background:${verdictColor}22">
            <span style="font-size:2rem;color:${verdictColor}">${verdictIcon}</span>
          </div>
        </div>
        <h2 class="text-4xl font-black tracking-tighter mb-1 uppercase" style="color:${verdictColor}">${verdictIcon} ${verdict.toUpperCase()}</h2>
        <p class="text-on-surface-variant text-sm font-medium">${reasons[0] || ''}</p>
      </div>

      <div class="grid grid-cols-2 gap-4">
        <div class="glass-panel p-4 rounded-xl">
          <div class="flex justify-between items-end mb-2">
            <span class="text-[10px] font-bold tracking-widest text-on-surface-variant uppercase">Fraud Score</span>
            <span class="text-xl font-black" style="color:${verdictColor}">${fraudScore}%</span>
          </div>
          <div class="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
            <div class="h-full rounded-full" style="width:${fraudScore}%;background:${verdictColor};box-shadow:0 0 10px ${verdictColor}"></div>
          </div>
        </div>
        <div class="glass-panel p-4 rounded-xl">
          <div class="flex justify-between items-end mb-2">
            <span class="text-[10px] font-bold tracking-widest text-on-surface-variant uppercase">Confidence</span>
            <span class="text-xl font-black text-primary">${confidence}%</span>
          </div>
          <div class="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
            <div class="h-full rounded-full bg-primary" style="width:${confidence}%;box-shadow:0 0 10px #8083ff"></div>
          </div>
        </div>
      </div>

      <div class="glass-panel rounded-xl overflow-hidden">
        <div class="px-6 py-4 border-b border-white/5">
          <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant">Verification Checks</h3>
        </div>
        <div class="divide-y divide-white/5">
          <div class="px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-4">
              <div class="w-[26px] h-[26px] rounded flex items-center justify-center font-bold text-sm" style="background:${vColor}22;color:${vColor}">${vBadge}</div>
              <div><div class="text-sm font-medium">Verhoeff Checksum</div><div class="text-xs text-on-surface-variant">${vLabel}</div></div>
            </div>
          </div>
          <div class="px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-4">
              <div class="w-[26px] h-[26px] rounded flex items-center justify-center font-bold text-sm" style="background:${qrColor}22;color:${qrColor}">${qrBadge}</div>
              <div><div class="text-sm font-medium">QR Consistency</div><div class="text-xs text-on-surface-variant">${qrLabel}</div></div>
            </div>
          </div>
          <div class="px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-4">
              <div class="w-[26px] h-[26px] rounded flex items-center justify-center font-bold text-sm" style="background:${tColor}22;color:${tColor}">${tBadge}</div>
              <div><div class="text-sm font-medium">Tampering Detection</div><div class="text-xs text-on-surface-variant">${tText}</div></div>
            </div>
          </div>
        </div>
      </div>

      <div>
        <div class="flex justify-between items-center mb-3">
          <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant">Digital Forensics</h3>
          <span class="text-xs font-bold" style="color:${flagColor}">${flags}/3 flags triggered</span>
        </div>
        <div class="grid grid-cols-3 gap-4">
          <div class="glass-panel p-4 rounded-xl text-center">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-2">ELA Score</div>
            <div class="text-lg font-black" style="color:${elaFlag?'#f87171':'#4ade80'}">${elaVal}/255</div>
            <div class="text-[9px] text-on-surface-variant/60 mt-1">${elaFlag?'⚠ Re-compression':'Normal'}</div>
          </div>
          <div class="glass-panel p-4 rounded-xl text-center">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-2">Noise CV</div>
            <div class="text-lg font-black" style="color:${noiseFlag?'#f87171':'#4ade80'}">${noiseVal}</div>
            <div class="text-[9px] text-on-surface-variant/60 mt-1">${noiseFlag?'⚠ Uneven':'Uniform'}</div>
          </div>
          <div class="glass-panel p-4 rounded-xl text-center">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-2">Sharpness CV</div>
            <div class="text-lg font-black" style="color:${sharpFlag?'#f87171':'#4ade80'}">${sharpVal}</div>
            <div class="text-[9px] text-on-surface-variant/60 mt-1">${sharpFlag?'⚠ Mismatch':'Consistent'}</div>
          </div>
        </div>
      </div>

      <div class="glass-panel p-6 rounded-xl">
        <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant mb-4">Key Findings</h3>
        <ul class="space-y-3">${findingsHtml}</ul>
      </div>

      <div class="glass-panel p-4 rounded-xl">
        <h3 class="text-sm font-bold tracking-wider uppercase text-on-surface-variant mb-3">Detected Regions</h3>
        <div class="rounded-lg overflow-hidden aspect-[1.6/1]">${annotatedHtml}</div>
      </div>
    `;

    // find or create the right column and replace its content
    let rightCol = document.getElementById('results-panel');
    if (!rightCol) {
      // first run — find the idle placeholder and replace it
      const idle = document.querySelector('.xl\\\\:col-span-7, .lg\\\\:col-span-7');
      if (idle) {
        idle.id = 'results-panel';
        rightCol = idle;
      }
    }
    if (rightCol) {
      rightCol.innerHTML = `<div class="space-y-6">${resultsHtml}</div>`;
    }
  }
})();
</script>
"""


if __name__ == "__main__":
    app.run(debug=False, port=5000)
