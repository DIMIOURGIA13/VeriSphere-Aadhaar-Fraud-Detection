"""
py -m streamlit run app.py
Aadhaar Fraud Detection — Streamlit UI
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from aadhaar_pipeline.pipeline import run_pipeline
from aadhaar_pipeline.detector import load_model
from aadhaar_pipeline.tampering import load_tampering_model

@st.cache_resource
def get_yolo():
    return load_model("aadhaar_best.pt")

@st.cache_resource
def get_resnet():
    resnet_path = next(
        (c for c in ["resnet_aadhaar.pth","resnet18_tampering.pth","resnet50_tampering.pth","resnet_tampering.pth"]
         if Path(c).exists()), None)
    return load_tampering_model(resnet_path, device="cpu")

st.set_page_config(
    page_title="Aadhaar Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #080b14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.topbar {
    background: linear-gradient(90deg,#0d1117,#161b27);
    border-bottom: 1px solid rgba(99,102,241,.25);
    padding: 18px 48px; display:flex; align-items:center; gap:14px;
}
.topbar-title {
    font-size:1.25rem; font-weight:700;
    background:linear-gradient(135deg,#a78bfa,#60a5fa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.topbar-sub { font-size:.78rem; color:#4b5563; margin-left:auto; }
.card { background:linear-gradient(145deg,#0f1623,#141d2e); border:1px solid rgba(255,255,255,.06); border-radius:16px; padding:20px 22px; margin-bottom:12px; }
.verdict-box { border-radius:16px; padding:28px 20px; text-align:center; margin-bottom:14px; }
.v-genuine { background:linear-gradient(135deg,#052e16,#064e3b); border:1.5px solid #10b981; box-shadow:0 0 40px rgba(16,185,129,.18); }
.v-suspicious { background:linear-gradient(135deg,#431407,#78350f); border:1.5px solid #f59e0b; box-shadow:0 0 40px rgba(245,158,11,.18); }
.v-fake { background:linear-gradient(135deg,#3b0a0a,#7f1d1d); border:1.5px solid #ef4444; box-shadow:0 0 40px rgba(239,68,68,.18); }
.v-icon { font-size:2.6rem; display:block; margin-bottom:6px; }
.v-label { font-size:1.6rem; font-weight:800; letter-spacing:2px; }
.v-genuine .v-label { color:#34d399; }
.v-suspicious .v-label { color:#fbbf24; }
.v-fake .v-label { color:#f87171; }
.v-sub { font-size:.82rem; margin-top:4px; opacity:.7; color:#e2e8f0; }
.bar-bg { background:rgba(255,255,255,.06); border-radius:99px; height:7px; overflow:hidden; margin-top:6px; }
.bar-fill { height:100%; border-radius:99px; }
.score-lbl { font-size:.7rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:.8px; }
.score-num { font-size:.82rem; font-weight:700; color:#e2e8f0; }
.chk-row { display:flex; align-items:center; gap:10px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,.04); }
.chk-row:last-child { border-bottom:none; }
.badge { width:26px; height:26px; border-radius:7px; display:flex; align-items:center; justify-content:center; font-size:.82rem; font-weight:700; flex-shrink:0; }
.bp { background:rgba(16,185,129,.15); color:#34d399; }
.bf { background:rgba(239,68,68,.15); color:#f87171; }
.bw { background:rgba(245,158,11,.15); color:#fbbf24; }
.bs { background:rgba(107,114,128,.15); color:#9ca3af; }
.chk-title { font-size:.82rem; font-weight:600; color:#e2e8f0; }
.chk-det { font-size:.72rem; color:#6b7280; margin-top:1px; }
.fld { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.05); border-radius:10px; padding:12px 14px; }
.fld-lbl { font-size:.67rem; font-weight:600; color:#6366f1; text-transform:uppercase; letter-spacing:.8px; margin-bottom:4px; }
.fld-val { font-size:.87rem; font-weight:500; color:#e2e8f0; word-break:break-word; }
.fld-empty { color:#374151; font-style:italic; }
.sec { font-size:.67rem; font-weight:700; color:#4b5563; text-transform:uppercase; letter-spacing:1px; margin:16px 0 8px; padding-bottom:6px; border-bottom:1px solid rgba(255,255,255,.05); }
[data-testid="stFileUploader"] { background:rgba(99,102,241,.05) !important; border:1.5px dashed rgba(99,102,241,.35) !important; border-radius:14px !important; }
.stButton > button { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important; color:white !important; border:none !important; border-radius:12px !important; font-weight:600 !important; font-size:.95rem !important; padding:12px 0 !important; width:100% !important; box-shadow:0 4px 20px rgba(99,102,241,.35) !important; }
[data-testid="stImage"] img { border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="topbar"><span style="font-size:1.6rem">🛡️</span><span class="topbar-title">Aadhaar Fraud Detection</span><span class="topbar-sub">YOLOv8 · EasyOCR · Verhoeff · QR · ResNet + Forensics</span></div>', unsafe_allow_html=True)
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

left, right = st.columns([4, 6], gap="large")

with left:
    st.markdown("<div style='padding:0 8px 0 32px'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, use_container_width=True, caption="Uploaded card")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    analyze = st.button("🔍  Analyze Card", use_container_width=True, disabled=uploaded is None)
    st.markdown("""
    <div class="card" style="margin-top:14px;">
      <div class="sec">How it works</div>
      <div style="font-size:.82rem;color:#6b7280;line-height:1.9;">
        <span style="color:#a78bfa">①</span> YOLO detects card regions<br>
        <span style="color:#60a5fa">②</span> EasyOCR reads text fields<br>
        <span style="color:#34d399">③</span> Verhoeff validates the UID<br>
        <span style="color:#fbbf24">④</span> QR code cross-checked<br>
        <span style="color:#f87171">⑤</span> ResNet + Forensics tampering check
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div style='padding:0 32px 0 8px'>", unsafe_allow_html=True)

    if not uploaded or not analyze:
        st.markdown("""
        <div style="height:400px;display:flex;flex-direction:column;align-items:center;justify-content:center;
             text-align:center;background:linear-gradient(145deg,#0f1623,#141d2e);
             border:1px dashed rgba(255,255,255,.07);border-radius:18px;">
          <div style="font-size:3.5rem;margin-bottom:12px;opacity:.25;">🛡️</div>
          <div style="font-size:1rem;font-weight:600;color:#374151;">Upload a card and click Analyze</div>
        </div>""", unsafe_allow_html=True)

    elif analyze and uploaded:
        with st.spinner("Analyzing..."):
            img_np  = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            temp    = Path("temp_upload.jpg")
            cv2.imwrite(str(temp), img_bgr)
            try:
                result = run_pipeline(
                    image_path=str(temp),
                    yolo_weights="aadhaar_best.pt",
                    resnet_weights=None,
                    device="cpu", verbose=True,
                    yolo_model=get_yolo(),
                    resnet_tuple=get_resnet())
                temp.unlink(missing_ok=True)
                error = None
            except Exception as e:
                temp.unlink(missing_ok=True)
                error = str(e)
                result = None

        if error:
            st.error(f"Pipeline error: {error}")
        else:
            verdict     = result["verdict"]
            fraud_score = result["fraud_score"]
            confidence  = result["confidence"]
            ocr         = result["ocr_fields"]
            verhoeff    = result["verhoeff"]
            consistency = result["consistency"]
            tampering   = result["tampering"]
            reasons     = result["reasons"]
            detections  = result["detections"]

            # ── verdict ──
            vcls  = {"Genuine":"genuine","Suspicious":"suspicious","Fake":"fake"}[verdict]
            vicon = {"Genuine":"✓","Suspicious":"⚠","Fake":"✗"}[verdict]
            vsub  = {"Genuine":"All checks passed","Suspicious":"Some checks raised concerns","Fake":"Multiple checks failed"}[verdict]
            st.markdown(f'<div class="verdict-box v-{vcls}"><span class="v-icon">{vicon}</span><div class="v-label">{verdict.upper()}</div><div class="v-sub">{vsub}</div></div>', unsafe_allow_html=True)

            # ── score bars ──
            fp = int(fraud_score * 100)
            cp = int(confidence * 100)
            bc = {"Genuine":"#10b981","Suspicious":"#f59e0b","Fake":"#ef4444"}[verdict]
            st.markdown(f"""
            <div class="card">
              <div style="display:flex;justify-content:space-between"><span class="score-lbl">Fraud Score</span><span class="score-num">{fp}%</span></div>
              <div class="bar-bg"><div class="bar-fill" style="width:{fp}%;background:{bc}"></div></div>
              <div style="display:flex;justify-content:space-between;margin-top:12px"><span class="score-lbl">Confidence</span><span class="score-num">{cp}%</span></div>
              <div class="bar-bg"><div class="bar-fill" style="width:{cp}%;background:#6366f1"></div></div>
            </div>""", unsafe_allow_html=True)

            # ── verification checks ──
            vp   = verhoeff.get("valid", False)
            vb   = "bp" if vp else "bf"
            vs   = "✓" if vp else "✗"
            vtxt = "Valid checksum" if vp else "Checksum failed"
            vdet = verhoeff.get("reason","")

            qa = consistency.get("qr_available", False)
            qr_fmt = consistency.get("qr_format", "")
            qr_match = consistency.get("qr_match_score", 0)
            qr_err = consistency.get("qr_error", "")
            if not qa:
                err_msg = qr_err if qr_err else "Could not read QR code"
                qb,qs,qt,qd = "bs","—","QR not decoded", err_msg
            elif qr_match >= 75:
                qb,qs,qt,qd = "bp","✓",f"{qr_match:.0f}% match",f"Format: {qr_fmt or 'unknown'}"
            elif qr_match >= 50 or verdict == "Genuine":
                # partial match on a genuine card — amber warning, not hard fail
                qb,qs,qt,qd = "bw","⚠",f"{qr_match:.0f}% match",f"Partial match — format: {qr_fmt or 'unknown'}"
            else:
                qb,qs,qt,qd = "bf","✗",f"{qr_match:.0f}% match",f"OCR/QR mismatch — format: {qr_fmt or 'unknown'}"

            tskip  = tampering.get("skipped", False)
            tlabel = tampering.get("label","unknown")
            tconf  = tampering.get("confidence", 0)
            tmethod= tampering.get("method","forensics")
            if tskip:
                tb,ts,tt,td = "bs","—","Not available","No model"
            elif tlabel == "real":
                tb,ts,tt,td = "bp","✓",f"Clean ({tconf:.0%})",tmethod
            elif tlabel == "fake":
                tb,ts,tt,td = "bf","✗",f"Tampered ({tconf:.0%})",tmethod
            elif tlabel == "suspicious":
                tb,ts,tt,td = "bw","⚠",f"Suspicious ({tconf:.0%})",tmethod
            else:
                tb,ts,tt,td = "bw","?","Unknown","Inconclusive"

            st.markdown(f"""
            <div class="card">
              <div class="sec">Verification Checks</div>
              <div class="chk-row"><div class="badge {vb}">{vs}</div><div><div class="chk-title">Verhoeff Checksum</div><div class="chk-det">{vtxt} — {vdet}</div></div></div>
              <div class="chk-row"><div class="badge {qb}">{qs}</div><div><div class="chk-title">QR Consistency</div><div class="chk-det">{qt} — {qd}</div></div></div>
              <div class="chk-row"><div class="badge {tb}">{ts}</div><div><div class="chk-title">Tampering Detection</div><div class="chk-det">{tt} — {td}</div></div></div>
            </div>""", unsafe_allow_html=True)

            # ── forensics breakdown — always show ──
            forensics_data = tampering.get("forensics", {})
            resnet_sc = tampering.get("resnet_scores", {})

            st.markdown('<div class="sec">Tampering Analysis</div>', unsafe_allow_html=True)

            if resnet_sc:
                rf = resnet_sc.get("fake", 0)
                rr = resnet_sc.get("real", 0)
                ri = "🔴" if rf > 0.5 else "🟢"
                st.markdown(f'<div class="card" style="margin-bottom:10px"><div class="fld-lbl">ResNet Score</div><div class="fld-val" style="color:#6b7280;font-style:italic">ResNet disabled — unreliable on this dataset</div></div>', unsafe_allow_html=True)

            if forensics_data:
                ela_mean = forensics_data.get("ela_mean", 0)
                ela_f    = forensics_data.get("ela_flagged", False)
                noise_cv = forensics_data.get("noise_cv", 0)
                noise_f  = forensics_data.get("noise_flagged", False)
                sharp_cv = forensics_data.get("sharpness_cv", 0)
                sharp_f  = forensics_data.get("sharpness_flagged", False)
                flags    = forensics_data.get("flags_triggered", 0)

                c1, c2, c3 = st.columns(3)
                with c1:
                    ind = "🔴" if ela_f else "🟢"
                    note = "⚠ Re-compression detected" if ela_f else "Normal compression"
                    st.markdown(f'<div class="fld"><div class="fld-lbl">ELA Score</div><div class="fld-val">{ind} {ela_mean:.2f} / 255</div><div style="font-size:.7rem;color:#6b7280;margin-top:3px">{note}</div></div>', unsafe_allow_html=True)
                with c2:
                    ind = "🔴" if noise_f else "🟢"
                    note = "⚠ Uneven noise pattern" if noise_f else "Uniform noise"
                    st.markdown(f'<div class="fld"><div class="fld-lbl">Noise Consistency</div><div class="fld-val">{ind} CV {noise_cv:.3f}</div><div style="font-size:.7rem;color:#6b7280;margin-top:3px">{note}</div></div>', unsafe_allow_html=True)
                with c3:
                    ind = "🔴" if sharp_f else "🟢"
                    note = "⚠ Sharpness mismatch" if sharp_f else "Consistent sharpness"
                    st.markdown(f'<div class="fld"><div class="fld-lbl">Sharpness Variance</div><div class="fld-val">{ind} CV {sharp_cv:.3f}</div><div style="font-size:.7rem;color:#6b7280;margin-top:3px">{note}</div></div>', unsafe_allow_html=True)

                flag_color = "#ef4444" if flags >= 2 else ("#f59e0b" if flags == 1 else "#10b981")
                flag_txt   = f"{flags}/3 forensic flags triggered"
                st.markdown(f'<div style="margin-top:8px;font-size:.78rem;color:{flag_color};font-weight:600">{flag_txt}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="fld"><div class="fld-val" style="color:#6b7280">Forensics data not available</div></div>', unsafe_allow_html=True)

            # ── findings ──
            all_ok = reasons == ["All checks passed."]
            rows = ""
            for r in reasons:
                dot = "#10b981" if all_ok else "#f59e0b"
                rows += f'<div style="display:flex;gap:10px;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.83rem;color:#cbd5e1"><div style="width:6px;height:6px;border-radius:50%;background:{dot};margin-top:6px;flex-shrink:0"></div><span>{r}</span></div>'
            st.markdown(f'<div class="card"><div class="sec">Findings</div>{rows}</div>', unsafe_allow_html=True)

            # ── annotated image ──
            annotated = img_np.copy()
            colors_rgb = {"aadhaar_number":(99,102,241),"address":(251,191,36),"dob":(52,211,153),"face":(248,113,113),"name":(167,139,250),"qr_code":(56,189,248)}
            for det in detections:
                x1,y1,x2,y2 = det["bbox"]
                col = colors_rgb.get(det["class_name"],(200,200,200))
                cv2.rectangle(annotated,(x1,y1),(x2,y2),col,2)
                lbl = f"{det['class_name']} {det['confidence']:.2f}"
                (tw,_),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
                cv2.rectangle(annotated,(x1,max(y1-18,0)),(x1+tw+6,max(y1,18)),col,-1)
                cv2.putText(annotated,lbl,(x1+3,max(y1-4,14)),cv2.FONT_HERSHEY_SIMPLEX,0.42,(10,10,10),1)

            st.markdown('<div class="sec">Detected Regions</div>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
