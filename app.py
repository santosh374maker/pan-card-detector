import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time

st.set_page_config(
    page_title="PAN Card Detector",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Sora', sans-serif; background: #080C14; color: #E8EDF5; }
.stApp { background: #080C14; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.hero { background: #0D1320; border-bottom: 1px solid #1E2D4A; padding: 2rem 3rem 1.5rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25);
    border-radius: 20px; padding: 4px 14px; font-size: 11px;
    font-family: 'Space Mono', monospace; color: #60A5FA;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.8rem;
}
.hero-badge span { width:6px; height:6px; border-radius:50%; background:#3B82F6; animation:pulse 2s infinite; }
.hero-title {
    font-size: clamp(1.6rem, 2.5vw, 2.2rem); font-weight: 700;
    letter-spacing: -0.03em; line-height: 1.1;
    background: linear-gradient(135deg, #E8EDF5 30%, #93C5FD 70%, #818CF8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.4rem;
}
.hero-sub { font-size: 0.88rem; color: #6B7FA3; font-weight: 300; }
.stats-row {
    display: flex; gap: 1rem; padding: 1rem 3rem;
    border-bottom: 1px solid #1E2D4A; background: #0A0F1C;
}
.stat-pill {
    display: flex; align-items: center; gap: 8px; padding: 5px 12px;
    background: #111827; border: 1px solid #1E2D4A; border-radius: 8px;
    font-size: 12px; color: #6B7FA3;
}
.stat-pill strong { color: #93C5FD; font-family: 'Space Mono', monospace; font-size: 12px; }
.panel { padding: 1.5rem 2rem; border-right: 1px solid #1E2D4A; }
.panel-right { border-right: none; }
.panel-label {
    font-size: 10px; font-family: 'Space Mono', monospace;
    letter-spacing: 0.12em; text-transform: uppercase; color: #3B82F6;
    margin-bottom: 0.8rem; display: flex; align-items: center; gap: 8px;
}
.panel-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, #1E2D4A, transparent); }
div[data-testid="stImage"] img { max-width: 380px !important; border-radius: 10px !important; border: 1px solid #1E2D4A !important; }
.result-card { border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; }
.result-card.success { background: #051B11; border: 1px solid rgba(34,197,94,0.25); }
.result-card.fail    { background: #1A0A0A; border: 1px solid rgba(239,68,68,0.25); }
.result-card.waiting { background: #0D1320; border: 1px solid #1E2D4A; }
.result-icon  { font-size: 1.8rem; margin-bottom: 0.4rem; }
.result-status { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.result-status.ok   { color: #4ADE80; }
.result-status.bad  { color: #F87171; }
.result-status.wait { color: #4A5F80; }
.result-sub { font-size: 0.80rem; color: #4A5F80; }
.conf-row { display:flex; align-items:center; justify-content:space-between; margin:0.8rem 0 0.3rem; }
.conf-label { font-size:11px; color:#6B7FA3; font-family:'Space Mono',monospace; }
.conf-value { font-size:13px; font-weight:600; color:#4ADE80; font-family:'Space Mono',monospace; }
.conf-track { height:4px; background:#1E2D4A; border-radius:2px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:2px; background:linear-gradient(90deg,#22C55E,#4ADE80); }
.meta-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:0.8rem; }
.meta-item { background:#080C14; border:1px solid #1A2540; border-radius:8px; padding:8px 10px; }
.meta-key  { font-size:10px; color:#3D4F6E; font-family:'Space Mono',monospace; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:3px; }
.meta-val  { font-size:12px; font-weight:500; color:#93C5FD; }
div[data-testid="stFileUploader"] > div {
    background: #0D1320 !important; border: 1.5px dashed #1E2D4A !important;
    border-radius: 12px !important; padding: 1.5rem !important;
}
div[data-testid="stFileUploader"] > div:hover { border-color: #3B82F6 !important; }
div[data-testid="stFileUploader"] label { display: none !important; }
button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB, #4F46E5) !important;
    border: none !important; color: white !important; border-radius: 8px !important;
    font-weight: 600 !important; box-shadow: 0 4px 20px rgba(37,99,235,0.3) !important;
}
button[kind="secondary"] {
    background: transparent !important; border: 1px solid #1E2D4A !important;
    color: #6B7FA3 !important; border-radius: 8px !important;
}
div[data-testid="column"] { padding: 0 !important; }
hr { border-color: #1E2D4A !important; margin: 1rem 0 !important; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
.animate-in { animation: fadeIn 0.3s ease forwards; }
</style>
""", unsafe_allow_html=True)


# ── Drawing helpers (Pillow only — no cv2) ────────────────────────────────────

def draw_single_box(pil_img: Image.Image, box, conf: float) -> Image.Image:
    """Draw ONE yellow bounding box using Pillow."""
    out  = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = map(int, box)

    # Yellow box
    for t in range(3):
        draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=(0, 220, 220))

    # Label background + text
    label    = f"PAN Card  {conf:.0%}"
    fontsize = max(16, int((x2 - x1) / 10))
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()

    bbox    = draw.textbbox((0, 0), label, font=font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad     = 6
    draw.rectangle([x1, y1 - th - pad*2, x1 + tw + pad*2, y1], fill=(0, 220, 220))
    draw.text((x1 + pad, y1 - th - pad), label, fill=(0, 0, 0), font=font)
    return out


def draw_not_pan(pil_img: Image.Image) -> Image.Image:
    """Red overlay with 'Not a PAN Card' text using Pillow."""
    out  = pil_img.copy().convert("RGBA")
    w, h = out.size

    # Semi-transparent red overlay
    overlay = Image.new("RGBA", (w, h), (180, 0, 0, 100))
    out     = Image.alpha_composite(out, overlay).convert("RGB")

    draw     = ImageDraw.Draw(out)
    msg      = "Not a PAN Card"
    fontsize = max(20, min(w, h) // 12)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()

    bbox   = draw.textbbox((0, 0), msg, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x      = (w - tw) // 2
    y      = (h - th) // 2

    # Shadow
    draw.text((x+2, y+2), msg, fill=(0, 0, 0), font=font)
    # White text
    draw.text((x, y),     msg, fill=(255, 255, 255), font=font)
    return out


def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def load_model(path: str):
    try:
        from ultralytics import YOLO
        return YOLO(path)
    except Exception:
        return None


# ── Session state ──────────────────────────────────────────────────────────────
for key, val in [("result", None), ("output_img", None),
                 ("total_scans", 0), ("pan_found", 0)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge"><span></span>AI-Powered Document Analysis</div>
  <div class="hero-title">PAN Card Detector</div>
  <div class="hero-sub">Upload any card image — detects Indian PAN cards and rejects all other document types.</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-row">
  <div class="stat-pill">🔍 <strong>{st.session_state.total_scans}</strong>&nbsp;Total Scans</div>
  <div class="stat-pill">✅ <strong>{st.session_state.pan_found}</strong>&nbsp;PAN Cards Found</div>
  <div class="stat-pill">🤖 <strong>YOLOv8n</strong>&nbsp;Model</div>
  <div class="stat-pill">⚡ <strong>&lt;1s</strong>&nbsp;Inference</div>
</div>
""", unsafe_allow_html=True)

# ── Two columns ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="small")

# ── LEFT ──────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">01 — Upload Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Upload",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:0.5rem 0;color:#3D4F6E;font-size:12px;">
            Drag & drop or click to upload · JPG · PNG · WEBP
        </div>""", unsafe_allow_html=True)

    if uploaded:
        pil_img = Image.open(uploaded)
        w, h    = pil_img.size
        size_kb = round(len(uploaded.getvalue()) / 1024, 1)

        st.markdown(f"""
        <div class="meta-grid" style="margin-bottom:0.8rem;">
            <div class="meta-item"><div class="meta-key">File</div><div class="meta-val" style="font-size:10px;word-break:break-all;">{uploaded.name}</div></div>
            <div class="meta-item"><div class="meta-key">Size</div><div class="meta-val">{size_kb} KB</div></div>
            <div class="meta-item"><div class="meta-key">Dimensions</div><div class="meta-val">{w}×{h}</div></div>
            <div class="meta-item"><div class="meta-key">Format</div><div class="meta-val">{uploaded.type.split('/')[-1].upper()}</div></div>
        </div>""", unsafe_allow_html=True)

        st.image(pil_img, width=380, caption="Input image")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="panel-label">02 — Settings</div>', unsafe_allow_html=True)

        weights_path = st.text_input(
            "Model weights path",
            value="runs/detect/pancard_detector6/weights/best.pt",
        )

        conf_thresh = 0.20

        st.markdown("<br>", unsafe_allow_html=True)
        detect_btn = st.button("🔎  Run Detection", type="primary", use_container_width=True)

        if detect_btn:
            # Downscale large images
            max_dim = 1200
            if max(w, h) > max_dim:
                scale   = max_dim / max(w, h)
                pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

            with st.spinner("Analysing…"):
                t_start      = time.time()
                model        = load_model(weights_path)
                pan_detected = False
                top_conf     = 0.0
                out_img      = pil_img.copy()

                if model:
                    img_rgb   = np.array(pil_img.convert("RGB"))
                    img_array = img_rgb[:, :, ::-1]  # RGB → BGR for YOLO)
                    results   = model.predict(img_array, conf=conf_thresh, verbose=False)
                    all_boxes = results[0].boxes
                    pan_dets  = [b for b in all_boxes
                                 if model.names[int(b.cls[0])] == "pancard"]

                    if pan_dets:
                        best_box     = max(pan_dets, key=lambda b: float(b.conf[0]))
                        top_conf     = float(best_box.conf[0])
                        pan_detected = True
                        out_img      = draw_single_box(
                            pil_img, best_box.xyxy[0].cpu().numpy(), top_conf
                        )
                    else:
                        out_img = draw_not_pan(pil_img)
                else:
                    st.error("Model not found — check weights path.")

                elapsed = time.time() - t_start

            st.session_state.total_scans += 1
            if pan_detected:
                st.session_state.pan_found += 1

            st.session_state.result = {
                "detected": pan_detected,
                "conf":     top_conf,
                "elapsed":  elapsed,
                "size":     f"{pil_img.width}×{pil_img.height}",
            }
            st.session_state.output_img = out_img
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ── RIGHT ──────────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="panel panel-right">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">03 — Detection Result</div>', unsafe_allow_html=True)

    r = st.session_state.result

    if r is None:
        st.markdown("""
        <div class="result-card waiting animate-in">
            <div class="result-icon">🪪</div>
            <div class="result-status wait">Awaiting Input</div>
            <div class="result-sub">Upload an image and click Run Detection.</div>
        </div>""", unsafe_allow_html=True)

    else:
        conf_pct = int(r["conf"] * 100)

        if r["detected"]:
            st.markdown(f"""
            <div class="result-card success animate-in">
                <div class="result-icon">✅</div>
                <div class="result-status ok">PAN Card Detected</div>
                <div class="result-sub">A valid Indian PAN card was identified.</div>
                <div class="conf-row">
                    <span class="conf-label">Confidence</span>
                    <span class="conf-value">{conf_pct}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{conf_pct}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card fail animate-in">
                <div class="result-icon">❌</div>
                <div class="result-status bad">Not a PAN Card</div>
                <div class="result-sub">No PAN card found. This is a different document type.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="meta-grid animate-in">
            <div class="meta-item"><div class="meta-key">Inference</div><div class="meta-val">{r['elapsed']*1000:.0f} ms</div></div>
            <div class="meta-item"><div class="meta-key">Confidence</div><div class="meta-val">{conf_pct}%</div></div>
            <div class="meta-item"><div class="meta-key">Image size</div><div class="meta-val">{r['size']}</div></div>
            <div class="meta-item"><div class="meta-key">Model</div><div class="meta-val">YOLOv8n</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="panel-label">04 — Output Image</div>', unsafe_allow_html=True)

        if st.session_state.output_img:
            st.image(st.session_state.output_img, width=380, caption="Detection result")
            st.download_button(
                label="⬇  Download Result",
                data=img_to_bytes(st.session_state.output_img),
                file_name="pan_detection_result.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1E2D4A; padding:1rem 3rem;
     display:flex; justify-content:space-between; background:#0A0F1C; margin-top:1rem;">
  <span style="font-size:11px;color:#2A3D5E;font-family:'Space Mono',monospace;">PAN CARD DETECTOR v1.0</span>
  <span style="font-size:11px;color:#2A3D5E;">YOLOv8 · Pillow · Streamlit</span>
</div>
""", unsafe_allow_html=True)