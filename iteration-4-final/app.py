import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="RoadBuddy",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================
# CSS: phone wrapper using :has(#phone-anchor)
# =========================
st.markdown(
    """
<style>
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

div[data-testid="stAppViewBlockContainer"]{
  padding-top: 0.25rem !important;
  padding-bottom: 0.75rem !important;
}

html, body, [data-testid="stAppViewContainer"] {
  background: #0b0b12 !important;
}

div[data-testid="stVerticalBlock"]:has(#phone-anchor){
  width: 390px;
  max-width: 390px;
  min-height: 760px;
  border-radius: 32px;
  border: 1px solid rgba(255,255,255,0.10);
  background:
    radial-gradient(900px 700px at 15% -10%, rgba(124,58,237,0.22), transparent 55%),
    linear-gradient(180deg, #0b0b12, #0e0e1a);
  box-shadow: 0 24px 80px rgba(0,0,0,0.55);
  padding: 18px 16px 22px 16px;
  position: relative;
  overflow: hidden;
  margin: 10px auto 0 auto;
}

div[data-testid="stVerticalBlock"]:has(#phone-anchor)::before{
  content:"";
  position:absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  width: 120px;
  height: 18px;
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.06);
  z-index: 2;
}

div[data-testid="stVerticalBlock"]:has(#phone-anchor) > div{
  margin-top: 18px;
}

.rb-title{ font-size: 1.55rem; font-weight: 850; color: #EAEAF2; letter-spacing: -0.3px; margin-top: 18px; }
.rb-sub{ color: rgba(234,234,242,0.70); font-size: 0.88rem; margin-top: 4px; margin-bottom: 12px; }

.rb-card{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 14px;
  margin-bottom: 12px;
}
.rb-muted{ color: rgba(234,234,242,0.70); font-size: 0.88rem; }

.stTabs [data-baseweb="tab-list"]{
  display:flex;
  gap: 6px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 6px;
  margin: 6px 0 14px 0;
  width: 100%;
}
.stTabs [data-baseweb="tab"]{
  flex: 1 1 0;
  justify-content: center;
  border-radius: 12px;
  padding: 10px 0;
  font-size: 0.86rem;
  font-weight: 650;
  color: rgba(234,234,242,0.75);
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(167,139,250,0.90)) !important;
  color: #ffffff !important;
  box-shadow: 0 8px 24px rgba(124,58,237,0.35);
}
.stTabs [data-baseweb="tab"] > div > p{ margin:0 !important; }

.stButton>button{
  border-radius: 14px !important;
  padding: 0.85rem 1rem !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: rgba(124,58,237,0.18) !important;
  color: #EAEAF2 !important;
  font-weight: 650 !important;
  width: 100% !important;
}
.stButton>button:hover{
  background: rgba(124,58,237,0.28) !important;
  border-color: rgba(167,139,250,0.35) !important;
}

label, .stCheckbox, .stRadio, .stSelectbox{ color:#EAEAF2 !important; }

[data-testid="stImage"] img{
  border-radius: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Session state
# =========================
def init_state():
    st.session_state.setdefault("cap", None)
    st.session_state.setdefault("monitoring", False)
    st.session_state.setdefault("show_video", True)

    st.session_state.setdefault("status", "Not started")
    st.session_state.setdefault("score", 0.0)

    st.session_state.setdefault("focused_s", 0.0)
    st.session_state.setdefault("distracted_s", 0.0)
    st.session_state.setdefault("last_tick", None)

    st.session_state.setdefault("distracted_streak_s", 0.0)
    st.session_state.setdefault("focused_streak_s", 0.0)
    st.session_state.setdefault("risk_s", 0.0)

    st.session_state.setdefault("prompt_text", "No prompts yet.")
    st.session_state.setdefault("prompt_active", False)
    st.session_state.setdefault("prompt_last_shown_at", 0.0)
    st.session_state.setdefault("snooze_until", 0.0)

    st.session_state.setdefault("last_frame_rgb", None)
    st.session_state.setdefault("last_rerun_ts", 0.0)

init_state()

# =========================
# Helpers
# =========================
def safe_rerun(interval=0.12):
    """Throttle reruns to reduce flicker but keep confidence responsive."""
    now = time.time()
    if now - st.session_state.last_rerun_ts >= interval:
        st.session_state.last_rerun_ts = now
        st.rerun()

def open_camera():
    if st.session_state.cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Close Teams/Zoom and try again.")
            return False

        # Reduce lag/flicker
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        st.session_state.cap = cap
    return True

def close_camera():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.session_state.cap = None

# =========================
# MediaPipe
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1

def infer_focus(frame) -> tuple[str, float]:
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if not result.multi_face_landmarks:
        return "No face", 0.0

    lm = result.multi_face_landmarks[0].landmark
    left_eye = lm[LEFT_EYE]
    right_eye = lm[RIGHT_EYE]
    nose = lm[NOSE_TIP]

    left_eye_pos = np.array([left_eye.x * w, left_eye.y * h])
    right_eye_pos = np.array([right_eye.x * w, right_eye.y * h])
    nose_pos = np.array([nose.x * w, nose.y * h])

    eye_center = (left_eye_pos + right_eye_pos) / 2
    diff_x = float(eye_center[0] - nose_pos[0])

    score = max(0.0, 1.0 - min(abs(diff_x), 80.0) / 80.0)
    status = "Focused" if abs(diff_x) < 30 else "Distracted"
    return status, score

def tick_once():
    if not st.session_state.monitoring:
        return None

    if not open_camera():
        st.session_state.monitoring = False
        return None

    ret, frame = st.session_state.cap.read()
    if not ret:
        return None

    status, score = infer_focus(frame)

    now = time.time()
    dt = 0.0 if st.session_state.last_tick is None else max(0.0, now - st.session_state.last_tick)
    st.session_state.last_tick = now

    # Summary time
    if status == "Focused":
        st.session_state.focused_s += dt
    elif status == "Distracted":
        st.session_state.distracted_s += dt

    # Temporal metrics (kept for explainability)
    if status == "Distracted":
        st.session_state.distracted_streak_s += dt
        st.session_state.focused_streak_s = max(0.0, st.session_state.focused_streak_s - dt)
        st.session_state.risk_s += dt
    elif status == "Focused":
        st.session_state.focused_streak_s += dt
        st.session_state.distracted_streak_s = max(0.0, st.session_state.distracted_streak_s - dt)
        st.session_state.risk_s = max(0.0, st.session_state.risk_s - dt * 1.25)
    else:
        st.session_state.risk_s += dt * 0.25

    st.session_state.status = status
    st.session_state.score = score

    # PROMPTS: show straight away on distraction (unless snoozed)
    if (
        status == "Distracted"
        and (not st.session_state.prompt_active)
        and now >= st.session_state.snooze_until
    ):
        st.session_state.prompt_text = (
            "I’m noticing distraction. If it’s safe, consider a brief pause and reset your focus."
        )
        st.session_state.prompt_active = True
        st.session_state.prompt_last_shown_at = now

    # Don't overwrite an active prompt
    if (not st.session_state.prompt_active) and (now >= st.session_state.snooze_until):
        st.session_state.prompt_text = "No prompt right now."

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame_rgb = frame_rgb
    return frame_rgb

# =========================
# PHONE APP CONTENT
# =========================
st.markdown('<div id="phone-anchor"></div>', unsafe_allow_html=True)

st.markdown('<div class="rb-title">RoadBuddy</div>', unsafe_allow_html=True)
st.markdown('<div class="rb-sub">Proof of concept · camera-based driver awareness</div>', unsafe_allow_html=True)

tab_setup, tab_drive, tab_prompt, tab_summary = st.tabs(["Setup", "Driving", "Prompt", "Summary"])

# ---- Setup ----
with tab_setup:
    st.markdown('<div class="rb-card">', unsafe_allow_html=True)
    st.markdown("**Setup**")
    st.markdown(
        '<div class="rb-muted">Position the camera so your face is visible. Good lighting helps.</div>',
        unsafe_allow_html=True,
    )

    st.session_state.show_video = st.checkbox("Show camera preview", value=st.session_state.show_video)
    preview = st.empty()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Test camera"):
            if open_camera():
                ret, frame = st.session_state.cap.read()
                if ret and st.session_state.show_video:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                st.success("Camera working.")
    with c2:
        if st.button("Start monitoring"):
            if open_camera():
                st.session_state.monitoring = True
                st.session_state.last_tick = time.time()

                st.session_state.focused_s = 0.0
                st.session_state.distracted_s = 0.0
                st.session_state.distracted_streak_s = 0.0
                st.session_state.focused_streak_s = 0.0
                st.session_state.risk_s = 0.0

                st.session_state.snooze_until = 0.0
                st.session_state.prompt_active = False
                st.session_state.prompt_text = "No prompt right now."
                st.session_state.status = "Starting…"

                st.success("Monitoring started. Open the Driving tab.")

    if st.button("Stop camera"):
        st.session_state.monitoring = False
        close_camera()
        st.info("Camera stopped.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Driving ----
with tab_drive:
    st.markdown('<div class="rb-card">', unsafe_allow_html=True)
    st.markdown("**Driving mode**")
    st.markdown(
        '<div class="rb-muted">Monitoring occurs passively with minimal on-screen distraction.</div>',
        unsafe_allow_html=True,
    )

    # IMPORTANT: process End journey BEFORE rerun scheduling
    end_clicked = st.button("End journey")
    if end_clicked:
        st.session_state.monitoring = False
        close_camera()
        st.success("Journey ended. Open the Summary tab.")

    header_row = st.container()
    indicator_row = st.container()
    video_row = st.container()

    with header_row:
        left, right = st.columns(2)
        status_ph = left.empty()
        conf_ph = right.empty()
        status_ph.markdown(f"**Status:** {st.session_state.status}")
        conf_ph.markdown(f"**Confidence:** {int(st.session_state.score * 100)}%")

    with indicator_row:
        indicator_ph = st.empty()
        if st.session_state.monitoring:
            if st.session_state.prompt_active:
                indicator_ph.warning("Safety prompt available (open the Prompt tab).")
            else:
                indicator_ph.caption("No safety prompts.")
        else:
            indicator_ph.info("Monitoring is off. Start it in Setup.")

    with video_row:
        video_box = st.empty()

    if st.session_state.monitoring:
        frame_rgb = tick_once()

        # Update header/indicator with latest values (same run)
        status_ph.markdown(f"**Status:** {st.session_state.status}")
        conf_ph.markdown(f"**Confidence:** {int(st.session_state.score * 100)}%")
        if st.session_state.prompt_active:
            indicator_ph.warning("Safety prompt available (open the Prompt tab).")
        else:
            indicator_ph.caption("No safety prompts.")

        if frame_rgb is None:
            frame_rgb = st.session_state.last_frame_rgb

        if frame_rgb is not None and st.session_state.show_video:
            disp = frame_rgb.copy()
            cv2.putText(
                disp,
                st.session_state.status,
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (180, 120, 255),
                2,
            )
            video_box.image(disp, channels="RGB", use_container_width=True)
        else:
            video_box.caption("Camera preview hidden (enable in Setup).")

        time.sleep(0.03)
        safe_rerun(interval=0.12)
    else:
        if st.session_state.last_frame_rgb is not None and st.session_state.show_video:
            video_box.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
        else:
            video_box.caption("Camera not running.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Prompt ----
with tab_prompt:
    st.markdown('<div class="rb-card">', unsafe_allow_html=True)
    st.markdown("**Safety prompt**")
    st.markdown(f'<div class="rb-muted">{st.session_state.prompt_text}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Dismiss"):
            st.session_state.risk_s = 0.0
            st.session_state.distracted_streak_s = 0.0
            st.session_state.prompt_active = False
            st.session_state.prompt_text = "Dismissed. Carry on."
            st.success("Dismissed.")
    with c2:
        if st.button("Snooze (2 min)"):
            st.session_state.snooze_until = time.time() + 120
            st.session_state.risk_s = 0.0
            st.session_state.distracted_streak_s = 0.0
            st.session_state.prompt_active = False
            st.session_state.prompt_text = "Snoozed. No prompts for 2 minutes."
            st.info("Snoozed for 2 minutes.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Summary ----
with tab_summary:
    st.markdown('<div class="rb-card">', unsafe_allow_html=True)
    st.markdown("**Journey summary**")

    total = max(0.1, st.session_state.focused_s + st.session_state.distracted_s)
    focus_pct = st.session_state.focused_s / total

    st.markdown(f"**Time focused:** {int(focus_pct * 100)}%")
    st.progress(focus_pct)

    st.markdown(f"Focused: **{int(st.session_state.focused_s)}s**")
    st.markdown(f"Distracted: **{int(st.session_state.distracted_s)}s**")

    if focus_pct > 0.8:
        reflection = "Your attention remained largely stable, with only brief moments of distraction."
    elif focus_pct > 0.6:
        reflection = "Your journey showed a balance of focus and distraction, suggesting opportunities for reflection."
    else:
        reflection = "Sustained distraction was observed. Reviewing conditions and habits may be helpful."

    st.markdown(f'<div class="rb-muted">{reflection}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
