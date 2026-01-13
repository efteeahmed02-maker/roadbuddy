import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# =========================
# Page config (Iteration 3: simple UI, no Summary tab, no phone wrapper)
# =========================
st.set_page_config(
    page_title="RoadBuddy",
    layout="centered",
    initial_sidebar_state="collapsed",
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

    st.session_state.setdefault("last_tick", None)
    st.session_state.setdefault("risk_s", 0.0)

    st.session_state.setdefault("prompt_text", "No prompts yet.")
    st.session_state.setdefault("prompt_active", False)
    st.session_state.setdefault("snooze_until", 0.0)

    st.session_state.setdefault("last_frame_rgb", None)
    st.session_state.setdefault("last_rerun_ts", 0.0)

    # Performance helpers
    st.session_state.setdefault("frame_i", 0)

init_state()

# =========================
# Helpers
# =========================
def safe_rerun(interval=0.15):
    """Throttle reruns to reduce flicker but keep status/confidence responsive."""
    now = time.time()
    if now - st.session_state.last_rerun_ts >= interval:
        st.session_state.last_rerun_ts = now
        st.rerun()

def open_camera():
    if st.session_state.cap is None:
        # Windows: CAP_DSHOW often reduces latency
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Could not open webcam. Close Teams/Zoom and try again.")
            return False

        # Reduce lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
    """Explainable rule: eye-centre vs nose horizontal deviation."""
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
    """Iteration 3 behaviour: prompt appears immediately on distraction (unless snoozed)."""
    if not st.session_state.monitoring:
        return None

    if not open_camera():
        st.session_state.monitoring = False
        return None

    # Drop buffered frames to keep feed close to real-time
    for _ in range(1):
        st.session_state.cap.grab()

    ret, frame = st.session_state.cap.read()
    if not ret:
        return None

    now = time.time()
    dt = 0.0 if st.session_state.last_tick is None else max(0.0, now - st.session_state.last_tick)
    st.session_state.last_tick = now

    # Run MediaPipe every N frames to reduce CPU load
    st.session_state.frame_i += 1
    RUN_EVERY_N = 3  # increase to 3 if still laggy

    if st.session_state.frame_i % RUN_EVERY_N == 0:
        status, score = infer_focus(frame)
        st.session_state.status = status
        st.session_state.score = score

        # Simple risk counter (for explainability)
        if status == "Distracted":
            st.session_state.risk_s += dt
        elif status == "Focused":
            st.session_state.risk_s = max(0.0, st.session_state.risk_s - dt)

        # Prompt immediately when distracted (persist until dismissed/snoozed/end journey)
        if (
            status == "Distracted"
            and (not st.session_state.prompt_active)
            and now >= st.session_state.snooze_until
        ):
            st.session_state.prompt_text = (
                "I’m noticing distraction. If it’s safe, consider refocusing."
            )
            st.session_state.prompt_active = True

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame_rgb = frame_rgb
    return frame_rgb

# =========================
# UI (Iteration 3: Setup / Driving / Prompt)
# =========================
st.title("RoadBuddy")

tab_setup, tab_drive, tab_prompt = st.tabs(["Setup", "Driving", "Prompt"])

# ---- Setup ----
with tab_setup:
    st.markdown("Use the camera to monitor attention. Prompts appear when distraction is detected.")

    st.session_state.show_video = st.checkbox("Show camera preview", value=st.session_state.show_video)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Test camera"):
            if open_camera():
                # Drop a frame or two before showing preview
                for _ in range(2):
                    st.session_state.cap.grab()
                ret, frame = st.session_state.cap.read()
                if ret and st.session_state.show_video:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                st.success("Camera working.")
    with c2:
        if st.button("Start monitoring"):
            if open_camera():
                st.session_state.monitoring = True
                st.session_state.last_tick = time.time()
                st.session_state.risk_s = 0.0
                st.session_state.frame_i = 0

                st.session_state.prompt_active = False
                st.session_state.prompt_text = "No prompt right now."
                st.session_state.snooze_until = 0.0

                st.session_state.status = "Starting…"
                st.session_state.score = 0.0
                st.success("Monitoring started. Open the Driving tab.")

    if st.button("Stop camera"):
        st.session_state.monitoring = False
        close_camera()
        st.info("Camera stopped.")

# ---- Driving ----
with tab_drive:
    # End journey MUST be handled before rerun scheduling
    end_clicked = st.button("End journey")
    if end_clicked:
        st.session_state.monitoring = False
        close_camera()
        st.success("Journey ended.")
        st.stop()

    # Single placeholders to prevent “second video feed”
    status_ph = st.empty()
    conf_ph = st.empty()
    indicator_ph = st.empty()
    video_ph = st.empty()

    status_ph.markdown(f"**Status:** {st.session_state.status}")
    conf_ph.markdown(f"**Confidence:** {int(st.session_state.score * 100)}%")

    if st.session_state.monitoring:
        frame_rgb = tick_once()

        status_ph.markdown(f"**Status:** {st.session_state.status}")
        conf_ph.markdown(f"**Confidence:** {int(st.session_state.score * 100)}%")

        if st.session_state.prompt_active:
            indicator_ph.warning("Safety prompt available (open the Prompt tab).")
        else:
            indicator_ph.caption("No safety prompts.")

        if frame_rgb is None:
            frame_rgb = st.session_state.last_frame_rgb

        if frame_rgb is not None and st.session_state.show_video:
            video_ph.image(frame_rgb, channels="RGB", use_container_width=True)
        else:
            video_ph.caption("Camera preview hidden (enable in Setup).")

        time.sleep(0.05)
        safe_rerun(interval=0.18)
    else:
        indicator_ph.info("Monitoring is off. Start it in Setup.")
        if st.session_state.last_frame_rgb is not None and st.session_state.show_video:
            video_ph.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
        else:
            video_ph.caption("Camera not running.")

# ---- Prompt ----
with tab_prompt:
    st.markdown("**Safety prompt**")

    if st.session_state.prompt_active:
        st.warning(st.session_state.prompt_text)
    else:
        st.info(st.session_state.prompt_text)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Dismiss"):
            st.session_state.prompt_active = False
            st.session_state.prompt_text = "Dismissed. Carry on."
            st.session_state.risk_s = 0.0
            st.success("Dismissed.")
    with c2:
        if st.button("Snooze (2 min)"):
            st.session_state.snooze_until = time.time() + 120
            st.session_state.prompt_active = False
            st.session_state.prompt_text = "Snoozed. No prompts for 2 minutes."
            st.session_state.risk_s = 0.0
            st.info("Snoozed for 2 minutes.")
