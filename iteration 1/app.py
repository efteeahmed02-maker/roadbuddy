import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

st.set_page_config(page_title="RoadBuddy", layout="centered")

# -------------------------
# Session state
# -------------------------
def init_state():
    st.session_state.setdefault("cap", None)
    st.session_state.setdefault("monitoring", False)
    st.session_state.setdefault("status", "Not started")
    st.session_state.setdefault("score", 0.0)
    st.session_state.setdefault("last_tick", None)

init_state()

# -------------------------
# Camera helpers
# -------------------------
def open_camera():
    if st.session_state.cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            return False
        st.session_state.cap = cap
    return True

def close_camera():
    if st.session_state.cap:
        st.session_state.cap.release()
    st.session_state.cap = None

# -------------------------
# MediaPipe
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

LEFT_EYE, RIGHT_EYE, NOSE_TIP = 33, 263, 1

def infer_focus(frame):
    h, w, _ = frame.shape
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return "No face", 0.0

    lm = result.multi_face_landmarks[0].landmark
    eye_center = np.array([
        (lm[LEFT_EYE].x + lm[RIGHT_EYE].x) * w / 2,
        (lm[LEFT_EYE].y + lm[RIGHT_EYE].y) * h / 2,
    ])
    nose = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])

    diff_x = eye_center[0] - nose[0]
    score = max(0.0, 1.0 - min(abs(diff_x), 80) / 80)
    status = "Focused" if abs(diff_x) < 30 else "Distracted"
    return status, score

def tick_once():
    if not st.session_state.monitoring:
        return None

    open_camera()
    ret, frame = st.session_state.cap.read()
    if not ret:
        return None

    status, score = infer_focus(frame)
    st.session_state.status = status
    st.session_state.score = score
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# -------------------------
# UI
# -------------------------
st.title("RoadBuddy")

tab_setup, tab_drive = st.tabs(["Setup", "Driving"])

with tab_setup:
    if st.button("Start monitoring"):
        st.session_state.monitoring = True
        st.success("Monitoring started.")

with tab_drive:
    st.markdown(f"**Status:** {st.session_state.status}")
    st.markdown(f"**Confidence:** {int(st.session_state.score * 100)}%")

    frame = tick_once()
    if frame is not None:
        st.image(frame)

    if st.button("End journey"):
        st.session_state.monitoring = False
        close_camera()
        st.success("Journey ended.")
