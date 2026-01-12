import time
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.set_page_config(page_title="Driver Focus Prototype", layout="centered")
st.title("Driver Focus Prototype")

run = st.toggle("Run camera", value=False)
show_video = st.checkbox("Show video", value=True)

status_box = st.empty()
frame_box = st.empty()

# ---------- YOUR ORIGINAL SETUP ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1
# ---------------------------------------

def process_frame(frame):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    status = "Distracted"

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]

        left_eye = face_landmarks.landmark[LEFT_EYE]
        right_eye = face_landmarks.landmark[RIGHT_EYE]
        nose = face_landmarks.landmark[NOSE_TIP]

        left_eye_pos = np.array([left_eye.x * w, left_eye.y * h])
        right_eye_pos = np.array([right_eye.x * w, right_eye.y * h])
        nose_pos = np.array([nose.x * w, nose.y * h])

        eye_center = (left_eye_pos + right_eye_pos) / 2
        diff_x = eye_center[0] - nose_pos[0]

        if abs(diff_x) < 30:
            status = "Focused"

    cv2.putText(
        frame,
        f"Status: {status}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return status, frame


if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam")
        st.stop()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        status, frame = process_frame(frame)

        if status == "Focused":
            status_box.success("Focused")
        else:
            status_box.error("Distracted")

        if show_video:
            frame_box.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB"
            )

        time.sleep(0.03)
else:
    status_box.info("Toggle 'Run camera' to start")
