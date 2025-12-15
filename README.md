# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tempfile

# ================== SETTINGS ==================
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance_web.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Load or create attendance CSV
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Load known faces
def load_known_faces():
    known_encodings = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            try:
                encoding = face_recognition.face_encodings(image)[0]
                name = os.path.splitext(filename)[0].replace("_", " ").title()
                known_encodings.append(encoding)
                known_names.append(name)
            except:
                st.warning(f"No face found in {filename}")
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# Mark attendance
def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    if ((df["Name"] == name) & (df["Date"] == today)).any():
        return False  # Already marked
    new_row = pd.DataFrame([{
        "Name": name,
        "Date": today,
        "Time": datetime.now().strftime("%H:%M:%S")
    }])
    new_row.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)
    return True

# Process image for recognition
def recognize_faces(image):
    rgb_image = image[:, :, ::-1]  # BGR to RGB
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    results = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            if mark_attendance(name):
                results.append(f"✓ {name} - Attendance Marked!")
            else:
                results.append(f"✓ {name} - Already Marked Today")
        else:
            results.append("✗ Unknown Person")
    
    # Draw boxes on image
    for (top, right, bottom, left), name in zip(face_locations, [known_names[matches.index(True)] if True in matches else "Unknown" for matches in [face_recognition.compare_faces(known_encodings, enc) for enc in face_encodings]]):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    return image, results, len(face_locations)

# Streamlit UI
st.set_page_config(page_title="AI Attendance System", layout="wide")
st.title("🧑‍🤝‍🧑 AI Face Recognition Attendance System")

tab1, tab2, tab3 = st.tabs(["Register New Person", "Mark Attendance", "View Attendance"])

with tab1:
    st.header("Register a New Person")
    name = st.text_input("Enter Person's Name")
    uploaded_files = st.file_uploader("Upload clear face photos (multiple allowed)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    if st.button("Register") and name and uploaded_files:
        for file in uploaded_files:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            image = face_recognition.load_image_file(tfile.name)
            if len(face_recognition.face_encodings(image)) == 0:
                st.error(f"No face detected in {file.name}")
            else:
                save_path = os.path.join(KNOWN_FACES_DIR, f"{name.replace(' ', '_')}.{file.name.split('.')[-1]}")
                cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        st.success(f"{name} registered successfully!")
        # Reload known faces
        known_encodings, known_names = load_known_faces()

with tab2:
    st.header("Mark Attendance")
    option = st.radio("Choose input method", ["Upload Group Photo", "Use Webcam (Real-time)"])
    
    if option == "Upload Group Photo":
        uploaded = st.file_uploader("Upload a photo with people", type=['jpg', 'png', 'jpeg'])
        if uploaded and st.button("Process Photo"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            image = cv2.imdecode(np.frombuffer(tfile.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_image, results, count = recognize_faces(image)
            st.image(processed_image, channels="BGR", caption=f"{count} People Detected")
            for r in results:
                st.write(r)
    
    else:  # Webcam
        st.write("Click 'Start Camera' → Look at camera → Attendance auto-marked!")
        if st.button("Start Camera"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed, results, count = recognize_faces(frame.copy())
                stframe.image(processed, channels="BGR")
                if results:
                    st.write("Detected:")
                    for r in results:
                        st.write(r)
            cap.release()

with tab3:
    st.header("Attendance Records")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        if not df.empty:
            st.dataframe(df.sort_values(by=["Date", "Time"], ascending=False))
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "attendance_records.csv", "text/csv")
        else:
            st.info("No attendance recorded yet.")
    else:
        st.info("No attendance file found.")

st.caption("Built with ❤️ using Streamlit + face_recognition")