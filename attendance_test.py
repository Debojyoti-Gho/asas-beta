import streamlit as st
st.set_page_config( 
    page_title="ASAS 2.0",  
    page_icon="🎓"   
) 
import sqlite3
import datetime
from datetime import date, datetime,timedelta,timezone
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet 
from reportlab.platypus import Image as PlatypusImage, Paragraph
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from io import BytesIO 
import matplotlib.pyplot as plt
import uuid
import requests
import numpy as np 
import time
import logging
import smtplib
from email.mime.text import MIMEText 
from email.mime.multipart import MIMEMultipart
from typing import Optional
from streamlit_cookies_manager import EncryptedCookieManager
from PIL import Image
import io
import streamlit.components.v1 as components
import json
import cv2
import mediapipe as mp
from scipy.spatial.distance import cosine
from deepface import DeepFace
import tempfile
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import re
from streamlit_lottie import st_lottie

def show_intro():
    if "intro_shown" not in st.session_state:
        st.session_state.intro_shown = False

    if not st.session_state.intro_shown:
        st.markdown(f"""
            <style>
                body {{
                    overflow: hidden;
                    margin: 0;
                }}

                .intro-container {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    z-index: 9999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    font-family: 'Bebas Neue', sans-serif;
                    padding: 0 20px;
                }}

                video.background-video {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    z-index: -1;
                }}

                .intro-text {{
                    font-size: 6em;
                    background: linear-gradient(45deg, #feda75, #fa7e1e, #d62976, #962fbf, #4f5bd5);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    animation: zoomFade 1.6s ease-in-out;
                    letter-spacing: 2px;
                    text-align: center;
                    max-width: 100%;
                    word-wrap: break-word;
                }}

                @keyframes zoomFade {{
                    0% {{ transform: scale(0.8); opacity: 0; }}
                    50% {{ transform: scale(1.05); opacity: 1; }}
                    100% {{ transform: scale(1); opacity: 1; }}
                }}

                .subtitle {{
                    font-size: 1.5em;
                    color: #ffffff;
                    margin-top: 10px;
                    opacity: 0.95;
                    text-align: center;
                }}

                .spinner {{
                    border: 4px solid rgba(255, 255, 255, 0.3);
                    border-top: 4px solid white;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin-top: 30px;
                }}

                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}

                @keyframes fadeOut {{
                    0% {{ opacity: 1; visibility: visible; }}
                    100% {{ opacity: 0; visibility: hidden; display: none; }}
                }}

                .fade-out {{
                    animation: fadeOut 1s ease forwards;
                    animation-delay: 6s;
                }}

                @keyframes flashWhite {{
                    0% {{ background: transparent; }}
                    100% {{ background: white; }}
                }}

                .white-flash {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    z-index: 10000;
                    animation: flashWhite 0.4s ease-in forwards;
                    animation-delay: 6.3s;
                }}

                @media (max-width: 768px) {{
                    .intro-text {{
                        font-size: 3em;
                    }}
                    .subtitle {{
                        font-size: 1.2em;
                    }}
                    .spinner {{
                        width: 30px;
                        height: 30px;
                        border-width: 3px;
                    }}
                }}
            </style>

            <!-- Font -->
            <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">

            <!-- Audio (separate background music) -->
            <audio id="bg-audio" autoplay>
                <source src="cinematic-intro-3-40041.mp3" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>

            <script>
                const audio = document.getElementById("bg-audio");
                document.addEventListener("click", () => {{
                    audio.play().catch(e => console.log("Autoplay blocked"));
                }});
            </script>

            <!-- Intro Container -->
            <div class="intro-container fade-out">
                <video class="background-video" autoplay muted playsinline>
                    <source src="https://cdnl.iconscout.com/lottie/premium/thumb/background-color-animation-download-in-lottie-json-gif-static-svg-file-formats--gradient-pack-patterns-animations-4330097.mp4" type="video/mp4">
                </video>
                <div class="intro-text">ASAS 2.0</div>
                <div class="subtitle">Advanced Student Attendance System</div>
                <div class="spinner"></div>
            </div>
        """, unsafe_allow_html=True)

        time.sleep(5)
        st.session_state.intro_shown = True
        st.rerun()

show_intro()



# Database setup
conn = sqlite3.connect("asasspecial.db", check_same_thread=False) 
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    user_id TEXT PRIMARY KEY,
    password TEXT,
    name TEXT,
    roll TEXT,
    section TEXT,
    email TEXT,
    enrollment_no TEXT,
    year TEXT,
    semester TEXT,
    device_id TEXT,
    student_face BLOB 
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS admin (
    admin_id TEXT PRIMARY KEY,
    password TEXT
    active INTEGER DEFAULT 1
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    student_id TEXT,
    date TEXT,
    day TEXT,
    
    period_1 INTEGER,
    period_2 INTEGER,
    period_3 INTEGER,
    period_4 INTEGER,
    period_5 INTEGER,
    period_6 INTEGER,
    period_7 INTEGER,
    PRIMARY KEY (student_id, date, day)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS admin_profile (
    admin_id TEXT PRIMARY KEY,
    name TEXT,
    department TEXT,
    designation TEXT,
    email TEXT,
    phone TEXT,
    face_encoding BLOB
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS semester_dates (
    year INTEGER,
    semester INTEGER,
    start_date DATE,
    end_date DATE,
    total_holidays INTEGER,
    total_classes INTEGER,
    total_periods INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS disputes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    date TEXT,
    reason TEXT,
    status TEXT DEFAULT 'Pending'
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS timetable (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL, -- e.g., Monday, Tuesday
    period TEXT NOT NULL, -- e.g., Period 1, Period 2
    subject TEXT NOT NULL,
    teacher TEXT NOT NULL
)              
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS admin_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_id TEXT NOT NULL,
    action TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (admin_id) REFERENCES admin (admin_id)
);
""")

# Commit the changes
conn.commit()

# Add default admin credentials
cursor.execute("INSERT OR IGNORE INTO admin (admin_id, password) VALUES ('admin', 'admin123')")
conn.commit()

# Register adapters and converters for SQLite
def adapt_date(d):
    return d.isoformat()

def adapt_datetime(dt):
    return dt.isoformat()

def convert_date(s):
    return date.fromisoformat(s.decode("utf-8"))

def convert_datetime(s):
    return datetime.fromisoformat(s.decode("utf-8"))

sqlite3.register_adapter(date, adapt_date)
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("DATE", convert_date)
sqlite3.register_converter("DATETIME", convert_datetime)

# Helper functions

def send_confirmation_email(admin_email, admin_name):
    # SMTP Configuration
    smtp_server = 'smtp-relay.brevo.com'
    smtp_port = 587
    smtp_user = '823c6b001@smtp-brevo.com'  # Replace with your SMTP username
    smtp_password = '6tOJHT2F4x8ZGmMw'      # Replace with your SMTP password
    from_email = 'debojyotighoshmain@gmail.com'  # Replace with your sending email address
    
    # Create the email content
    subject = "Welcome to the Admin Portal!"

    body = f"""
    Dear {admin_name},

    On behalf of the entire team, I am pleased to welcome you to the Admin Portal! We are thrilled to have you join our growing organization, where your role will be crucial in shaping our success.

    Your profile has been successfully created, and we would like to provide you with some important details about your account, features available to you, and your next steps:

    **Your Account Details:**
    - **Admin ID:** {new_admin_id}
    - **Name:** {new_name}
    - **Department:** {new_department}
    - **Designation:** {new_designation}
    - **Email:** {new_email}
    - **Phone:** {new_phone}

    **Features You Can Explore:**
    - **Profile Management:** Edit and manage your personal and professional details at any time.
    - **Data Analysis:** Gain insights from data reports and analytics dashboards.
    - **Report Generation:** Generate and share custom reports with other users.
    - **User Permissions:** Manage access rights and permissions for different users in the system.

    We believe that you will play a pivotal role in enhancing the experience for both administrators and users. Please take some time to familiarize yourself with the system, and if you have any questions or need assistance, do not hesitate to reach out to our support team.

    **Getting Started:**
    - Log in to the system using the credentials provided.
    - Begin by exploring the "Dashboard" for a high-level overview of key performance indicators and system notifications.
    - Customize your profile and update your contact preferences to stay informed about important updates.

    We are excited to have you as part of our organization, and we are confident that your expertise and leadership will help drive success. We look forward to working together and supporting you in your journey with us.

    Once again, welcome aboard!

    Best regards,
    The Admin Team
    """

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = admin_email
    msg['Subject'] = subject

    # Attach the body to the email
    msg.attach(MIMEText(body, 'plain'))

    # Establish a connection to the SMTP server
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade to secure connection
        server.login(smtp_user, smtp_password)
        server.sendmail(from_email, admin_email, msg.as_string())
        server.quit()  # Close the connection
        st.success("Email sent successfully.")
    except Exception as e:
        st.error(f"Failed to send email. Error: {str(e)}")
        
# SMTP Configuration
smtp_server = 'smtp-relay.brevo.com'
smtp_port = 587
smtp_user = '823c6b001@smtp-brevo.com'  # Replace with your SMTP username
smtp_password = '6tOJHT2F4x8ZGmMw'      # Replace with your SMTP password
from_email = 'debojyotighoshmain@gmail.com'  # Replace with your sending email address
    
# Function to calculate attendance percentage for each student
def get_low_attendance_students(threshold=75):
    # Fetch all students
    cursor.execute("SELECT user_id, name, email FROM students")
    all_students = cursor.fetchall()

    low_attendance = []

    for student in all_students:
        user_id, name, email = student

        # Fetch the student's year and semester
        cursor.execute("""
            SELECT year, semester FROM students WHERE user_id = ?
        """, (user_id,))
        student_semester = cursor.fetchone()

        if student_semester:
            year, semester = student_semester

            # Fetch total_classes and total_periods from semester_dates table
            cursor.execute("""
                SELECT total_classes, total_periods
                FROM semester_dates 
                WHERE year = ? AND semester = ?
            """, (year, semester))
            semester_details = cursor.fetchone()

            if semester_details:
                total_classes_in_semester, total_periods_in_semester = semester_details

                # Fetch the student's attendance records
                cursor.execute("SELECT * FROM attendance WHERE student_id = ?", (user_id,))
                attendance_records = cursor.fetchall()

                if attendance_records:
                    # Calculate total periods attended
                    total_present = sum(
                    sum(1 if record[i] == 1 else 0 for i in range(3, 10))  # Sum the periods attended (1 for present, 0 for absent)
                    for record in attendance_records
                )


                    # Calculate attendance percentage
                    if total_periods_in_semester > 0:  # Avoid division by zero
                        attendance_percentage = (total_present / total_periods_in_semester) * 100
                        
                        # Add to low attendance list if below the threshold
                        if attendance_percentage < threshold:
                            low_attendance.append((user_id, name, email, attendance_percentage))

    return low_attendance


def send_email(student_email, student_name, attendance_percentage):

    # Email Content
    subject = f"Attendance Alert - Immediate Attention Required"
    body = f"""
    Dear {student_name},

    This is to inform you that your attendance is currently at {attendance_percentage:.2f}%, 
    which is below the minimum required threshold of 75%.

    You are hereby advised to attend more classes and meet the required attendance criteria 
    to avoid being debarred from examinations or other academic activities.

    Regards,
    Admin Team
    Your Institution Name
    """

    try:
        # Create Email Message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = student_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connect to SMTP Server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Start TLS Encryption
            server.login(smtp_user, smtp_password)  # Login with SMTP credentials
            server.sendmail(from_email, student_email, msg.as_string())  # Send email

        st.success(f"Email successfully sent to {student_name} ({student_email}).")
    except smtplib.SMTPAuthenticationError:
        st.error(f"Failed to authenticate with SMTP server. Check your username and password.")
    except Exception as e:
        st.error(f"Failed to send email to {student_name} ({student_email}). Error: {str(e)}")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# def capture_and_detect_faces():
#     cap = cv2.VideoCapture(0)  # Open webcam
#     if not cap.isOpened():
#         st.error("Cannot access the webcam.")
#         return None, None

#     face_encodings = []
#     face_locations = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture a frame from the webcam.")
#             break

#         # Convert the frame from BGR to RGB for face_recognition
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect faces in the frame
#         face_locations = face_recognition.face_locations(rgb_frame)
#         for top, right, bottom, left in face_locations:
#             # Draw rectangles around detected faces
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

#         # Convert frame to JPEG for Streamlit display
#         _, jpeg_frame = cv2.imencode('.jpg', frame)
#         st.image(jpeg_frame.tobytes(), channels="BGR", use_column_width=True, caption="Real-Time Face Detection")

#         # Stop the loop if a face is detected
#         if face_locations:
#             try:
#                 face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#                 logging.info(f"Detected {len(face_encodings)} face(s).")
#                 break  # Stop after capturing faces
#             except Exception as e:
#                 logging.error(f"Error generating face encodings: {e}")
#                 break

#         # Add a "Stop" button to break the loop manually
#         if st.button("Stop Capture"):
#             break

#     cap.release()
#     return face_encodings, face_locations


# def cluster_faces(face_encodings, eps=0.6, min_samples=2):
#     """
#     Cluster detected faces into groups of distinct individuals using DBSCAN.
#     """
#     if not face_encodings:
#         logging.warning("No face encodings provided for clustering.")
#         return 0, []

#     db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
#     cluster_labels = db.fit_predict(face_encodings)

#     num_people = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # -1 is noise
#     logging.info(f"Identified {num_people} distinct individual(s).")
#     return num_people, cluster_labels


# def create_kdtree(database_encodings):
#     # Explicitly check for an empty list
#     if len(database_encodings) == 0:
#         logging.error("Empty database encodings provided. Cannot create KD-Tree.")
#         return None

#     try:
#         return KDTree(database_encodings)
#     except Exception as e:
#         logging.error(f"Error creating KD-Tree: {e}")
#         return None


# def match_faces_with_database(captured_faces):
#     """
#     Match captured face encodings with entries in the database using a KD-Tree.
#     """
#     # Retrieve all face encodings and user details from the database
#     cursor.execute("SELECT user_id, name, student_face FROM students")
#     students = cursor.fetchall()

#     if not students:
#         logging.warning("No students found in the database.")
#         return []

#     database_encodings = []
#     user_info = []

#     for student in students:
#         user_id, name, stored_face_encoding = student
#         if stored_face_encoding:
#             try:
#                 # Log the raw stored encoding data
#                 logging.debug(f"Processing face encoding for {name}, user_id: {user_id}")
                
#                 # Deserialize face encoding from BLOB to numpy array
#                 stored_encoding = np.frombuffer(stored_face_encoding, dtype=np.float64)
                
#                 # Verify the shape and size of the encoding
#                 if stored_encoding.size != 128:  # Expected size of face encoding
#                     logging.warning(f"Face encoding for {name} seems to be an invalid size: {stored_encoding.size}")
#                     continue

#                 database_encodings.append(stored_encoding)
#                 user_info.append((user_id, name))
#             except Exception as e:
#                 logging.error(f"Error processing face encoding for {name}: {e}")

#     # Log the number of valid encodings retrieved
#     logging.info(f"Retrieved {len(database_encodings)} valid face encodings from the database.")

#     if not database_encodings:
#         logging.warning("No valid face encodings found in the database.")
#         return []

#     # Build a KD-Tree for efficient matching
#     kdtree = create_kdtree(np.array(database_encodings))

#     matched_students = []
#     for captured_face in captured_faces:
#         # Find the nearest neighbor in the database
#         dist, index = kdtree.query([captured_face], k=1)
#         if dist[0][0] <= 0.6:  # Match threshold
#             matched_students.append(user_info[index[0][0]])
#             logging.info(f"Match found: {user_info[index[0][0]]}")

#     return matched_students


# # Load the pre-trained MiDaS model for depth estimation on appropriate device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# model.eval().to(device)
# def capture_face():
#     st.write("Automatically turnig on your camera to capture the face.")
#     cap = cv2.VideoCapture(0)  # Open the camera
    
#     # Check if the camera is successfully opened
#     if not cap.isOpened():
#         st.error("Could not open camera.")
#         return None

#     st_frame = st.image([])

#     captured_face = None  # Variable to store the captured face

#     while True:
#         ret, frame = cap.read()
        
#         # Check if frame was successfully captured
#         if not ret or frame is None or frame.size == 0:
#             st.error("Failed to capture frame from the camera.")
#             break
        
#         st_frame.image(frame, channels="BGR")

#         # Convert to RGB for face recognition
#         rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_image)
        
#         # If at least one face is detected, capture the first one
#         if len(face_locations) > 0:
#             captured_face = frame
#             st.write("Face captured successfully!")
#             break  # Exit the loop once a face is captured

#     cap.release()  # Release the camera

#     if captured_face is None:
#         st.error("No valid face detected. Try again.")
#         return None

#     # Check if captured face is valid before proceeding with depth estimation
#     if captured_face is None or captured_face.size == 0:
#         st.error("Captured face is empty or invalid.")
#         return None

#     # Depth estimation: Convert the captured frame to RGB for MiDaS model
#     rgb_frame = cv2.cvtColor(captured_face, cv2.COLOR_BGR2RGB)

#     # Define the transformation (resize and normalization)
#     transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(384),  # Ensure it resizes to 384x384
#     transforms.CenterCrop(384),  # Ensure the size matches the expected dimensions
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

#     # Apply the transformations
#     input_tensor = transform(rgb_frame).unsqueeze(0)

#     # Predict the depth
#     with torch.no_grad():
#         depth_map = model(input_tensor)

#     # Normalize the depth map for display
#     depth_map = depth_map.squeeze().cpu().numpy()
#     depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
#     depth_map = np.uint8(depth_map)

#     # Show depth map for debugging (optional)
#     st.image(depth_map, channels="GRAY", caption="Depth Map")

#     # Check if the depth map has sufficient variation (meaningful 3D object)
#     depth_variation = np.std(depth_map)

#     # Enhance the check to account for larger regions
#     mean_depth = np.mean(depth_map)
#     depth_threshold = 80  # Adjust threshold for a more reliable check

#     # Check if depth variation and mean depth are consistent with 3D face
#     if depth_variation < depth_threshold or mean_depth < 20:
#         st.error("Depth variation too low or invalid depth detected! This might be a 2D image.")
#         captured_face = None
#         return None

#     return captured_face  # Return the captured frame if everything is valid

# # Preprocess face image to get encoding
# def get_face_encoding(image):
#     # Check if the image is None or empty
#     if image is None or image.size == 0:
#         st.error("Failed to capture face image. The image is empty.")
#         return None
    
#     # Convert the image to RGB (required by face_recognition)
#     try:
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     except cv2.error as e:
#         st.error(f"OpenCV error during color conversion: {e}")
#         return None
    
#     # Detect face locations
#     face_locations = face_recognition.face_locations(rgb_image)
    
#     if len(face_locations) == 0:
#         st.error("No face detected. Try again!")
#         return None
    
#     # Get face encodings for the detected faces
#     face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
#     if len(face_encodings) > 0:
#         return face_encodings[0]  # Return the encoding of the first detected face
#     else:
#         st.error("No face encoding found.")
#         return None
    
# def authenticate_with_face(captured_encoding, stored_encoding, threshold=0.6):
#     # Calculate the Euclidean distance between the two encodings
#     distance = euclidean(captured_encoding, stored_encoding)

#     # Log the distance for debugging purposes
#     st.write(f"Distance between captured and stored encoding: {distance}")

#     # Compare the distance with a threshold
#     if distance < threshold:
#         return True
#     return False

# Function to load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 📌 Load Haar Cascade model for face detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 📌 Define class periods for attendance marking
PERIOD_TIMES = {
    "Period 1": ("08:00", "09:00"),
    "Period 2": ("09:15", "10:15"),
    "Period 3": ("10:30", "11:30"),
    "Period 4": ("11:45", "12:45"),
    "Period 5": ("14:00", "15:00"),
    "Period 6": ("15:15", "16:15"),
    "Period 7": ("16:30", "17:30"),
}

# 📌 Function to detect faces using Haar Cascade
def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return [image[y:y+h, x:x+w] for (x, y, w, h) in faces]

# 📌 Function to capture and detect faces
def capture_and_detect_faces():
    img_file = st.camera_input("📸 Capture an Image")

    if img_file:
        nparr = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detected_faces = detect_faces_haar(frame)

        return detected_faces if detected_faces else None
    return None

# 📌 Function to determine the current class period
def get_current_period():
    now = datetime.now().strftime("%H:%M")
    for period, (start, end) in PERIOD_TIMES.items():
        if start <= now <= end:
            return period
    return None

def match_faces_with_db():
    if "detected_faces" not in st.session_state or not st.session_state.detected_faces:
        st.error("❌ No faces available for matching. Please capture an image first.")
        return []
        
    verified_students = []

    cursor.execute("SELECT user_id, name, student_face FROM students")
    students_data = cursor.fetchall()

    for face in st.session_state.detected_faces:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, face)
            face_path = temp_file.name

        for user_id, name, stored_face_blob in students_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_stored:
                temp_stored.write(stored_face_blob)
                stored_face_path = temp_stored.name
            try:
                result = DeepFace.verify(face_path, stored_face_path, model_name="Facenet", distance_metric="cosine")
                if result["verified"] and result["distance"] < 0.7:
                    verified_students.append((user_id, name))
                    break
            except Exception:
                continue

    # ✅ Display recognized students in Streamlit UI
    if verified_students:
        st.success(f"✅ Recognized {len(verified_students)} student(s):")
        for user_id, name in verified_students:
            st.write(f"📌 **User ID:** `{user_id}` | **Name:** {name}")
    else:
        st.warning("⚠ No matching students found. Try again with a clearer image.")

    return verified_students

    

def record_attendance_for_batch(student_data):
    if not student_data:
        st.warning("⚠ No recognized faces. Attendance not marked.")
        return

    current_period = get_current_period()
    if not current_period:
        st.warning("No active class period at the moment.")
        return

    current_day = datetime.now().strftime("%A")
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    st.success(f"Attendance for {current_period} is being marked!")
    
    try:
        cursor.execute("""
            SELECT subject, teacher FROM timetable 
            WHERE day = ? AND period = ?
        """, (current_day, current_period))
        period_details = cursor.fetchone()

        if period_details:
            subject, teacher = period_details
            st.info(f"Subject: {subject} | Teacher: {teacher}")
            
            period_index = list(PERIOD_TIMES.keys()).index(current_period) + 1
            period_column = f"period_{period_index}"
            
            marked_students = []
            
            for student_id, name in student_data:
                cursor.execute("""
                    SELECT * FROM attendance WHERE student_id = ? AND date = ? AND day = ?
                """, (student_id, today_date, current_day))
                existing_record = cursor.fetchone()
                
                if existing_record:
                    cursor.execute(f"""
                        UPDATE attendance 
                        SET {period_column} = 1
                        WHERE student_id = ? AND date = ? AND day = ?
                    """, (student_id, today_date, current_day))
                    conn.commit()
                else:
                    attendance_data = {period: 0 for period in PERIOD_TIMES.keys()}
                    attendance_data[current_period] = 1
                    cursor.execute("""
                        INSERT INTO attendance (student_id, date, day, period_1, period_2, period_3, period_4, period_5, period_6, period_7)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (student_id, today_date, current_day, *attendance_data.values()))
                    conn.commit()
                
                marked_students.append(name)
            
            st.success(f"🎉 Attendance marked for: **{', '.join(marked_students)}**")
        else:
            st.error(f"No timetable entry found for {current_period} on {current_day}.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    
    
# Resize function for consistent dimensions
def resize_face(image, target_size=(224, 224)):
    return cv2.resize(image, target_size)

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to extract face features
def extract_face_features(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Ensure correct color format

    with mp_face_detection.FaceDetection(min_detection_confidence=0.9) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            detection = results.detections[0]  # Use the first detected face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            # Ensure bounding box is within the image limits
            x, y, w, h = max(0, x), max(0, y), min(iw, w), min(ih, h)

            face_image = image[y:y+h, x:x+w]

            if face_image.size == 0:
                raise ValueError("Extracted face is empty.")

            # Resize face image to standard size for feature extraction
            return resize_face(face_image)
        else:
            return None  # No face detected

# Flatten the face images
def flatten_face(image):
    return image.flatten()

# Cosine similarity calculation
def calculate_cosine_similarity(stored_face, captured_face):
    if stored_face is None or captured_face is None:
        raise ValueError("Invalid face images provided for comparison.")

    # Resize both images to the same size
    stored_face_resized = resize_face(stored_face)
    captured_face_resized = resize_face(captured_face)

    # Flatten both images
    stored_face_flat = flatten_face(stored_face_resized)
    captured_face_flat = flatten_face(captured_face_resized)

    # Ensure the feature vectors have the same length
    if stored_face_flat.shape[0] != captured_face_flat.shape[0]:
        raise ValueError("Feature vector sizes do not match!")

    # Calculate cosine similarity
    return 1 - cosine(stored_face_flat, captured_face_flat)
    

# MiDaS model path (saved in the root directory)
MIDAS_MODEL_PATH = "midas_model.pt"

def download_midas_model():
    """Downloads the MiDaS model if it is missing."""
    if not os.path.exists(MIDAS_MODEL_PATH):
        print("Downloading MiDaS model...")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        torch.save(model.state_dict(), MIDAS_MODEL_PATH)  # Save in root directory
        print("MiDaS model downloaded and saved successfully!")

def load_midas_model():
    """Loads the MiDaS model from the saved file."""
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False)  # Initialize model
    model.load_state_dict(torch.load(MIDAS_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

def estimate_depth(image, model):
    """Performs depth estimation on an image using MiDaS."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        depth_map = model(image)

    depth_map = depth_map.squeeze().numpy()  # Remove batch dimension
    depth_variance = np.var(depth_map)  # Compute depth variance

    return depth_variance

def detect_spoof(image_path):
    """Detects spoofing using sharpness, edge count, and depth estimation."""
    
    # Ensure the MiDaS model is available
    if not os.path.exists(MIDAS_MODEL_PATH):
        download_midas_model()
    
    model = load_midas_model()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        st.error("Failed to load image.")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness check using Laplacian (variance of image sharpness)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Edge detection using Canny edge detector
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)  # Counting non-zero pixels (edges)

    # 3. Depth estimation
    depth_variance = estimate_depth(image, model)

    # Display values in Streamlit
    st.text(f"Variance (sharpness): {variance}")
    st.text(f"Edge count: {edge_count}")
    st.text(f"Depth variance: {depth_variance}")

    # Thresholds for spoof detection
    sharpness_threshold = 60
    edge_threshold = 4000
    depth_threshold = 500  # Higher variance means real 3D depth

    # Decision-making based on all three factors
    if variance < sharpness_threshold:
        st.warning("This looks like a printed photo or screen capture (low sharpness).")
        return False
    elif edge_count < edge_threshold:
        st.warning("This looks like a printed photo or screen capture (low edge count).")
        return False
    elif depth_variance < depth_threshold:
        st.warning("Depth analysis suggests a flat image (possible spoof attempt).")
        return False
    else:
        st.success("This seems like a real captured photo.")
        return True


# Function to verify if the captured face is registered using DeepFace
def is_face_registered(face_blob):
    new_face_path = "/tmp/new_face.jpg"

    with open(new_face_path, "wb") as f:
        f.write(face_blob)

    # Step 1: Anti-Spoofing (Detect if it's a printed photo or screen)
    if not detect_spoof(new_face_path):
        st.error("Possible scam detected! This appears to be a printed photo or screen image.")
        st.stop()

    # Step 2: Face Verification (Check if the face is already registered)
    cursor.execute("SELECT student_face FROM students")
    stored_faces = cursor.fetchall()

    THRESHOLD = 0.7

    for stored_face in stored_faces:
        stored_face_path = "/tmp/stored_face.jpg"
        with open(stored_face_path, "wb") as f:
            f.write(stored_face[0])

        try:
            # Verify if the captured face matches a registered face
            result = DeepFace.verify(new_face_path, stored_face_path, model_name="Facenet", distance_metric="cosine")
            similarity_score = result["distance"]

            st.write(f"Face match result: {result}")
            st.write(f"Similarity score: {similarity_score}")

            if result["verified"] and similarity_score < THRESHOLD:
                return True  # Face is already registered

        except Exception as e:
            st.error(f"Error during face verification: {e}")

    return False  # No match found


# Function to verify the captured face with the stored face (BLOB)
def verify_face(captured_face_blob, stored_face_blob):
    try:
        # Convert BLOB to image (temporary file to use with DeepFace)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as captured_face_file:
            captured_face_file.write(captured_face_blob)
            captured_face_path = captured_face_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as stored_face_file:
            stored_face_file.write(stored_face_blob)
            stored_face_path = stored_face_file.name

        # Use DeepFace to verify the face similarity
        result = DeepFace.verify(captured_face_path, stored_face_path, model_name="Facenet", distance_metric="cosine")
        
        st.write(f"Face match result: {result}")
        similarity_score = result["distance"]
        st.write(f"Similarity score: {similarity_score}")

        # Threshold for similarity score (lower means more similar)
        THRESHOLD = 0.7
        if result["verified"] and similarity_score < THRESHOLD:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error during face verification: {e}")
        return False


# Database setup to store device IDs
def create_connection():
    conn = sqlite3.connect("device_ids.db")
    return conn

def create_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS device_ids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT UNIQUE,
                created_at DATETIME)''')
    conn.commit()
    conn.close()

def insert_device_id(device_id):
    conn = create_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO device_ids (device_id, created_at) VALUES (?, ?)",
                  (device_id, datetime.now()))
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

# Initialize the database table
create_table()

# Initialize cookies manager
cookies = EncryptedCookieManager(
    prefix="my_app_",
    password="your_secret_key"  # Use a secure password for encryption
)

# Ensure cookies are loaded
if not cookies.ready():
    st.stop()

# Check for an existing device ID in cookies
device_id_from_cookies = cookies.get("device_id")  # Use a different name to avoid conflicts

if not device_id_from_cookies:
    # Generate a new UUID if not found
    device_id_from_cookies = str(uuid.uuid4())
    cookies["device_id"] = device_id_from_cookies  # Save to cookies
    cookies.save()  # Commit changes

# Insert the device ID into the database
insert_device_id(device_id_from_cookies)


def get_precise_location(api_key=None):
    if api_key:
        # Google Maps Geocode API URL
        google_maps_url = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={{LATITUDE}},{{LONGITUDE}}&key={api_key}'
        
        # For now, we'll use some dummy coordinates (you can replace this with dynamic geolocation)
        latitude, longitude = 22.5726, 88.3639  # Example: Coordinates for Kolkata, India

        # Request to Google Maps Geocode API to fetch the precise address
        try:
            st.info("Requesting location from Google Maps API...")
            response = requests.get(google_maps_url.format(LATITUDE=latitude, LONGITUDE=longitude))

            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    # Parsing the address components
                    address_components = data['results'][0]['address_components']
                    full_address = {
                        'street_number': '',
                        'street_name': '',
                        'city': '',
                        'state': '',
                        'country': '',
                        'postal_code': ''
                    }

                    for component in address_components:
                        types = component['types']
                        if 'street_number' in types:
                            full_address['street_number'] = component['long_name']
                        elif 'route' in types:
                            full_address['street_name'] = component['long_name']
                        elif 'locality' in types:
                            full_address['city'] = component['long_name']
                        elif 'administrative_area_level_1' in types:
                            full_address['state'] = component['long_name']
                        elif 'country' in types:
                            full_address['country'] = component['long_name']
                        elif 'postal_code' in types:
                            full_address['postal_code'] = component['long_name']

                    detailed_address = (
                        f"Street: {full_address['street_number']} {full_address['street_name']}, "
                        f"City: {full_address['city']}, "
                        f"State: {full_address['state']}, "
                        f"Country: {full_address['country']}, "
                        f"Postal Code: {full_address['postal_code']}"
                    )
                    st.success("Google Maps API used successfully.")
                    return detailed_address
                else:
                    st.warning("Google Maps API did not return a valid address.")
                    return "Error fetching precise location from Google Maps API."
            else:
                st.error(f"Google Maps API request failed with status code {response.status_code}.")
                return "Error with Google Maps API request."
        except requests.exceptions.RequestException as e:
            st.error(f"Google Maps API request failed: {str(e)}")
            return "Error with Google Maps API request."

    # If no API key is provided, use ip-api as a fallback
    else:
        url = 'http://ip-api.com/json'  # Using ip-api for location lookup
        try:
            st.info("Requesting location from ip-api as fallback...")
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                city = data.get('city', 'Unknown city')  # Fetch city name
                st.info("Using ip-api as fallback for location as Google maps API is unavailable.")
                return city
            else:
                st.error("Error fetching location from ip-api.")
                return "Error fetching location."
        except requests.exceptions.RequestException as e:
            st.error(f"ip-api request failed: {str(e)}")
            return "Error with ip-api request."


import streamlit as st
import streamlit.components.v1 as components
import json

st.title("📡 BLE Device Verifier")

# Constants
REQUIRED_DEVICE_ID = "76:6B:E1:0F:92:09"
REQUIRED_DEVICE_NAME = "INSTITUTE BLE VERIFY SIGNA"

# --- Session state initialization ---
if "scanned_devices" not in st.session_state:
    st.session_state.scanned_devices = []
if "verified" not in st.session_state:
    st.session_state.verified = False
if "last_scanned_device" not in st.session_state:
    st.session_state.last_scanned_device = {"name": "", "id": ""}
if "show_json" not in st.session_state:
    st.session_state.show_json = ""

# --- BLE Scan JS without filters ---
js_code = """
<script>
    function storeDevice(deviceInfo) {
        const data = {
            name: deviceInfo.name,
            id: deviceInfo.id
        };
        parent.document.querySelector('textarea[aria-label="Scanned device JSON:"]').value = JSON.stringify(data, null, 2);
        parent.document.querySelector('textarea[aria-label="Scanned device JSON:"]').dispatchEvent(new Event('input', { bubbles: true }));
    }

    async function scanBLE() {
        try {
            const device = await navigator.bluetooth.requestDevice({
                acceptAllDevices: true
            });

            const deviceInfo = {
                name: device.name || "Unnamed Device",
                id: device.id
            };
            storeDevice(deviceInfo);
        } catch (e) {
            alert("Scan cancelled or failed.");
            console.error(e);
        }
    }
</script>
<button onclick="scanBLE()">🔎 Scan Bluetooth Device</button>
"""

components.html(js_code, height=60)

# --- Text Area: Bound only by key ---
st.text_area(
    "Scanned device JSON:",
    height=100,
    key="show_json"
)

# --- Verify button ---
if st.button("🔒 Verify Device"):
    if st.session_state.show_json:
        try:
            device = json.loads(st.session_state.show_json)
            device_name = device.get("name", "").strip()
            device_id = device.get("id", "").strip()

            # Save scanned data
            st.session_state.last_scanned_device = {"name": device_name, "id": device_id}

            # Store if new
            if not any(d["id"] == device_id for d in st.session_state.scanned_devices):
                st.session_state.scanned_devices.append({
                    "name": device_name,
                    "id": device_id
                })
                st.success(f"✅ Device added: {device_name} ({device_id})")

            # Update the JSON back to text area (keeps formatting clean)
            st.session_state.show_json = json.dumps({
                "name": device_name,
                "id": device_id
            }, indent=2)

        except Exception as e:
            st.error(f"❌ Invalid JSON: {e}")

    # --- Perform verification ---
    if st.session_state.scanned_devices:
        match_found = any(
            d["id"] == REQUIRED_DEVICE_ID or d["name"] == REQUIRED_DEVICE_NAME
            for d in st.session_state.scanned_devices
        )

        if match_found:
            matched_device = next(
                d for d in st.session_state.scanned_devices
                if d["id"] == REQUIRED_DEVICE_ID or d["name"] == REQUIRED_DEVICE_NAME
            )
            st.session_state.verified = True
            st.success(f"✅ Verified: {matched_device['name']} ({matched_device['id']})")
        else:
            st.error("❌ Verification failed. Required device not found.")
    else:
        st.warning("⚠️ No scanned devices available.")

# --- Last Scanned Device Display ---
if st.session_state.last_scanned_device["name"]:
    st.subheader("📍 Last Scanned Device")
    st.json(st.session_state.last_scanned_device)

# --- All Devices History ---
if st.session_state.scanned_devices:
    st.subheader("📋 All Scanned Devices")
    for i, device in enumerate(st.session_state.scanned_devices, 1):
        st.write(f"{i}. *{device['name']}* ({device['id']})")



def measure_latency(flask_server_url):
    """
    Measure the network latency between the Streamlit app and Flask server.
    Returns latency in milliseconds.
    """
    start_time = time.time()
    try:
        response = requests.get(flask_server_url)
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        return latency
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to server: {e}")
        return None

def get_ble_signal_from_api():
    """
    Fetch BLE signals by making a GET request to the Flask BLE API server.
    Only proceeds if the network latency is below the threshold (considered within 10 meters).
    """
    flask_server_url = "https://fresh-adjusted-spider.ngrok-free.app/scan_ble"  # Your Flask API URL
    
    # Measure the latency first
    latency = measure_latency(flask_server_url)
    
    if latency is None:
        st.error("Unable to measure latency. Skipping BLE signal fetch.")
        return None

    st.write(f"Network latency: {latency:.2f} ms")
    
    # If latency is above the threshold (50 ms), assume the devices are too far
    latency_threshold_ms = 7000  # Adjust this as needed (for example, 50 ms threshold for 10 meters)
    if latency > latency_threshold_ms:
        st.error("Devices are too far from your classroom of your institution.")
        return None
    
    # Proceed to fetch BLE signals if latency is within range
    try:
        response = requests.get(flask_server_url)

        if response.status_code == 200:
            try:
                return response.json()  # Parse and return the JSON response from the Flask server
            except ValueError:
                st.error("Error: Received an invalid JSON response from the server.")
                return None
        else:
            st.error(f"Failed to fetch BLE devices. Status Code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the BLE API: {e}")
        return None


def get_current_period():
    """
    Get the current active period based on the current time, considering timezones.

    This function maps current time to the relevant period, taking into account
    the user's timezone (if available) or a specified default timezone.
    """

    # Specify the desired timezone (e.g., IST for Indian Standard Time)
    desired_timezone = timezone(timedelta(hours=5, minutes=30))  # Adjust as needed

    # Try to get the user's timezone (optional)
    # This might not be reliable in all cases
    # user_timezone = st.session_state.get('user_timezone', desired_timezone)

    current_time = datetime.now(desired_timezone).time()
    st.info(f"The current time (in {desired_timezone}): {current_time}")

    # Convert current_time to minutes for easier comparison
    current_minutes = current_time.hour * 60 + current_time.minute

    period_times = {
        "Period 1": ("09:30", "10:20"),
        "Period 2": ("10:20", "11:10"),
        "Period 3": ("11:10", "12:00"),
        "Period 4": ("12:00", "12:50"),
        "Period 5": ("13:20", "14:30"),
        "Period 6": ("14:30", "15:20"),
        "Period 7": ("15:20", "21:10")  # Update end time for Period 7 if needed
    }

    for period, times in period_times.items():
        start_time = datetime.strptime(times[0], "%H:%M").time()
        end_time = datetime.strptime(times[1], "%H:%M").time()

        # Convert start_time and end_time to minutes
        start_minutes = start_time.hour * 60 + start_time.minute
        end_minutes = end_time.hour * 60 + end_time.minute

        # Check if current time is within the period's range
        if start_minutes <= (current_minutes + current_time.second / 60) <= end_minutes:
            return period

    return None

# WebAuthn Registration Script (JavaScript) for capturing fingerprint data
def webauthn_register_script():
    script = """
    <script>
        async function registerFingerprint() {
            try {
                // Generate WebAuthn registration options
                const publicKey = {
                    challenge: Uint8Array.from('someRandomChallenge123', c => c.charCodeAt(0)),
                    rp: { name: 'WebAuthn Example' },
                    user: {
                        id: new Uint8Array(16),
                        name: 'asas-beta@user.com',
                        displayName: 'Student User'
                    },
                    pubKeyCredParams: [{ type: 'public-key', alg: -7 }],
                    authenticatorAttachment: 'platform',
                    timeout: 60000,
                    userVerification: 'required'
                };

                // Call WebAuthn API to register the credential
                const credential = await navigator.credentials.create({ publicKey });

                // Store the registration response (public key and credential ID)
                localStorage.setItem('credentialId', credential.id);
                localStorage.setItem('publicKey', JSON.stringify(credential.response.attestationObject));

                document.getElementById('registration-result').innerHTML = 'Registration successful!';
            } catch (error) {
                document.getElementById('registration-result').innerHTML = 'Registration failed: ' + error;
            }
        }
    </script>
    <button onclick="registerFingerprint()">Register Fingerprint</button>
    <p id="registration-result"></p>
    """
    return script

    
# Placeholder for WebAuthn integration script
def webauthn_script():
    script = """
    <script>
        async function authenticate() {
            const publicKey = {
                challenge: Uint8Array.from("YourServerChallenge", c => c.charCodeAt(0)), // Replace with a secure challenge
                timeout: 60000,
                userVerification: "required"
            };

            try {
                const credential = await navigator.credentials.get({ publicKey });
                document.getElementById("webauthn-result").innerHTML = 'Authentication successful! Please wait for the next steps.';
                document.getElementById("webauthn-status").value = "success";

                // Notify Streamlit about success
                window.parent.postMessage({ status: "success" }, "*");
            } catch (error) {
                document.getElementById("webauthn-result").innerHTML = "Authentication failed: " + error;
                document.getElementById("webauthn-status").value = "failed";

                // Notify Streamlit about failure
                window.parent.postMessage({ status: "failed" }, "*");
            }
        }

        // Listen for messages from parent (Streamlit)
        window.addEventListener("message", function(event) {
            if (event.data && event.data.status) {
                const status = event.data.status;
                document.getElementById("webauthn-status").value = status;
            }
        });
    </script>
    <button onclick="authenticate()">Authenticate with Fingerprint</button>
    <p id="webauthn-result"></p>
    <input type="hidden" id="webauthn-status" name="webauthn-status" value="pending">
    """
    return script


# Function to handle the authentication result
def handle_authentication_status():
    if st.session_state.auth_status == "success":
        st.success("Fingerprint authentication successful!")
        return True
    elif st.session_state.auth_status == "failed":
        st.error("Fingerprint authentication failed!")
        return False
    else:
        st.warning("Awaiting authentication status...")
        return False


def notifications():
    script = """
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        if (!("Notification" in window)) {
          alert("⚠️ Your browser does not support notifications.");
          return;
        }

        async function requestNotificationPermission() {
          if (Notification.permission === "granted") {
            alert("✅ Notifications are already enabled!");
            sendNotification();
          } else if (Notification.permission !== "denied") {
            const permission = await Notification.requestPermission();
            if (permission === "granted") {
              sendNotification();
            } else {
              alert("❌ Notification permission denied.");
            }
          } else {
            alert("❌ You have blocked notifications. Enable them in browser settings.");
          }
        }

        function sendNotification() {
          if (Notification.permission === "granted") {
            new Notification("🔔 Notification Enabled!", {
              body: "You have successfully subscribed to notifications.",
              icon: "https://www.google.com/favicon.ico"
            });
          }
        }

        window.subscribeToNotifications = requestNotificationPermission;
      });
    </script>

    <!-- Subscribe Button -->
    <button onclick="subscribeToNotifications()" style="padding: 10px; font-size: 16px; background-color: red; color: white; border: none; cursor: pointer;">
      Enable Notifications
    </button>
    """
    return script
    

# Function to send custom notifications (for teachers)
def send_custom_notification():
    script = """
    <script>
      function sendCustomNotification() {
        if (!("Notification" in window)) {
          alert("⚠️ Your browser does not support notifications.");
          return;
        }

        let message = document.getElementById("notificationMessage").value;
        if (message.trim() === "") {
          alert("⚠️ Please enter a message before sending.");
          return;
        }

        if (Notification.permission === "granted") {
          new Notification("📢 Message from Teacher", {
            body: message,
            icon: "https://www.google.com/favicon.ico"
          });
        } else if (Notification.permission !== "denied") {
          Notification.requestPermission().then(permission => {
            if (permission === "granted") {
              new Notification("📢 Message from Teacher", {
                body: message,
                icon: "https://www.google.com/favicon.ico"
              });
            } else {
              alert("❌ Notification permission denied.");
            }
          });
        } else {
          alert("❌ You have blocked notifications. Enable them in browser settings.");
        }
      }
    </script>

    <input type="text" id="notificationMessage" placeholder="Enter your message..." 
      style="padding: 10px; font-size: 16px; width: 80%; margin-bottom: 10px;">

    <button onclick="sendCustomNotification()" 
      style="padding: 10px; font-size: 16px; background-color: blue; color: white; border: none; cursor: pointer;">
      Send Notification
    </button>
    """
    return script


from datetime import datetime, timedelta

# Initialize the database and create the notifications table with an expiration column
def init_db():
    conn = sqlite3.connect('notifications.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Function to retrieve notifications from the database
def get_notifications():
    conn = sqlite3.connect('notifications.db', check_same_thread=False)
    c = conn.cursor()
    
    # Remove expired notifications
    c.execute("DELETE FROM notifications WHERE expires_at < datetime('now')")
    conn.commit()
    
    # Retrieve remaining notifications
    c.execute('SELECT id, message, expires_at FROM notifications ORDER BY id')
    rows = c.fetchall()
    conn.close()
    # Return a list of dicts for easier handling
    return [{"id": row[0], "message": row[1], "expires_at": row[2]} for row in rows]

# Function to add a new notification to the database with a configurable duration
def add_notification(message, duration_value=24, duration_unit="Hours"):
    # Convert the duration to hours based on the provided unit
    if duration_unit.lower() == "hours":
        duration_hours = duration_value
    elif duration_unit.lower() == "days":
        duration_hours = duration_value * 24
    elif duration_unit.lower() == "months":
        duration_hours = duration_value * 24 * 30  # Approximation: 30 days per month
    else:
        duration_hours = duration_value  # Fallback to hours if unit is unrecognized

    expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
    expires_at_str = expires_at.strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect('notifications.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('INSERT INTO notifications (message, expires_at) VALUES (?, ?)', (message, expires_at_str))
    conn.commit()
    conn.close()

# Function to remove a notification by its id
def remove_notification(notification_id):
    conn = sqlite3.connect('notifications.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('DELETE FROM notifications WHERE id = ?', (notification_id,))
    conn.commit()
    conn.close()

# Function to clear all notifications
def clear_all_notifications():
    conn = sqlite3.connect('notifications.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('DELETE FROM notifications')
    conn.commit()
    conn.close()



def is_strong_password(password):
    if len(password) < 8:
        return "❌ Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "❌ Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "❌ Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "❌ Password must contain at least one number."
    if not re.search(r"[!_@#$%^&*(),.?\":{}|<>]", password):
        return "❌ Password must contain at least one special character."
    return "✅ Strong password!"
    

# Streamlit UI
menu = st.sidebar.selectbox("Navigation Menu", ["Home", "Student's Registration", "Student's Login", "Teacher's Login", "Admin Management", "Lab Examination System", "Teacher's Registration", "Notification Center"])

if menu == "Home":
    st.warning("🔔 You must subscribe to notifications for future updates!")
    # Use Streamlit's `st.components.v1.html()` to execute JavaScript properly
    st.components.v1.html(notifications(), height=50)
    st.image('WhatsApp Image 2025-01-24 at 18.06.51.jpeg', width=200)
    st.title("ADVANCED STUDENT ATTENDANCE SYSTEM")
    st.subheader("developed by Debojyoti Ghosh")
    st.write("Welcome to the Student Management System!")
    st.subheader("Frequently Asked Questions (FAQs)")

    # FAQ Section
    faq = [
        {"question": "How do I register as a student?", 
         "answer": "To register as a student, go to the 'Register' section and fill in your details. You will need to provide basic information like name, roll number, and email."},
        
        {"question": "How do I log in as a student?", 
         "answer": "To log in as a student, go to the 'Student Login' section and enter your registered email and password."},
        
        {"question": "How do I mark attendance?", 
         "answer": "Once logged in, you can access the attendance page where your attendance is automatically marked based on your participation."},
        
        {"question": "What do I do if I forget my password?", 
         "answer": "If you've forgotten your password, go to the 'Forgot Password' section on the login page and follow the instructions to reset it."},
        
        {"question": "How can an admin manage students?", 
         "answer": "Admins can manage students' records by logging into the 'Admin Login' section. Admins have the ability to view, update, and delete student information."},
        
        {"question": "Can I track my exam schedule?", 
         "answer": "Yes, the 'Lab Examination System' section allows students to view and track their exam schedule."},
        
        {"question": "How can I contact support?", 
         "answer": "If you need help, you can contact us by sending an email to support@asas.com."}
    ]
    
    # Display the FAQs
    for item in faq:
        st.subheader(item["question"])
        st.write(item["answer"])
    
# Main registration logic
elif menu == "Student's Registration":
    st.header("Student Registration")
    
    # Init session state variables
    for key, default in {
        "step": 1,
        "email_otp": None,
        "email_verified": False,
        "face_blob": None,
        "face_verified": False,
        "form_data": {}
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    # ---- STEP 1: Student Details ----
    if st.session_state.step == 1:
        with st.form("student_details_form"):
            st.subheader("Student Details")
    
            # Persist inputs in session_state
            def save_student_details():
                st.session_state.form_data = {
                    "name": name,
                    "roll": roll,
                    "section": section,
                    "email": email,
                    "enrollment_no": enrollment_no,
                    "year": year,
                    "semester": semester,
                    "user_id": user_id,
                    "password": password
                }
                st.session_state.step = 2
    
            name = st.text_input("Name")
            roll = st.text_input("Roll Number")
            section = st.selectbox("Select Section", ["A", "B", "C", "D"])
            email = st.text_input("Email")
            enrollment_no = st.text_input("Enrollment Number")
            year = st.selectbox("Select Year", [1, 2, 3, 4])
            semester = st.selectbox("Select Semester", list(range(1, 9)))
            user_id = st.text_input("User ID")
            password = st.text_input("Password", type="password")
    
            st.markdown("""
            **Password must meet the following criteria:**  
            - ✅ At least **8 characters** long  
            - ✅ Contains **one uppercase letter** (A-Z)  
            - ✅ Contains **one lowercase letter** (a-z)  
            - ✅ Contains **one number** (0-9)  
            - ✅ Contains **one special character** (@, #, $, etc.)
            """, unsafe_allow_html=True)
    
            
            if st.form_submit_button("Next"):
               if is_strong_password(password) == "✅ Strong password!":
                  st.info(is_strong_password(password))
                  st.success("now you can click next to proceed with the registration")
                  save_student_details()
               else :
                  st.info(is_strong_password(password))
                
                    
    
    # ---- STEP 2: Face Capture ----
    elif st.session_state.step == 2:
        st.subheader("Capture Your Face")
        st.warning("Please set your screen brightness to maximum for better biometric capture.")
    
        with st.form("face_capture_form"):
            face_image = st.camera_input("Capture your face")
    
            submitted = st.form_submit_button("Capture and Continue")
    
            if submitted:
                if face_image:
                    img = Image.open(face_image)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="JPEG")
                    face_blob = img_bytes.getvalue()
    
                    if is_face_registered(face_blob):
                        st.error("This face is already registered. Please try again with a different face.")
                    else:
                        st.success("Face captured successfully!")
                        st.session_state.face_blob = face_blob
                        st.session_state.face_verified = True
                        st.session_state.step = 3
                        st.rerun()
                else:
                    st.error("Please capture your face before continuing.")
    
    
    # ---- STEP 3: WebAuthn / Fingerprint ----
    elif st.session_state.step == 3:
        st.subheader("🔐 Fingerprint Verification")
        st.warning("Complete fingerprint registration within 15 seconds.")
    
        # Inject WebAuthn script
        st.components.v1.html(webauthn_register_script())
    
        # Add animated countdown timer
        st.components.v1.html("""
            <div style="text-align: center;">
                <h3 id="timer" style="font-size: 48px; color: #ff4b4b; font-family: monospace;">15</h3>
                <p style="color: gray;">Time remaining to complete fingerprint verification</p>
            </div>
            <script>
                let count = 15;
                let timerEl = document.getElementById("timer");
                let interval = setInterval(() => {
                    count--;
                    timerEl.textContent = count;
                    if (count <= 0) {
                        clearInterval(interval);
                        window.parent.postMessage({ stepComplete: true }, "*");
                    }
                }, 1000);
            </script>
        """, height=150)
    
        # JS -> Streamlit message listener (via query param rerun or state is tricky)
        # So instead, use a Streamlit timer fallback:
        
        st.toast("⏳ Waiting for fingerprint input.", icon="⏳")
        time.sleep(2)
        st.toast("⏳ Waiting for fingerprint input...", icon="⏳")
        time.sleep(2)
        st.toast("⏳ Waiting for fingerprint input.....", icon="⏳")

        time.sleep(11)  # Sleep slightly more than countdown (non-blocking workaround)
        st.session_state.step = 4
        st.rerun()
    
    # ---- STEP 4: Email OTP Verification ----
    elif st.session_state.step == 4:
        st.subheader("Email Verification")
        email = st.session_state.form_data["email"]
    
        if not st.session_state.email_verified:
            if st.button("Send OTP"):
                otp = str(np.random.randint(100000, 999999))
                st.session_state.email_otp = otp
    
                try:
                    message = MIMEMultipart()
                    message["From"] = 'debojyotighoshmain@gmail.com'
                    message["To"] = email
                    message["Subject"] = "Your OTP for Student Registration"
                    message.attach(MIMEText(f"Your OTP is: {otp}", "plain"))
    
                    with smtplib.SMTP('smtp-relay.brevo.com', 587) as server:
                        server.starttls()
                        server.login('823c6b001@smtp-brevo.com', '6tOJHT2F4x8ZGmMw')
                        server.sendmail('debojyotighoshmain@gmail.com', email, message.as_string())
    
                    st.success(f"OTP sent to {email}")
                except Exception as e:
                    st.error(f"Failed to send OTP: {e}")
    
            otp_input = st.text_input("Enter the OTP sent to your email")
            if st.button("Verify OTP"):
                if otp_input == st.session_state.email_otp:
                    st.session_state.email_verified = True
                    st.success("Email verified successfully!")
                    st.session_state.step = 5
                    st.rerun()
                else:
                    st.error("Invalid OTP. Please try again.")
    
    # ---- STEP 5: Final Registration ----
    elif st.session_state.step == 5:
        st.subheader("Final Step: Complete Registration")
        form = st.session_state.form_data
        face_blob = st.session_state.face_blob
        device_id = device_id_from_cookies
    
        if st.button("Register"):
            if not device_id:
                st.error("Could not fetch device ID. Please try again.")
            else:
                # Check for duplicates
                cursor.execute("SELECT * FROM students WHERE device_id = ?", (device_id,))
                if cursor.fetchone():
                    st.error("Device already registered.")
                elif any(cursor.execute(f"SELECT * FROM students WHERE {field} = ?", (form[field],)).fetchone()
                         for field in ["email", "roll", "user_id", "enrollment_no"]):
                    st.error("One of the provided details is already registered.")
                else:
                    # Insert into DB
                    cursor.execute("""
                        INSERT INTO students 
                        (user_id, password, name, roll, section, email, enrollment_no, year, semester, device_id, student_face)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (form["user_id"], form["password"], form["name"], form["roll"], form["section"],
                          form["email"], form["enrollment_no"], form["year"], form["semester"],
                          device_id, face_blob))
                    conn.commit()
                    st.success("Registration completed successfully!")
                    st.warning("from now on this will be your only verified device for future logins!")
                    st.info("Proceed to Student Login page.")
                    st.session_state.step = 1  # Reset flow
                                            

# Student Login Page Logic
elif menu == "Student's Login":
    
    st.header("Student Login")
    st.success(f"Your unique device ID is: {device_id_from_cookies}")
    
    # Initialize session state for authentication
    if "auth_status" not in st.session_state:
        st.session_state.auth_status = "pending"
    
    # WebAuthn Integration
    st.subheader("Fingerprint Authentication")
    st.warning("Please proceed with the fingerprint authentication first to continue with login!")
    st.info("Passkeys will only be available if you have registered through desktop. In case of unavailability, this step will be bypassed. Please wait!!")
    
    # Inject WebAuthn script (from your separate function)
    auth_script = webauthn_script()
    st.components.v1.html(auth_script)
    
    # JavaScript message listener to capture authentication result
    st.markdown(
        """
        <script>
            window.addEventListener("message", (event) => {
                if (event.data.status === "success") {
                    window.parent.postMessage({ auth_result: "success" }, "*");
                } else if (event.data.status === "failed") {
                    window.parent.postMessage({ auth_result: "failed" }, "*");
                }
            });
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Add animated countdown timer
    st.components.v1.html("""
        <div style="text-align: center;">
            <h3 id="timer" style="font-size: 48px; color: #ff4b4b; font-family: monospace;">10</h3>
            <p style="color: gray;">Time remaining to complete fingerprint verification</p>
        </div>
        <script>
            let count = 10;
            let timerEl = document.getElementById("timer");
            let interval = setInterval(() => {
                count--;
                timerEl.textContent = count;
                if (count <= 0) {
                    clearInterval(interval);
                    window.parent.postMessage({ stepComplete: true }, "*");
                }
            }, 1000);
        </script>
    """, height=100)
    
    # Wait for WebAuthn result
    time.sleep(10)  # Delay to allow authentication
    
    # Check if authentication succeeded or needs bypassing
    auth_result = st.query_params.get("auth_result", "pending")
    
    if auth_result == "success":
        st.session_state.auth_status = "success"
        st.success("Fingerprint accepted. Waiting for server confirmation!!")
    else:
        st.session_state.auth_status = "bypass"
        st.info("Fingerprint authentication is still under development for Android, so it can be bypassed sometimes. However, later on this will be mandatory!!")
        
    # Create a login form with User ID, Password, and Face capture
    with st.form(key='login_form'):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        face_image = st.camera_input("Capture your face")
        st.warning("you are always recommended to set your screen brightness to maximum level while biometric verification!!")
        submit_button = st.form_submit_button("Login")

    # Proceed after the login button is clicked
    if submit_button:
        # Load Lottie animation
        chill_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json")
        
        # Display header and animation
        st.markdown("""
            <div style="text-align: center;">
                <h2 style="color: #00c2cb;">🚀 All set!</h2>
                <p style="font-size: 18px;">Just chill, we’re taking over now 😎</p>
            </div>
        """, unsafe_allow_html=True)
        
        if chill_lottie:
            st_lottie(chill_lottie, height=300, key="chill")
        else:
            st.warning("Couldn’t load animation, but we're still vibin'.")
        
        # Toast-style messages
        messages = [
            "✨ You vibe, we verify.",
            "📋 Attendance? We ate. No crumbs left.",
            "😎 Chillax, we're on attendance duty."
        ]
        
        for msg in messages:
            st.toast(msg)
            time.sleep(2.2)  # Toast duration default is ~2 seconds
                
        # Fetch the device ID (IP address)
        device_id = device_id_from_cookies
        st.success(f"Your unique device ID is: {device_id_from_cookies}")

        if not device_id:
            st.error("Could not fetch device Id. Login cannot proceed.")
        else:
            cursor.execute("SELECT * FROM students WHERE user_id = ? AND password = ?", (user_id, password))
            user = cursor.fetchone()

            if user:
                if user[9] == device_id:  # Match device_id from IP address
                    location = get_precise_location()
                    st.write(f"Your current location is: {location}")

                    if location and "The Dalles" in location:
                        time.sleep(2)
                        st.success("User ID and password verification successful!")

                        if face_image:
                            # Convert the captured face image to binary
                            img = Image.open(face_image)
                            img_bytes = io.BytesIO()
                            img.save(img_bytes, format="JPEG")
                            captured_face_blob = img_bytes.getvalue()

                            # Save the captured image temporarily for anti-spoofing check
                            captured_face_path = "captured_face.jpg"
                            with open(captured_face_path, "wb") as f:
                                f.write(captured_face_blob)

                            # Anti-spoofing check
                            if not detect_spoof(captured_face_path):
                                st.error("Spoofed image detected. Please use a real face photo.")
                                st.stop()  # Stop execution

                            # Save the captured image temporarily for DeepFace comparison
                            captured_face_path = "captured_face.jpg"
                            with open(captured_face_path, "wb") as f:
                                f.write(captured_face_blob)

                            # Now compare the captured face with the stored face during registration
                            stored_face_image = user[10]  # Example: Stored in the database as a blob

                            # Save the stored face image temporarily for DeepFace comparison
                            stored_face_path = "stored_face.jpg"
                            with open(stored_face_path, "wb") as f:
                                f.write(stored_face_image)

                            # Use DeepFace to compare the faces
                            result = DeepFace.verify(captured_face_path, stored_face_path, model_name="Facenet", enforce_detection=False)

                            # Get similarity score from the result
                            similarity_score = result["distance"]  # Lower values indicate higher similarity

                            threshold = 0.4  # Adjust this threshold as necessary
                            if similarity_score < threshold:
                                st.success("Face recognized successfully!")
                                # Proceed with the rest of the login process (location, Bluetooth, etc.)
                                st.success("You have passed the location check, and your location has been verified.")
                        
                                st.success("Your registered device has been verified successfully.")
                                
                                st.success("Fingerprint authentication successful.")
                                
                                st.success(f"Login successful! Welcome, {user[2]}")

                                # Check for Bluetooth signal during login session
                                st.info("Just a step away from your dashboard!! Scanning for physical verification devices...")

                                # Replace the original BLE signal detection logic
                                ble_signal = get_ble_signal_from_api()
                                time.sleep(3)

                                if ble_signal:
                                    if isinstance(ble_signal, dict) and "status" in ble_signal:
                                        # Handle API status response (e.g., Bluetooth is off)
                                        st.warning(ble_signal["status"])
                                    else:
                                        st.success("You are in your classroom. Have a nice study time! We will mark your attendance shortly.")
                                        st.info("Verification devices found. Listing all available devices...")

                                        # Display all available Bluetooth devices
                                        st.write("Available physical verification devices:")
                                        for device_name, mac_address in ble_signal.items():
                                            st.write(f"Device Name: {mac_address}, MAC Address: {device_name}")

                                        # Automatically check if the required Bluetooth device is in the list
                                        required_device_name = "76:6B:E1:0F:92:09"
                                        required_mac_id = "INSTITUTE BLE VERIFY SIGNA"  # Replace with the actual MAC address if known

                                        found_device = False
                                        for device_name, mac_address in ble_signal.items():
                                            if required_device_name in device_name or mac_address == required_mac_id:
                                                st.success(f"Required verifying device found! MAC Address: {device_name}, Device Name: {mac_address}")
                                                found_device = True
                                                break

                                        if found_device:
                                            # Save user login to session state
                                            st.session_state.logged_in = True
                                            st.session_state.user_id = user_id  # Replace with actual user ID if available
                                            st.session_state.bluetooth_selected = True  # Mark Bluetooth as selected
                                        else:
                                            st.error("Required verifying device not found. Login failed.")
                                            st.stop()

                                    # Define constant for period times
                                    PERIOD_TIMES = {
                                        "Period 1": ("09:30", "10:20"),
                                        "Period 2": ("10:20", "11:10"),
                                        "Period 3": ("11:10", "12:00"),
                                        "Period 4": ("12:00", "12:50"),
                                        "Period 5": ("13:40", "14:30"),
                                        "Period 6": ("14:30", "15:20"),
                                        "Period 7": ("15:20", "16:10")
                                    }

                                    # Attendance Marking Logic
                                    current_period = get_current_period()

                                    if current_period:
                                        st.success(f"Attendance for {current_period} is being marked automatically. Waiting for confirmation from the server!")

                                        current_day = datetime.now().strftime("%A")  # Get current weekday name

                                        try:
                                            # Fetch timetable details
                                            cursor.execute("""
                                                SELECT subject, teacher FROM timetable 
                                                WHERE day = ? AND period = ?
                                            """, (current_day, current_period))
                                            period_details = cursor.fetchone()

                                            if period_details:
                                                subject, teacher = period_details
                                                st.info(f"Subject: {subject} | Teacher: {teacher}")

                                                # Check for existing attendance record
                                                cursor.execute("""
                                                    SELECT * FROM attendance WHERE student_id = ? AND date = ? AND day = ?
                                                """, (user_id, datetime.now().strftime('%Y-%m-%d'), current_day))
                                                existing_record = cursor.fetchone()

                                                period_column = f"period_{list(PERIOD_TIMES.keys()).index(current_period) + 1}"

                                                if existing_record:
                                                    # Update attendance
                                                    cursor.execute(f"""
                                                        UPDATE attendance 
                                                        SET {period_column} = 1, subject = ?, teacher = ?
                                                        WHERE student_id = ? AND date = ? AND day = ?
                                                    """, (subject, teacher, user_id, datetime.now().strftime('%Y-%m-%d'), current_day))
                                                    conn.commit()
                                                    st.success(f"Attendance updated for {current_period} ({subject}) by {teacher} on {current_day}!")
                                                else:
                                                    # Prepare attendance data
                                                    attendance_data = {period: 0 for period in PERIOD_TIMES.keys()}
                                                    attendance_data[current_period] = 1

                                                    # Insert new attendance record
                                                    cursor.execute("""
                                                        INSERT INTO attendance (student_id, date, day, period_1, period_2, period_3, period_4, period_5, period_6, period_7, subject, teacher)
                                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                                    """, (user_id, datetime.now().strftime('%Y-%m-%d'), current_day, *attendance_data.values(), subject, teacher))
                                                    conn.commit()
                                                    st.success(f"Attendance for {current_period} ({subject}) by {teacher} marked successfully for {current_day}!")
                                            else:
                                                st.error(f"No timetable entry found for {current_period} on {current_day}.")
                                        except Exception as e:
                                            st.error(f"An error occurred: {e}")
                                    else:
                                        st.warning("No active class period at the moment.")
                                else:
                                    st.error("No verifying devices found. Maybe you are not in your institution.")
                            else:
                                st.error(f"Face recognition failed. Cosine similarity score: {similarity_score}. Please try again.")
                        else:
                            st.error("Please capture your face.")
                    else:
                        st.error("You must be in Kolkata to login.")
                else:
                    st.error("Device ID does not match. Please login from your registered device.")
            else:
                st.error("Invalid User ID or Password. Please try again.")
                        
    
    # Display student attendance search form
    if st.session_state.get('logged_in', False):
        st.subheader("Search Attendance Records")

        # Dynamically generate the last 10 years for selection
        current_year = datetime.now().year
        years = [str(year) for year in range(current_year, current_year - 10, -1)]

        # Select Year, Month, Day of the Week, and Date
        year = st.selectbox("Select Year", years)
        month = st.selectbox("Select Month", [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ])
        day_of_week = st.selectbox("Select Day of the Week (Optional)", [
            "Any", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
        ])
        include_date = st.checkbox("Filter by Specific Date?")
        selected_date = None

        # Allow date selection if checkbox is enabled
        if include_date:
            selected_date = st.date_input("Select Date")

        # Convert month to numeric value for filtering
        month_num = datetime.strptime(month, "%B").month

        # Search Button
        if st.button("Search Attendance"):
            # Build SQL query dynamically based on user input
            query = """
                SELECT * FROM attendance 
                WHERE student_id = ? 
                AND strftime('%Y', date) = ? 
                AND strftime('%m', date) = ?
            """
            params = [st.session_state.user_id, year, f"{month_num:02d}"]

            # Add day of the week filter if selected
            if day_of_week != "Any":
                query += " AND day = ?"
                params.append(day_of_week)
            
            # Add specific date filter if enabled
            if include_date and selected_date:
                query += " AND date = ?"
                params.append(selected_date.strftime('%Y-%m-%d'))

            # Execute the query
            cursor.execute(query, tuple(params))
            attendance_records = cursor.fetchall()

            # Display attendance records or a message if none are found
            if attendance_records:
                st.write("### Attendance Records Found:")
                for record in attendance_records:
                    st.write(f"Date: {record[1]}, Day: {record[2]}")
                    for i in range(3, 10):  # Periods are in columns 3 to 9
                        st.write(f"Period {i-2}: {'Present' if record[i] == 1 else 'Absent'}")
            else:
                st.write("No attendance records found for the selected filters.")
                
        # Add dispute submission section
        st.subheader("Raise Attendance Dispute")
        st.info("If you feel there's a discrepancy in your attendance, please submit a dispute.")

        # Dispute Form
        dispute_reason = st.text_area("Describe the issue")
        submit_dispute = st.button("Submit Dispute")

        if submit_dispute:
            if dispute_reason.strip():
                # Insert dispute into the database
                cursor.execute("""
                    INSERT INTO disputes (student_id, date, reason)
                    VALUES (?, ?, ?)
                """, (user_id, datetime.now().strftime('%Y-%m-%d'), dispute_reason))
                conn.commit()
                st.success("Your dispute has been submitted successfully!")
            else:
                st.error("Please provide a reason for the dispute.")

        # Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.bluetooth_selected = False  # Reset Bluetooth selection flag
            st.session_state.attendance_saved = False  # Reset attendance saved flag
            st.rerun()

# Admin Login Flow
elif menu == "Teacher's Login":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False  # Initially, no one is logged in
    
    if 'admin_id' not in st.session_state:  # Ensure admin_id is initialized
        st.session_state.admin_id = None

    if st.session_state.logged_in:
        # Admin is logged in, show the dashboard
        st.warning("ID and password verification in progress!")
        time.sleep(2)

        # Admin Profile Section
        if st.session_state.admin_id:  # Check if admin_id exists before querying the database
            try:
                cursor.execute("SELECT * FROM admin_profile WHERE admin_id = ?", (st.session_state.admin_id,))
                profile = cursor.fetchone()
        
                if profile:
                    time.sleep(2)
                    st.info("The ID and password verification is successful!")
                    st.success("Admin login successful!")
                    st.success(f"Welcome back to your dashboard, {profile[1]}!!")
                    st.header("Admin Profile")
                    st.write(f"Name: {profile[1]}")
                    st.write(f"Department: {profile[2]}")
                    st.write(f"Designation: {profile[3]}")
                    st.write(f"Email: {profile[4]}")
                    st.write(f"Phone: {profile[5]}")
        
                    # Option to update profile
                    st.write("---")
                    st.subheader("Update Profile")
                    with st.form(key="update_profile_form"):
                        new_name = st.text_input("New Name", profile[1])
                        new_department = st.text_input("New Department", profile[2])
                        new_designation = st.text_input("New Designation", profile[3])
                        new_email = st.text_input("New Email", profile[4])
                        new_phone = st.text_input("New Phone", profile[5])
                        st.error("Please note: For safety, admin ID and password need to be changed together and once changed, the profile setup has to be done again!")
        
                        # Option to update Admin ID and Password
                        new_admin_id = st.text_input("New Admin ID", st.session_state.admin_id)
                        new_password = st.text_input("New Password", type="password")
        
                        # Option to capture a new face image
                        st.warning("Capture your face to update authentication.")
                        captured_face = st.camera_input("Capture Face Image (Optional)")
        
                        submit_button = st.form_submit_button("Save Changes")
        
                        if submit_button:
                            # Check if new admin ID already exists
                            cursor.execute("SELECT * FROM admin WHERE admin_id = ?", (new_admin_id,))
                            existing_admin = cursor.fetchone()
        
                            if existing_admin and new_admin_id != st.session_state.admin_id:
                                st.error("Admin ID already exists. Please choose a different one.")
                            else:
                                # Handle face image if captured
                                new_face_blob = None
                                if captured_face:
                                    new_face_blob = captured_face.getvalue()  # Store raw image as BLOB
        
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                                        temp_file.write(new_face_blob)
                                        temp_image_path = temp_file.name
        
                                    # Check if the face is real (anti-spoofing)
                                    if not detect_spoof(temp_image_path):
                                        st.error("Possible spoofing detected! Use a real captured image.")
                                        st.stop()
                                    st.success("Face successfully verified!")
        
                                # Update admin profile
                                cursor.execute(""" 
                                    UPDATE admin_profile 
                                    SET name = ?, department = ?, designation = ?, email = ?, phone = ? 
                                    WHERE admin_id = ? 
                                """, (
                                    new_name,
                                    new_department,
                                    new_designation,
                                    new_email,
                                    new_phone,
                                    st.session_state.admin_id,
                                ))
        
                                # Update admin login credentials (admin ID and password)
                                if new_admin_id != st.session_state.admin_id:
                                    cursor.execute(""" 
                                        UPDATE admin 
                                        SET admin_id = ?, password = ? 
                                        WHERE admin_id = ? 
                                    """, (new_admin_id, new_password, st.session_state.admin_id))
                                    st.session_state.admin_id = new_admin_id  # Update the session ID
        
                                if new_face_blob:
                                    cursor.execute(""" 
                                        UPDATE admin_profile 
                                        SET face_encoding = ? 
                                        WHERE admin_id = ? 
                                    """, (new_face_blob, st.session_state.admin_id))
        
                                conn.commit()
                                st.success("Profile and login credentials updated successfully!")
        
                else:
                    st.error("Admin profile not found. Please complete your profile setup.")
                    st.write("---")
                    st.subheader("Complete Profile Setup")
        
                    # If no profile found, allow admin to create one
                    with st.form(key="create_profile_form"):
                        new_name = st.text_input("Name")
                        new_department = st.text_input("Department")
                        new_designation = st.text_input("Designation")
                        new_email = st.text_input("Email")
                        new_phone = st.text_input("Phone")
                        new_admin_id = st.text_input("Admin ID")
                        new_password = st.text_input("Password", type="password")
        
                        # Option to capture a new face image
                        st.warning("Capture your face to complete profile setup.")
                        captured_face = st.camera_input("Capture Face Image (Required)")
        
                        submit_button = st.form_submit_button("Save Profile")
        
                        if submit_button:
                            if new_name and new_department and new_designation and new_email and new_phone and new_admin_id and new_password and captured_face:
                                cursor.execute("SELECT * FROM admin WHERE admin_id = ?", (new_admin_id,))
                                existing_admin = cursor.fetchone()
        
                                if existing_admin:
                                    st.error("Admin ID already exists. Please choose a different one.")
                                else:
                                    face_blob = captured_face.getvalue()  # Store raw image as BLOB
        
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                                        temp_file.write(face_blob)
                                        temp_image_path = temp_file.name
        
                                    # Check if the face is real (anti-spoofing)
                                    if not detect_spoof(temp_image_path):
                                        st.error("Possible spoofing detected! Use a real captured image.")
                                        st.stop()
                                    st.success("Face successfully verified!")
        
                                    cursor.execute(""" 
                                        INSERT INTO admin_profile (admin_id, name, department, designation, email, phone, face_encoding) 
                                        VALUES (?, ?, ?, ?, ?, ?, ?) 
                                    """, (new_admin_id, new_name, new_department, new_designation, new_email, new_phone, face_blob))
        
                                    cursor.execute(""" 
                                        INSERT INTO admin (admin_id, password) 
                                        VALUES (?, ?) 
                                    """, (new_admin_id, new_password))
        
                                    conn.commit()
                                    st.success("Profile and login credentials created successfully!")
                                    # Send the confirmation email
                                    send_confirmation_email(new_email, new_name)
                            else:
                                st.error("Please fill all fields and capture a face image.")
        
            except sqlite3.OperationalError as e:
                st.error(f"Database error: {e}")
            except sqlite3.IntegrityError as e:
                st.error(f"Integrity error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                
        st.markdown("---")
        # Advanced Search Section for Registered Students
        st.subheader("🔍 Advanced Search for Registered Students")
        
        # === Layout: Group inputs into 3 clean columns ===
        col1, col2, col3 = st.columns(3)
        
        with col1:
            student_name = st.text_input("👤 Name")
            student_id = st.text_input("🆔 Student ID")
        
        with col2:
            student_department = st.text_input("🏫 Department")
            student_year = st.selectbox("🎓 Year", ["All", "1st Year", "2nd Year", "3rd Year", "4th Year"])
        
        with col3:
            search_by_month = st.selectbox("📅 Month", [
                "All", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ])
            search_by_year = st.selectbox("📆 Attendance Year", ["All", "2024", "2023", "2022", "2021", "2020"])
            search_by_day = st.selectbox("🗓️ Day", [
                "All", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            ])
            search_by_date = st.date_input("📌 Specific Date", value=None)
        
        st.markdown("")
        
        # Search Button
        if st.button("🔍 Search Students"):
            # SQL Base
            query = """
                SELECT DISTINCT students.* 
                FROM students
                LEFT JOIN attendance ON students.user_id = attendance.student_id
                WHERE 1=1
            """
            params = []
        
            # Apply filters
            if student_name:
                query += " AND students.name LIKE ?"
                params.append(f"%{student_name}%")
            if student_id:
                query += " AND students.user_id = ?"
                params.append(student_id)
            if student_department:
                query += " AND students.section LIKE ?"
                params.append(f"%{student_department}%")
            if student_year and student_year != "All":
                query += " AND students.year = ?"
                params.append(student_year)
            if search_by_month != "All":
                month_number = {
                    "January": "01", "February": "02", "March": "03", "April": "04", 
                    "May": "05", "June": "06", "July": "07", "August": "08", 
                    "September": "09", "October": "10", "November": "11", "December": "12"
                }[search_by_month]
                query += " AND strftime('%m', attendance.date) = ?"
                params.append(month_number)
            if search_by_year != "All":
                query += " AND strftime('%Y', attendance.date) = ?"
                params.append(search_by_year)
            if search_by_day != "All":
                day_number = {
                    "Sunday": "0", "Monday": "1", "Tuesday": "2", "Wednesday": "3", 
                    "Thursday": "4", "Friday": "5", "Saturday": "6"
                }[search_by_day]
                query += " AND strftime('%w', attendance.date) = ?"
                params.append(day_number)
            if search_by_date:
                query += " AND attendance.date = ?"
                params.append(search_by_date.strftime('%Y-%m-%d'))
        
            # Execute query
            cursor.execute(query, params)
            filtered_students = cursor.fetchall()
        
            # Save to session_state
            st.session_state["search_results"] = filtered_students
        
        # === Display Results (If Available) ===
        if "search_results" in st.session_state:
            filtered_students = st.session_state["search_results"]
        
            st.subheader("📋 Filtered Students")
        
            if filtered_students:
                for student in filtered_students:
                    student_id = student[0]
                    student_name = student[2]
        
                    with st.expander(f"👨‍🎓 {student_name} ({student_id})"):
                        st.markdown(f"**🎓 Roll:** {student[3]}")
                        st.markdown(f"**🏫 Department:** {student[4]}")
                        st.markdown(f"**📧 Email:** {student[5]}")
                        st.markdown(f"**🆕 Enrollment No:** {student[6]}")
                        st.markdown(f"**📘 Year:** {student[7]} | **Semester:** {student[8]}")
                        st.markdown(f"**🔐 Device ID:** `{student[9]}`")
        
                        # Get year & semester
                        cursor.execute("SELECT year, semester FROM students WHERE user_id = ?", (student_id,))
                        student_details = cursor.fetchone()
        
                        if student_details:
                            year, semester = student_details
        
                            # Semester info
                            cursor.execute("""
                                SELECT total_classes, total_periods 
                                FROM semester_dates 
                                WHERE year = ? AND semester = ?
                            """, (year, semester))
                            semester_details = cursor.fetchone()
        
                            if semester_details:
                                total_classes_in_semester, total_periods_in_semester = semester_details
        
                                # Attendance data
                                cursor.execute("SELECT * FROM attendance WHERE student_id = ?", (student_id,))
                                attendance_records = cursor.fetchall()
        
                                if attendance_records:
                                    total_classes_attended = len(attendance_records)
                                    total_periods_attended = sum(
                                        sum(1 if record[i] == 1 else 0 for i in range(3, 10))
                                        for record in attendance_records
                                    )
        
                                    # Attendance % calculation
                                    attendance_percentage = (
                                        (total_periods_attended / total_periods_in_semester) * 100
                                        if total_periods_in_semester > 0 else 0.0
                                    )
        
                                    st.success(f"✅ **Attendance Percentage:** {attendance_percentage:.2f}%")
                                    st.write(f"📅 Total Classes in Semester: {total_classes_in_semester}")
                                    st.write(f"⏰ Total Periods: {total_periods_in_semester}")
                                    st.write(f"📚 Attended Classes: {total_classes_attended}")
                                    st.write(f"📖 Attended Periods: {total_periods_attended}")
        
                                    # Unique toggle key for each student
                                    toggle_key = f"{student_id}_detailed_attendance"
        
                                    # Show checkbox (preserves state)
                                    if st.checkbox("🗂️ Show Detailed Attendance Records", key=toggle_key):
                                        for record in attendance_records:
                                            st.markdown(f"**🗓️ Date:** {record[1]} | **📌 Day:** {record[2]}")
                                            for i in range(3, 10):
                                                status = record[i]
                                                period_status = (
                                                    "✅ Present" if status == 1 else
                                                    "❌ Absent" if status == 0 else "⚪ N/A"
                                                )
                                                st.markdown(f"• Period {i-2}: {period_status}")
                                else:
                                    st.warning("⚠️ No attendance records found.")
                            else:
                                st.error("❌ Semester info not available.")
                        else:
                            st.error("❌ Student details missing.")
            else:
                st.info("🔎 No students found matching the criteria.")

                        
        st.markdown("---")
        st.markdown("---")
        # Section to View All Registered Students and Attendance Analysis
        st.subheader("Registered Students and Attendance Analysis")

        # Fetch all registered students
        cursor.execute("SELECT * FROM students")
        students = cursor.fetchall()

        if students:
            for student in students:
                student_id = student[0]
                student_name = student[2]
                st.write(f"Student ID: {student_id}, Name: {student_name}")

                # Button to view student details with a unique key
                if st.button(f"View Details for {student_name}", key=f"details_{student_id}"):
                    # Display student details
                    st.write(f"Roll: {student[3]}")
                    st.write(f"Section: {student[4]}")
                    st.write(f"Email: {student[5]}")
                    st.write(f"Enrollment No: {student[6]}")
                    st.write(f"Year: {student[7]}")
                    st.write(f"Semester: {student[8]}")

                    # Fetch the student's face image from the database (stored as a BLOB in student[10])
                    face_blob = student[10]  # Assuming 'student_face' is at index 10
                    
                    # Display the face image if available
                    if face_blob:
                        face_image = Image.open(io.BytesIO(face_blob))
                        
                        # Resize image to passport size (e.g., 200x200 pixels)
                        face_image = face_image.resize((100, 100))
                        st.image(face_image, caption="Current Face Image", use_container_width=True)
                    else:
                        st.write("No face image available for this student.")
                                    
                    # Attendance Analysis for the student
                    st.subheader(f"Attendance Analysis for {student_name}")

                    # Fetch the student's year and semester from the students table
                    cursor.execute("""
                        SELECT year, semester FROM students WHERE user_id = ?
                    """, (student_id,))
                    student_semester = cursor.fetchone()

                    if student_semester:
                        year, semester = student_semester

                        # Fetch total_classes and total_periods from semester_dates table based on year and semester
                        cursor.execute("""
                            SELECT total_classes, total_periods
                            FROM semester_dates 
                            WHERE year = ? AND semester = ?
                        """, (year, semester))
                        semester_details = cursor.fetchone()

                        if semester_details:
                            total_classes_in_semester, total_periods_in_semester = semester_details
                            
                            # Fetch the student's attendance records
                            cursor.execute("SELECT * FROM attendance WHERE student_id = ?", (student_id,))
                            attendance_records = cursor.fetchall()

                            if attendance_records:
                                total_days = len(attendance_records)  # Number of days attended
                                
                                # Calculate total periods attended, treating None as 0 (absent)
                                total_present = sum(
                                    sum(1 if record[i] == 1 else 0 for i in range(3, 10))  # Sum the periods attended (1 for present, 0 for absent)
                                    for record in attendance_records
                                )

                                # Calculate attendance percentage based on total periods in the semester
                                attendance_percentage = (total_present / total_periods_in_semester) * 100

                                # Display detailed information for the attendance analysis
                                st.write(f"**Total class days in Semester:** {total_classes_in_semester}")
                                st.write(f"**Total Periods or classes in Semester:** {total_periods_in_semester}")
                                st.write(f"**Total Attendance Days:** {total_days}")
                                st.write(f"**Total Periods or classes Attended:** {total_present}")
                                st.write(f"**Attendance Percentage:** {attendance_percentage:.2f}%")

                                # Display Attendance Breakdown (Day-wise and Period-wise)
                                st.write("### Detailed Attendance Breakdown:")
                                for record in attendance_records:
                                    st.write(f"**Date:** {record[1]}, **Day:** {record[2]}")  # Date and Day
                                    
                                    # Fetch subject and teacher information for each period
                                    for i in range(3, 10):  # Assuming period information starts from index 3 to 9
                                        if record[i] == 1:  # Check if the student was present
                                            period = f"Period {i-2}"  # Periods are labeled starting from Period 1
                                            
                                            # Fetch the subject and teacher for the corresponding period
                                            period_number = i - 2  # Adjust the index to match periods 1-7
                                            
                                            cursor.execute("""
                                                SELECT subject, teacher 
                                                FROM timetable
                                                WHERE day = ? AND period = ?
                                            """, (record[2], period))  # Using the formatted period like "Period 1"
                                            
                                            timetable_entry = cursor.fetchone()
                                            
                                            if timetable_entry:
                                                subject, teacher = timetable_entry
                                                st.write(f"**{period}:** Present | **Subject:** {subject} | **Teacher:** {teacher}")
                                            else:
                                                st.write(f"**{period}:** Present | No timetable entry found.")


                                        else:
                                            period = f"Period {i-2}"  # Periods are labeled starting from 1
                                            st.write(f"**{period}:** Absent")
                            else:
                                st.write("No attendance records found for this student.")
                        else:
                            st.write("No semester details found for this student.")
                    else:
                        st.write("No year or semester information found for this student.")


                # Initialize session state for form visibility
                if f"form_shown_{student_id}" not in st.session_state:
                    st.session_state[f"form_shown_{student_id}"] = False
                
                # Button to toggle form visibility
                if st.button(f"View/Edit Details for {student_name}", key=f"edit_{student_id}"):
                    st.session_state[f"form_shown_{student_id}"] = not st.session_state[f"form_shown_{student_id}"]
                
                # Display the form if it's active
                if st.session_state[f"form_shown_{student_id}"]:
                    st.header(f"Edit Details for {student_name}")
                
                    # Retrieve and display the student's face image from the database
                    try:
                        cursor.execute("SELECT student_face FROM students WHERE user_id = ?", (student_id,))
                        result = cursor.fetchone()
                
                        if result and result[0]:  # Check if a face image is present
                            face_blob = result[0]
                            face_image = Image.open(io.BytesIO(face_blob))
                            # Resize image to passport size (e.g., 200x200 pixels)
                            face_image = face_image.resize((100, 100))
                            st.image(face_image, caption="Current Face Image", use_container_width=True)
                
                        else:
                            st.warning("No face image found for this student.")
                    except Exception as e:
                        st.error(f"Error retrieving face image: {e}")
                
                    # Add option to upload or capture a new face image
                    st.subheader("Update Face Image")
                    new_face_image = st.camera_input("Capture a new face image", key="new_face_capture") or st.file_uploader("Upload a new face image", type=["png", "jpg", "jpeg"])
                
                    # Pre-fill existing student data into form fields
                    new_name = st.text_input("Name", value=student[2], key=f"name_{student_id}")
                    new_roll = st.text_input("Roll Number", value=student[3], key=f"roll_{student_id}")
                    new_section = st.text_input("Section", value=student[4], key=f"section_{student_id}")
                    new_email = st.text_input("Email", value=student[5], key=f"email_{student_id}")
                    new_enrollment_no = st.text_input("Enrollment Number", value=student[6], key=f"enrollment_{student_id}")
                    new_year = st.selectbox(
                        "Year", ["1", "2", "3", "4"],
                        index=["1", "2", "3", "4"].index(student[7]),
                        key=f"year_{student_id}"
                    )
                    new_semester = st.text_input("Semester", value=student[8], key=f"semester_{student_id}")
                    new_user_id = st.text_input("User ID", value=student[0], key=f"user_id_{student_id}")
                    new_password = st.text_input("Password", type="password", value=student[1], key=f"password_{student_id}")
                    new_device_ip = st.text_input("Device IP", value=student[9], key=f"device_ip_{student_id}")
                
                    # Save changes button
                    if st.button("Save Changes", key=f"save_changes_{student_id}"):
                        try:
                            # Validate required fields
                            if not new_user_id or not new_password or not new_device_ip:
                                st.error("User ID, Password, and Device IP are required fields.")
                            
                
                            # Process the new face image if provided
                            if new_face_image:
                                img = Image.open(new_face_image)
                                img_bytes = io.BytesIO()
                                img.save(img_bytes, format="JPEG")  # Save the image as bytes
                                face_blob = img_bytes.getvalue()  # Convert to BLOB
                
                                # Update the database with the new face image and details
                                cursor.execute("""
                                    UPDATE students
                                    SET user_id = ?, password = ?, device_id = ?, name = ?, roll = ?, section = ?, email = ?, enrollment_no = ?, year = ?, semester = ?, student_face = ?
                                    WHERE user_id = ?
                                """, (
                                    new_user_id, new_password, new_device_ip, new_name, new_roll, new_section,
                                    new_email, new_enrollment_no, new_year, new_semester, face_blob, student_id
                                ))
                            else:
                                # Update details without changing the face image
                                cursor.execute("""
                                    UPDATE students
                                    SET user_id = ?, password = ?, device_id = ?, name = ?, roll = ?, section = ?, email = ?, enrollment_no = ?, year = ?, semester = ?
                                    WHERE user_id = ?
                                """, (
                                    new_user_id, new_password, new_device_ip, new_name, new_roll, new_section,
                                    new_email, new_enrollment_no, new_year, new_semester, student_id
                                ))
                
                            conn.commit()
                
                            # Confirm success
                            if cursor.rowcount > 0:
                                st.success(f"Details for {student_name} have been successfully updated!")
                            else:
                                st.error("No changes were made. Please verify the details.")
                
                            # Close the form after successful update
                            st.session_state[f"form_shown_{student_id}"] = False
                
                        except Exception as e:
                            st.error(f"An error occurred while updating the details: {str(e)}")
                
                    # Cancel button to hide the form
                    if st.button("Cancel", key=f"cancel_edit_{student_id}"):
                        st.session_state[f"form_shown_{student_id}"] = False

        

                # Button to deregister a student
                if st.button(f"Deregister {student_name}", key=f"deregister_{student_id}"):
                    cursor.execute("DELETE FROM students WHERE user_id = ?", (student_id,))
                    cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
                    conn.commit()
                    st.warning(f"Student {student_name} has been deregistered!")

                # Initialize session state for attendance form visibility
                if f"attendance_form_shown_{student_id}" not in st.session_state:
                    st.session_state[f"attendance_form_shown_{student_id}"] = False

                # Button to toggle attendance form visibility
                if st.button(f"Overwrite Attendance for {student_name}", key=f"toggle_attendance_{student_id}"):
                    st.session_state[f"attendance_form_shown_{student_id}"] = not st.session_state[f"attendance_form_shown_{student_id}"]

                # Display the attendance form if it's active
                if st.session_state[f"attendance_form_shown_{student_id}"]:
                    st.header(f"Manually Update Attendance for {student_name}")
                    st.error("PLEASE NOTE : only overwrite attendance in case of student unable to give attendance though present because any action otherwise can lead to improper calculation of attendance!!!!")

                    # Date input for selecting the attendance date
                    selected_date = st.date_input("Select Date", key=f"attendance_date_{student_id}")
                    selected_day = selected_date.strftime("%A")

                    # Fetch existing attendance for the selected date
                    cursor.execute("""
                        SELECT period_1, period_2, period_3, period_4, period_5, period_6, period_7
                        FROM attendance
                        WHERE student_id = ? AND date = ?
                    """, (student_id, selected_date))
                    existing_attendance = cursor.fetchone()

                    # Pre-fill attendance status for each period
                    attendance = []
                    for period in range(1, 8):
                        prefilled_status = "Present" if existing_attendance and existing_attendance[period - 1] == 1 else "Absent"
                        status = st.selectbox(
                            f"Period {period} Status",
                            ["Absent", "Present"],
                            index=["Absent", "Present"].index(prefilled_status),
                            key=f"attendance_period_{period}_{student_id}"
                        )
                        attendance.append(1 if status == "Present" else 0)

                    # Save changes button
                    if st.button("Save Attendance", key=f"save_attendance_{student_id}"):
                        try:
                            # Update attendance in the database
                            cursor.execute("""
                                REPLACE INTO attendance (student_id, date, day, period_1, period_2, period_3, period_4, period_5, period_6, period_7)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (student_id, selected_date, selected_day, *attendance))
                            conn.commit()

                            # Confirm success
                            if cursor.rowcount > 0:
                                st.success(f"Attendance for {student_name} on {selected_date} has been successfully updated!")
                            else:
                                st.error("No changes were made. Please verify the details.")

                            # Close the form after successful update
                            st.session_state[f"attendance_form_shown_{student_id}"] = False
                        except Exception as e:
                            st.error(f"An error occurred while updating the attendance: {str(e)}")

                    # Cancel button to hide the form
                    if st.button("Cancel", key=f"cancel_attendance_{student_id}"):
                        st.session_state[f"attendance_form_shown_{student_id}"] = False

        else:
            st.write("No registered students found.")
            
        st.markdown("---")
        # Helper function to style paragraphs
        def styled_paragraph(text, style):
            return Paragraph(text, style)

        def generate_pdf(students, start_date, end_date, admin_details, total_classes_in_semester, total_periods_in_semester):
            # Create a file-like buffer to hold the PDF
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
        
            # Styles
            styles = getSampleStyleSheet()
            normal_style = styles['Normal']
            heading_style = styles['Heading1']
        
            # PDF content
            content = []
        
            # Admin details
            content.append(Paragraph(f"Generated by: {admin_details.get('name', 'Unknown')}", normal_style))
            content.append(Paragraph(f"Department: {admin_details.get('department', 'Not Present')}", normal_style))
            content.append(Paragraph(f"Designation: {admin_details.get('designation', 'Not Present')}", normal_style))
            content.append(Paragraph(f"Email: {admin_details.get('email', 'Unknown')}", normal_style))
            content.append(Paragraph(f"Phone: {admin_details.get('phone', 'Not Present')}", normal_style))
            content.append(Paragraph(f"Date Range: {start_date} to {end_date}", normal_style))
        
            # Semester details
            content.append(Paragraph(f"Semester: {start_date.year} - {end_date.year}", normal_style))
            content.append(Paragraph(f"Total Class Days: {total_classes_in_semester}", normal_style))
            content.append(Paragraph(f"Total Periods: {total_periods_in_semester}", normal_style))
        
            # Student attendance report
            for student in students:
                content.append(Paragraph(f"Student: {student['name']}", heading_style))
        
                # Table data for attendance
                data = [["Date", "Attendance"]]
                total_present_classes, total_present_periods = 0, 0
                attendance_periods = {}
        
                for record in student['attendance']:
                    record_date = datetime.strptime(record['date'], '%Y-%m-%d').date() if isinstance(record['date'], str) else record['date']
                    if start_date <= record_date <= end_date:
                        data.append([record_date.strftime('%Y-%m-%d'), "Present" if record['status'] else "Absent"])
                        if record['status']:
                            total_present_classes += 1
                            total_present_periods += 1
        
                        period_key = record_date.strftime('%Y-%m')
                        if period_key not in attendance_periods:
                            attendance_periods[period_key] = {"present": 0, "total": 0}
                        attendance_periods[period_key]["total"] += 1
                        if record['status']:
                            attendance_periods[period_key]["present"] += 1
        
                # Add attendance table
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                content.append(table)
        
                # Attendance summary
                attendance_percentage_classes = (total_present_classes / total_classes_in_semester) * 100 if total_classes_in_semester else 0
                attendance_percentage_periods = (total_present_periods / total_periods_in_semester) * 100 if total_periods_in_semester else 0
                pass_fail_classes = "Pass" if attendance_percentage_classes >= 75 else "Fail"
                pass_fail_periods = "Pass" if attendance_percentage_periods >= 75 else "Fail"
        
                content.append(Paragraph(f"Attendance Percentage (Class days): {attendance_percentage_classes:.2f}%", normal_style))
                content.append(Paragraph(f"Pass/Fail Status (Class days): {pass_fail_classes}", normal_style))
                content.append(Paragraph(f"Attendance Percentage (Periods): {attendance_percentage_periods:.2f}%", normal_style))
                content.append(Paragraph(f"Pass/Fail Status (Periods): {pass_fail_periods}", normal_style))
        
                # Period-wise attendance table
                period_data = [["Period", "Present", "Total"]]
                for period, values in attendance_periods.items():
                    period_data.append([period, values["present"], values["total"]])
        
                period_table = Table(period_data)
                period_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                content.append(period_table)
        
                # Generate attendance chart
                fig, ax = plt.subplots(figsize=(6, 4))
                periods = list(attendance_periods.keys())
                presents = [attendance_periods[period]["present"] for period in periods]
                totals = [attendance_periods[period]["total"] for period in periods]
        
                ax.bar(periods, presents, label="Present", alpha=0.7, color='g')
                ax.bar(periods, [totals[i] - presents[i] for i in range(len(periods))], label="Absent", alpha=0.7, color='r')
                ax.set_xlabel('Period')
                ax.set_ylabel('Attendance Count')
                ax.set_title(f"Attendance Analysis for {student['name']}")
                ax.legend()
        
                # Save chart to buffer
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format="png")
                img_buffer.seek(0)
                plt.close(fig)  # Close the figure to prevent memory leaks
        
                # Append image to PDF content
                content.append(Paragraph("<br/>", normal_style))
                content.append(PlatypusImage(img_buffer, width=400, height=300))
        
            # Build PDF
            doc.build(content)
            buffer.seek(0)
            return buffer
            
        # Section to select students and time period
        st.subheader("Generate Attendance Report (PDF)")

        # Admin details (fetched from the database)
        cursor.execute("SELECT * FROM admin_profile WHERE admin_id = '1'")  # Fetch admin details based on admin_id
        admin_data = cursor.fetchone()

        # Ensure admin_data is fetched correctly
        if admin_data:
            admin_details = {
                "name": admin_data['name'],
                "department": admin_data['department'],
                "designation": admin_data['designation'],
                "email": admin_data['email'],
                "phone": admin_data['phone']
            }
        else:
            admin_details = {
                "name": "Unknown",
                "department": "Unknown",
                "designation": "Unknown",
                "email": "Unknown",
                "phone": "Unknown"
            }

        # Filter by time period (start and end date)
        start_date = st.date_input("Select Start Date", date(2024, 1, 1))
        end_date = st.date_input("Select End Date", date(2024, 12, 31))

        # Dropdown to select semester and year
        cursor.execute("SELECT DISTINCT year, semester FROM semester_dates")
        semester_data = cursor.fetchall()
        semesters = [(f"Year: {item[0]}, Semester: {item[1]}", item[0], item[1]) for item in semester_data]
        selected_semester = st.selectbox("Select Semester", semesters, format_func=lambda x: x[0])

        # Ensure selected_semester has a valid value
        if selected_semester:
            # Unpack year and semester from the selected value
            year, semester = selected_semester[1], selected_semester[2]
            
            # Fetch the total_classes and total_periods for the selected semester
            cursor.execute("""
                SELECT total_classes, total_periods 
                FROM semester_dates 
                WHERE year = ? AND semester = ?
            """, (year, semester))
            semester_details = cursor.fetchone()

            if semester_details:
                total_classes_in_semester = semester_details[0]  # First element in tuple
                total_periods_in_semester = semester_details[1]  # Second element in tuple
            else:
                total_classes_in_semester = 0
                total_periods_in_semester = 0
        else:
            st.error("Please select a valid semester.")


        # Dropdown to select students
        cursor.execute("SELECT * FROM students")
        students_data = cursor.fetchall()
        student_names = [f"{student[2]} ({student[0]})" for student in students_data]
        selected_students = st.multiselect("Select Students", student_names)

        # Button to generate PDF
        if st.button("Generate Attendance Report"):
            # Filter the selected students' data
            students = []
            for student in students_data:
                if f"{student[2]} ({student[0]})" in selected_students:
                    student_id = student[0]
                    cursor.execute("SELECT * FROM attendance WHERE student_id = ?", (student_id,))
                    attendance_records = cursor.fetchall()
                    attendance = [{"date": record[1], "status": sum(int(x) if str(x).isdigit() else 0 for x in record[3:]) > 0} for record in attendance_records]
                    students.append({
                        "name": student[2],
                        "attendance": attendance
                    })
            
            if students:
                # Generate and download the PDF
                pdf_buffer = generate_pdf(students, start_date, end_date, admin_details, total_classes_in_semester, total_periods_in_semester)
                
                # Allow the admin to download the PDF
                st.download_button(
                    label="Download Attendance Report",
                    data=pdf_buffer,
                    file_name="attendance_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("No students selected or no attendance records available for the selected period.")

                
        st.markdown("---")    
        # Function to calculate the number of weekdays (excluding weekends) between two dates
        def calculate_weekdays(start_date, end_date):
            weekdays = 0
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # 0-4 are weekdays (Monday to Friday)
                    weekdays += 1
                current_date += timedelta(days=1)
            return weekdays

        # Function to calculate the total number of periods (assuming 7 periods per weekday)
        def calculate_total_periods(start_date, end_date, total_holidays):
            weekdays = calculate_weekdays(start_date, end_date)
            total_periods = weekdays * 7  # Assuming 7 periods per day
            total_periods -= total_holidays * 7  # Subtract holidays (7 periods per holiday)
            return total_periods

        # Function to display the form for adding start and end times for the selected semester
        def display_semester_form():
            st.title("Add Semester Details")

            # Dropdown for selecting Year (1-4)
            year = st.selectbox("Select Year", [1, 2, 3, 4])

            # Dropdown for selecting Semester (1 or 2)
            semester = st.selectbox("Select Semester", [1, 2, 3, 4, 5, 6, 7, 8])

            # Input fields for start date, end date, and total holidays
            start_date = st.date_input(f"Start Date (Year {year}, Semester {semester})")
            end_date = st.date_input(f"End Date (Year {year}, Semester {semester})")
            total_holidays = st.number_input(f"Total Holidays (Year {year}, Semester {semester})", min_value=0, value=0)

            # Button to save the semester details
            if st.button(f"Save Semester {semester} for Year {year}"):

                # Ensure that start_date is before end_date
                if start_date > end_date:
                    st.error("Start date cannot be after end date.")
                    return
                
                total_periods = calculate_total_periods(start_date, end_date, total_holidays)
                total_classes = total_periods // 7  # Assuming 7 periods per class

                # Save the data to the database
                try:
                    cursor.execute("""
                        INSERT INTO semester_dates (year, semester, start_date, end_date, total_holidays, total_classes, total_periods)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (year, semester, start_date, end_date, total_holidays, total_classes, total_periods))
                    conn.commit()

                    # Show confirmation
                    st.success(f"Semester {semester} for Year {year} saved successfully with {total_classes} class days and {total_periods} periods or classes.")
                    st.write(f"Total Class days: {total_classes}, Total Periods or classes: {total_periods}")
                except Exception as e:
                    st.error(f"Error saving semester details: {e}")

        # Call the function to display the form
        display_semester_form()
        
        st.markdown("---")
        # Create a Streamlit interface
        st.title("📷 One-Shot Attendance System")
        st.write("Click the button below to start face detection and mark attendance automatically.")
        
        # 📌 Store detected faces in session state
        if "detected_faces" not in st.session_state:
            st.session_state.detected_faces = None
        
        # 🔹 Capture Image
        img_file = st.camera_input("Take a photo")
        
        if img_file:
            nparr = np.frombuffer(img_file.read(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            detected_faces = detect_faces_haar(frame)
        
            if detected_faces:
                st.session_state.detected_faces = detected_faces
                st.success(f"✅ Detected {len(detected_faces)} face(s). Click 'Process Attendance'.")
            else:
                st.session_state.detected_faces = None
                st.warning("⚠ No faces detected. Try again.")
        
        # 🔹 Process Attendance Button
        if st.session_state.detected_faces and st.button("🚀 Process Attendance"):
            matched_students = match_faces_with_db()
            record_attendance_for_batch(matched_students)
            
        st.markdown("---")
        st.title("SEND ONLINE APP NOTIFICATIONS(FOR DESKTOP)")

        # Section for teachers to send notifications
        st.subheader("✉️ Send Custom Notifications (For Teachers)")
        st.components.v1.html(send_custom_notification(), height=150)
        
        # Inject custom CSS for an ultra modern design
        st.markdown(
            """
            <style>
            /* Global styling */
            body {
                background: linear-gradient(135deg, #ece9e6, #ffffff);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            /* Main container styling */
            .main-container {
                background-color: #fff;
                border-radius: 10px;
                padding: 2rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin: 2rem auto;
                max-width: 900px;
            }
            /* Title styling */
            h1 {
                text-align: center;
                color: #333;
                font-size: 2.8rem;
                margin-bottom: 0.5rem;
            }
            h3 {
                color: #555;
                margin-top: 1.5rem;
            }
            /* Notification card styling */
            .notification-card {
                /* Example: a vertical gradient from blue to green */
                background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
                
                /* Make text clearly visible on the gradient */
                color: #FFFFFF;
                
                /* Keep or remove the border-left style as you see fit */
                border-left: 5px solid #4caf50;
                
                /* Padding, margin, and rounding for a “card” look */
                padding: 1.2rem;
                margin-bottom: 1rem;
                border-radius: 8px;
                
                /* Subtle transition effects for hover */
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .notification-card:hover {
                transform: translateX(5px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            /* Custom button styles */
            .custom-btn {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            .custom-btn:hover {
                background-color: #1976D2;
            }
            .clear-btn {
                background-color: #e91e63;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            .clear-btn:hover {
                background-color: #d81b60;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Main container with modern card styling
        with st.container():
            # Header Title
            st.markdown("<h1>Notification Center (for Mobile & Desktop) </h1>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Section: Add New Notification
            st.markdown("<h3>Add a New Notification</h3>", unsafe_allow_html=True)
            new_notif = st.text_input(
                "Enter your notification:",
                placeholder="Type your notification message here...",
                key="notif_input"
            )
            
            # Duration input: value and unit
            duration_value = st.number_input(
                "Enter duration value:",
                min_value=1,
                value=24,
                step=1,
                key="duration_value"
            )
            duration_unit = st.selectbox(
                "Select duration unit:",
                options=["Hours", "Days", "Months"],
                key="duration_unit"
            )
            
            if st.button("Add Notification", key="add_notif_button"):
                if new_notif.strip():
                    # Call the updated function with the proper keyword arguments
                    add_notification(new_notif.strip(), duration_value=duration_value, duration_unit=duration_unit)
                    st.success("Notification added successfully!")
                    st.rerun()  # Refresh the page to show new notification
                else:
                    st.error("Please enter a notification message.")
            
            st.markdown("<hr>", unsafe_allow_html=True)
        
            
            # Section: Display All Notifications
            st.markdown("<h3>All Notifications</h3>", unsafe_allow_html=True)
            notifications = get_notifications()
            
            if notifications:
                if st.button("Clear All Notifications", key="clear_all_notifications"):
                    clear_all_notifications()
                    st.success("All notifications cleared!")
                    st.rerun()
                
                # Display each notification with a modern card look and a removal button
                for notif in notifications:
                    col1, col2 = st.columns([0.85, 0.15])
                    with col1:
                        st.markdown(
                            f"""
                            <div class="notification-card">
                                <p><strong>Notification {notif['id']}:</strong> {notif['message']}</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    with col2:
                        if st.button("Remove", key=f"remove_{notif['id']}"):
                            remove_notification(notif["id"])
                            st.success("Notification removed!")
                            st.rerun()
            else:
                st.info("No notifications yet. Add your first notification above!")
            
            st.markdown("</div>", unsafe_allow_html=True)
        # # Button to start capture and retry mechanism
        # retry = True
        # retry_count = 0  # Counter to make button keys unique

        # while retry:
        #     # Add a button to start the face capture process with a unique key
        #     start_button = st.button("Start Capture / reset field for New Shot", key=f"start_capture_button_{retry_count}")

        #     if start_button:
        #         # Capture and detect faces
        #         captured_faces, face_locations = capture_and_detect_faces()

        #         if captured_faces:
        #             # Cluster detected faces to group them by person
        #             num_people, cluster_labels = cluster_faces(captured_faces)

        #             st.write(f"Detected {len(captured_faces)} faces in total.")
        #             st.write(f"Identified {num_people} distinct people.")

        #             # Match captured faces with the database
        #             matched_students = match_faces_with_database(captured_faces)

        #             if matched_students:
        #                 logging.info(f"Matched Students: {matched_students}")

        #                 # Get current period and mark attendance
        #                 current_period = get_current_period()

        #                 if current_period:
        #                     st.success(f"Attendance for {current_period} is being marked automatically, waiting for confirmation from server!")

        #                     # Define period times for attendance marking
        #                     period_times = {
        #                         "Period 1": ("09:30", "10:20"),
        #                         "Period 2": ("10:20", "11:10"),
        #                         "Period 3": ("11:10", "12:00"),
        #                         "Period 4": ("12:00", "12:50"),
        #                         "Period 5": ("13:20", "14:30"),
        #                         "Period 6": ("14:30", "15:20"),
        #                         "Period 7": ("15:20", "16:10")
        #                     }

        #                     # Initialize attendance data for all periods as False (Absent)
        #                     attendance_data = {period: False for period in period_times.keys()}

        #                     # Mark the current period as present (True)
        #                     attendance_data[current_period] = True

        #                     # Define the days array
        #                     days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        #                     # Automatically set the current day from the system date
        #                     current_day = datetime.now().strftime("%A")  # Get the full weekday name (e.g., "Monday", "Tuesday")

        #                     # Check if today is a weekday
        #                     if current_day in days:
        #                         # Display the day selector with the current day pre-selected
        #                         selected_day = st.selectbox("Select Day", days, index=days.index(current_day))  # Pre-select current day

        #                         for user_id, name in matched_students:
        #                             # Check if attendance record already exists for the student
        #                             cursor.execute("""
        #                                 SELECT * FROM attendance WHERE student_id = ? AND date = ? AND day = ?
        #                             """, (user_id, datetime.now().strftime('%Y-%m-%d'), selected_day))
        #                             existing_record = cursor.fetchone()

        #                             if existing_record:
        #                                 # Update the attendance for the current period
        #                                 period_column = f"period_{list(period_times.keys()).index(current_period) + 1}"
        #                                 cursor.execute(f"""
        #                                     UPDATE attendance 
        #                                     SET {period_column} = 1
        #                                     WHERE student_id = ? AND date = ? AND day = ?
        #                                 """, (user_id, datetime.now().strftime('%Y-%m-%d'), selected_day))
        #                                 conn.commit()
        #                                 st.success(f"Attendance updated for {name} in {current_period} on {selected_day}!")
        #                             else:
        #                                 # Insert new attendance record
        #                                 cursor.execute("""
        #                                     INSERT INTO attendance (student_id, date, day, period_1, period_2, period_3, period_4, period_5, period_6, period_7)
        #                                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        #                                 """, (user_id, datetime.now().strftime('%Y-%m-%d'), selected_day, *attendance_data.values()))
        #                                 conn.commit()
        #                                 st.success(f"Attendance for {name} marked successfully in {current_period} on {selected_day}!")
        #                                 st.write("Face capture process completed.")
        #                     else:
        #                         # Display a warning if today is a weekend
        #                         st.warning(f"No attendance marking allowed on {current_day} as it is a holiday.")
        #                         st.write("Face capture process completed.")
        #                 else:
        #                     st.warning("No active class period at the moment.")
        #                     st.write("Face capture process completed.")
        #             else:
        #                 logging.info("No matches found.")
        #                 st.write("Face capture process completed.")
        #         else:
        #             logging.info("No faces detected or processed.try again!")
        #             st.write("Face capture process completed.")

        #     # Option to retry capturing faces if needed with a unique key
        #     retry_count += 1  # Increment retry counter for unique key
        #     retry = st.button("enhanced scanning coming soon!!", key=f"retry_capture_button_{retry_count}")

        #     if not retry:
        #         st.write("mark attendance of upto 30 people!")
                
        st.markdown("---")       
        # Streamlit interface
        st.title("Debarred Students - Attendance Alert System")

        # Fetch students with attendance below 75%
        low_attendance_students = get_low_attendance_students()

        if low_attendance_students:
            st.subheader("Debarred Students")
            for student in low_attendance_students:
                user_id, name, email, attendance_percentage = student
                st.write(f"Name: {name}, Email: {email}, Attendance: {attendance_percentage:.2f}%")
                
                # Button to send alert email
                if st.button(f"Alert {name}", key=f"alert_{user_id}"):
                    send_email(email, name, attendance_percentage)
        else:
            st.info("No students have attendance below the threshold.")
            
        st.markdown("---")   
         # Notification Center for Disputes
        st.subheader("Dispute Notifications")
        st.info("Below are the pending disputes raised by students.")

        # Fetch pending disputes with student details
        cursor.execute("""
            SELECT d.id, d.student_id, s.name, s.enrollment_no, s.roll, d.date, d.reason 
            FROM disputes d
            JOIN students s ON d.student_id = s.user_id
            WHERE d.status = 'Pending'
        """)
        disputes = cursor.fetchall()

        if disputes:
            for dispute in disputes:
                dispute_id, student_id, student_name, enrollment_no, roll_no, date, reason = dispute
                st.write(f"**Dispute ID:** {dispute_id}")
                st.write(f"**Student ID:** {student_id}")
                st.write(f"**Name:** {student_name}")
                st.write(f"**Enrollment No.:** {enrollment_no}")
                st.write(f"**Roll No.:** {roll_no}")
                st.write(f"**Date:** {date}")
                st.write(f"**Reason:** {reason}")

                # Button to mark the dispute as resolved
                if st.button(f"Resolve Dispute {dispute_id}"):
                    cursor.execute("UPDATE disputes SET status = 'Resolved' WHERE id = ?", (dispute_id,))
                    conn.commit()
                    st.success(f"Dispute ID {dispute_id} marked as resolved.")
        else:
            st.write("No pending disputes.")
            
        st.markdown("---")
        # Timetable Entry Form
        DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        PERIODS = ["Period 1", "Period 2", "Period 3", "Period 4", "Period 5", "Period 6", "Period 7"]

        def get_timetable(day):
            cursor.execute("""
                SELECT period, subject, teacher FROM timetable WHERE day = ? ORDER BY period
            """, (day,))
            return cursor.fetchall()

        def add_timetable_entry(day, period, subject, teacher):
            try:
                cursor.execute("""
                    SELECT * FROM timetable WHERE day = ? AND period = ?
                """, (day, period))
                existing_entry = cursor.fetchone()

                if existing_entry:
                    st.error(f"An entry already exists for {day} - {period}.") 
                    return False

                cursor.execute("""
                    INSERT INTO timetable (day, period, subject, teacher) 
                    VALUES (?, ?, ?, ?)
                """, (day, period, subject, teacher))
                conn.commit()
                return True
            except Exception as e:
                st.error(f"Error adding entry: {e}") 
                return False

        def update_timetable_entry(day, period, subject, teacher):
            try:
                cursor.execute("""
                    UPDATE timetable SET subject = ?, teacher = ? 
                    WHERE day = ? AND period = ?
                """, (subject, teacher, day, period))
                conn.commit()
                st.success(f"Timetable entry for {day} - {period} updated.")
                return True
            except Exception as e:
                st.error(f"Error updating entry: {e}") 
                return False

        def delete_timetable_entry(day, period):
            try:
                cursor.execute("""
                    DELETE FROM timetable WHERE day = ? AND period = ?
                """, (day, period))
                conn.commit()
                st.success(f"Timetable entry for {day} - {period} deleted.")
                return True
            except Exception as e:
                st.error(f"Error deleting entry: {e}") 
                return False

        st.title("Timetable Management")

        with st.form("add_timetable"):
            st.subheader("Add Timetable Entry")
            day = st.selectbox("Select Day", DAYS_OF_WEEK)
            period = st.selectbox("Select Period", PERIODS)
            subject = st.text_input("Subject Name")
            teacher = st.text_input("Teacher Name")
            submit_add = st.form_submit_button("Add Entry")

            if submit_add:
                if subject and teacher:
                    if add_timetable_entry(day, period, subject, teacher):
                        st.success(f"Timetable entry added: {day} - {period} - {subject} - {teacher}")

        with st.form("update_timetable"):
            st.subheader("Update Timetable Entry")
            day = st.selectbox("Select Day", DAYS_OF_WEEK)
            period = st.selectbox("Select Period", PERIODS)
            subject = st.text_input("Subject Name")
            teacher = st.text_input("Teacher Name")
            submit_update = st.form_submit_button("Update Entry")

            if submit_update:
                if subject and teacher:
                    if update_timetable_entry(day, period, subject, teacher):
                        st.success(f"Timetable entry updated: {day} - {period} - {subject} - {teacher}")

        with st.form("delete_timetable"):
            st.subheader("Delete Timetable Entry")
            day = st.selectbox("Select Day", DAYS_OF_WEEK)
            period = st.selectbox("Select Period", PERIODS)
            submit_delete = st.form_submit_button("Delete Entry")

            if submit_delete:
                if st.button("Confirm Delete"): 
                    if delete_timetable_entry(day, period):
                        st.success(f"Timetable entry for {day} - {period} deleted.")

        st.header("View Timetable")
        selected_day = st.selectbox("Select Day to View", DAYS_OF_WEEK)
        if selected_day:
            timetable = get_timetable(selected_day) 
            if timetable:
                st.table(timetable)
            else:
                st.warning(f"No timetable found for {selected_day}.")
            
        # Logout Button
        if st.button("Logout"):
            st.session_state.logged_in = False  # Reset login state
            st.session_state.admin_id = None  # Clear stored admin ID
            st.session_state.admin_name = None  # Clear stored admin name
            st.session_state.profile_data = None  # Clear profile data
            st.rerun()  # Refresh the page after logout

    # Admin Login with Face Verification
    else:
        # Admin Login Form
        st.header("Admin Login")
        with st.form("Login Form"):
            admin_id = st.text_input("Admin ID", key="login_admin_id")
            admin_password = st.text_input("Admin Password", type="password", key="login_admin_password")
        
            # Face Capture using Streamlit's camera_input
            captured_face = st.camera_input("Capture your face for verification")
        
            # Session State Initialization for OTP-related session management
            if "otp" not in st.session_state:
                st.session_state.otp = None
            if "otp_verified" not in st.session_state:
                st.session_state.otp_verified = False
            if "reset_admin_id" not in st.session_state:
                st.session_state.reset_admin_id = None
        
            # Admin Login Button
            submit_login = st.form_submit_button("Login")
        
            if submit_login:
                # Check if the admin exists in the database (admin table)
                cursor.execute("SELECT * FROM admin WHERE admin_id = ?", (admin_id,))
                admin = cursor.fetchone()
        
                if admin:
                    admin_db_password = admin[1]  # Assuming the 2nd column is the password
                    active_status = admin[2]  # Assuming the 3rd column is the active status
        
                    if admin_id == "admin":
                        # Skip face verification for the admin user and directly check password
                        if admin_password == admin_db_password:
                            if active_status == 0:  # Account is deactivated
                                st.error("Your account has been deactivated. Please contact the system administrator.")
                            else:
                                # Admin login successful
                                st.session_state.logged_in = True
                                st.session_state.admin_id = admin_id
                                st.success("Login successful!")
                                st.rerun()  # Refresh the page to show the admin dashboard
                        else:
                            st.error("Invalid admin ID or password.")
                    else:
                        # Perform face verification and spoof check for other admin IDs
                        if captured_face is None:
                            st.error("Please capture your face for verification.")
                        else:
                            # Convert the captured face image to binary as done in your example
                            img = Image.open(captured_face)
                            img_bytes = io.BytesIO()
                            img.save(img_bytes, format="JPEG")
                            captured_face_blob = img_bytes.getvalue()
        
                            # Save the captured image temporarily for anti-spoofing check
                            captured_face_path = "captured_face.jpg"
                            with open(captured_face_path, "wb") as f:
                                f.write(captured_face_blob)
        
                            # Anti-spoofing check
                            if not detect_spoof(captured_face_path):
                                st.error("Spoofed image detected. Please use a real face photo.")
                                st.stop()  # Stop execution
        
                            # Save the captured image temporarily for DeepFace comparison
                            captured_face_path = "captured_face.jpg"
                            with open(captured_face_path, "wb") as f:
                                f.write(captured_face_blob)
        
                            # Verify face using DeepFace if spoof detection passes
                            cursor.execute("SELECT face_encoding FROM admin_profile WHERE admin_id = ?", (admin_id,))
                            stored_face_blob = cursor.fetchone()
        
                            if stored_face_blob:
                                stored_face_blob = stored_face_blob[0]  # Extract the BLOB value
                                # Save the stored face temporarily for DeepFace comparison
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as stored_face_file:
                                    stored_face_file.write(stored_face_blob)
                                    stored_face_path = stored_face_file.name
        
                                # Use DeepFace to verify the face similarity
                                result = DeepFace.verify(captured_face_path, stored_face_path, model_name="Facenet", distance_metric="cosine")
                                
                                st.write(f"Face match result: {result}")
                                similarity_score = result["distance"]
                                st.write(f"Similarity score: {similarity_score}")
        
                                # Threshold for similarity score (lower means more similar)
                                THRESHOLD = 0.7
                                if result["verified"] and similarity_score < THRESHOLD:
                                    # Check password from the admin table
                                    if admin_password == admin_db_password:
                                        if active_status == 0:  # Account is deactivated
                                            st.error("Your account has been deactivated. Please contact the system administrator.")
                                        else:
                                            # Admin login successful
                                            st.session_state.logged_in = True
                                            st.session_state.admin_id = admin_id
                                            st.success("Login successful!")
                                            st.rerun()  # Refresh the page to show the admin dashboard
                                    else:
                                        st.error("Invalid admin ID or password.")
                                else:
                                    st.error("Face verification failed. Please try again.")
                            else:
                                st.error("Face image not found in the database.")
                else:
                    st.error("Admin ID not found in the database.")
            
        # Forgot Password Form (Separate Form)
        st.header("Forgot Password?")
        with st.form("Forgot Password Form"):
            reset_admin_id = st.text_input("Enter your Admin ID to reset password", key="reset_admin_id_input")
            submit_forgot_password = st.form_submit_button("Generate OTP")
        
            if submit_forgot_password:
                cursor.execute("SELECT email FROM admin_profile WHERE admin_id = ?", (reset_admin_id,))
                admin_email = cursor.fetchone()
        
                if admin_email:
                    admin_email = admin_email[0]
                    otp = str(np.random.randint(100000, 999999))  # Generate 6-digit OTP
                    st.session_state.otp = otp  # Store OTP in session state
                    st.session_state.reset_admin_id = reset_admin_id  # Save the admin ID for verification
        
                    # SMTP Configuration
                    smtp_server = 'smtp-relay.brevo.com'
                    smtp_port = 587
                    smtp_user = '823c6b001@smtp-brevo.com'
                    smtp_password = '6tOJHT2F4x8ZGmMw'
                    from_email = 'debojyotighoshmain@gmail.com'
        
                    # Send OTP to email
                    try:
                        message = MIMEMultipart()
                        message["From"] = from_email
                        message["To"] = admin_email
                        message["Subject"] = "Your OTP for Admin Login"
        
                        body = f"Your OTP for Admin login is: {otp}\n\nPlease use this OTP to complete your login."
                        message.attach(MIMEText(body, "plain"))
        
                        with smtplib.SMTP(smtp_server, smtp_port) as server:
                            server.starttls()
                            server.login(smtp_user, smtp_password)
                            server.sendmail(from_email, admin_email, message.as_string())
        
                        st.success(f"OTP sent to {admin_email}. Please check your email.")
                    except Exception as e:
                        st.error(f"Error sending OTP: {e}")
                else:
                    st.error("Admin ID not found in the database.")
        
        # OTP Verification Form (Separate Form)
        if st.session_state.otp:
            with st.form("OTP Verification Form"):
                st.subheader("OTP Verification")
                otp_input = st.text_input("Enter the OTP sent to your email", key="otp_input")
                submit_otp = st.form_submit_button("Verify OTP")
        
                if submit_otp:
                    if otp_input == st.session_state.otp:
                        st.success("OTP verified successfully!")
                        st.session_state.otp_verified = True  # Mark OTP as verified
                    else:
                        st.error("Invalid OTP. Please try again.")


                
# Section for Admin Management
elif menu == "Admin Management":
    st.header("Admin Account Management")

    # Session state for OTP, Admin login, and Admin credentials
    if "admin_otp" not in st.session_state:
        st.session_state.admin_otp = None
    if "otp_verified" not in st.session_state:
        st.session_state.otp_verified = False
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    # Admin Login (Admin ID & Password)
    if not st.session_state.admin_authenticated:
        st.subheader("Admin Login")

        admin_id = st.text_input("Enter Admin ID")
        admin_password = st.text_input("Enter Admin Password", type="password")

        if st.button("Login"):
            cursor.execute("SELECT * FROM admin WHERE admin_id = ? AND password = ?", (admin_id, admin_password))
            admin_data = cursor.fetchone()
            if admin_data:
                st.session_state.admin_authenticated = True
                st.session_state.logged_in_admin_id = admin_id  # Store logged-in admin ID
                st.success("Login successful.")
            else:
                st.error("Invalid Admin ID or Password.")

    # Admin Management (only available after successful login)
    if st.session_state.admin_authenticated:
        st.subheader("Registered Admin Accounts")

        # Fetch all registered admin accounts
        cursor.execute(""" 
        SELECT a.admin_id, a.active, ap.email 
        FROM admin a 
        LEFT JOIN admin_profile ap ON a.admin_id = ap.admin_id
        """)
        admins = cursor.fetchall()

        if admins:
            for admin in admins:
                admin_id, active, email = admin
                status = "Active" if active else "Inactive"

                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                with col1:
                    st.text(f"Admin ID: {admin_id}")
                with col2:
                    st.text(f"Status: {status}")
                with col3:
                    if st.button(f"Request OTP for {admin_id}") and email:
                        # Check if the admin ID is 'admin' (the logged-in admin ID)
                        if admin_id == "admin":
                            # Send OTP to logged-in admin's email
                            logged_in_admin_email = st.session_state.logged_in_admin_id

                            cursor.execute("SELECT email FROM admin_profile WHERE admin_id = ?", (logged_in_admin_email,))
                            logged_in_admin_email_result = cursor.fetchone()
                            logged_in_admin_email = logged_in_admin_email_result[0] if logged_in_admin_email_result else None
                        else:
                            # Send OTP to the respective admin's email
                            logged_in_admin_email = email

                        if logged_in_admin_email:
                            otp = str(np.random.randint(100000, 999999))
                            st.session_state.admin_otp = otp

                            # SMTP Configuration
                            smtp_server = 'smtp-relay.brevo.com'
                            smtp_port = 587
                            smtp_user = '823c6b001@smtp-brevo.com'
                            smtp_password = '6tOJHT2F4x8ZGmMw'
                            from_email = 'debojyotighoshmain@gmail.com'

                            try:
                                message = MIMEMultipart()
                                message["From"] = from_email
                                message["To"] = logged_in_admin_email
                                message["Subject"] = "Admin Action OTP Verification"

                                body = f"Your OTP for the requested admin action is: {otp}\n\nPlease enter this OTP to verify your request."
                                message.attach(MIMEText(body, "plain"))

                                with smtplib.SMTP(smtp_server, smtp_port) as server:
                                    server.starttls()
                                    server.login(smtp_user, smtp_password)
                                    server.sendmail(from_email, logged_in_admin_email, message.as_string())

                                st.success(f"OTP sent to your email {logged_in_admin_email}. Please check your email.")
                            except Exception as e:
                                st.error(f"Failed to send OTP: {e}")
                        else:
                            st.error("No email found for this admin. Cannot send OTP.")

                    # Verify OTP
                    otp_input = st.text_input(f"Enter OTP for {admin_id}", key=f"otp_{admin_id}")
                    if st.button(f"Verify OTP for {admin_id}"):
                        if otp_input == st.session_state.admin_otp:
                            st.session_state.otp_verified = True
                            st.success("OTP verified successfully!")
                        else:
                            st.error("Invalid OTP. Please try again.")

                    # Activate/Deactivate/Remove buttons (only enabled after OTP verification)
                    if st.session_state.otp_verified:
                        if active:
                            if st.button(f"Deactivate {admin_id}"):
                                cursor.execute("UPDATE admin SET active = 0 WHERE admin_id = ?", (admin_id,))
                                cursor.execute("INSERT INTO admin_audit (admin_id, action, timestamp) VALUES (?, 'Deactivated', ?)", (admin_id, datetime.now()))
                                conn.commit()
                                st.success(f"Admin account '{admin_id}' has been deactivated.")
                        else:
                            if st.button(f"Activate {admin_id}"):
                                cursor.execute("UPDATE admin SET active = 1 WHERE admin_id = ?", (admin_id,))
                                cursor.execute("INSERT INTO admin_audit (admin_id, action, timestamp) VALUES (?, 'Activated', ?)", (admin_id, datetime.now()))
                                conn.commit()
                                st.success(f"Admin account '{admin_id}' has been activated.")

                        if st.button(f"Remove {admin_id}"):
                            cursor.execute("DELETE FROM admin WHERE admin_id = ?", (admin_id,))
                            cursor.execute("DELETE FROM admin_profile WHERE admin_id = ?", (admin_id,))
                            cursor.execute("INSERT INTO admin_audit (admin_id, action, timestamp) VALUES (?, 'Removed', ?)", (admin_id, datetime.now()))
                            conn.commit()
                            st.success(f"Admin account '{admin_id}' and its profile have been removed.")
        else:
            st.info("No admin accounts found.")

        # Activity History Section
        st.subheader(f"Activity History for Admin: {st.session_state.logged_in_admin_id}")

        # Fetch the audit trail for the logged-in admin
        cursor.execute("""
        SELECT action, timestamp FROM admin_audit
        WHERE admin_id = ?
        ORDER BY timestamp DESC
        """, (st.session_state.logged_in_admin_id,))
        activities = cursor.fetchall()

        if activities:
            for activity in activities:
                action, timestamp = activity
                st.write(f"**Action:** {action}")
                st.write(f"**Timestamp:** {timestamp}")
                st.write("-" * 50)
        else:
            st.info("No activity history found for this admin.")

        # Option to reset the default admin account if needed
        st.subheader("Restore Default Admin Account")
        if st.button("Request OTP for Restore Default Admin"):
            # Generate OTP for restoring the default admin account
            otp = str(np.random.randint(100000, 999999))
            st.session_state.admin_otp = otp

            # Fetch the logged-in admin's email
            cursor.execute("SELECT email FROM admin_profile WHERE admin_id = ?", (st.session_state.logged_in_admin_id,))
            admin_email = cursor.fetchone()
            admin_email = admin_email[0] if admin_email else None

            if admin_email:
                # Send OTP to the logged-in admin
                try:
                    message = MIMEMultipart()
                    message["From"] = from_email
                    message["To"] = admin_email
                    message["Subject"] = "Admin Action OTP Verification"

                    body = f"Your OTP for restoring the default admin account is: {otp}\n\nPlease enter this OTP to verify your request."
                    message.attach(MIMEText(body, "plain"))

                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls()
                        server.login(smtp_user, smtp_password)
                        server.sendmail(from_email, admin_email, message.as_string())

                    st.success(f"OTP sent to your email {admin_email}. Please check.")
                except Exception as e:
                    st.error(f"Failed to send OTP: {e}")
            else:
                st.error("No email found for the logged-in admin.")

        otp_input_restore = st.text_input("Enter OTP to Restore Default Admin")
        if st.button("Verify OTP to Restore"):
            if otp_input_restore == st.session_state.admin_otp:
                st.session_state.otp_verified = True
                st.success("OTP verified successfully!")
            else:
                st.error("Invalid OTP. Please try again.")

        if st.session_state.otp_verified:
            if st.button("Restore Default Admin"):
                cursor.execute(""" 
                INSERT OR IGNORE INTO admin (admin_id, password, active) 
                VALUES ('admin', 'admin123', 1)
                """)
                cursor.execute(""" 
                INSERT OR IGNORE INTO admin_profile (admin_id, name, department, designation, email, phone, face_encoding) 
                VALUES ('admin', 'Default Admin', 'IT', 'Administrator', 'defaultadmin@example.com', '1234567890', NULL)
                """)
                cursor.execute("INSERT INTO admin_audit (admin_id, action, timestamp) VALUES ('admin', 'Restored', ?)", (datetime.now(),))
                conn.commit()
                st.success("Default admin account has been restored.")

elif menu == "Lab Examination System" : 
    st.title("AI-Enabled Lab Examination System")
    st.title("AILES")
    st.success("Welcome to the Lab Examination System. Get started with your lab journey!")
    st.info("This system allows you to conduct lab examinations for students.")
    st.info("You can add questions, conduct exams, and view results here with a lot of patented features.")
    # Add a button with a modern design
    st.markdown(
        """
        <style>
        .redirect-button {
            display: inline-block;
            background-color:rgb(16, 55, 117);
            color:rgb(17, 42, 143);
            font-size: 18px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s ease;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .redirect-button:hover {
            background-color:rgb(136, 41, 173);
            text-decoration: none;
        }
        </style>
        <a href="https://les-beta.netlify.app/" target="_blank" class="redirect-button">Let's Go!</a>
        """,
        unsafe_allow_html=True,
    )

elif menu == "Teacher's Registration" :
    st.title("Teacher's Registration")
    st.info("comming soon!!")

elif menu == "Notification Center" :

    # Inject custom CSS for an ultra modern design
    st.markdown(
        """
        <style>
        /* Global styling */
        body {
            background: linear-gradient(135deg, #ece9e6, #ffffff);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Main container styling */
        .main-container {
            background-color: #fff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin: 2rem auto;
            max-width: 900px;
        }
        /* Title styling */
        h1 {
            text-align: center;
            color: #333;
            font-size: 2.8rem;
            margin-bottom: 0.5rem;
        }
        h3 {
            color: #555;
            margin-top: 1.5rem;
        }
        /* Notification card styling */
        .notification-card {
            /* Example: a vertical gradient from blue to green */
            background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
            
            /* Make text clearly visible on the gradient */
            color: #FFFFFF;
            
            /* Keep or remove the border-left style as you see fit */
            border-left: 5px solid #4caf50;
            
            /* Padding, margin, and rounding for a “card” look */
            padding: 1.2rem;
            margin-bottom: 1rem;
            border-radius: 8px;
            
            /* Subtle transition effects for hover */
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .notification-card:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        /* Custom button styles */
        .custom-btn {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .custom-btn:hover {
            background-color: #1976D2;
        }
        .clear-btn {
            background-color: #e91e63;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .clear-btn:hover {
            background-color: #d81b60;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Main container with modern card styling
    with st.container():
        st.markdown("<hr>", unsafe_allow_html=True)
                    
        # Section: Display All Notifications
        st.markdown("<h3>All Notifications</h3>", unsafe_allow_html=True)
        notifications = get_notifications()
        
        if notifications:
            # Display each notification with a modern card look
            for notif in notifications:
                st.markdown(
                    f"""
                    <div class="notification-card">
                        <p><strong>Notification {notif['id']}:</strong> {notif['message']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
        else:
            st.info("No notifications yet!")
        
        st.markdown("</div>", unsafe_allow_html=True)
