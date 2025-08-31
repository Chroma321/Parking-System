import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import time
import os
import mysql.connector
from datetime import datetime
import threading

class EntryCameraANPR:
    def __init__(self, camera_source=0):
        # Print version info
        print("=== ENTRY CAMERA INITIALIZED ===")
        print("opencv version:", cv2.__version__)
        print("ultralytics version:", YOLO._version)
        print("easyocr version:", easyocr.__version__)
        print("numpy version:", np.__version__)
        print("mysql-connector-python version:", mysql.connector.__version__)
        
        # Camera source configuration
        self.camera_source = camera_source
        self.camera_type = "ENTRY"
        
        # Setup directories
        base_dir = os.path.abspath(os.path.dirname(__file__))
        project_dir = os.path.dirname(os.path.dirname(base_dir))
        self.image_dir = os.path.join(project_dir, "Captured Image", "Entry")
        self.log_dir = os.path.join(project_dir, "HasilDeteksi")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, "Entry_Captured_License.txt")
        
        # Load model and OCR
        model_path = os.path.join(project_dir, "yolov10", "runs", "detect", "train10", "weights", "best.pt")
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en', 'id'])
        self.detection_cooldown = 5.0
        self.should_stop = False
        
    def log_entry_access(self, plate_number):
        """Log license plate entry to database"""
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                database="gate_access"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM member_list WHERE plate_number = %s", (plate_number,))
            member = cursor.fetchone()
            status = 'member' if member else 'guest'
            
            query = """
                INSERT INTO access_log (plate_number, status, event_type, camera_location, timestamp)
                VALUES (%s, %s, 'entry', 'main_entrance', %s)
            """
            cursor.execute(query, (plate_number, status, datetime.now()))
            
            # Check for existing active session
            cursor.execute("""
                SELECT id FROM vehicle_sessions 
                WHERE plate_number = %s AND status = 'active'
                ORDER BY entry_time DESC LIMIT 1
            """, (plate_number,))
            
            existing_session = cursor.fetchone()
            
            if existing_session:
                # Update existing session entry time
                cursor.execute("""
                    UPDATE vehicle_sessions 
                    SET entry_time = %s, updated_at = %s
                    WHERE id = %s
                """, (datetime.now(), datetime.now(), existing_session[0]))
                print(f"🔄 [ENTRY-UPDATE] {plate_number} session updated")
            else:
                # Create new session
                session_query = """
                    INSERT INTO vehicle_sessions (plate_number, entry_time, member_status, status)
                    VALUES (%s, %s, %s, 'active')
                """
                cursor.execute(session_query, (plate_number, datetime.now(), status))
                print(f"🚪 [ENTRY-NEW] {plate_number} new session started")
            
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ [ENTRY-{status.upper()}] {plate_number} entered and logged to database.")
        except Exception as e:
            print(f"Database error (Entry): {e}")

    def detect_from_camera(self):
        """Main detection loop for entry camera"""
        # Camera connection logic (same as before but with entry-specific settings)
        if isinstance(self.camera_source, str) and self.camera_source.startswith('http'):
            possible_urls = [
                self.camera_source,
                f"{self.camera_source}/video",
                f"{self.camera_source}/videofeed",
                f"{self.camera_source}/mjpg/video.mjpg"
            ]
            
            cap = None
            for url in possible_urls:
                print(f"[ENTRY] Trying to connect to: {url}")
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret:
                        print(f"✅ [ENTRY] Successfully connected to: {url}")
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    cap = None
                    
            if cap is None:
                print(f"❌ [ENTRY] Error: Could not connect to camera: {self.camera_source}")
                return
        else:
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                print(f"❌ [ENTRY] Error: Could not open camera: {self.camera_source}")
                return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        print(f"✅ [ENTRY] Camera opened successfully")
        
        detection_active = False
        last_detection_time = time.time()
        
        while True:
            if self.should_stop:
                print("[ENTRY] Detection stopped by user")
                break
                
            ret, frame = cap.read()
            if not ret:
                print("[ENTRY] Error: Failed to read frame")
                break
            
            current_time = time.time()
            if not detection_active and (current_time - last_detection_time) > self.detection_cooldown:
                results = self.model.predict(frame, conf=0.25, verbose=False)
                result = results[0]
                boxes = result.boxes.xyxy
                
                if len(boxes) > 0:
                    detection_active = True
                    last_detection_time = current_time
                    
                    x1, y1, x2, y2 = map(int, boxes[0])
                    cropped_image = frame[y1:y2, x1:x2]
                    
                    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    enhanced = cv2.convertScaleAbs(thresh, alpha=1.2, beta=10)
                    
                    ocr_results = self.reader.readtext(enhanced)
                    combined_texts = []
                    fallback_text = ""
                    min_conf = 0.2
                    
                    if ocr_results:
                        for (_, text, prob) in ocr_results:
                            cleaned_text = text.replace(" ", "").strip().upper()
                            if prob >= min_conf:
                                combined_texts.append(cleaned_text)
                            fallback_text = cleaned_text
                        
                        final_text = "".join(combined_texts) if combined_texts else fallback_text
                        print(f"[ENTRY] License Plate: {final_text}")
                        
                        # Save to file and database
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                            log_file.write(f"[{timestamp}] [ENTRY] License Plate: {final_text}\n")
                        
                        self.log_entry_access(final_text)
                        
                        timestamp_img = time.strftime("%Y%m%d_%H%M%S")
                        image_path = os.path.join(self.image_dir, f"entry_plate_{timestamp_img}.png")
                        cv2.imwrite(image_path, cropped_image)
                        print(f"[ENTRY] Image saved: {image_path}")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "ENTRY - Plate Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add ENTRY label to frame
            cv2.putText(frame, "ENTRY CAMERA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Entry Detection", frame)
            
            if detection_active and (current_time - last_detection_time) > self.detection_cooldown:
                detection_active = False
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
