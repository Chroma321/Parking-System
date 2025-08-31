import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import time
import os
import mysql.connector
from datetime import datetime
import threading

class ExitCameraANPR:
    def __init__(self, camera_source=1):
        # Print version info
        print("=== EXIT CAMERA INITIALIZED ===")
        print("opencv version:", cv2.__version__)
        print("ultralytics version:", YOLO._version)
        print("easyocr version:", easyocr.__version__)
        print("numpy version:", np.__version__)
        print("mysql-connector-python version:", mysql.connector.__version__)
        
        # Camera source configuration
        self.camera_source = camera_source
        self.camera_type = "EXIT"
        
        # Setup directories
        base_dir = os.path.abspath(os.path.dirname(__file__))
        project_dir = os.path.dirname(os.path.dirname(base_dir))
        self.image_dir = os.path.join(project_dir, "Captured Image", "Exit")
        self.log_dir = os.path.join(project_dir, "HasilDeteksi")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, "Exit_Captured_License.txt")
        
        # Load model and OCR
        model_path = os.path.join(project_dir, "yolov10", "runs", "detect", "train10", "weights", "best.pt")
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en', 'id'])
        self.detection_cooldown = 5.0
        self.should_stop = False
        
    def log_exit_access(self, plate_number):
        """Log license plate exit to database"""
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
                VALUES (%s, %s, 'exit', 'main_exit', %s)
            """
            cursor.execute(query, (plate_number, status, datetime.now()))
            
            # Find and complete the active session
            cursor.execute("""
                SELECT id, entry_time FROM vehicle_sessions 
                WHERE plate_number = %s AND status = 'active'
                ORDER BY entry_time DESC LIMIT 1
            """, (plate_number,))
            
            active_session = cursor.fetchone()
            
            if active_session:
                session_id, entry_time = active_session
                exit_time = datetime.now()
                
                # Calculate duration in minutes
                duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
                
                # Update session as completed
                cursor.execute("""
                    UPDATE vehicle_sessions 
                    SET exit_time = %s, duration_minutes = %s, status = 'completed', updated_at = %s
                    WHERE id = %s
                """, (exit_time, duration_minutes, exit_time, session_id))
                
                print(f"🚪 [EXIT-COMPLETE] {plate_number} session completed - Duration: {duration_minutes} minutes")
            else:
                # No active session found, create incomplete exit record
                session_query = """
                    INSERT INTO vehicle_sessions (plate_number, exit_time, member_status, status)
                    VALUES (%s, %s, %s, 'incomplete')
                """
                cursor.execute(session_query, (plate_number, datetime.now(), status))
                print(f"⚠️ [EXIT-INCOMPLETE] {plate_number} exit without entry record")
            
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ [EXIT-{status.upper()}] {plate_number} exited and logged to database.")
        except Exception as e:
            print(f"Database error (Exit): {e}")

    def detect_from_camera(self):
        """Main detection loop for exit camera"""
        # Camera connection logic
        if isinstance(self.camera_source, str) and self.camera_source.startswith('http'):
            possible_urls = [
                self.camera_source,
                f"{self.camera_source}/video",
                f"{self.camera_source}/videofeed",
                f"{self.camera_source}/mjpg/video.mjpg"
            ]
            
            cap = None
            for url in possible_urls:
                print(f"[EXIT] Trying to connect to: {url}")
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret:
                        print(f"✅ [EXIT] Successfully connected to: {url}")
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    cap = None
                    
            if cap is None:
                print(f"❌ [EXIT] Error: Could not connect to camera: {self.camera_source}")
                return
        else:
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                print(f"❌ [EXIT] Error: Could not open camera: {self.camera_source}")
                return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        print(f"✅ [EXIT] Camera opened successfully")
        
        detection_active = False
        last_detection_time = time.time()
        
        while True:
            if self.should_stop:
                print("[EXIT] Detection stopped by user")
                break
                
            ret, frame = cap.read()
            if not ret:
                print("[EXIT] Error: Failed to read frame")
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
                        print(f"[EXIT] License Plate: {final_text}")
                        
                        # Save to file and database
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                            log_file.write(f"[{timestamp}] [EXIT] License Plate: {final_text}\n")
                        
                        self.log_exit_access(final_text)
                        
                        timestamp_img = time.strftime("%Y%m%d_%H%M%S")
                        image_path = os.path.join(self.image_dir, f"exit_plate_{timestamp_img}.png")
                        cv2.imwrite(image_path, cropped_image)
                        print(f"[EXIT] Image saved: {image_path}")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "EXIT - Plate Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add EXIT label to frame
            cv2.putText(frame, "EXIT CAMERA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Exit Detection", frame)
            
            if detection_active and (current_time - last_detection_time) > self.detection_cooldown:
                detection_active = False
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
