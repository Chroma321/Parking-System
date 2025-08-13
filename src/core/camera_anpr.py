import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import time
import os
import mysql.connector
from datetime import datetime

class CameraANPR:
    def __init__(self):
        # Print version info
        print("opencv version:", cv2.__version__)
        print("ultralytics version:", YOLO._version)
        print("easyocr version:", easyocr.__version__)
        print("numpy version:", np.__version__)
        print("mysql-connector-python version:", mysql.connector.__version__)
        
        # Setup directories
        base_dir = os.path.abspath(os.path.dirname(__file__))
        project_dir = os.path.dirname(os.path.dirname(base_dir))
        self.image_dir = os.path.join(project_dir, "Captured Image")
        self.log_dir = os.path.join(project_dir, "HasilDeteksi")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, "Captured License.txt")
        
        # Load model and OCR
        model_path = os.path.join(project_dir, "yolov10", "runs", "detect", "train10", "weights", "best.pt")
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en', 'id'])
        self.detection_cooldown = 5.0

    def log_basic_access(self, plate_number):
        """Log license plate access to database"""
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
            INSERT INTO access_log (plate_number, status, timestamp)
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (plate_number, status, datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[{status.upper()}] {plate_number} tercatat ke database.")

    def detect_from_camera(self):
        """Main detection loop from camera"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Kamera tidak dapat dibuka.")
            return
        
        detection_active = False
        last_detection_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Gagal membaca frame.")
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
                        print(f"Teks Plat Nomor : {final_text}")
                        
                        # Save to file and database
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                            log_file.write(f"[{timestamp}] Teks Plat Nomor : {final_text}\n")
                        
                        self.log_basic_access(final_text)
                        
                        timestamp_img = time.strftime("%Y%m%d_%H%M%S")
                        image_path = os.path.join(self.image_dir, f"detected_plate_{timestamp_img}.png")
                        cv2.imwrite(image_path, cropped_image)
                        print(f"Gambar disimpan sebagai: {image_path}")
                    else:
                        print("Tidak ada teks yang terdeteksi.")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Plate Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Live Detection", frame)
            
            if detection_active and (current_time - last_detection_time) > self.detection_cooldown:
                detection_active = False
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()