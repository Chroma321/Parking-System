
from nicegui import ui
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.entry_camera_anpr import EntryCameraANPR
from src.core.exit_camera_anpr import ExitCameraANPR
import mysql.connector
import cv2
import base64
import asyncio
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import threading
import time

class ANPRApp:
    def __init__(self):
        # Entry camera system
        self.entry_anpr = None
        self.entry_cap = None
        self.entry_detection_running = False
        self.entry_camera_feed_active = False
        self.entry_camera_source = 0  # Default: local camera (0)
        
        # Exit camera system
        self.exit_anpr = None
        self.exit_cap = None
        self.exit_detection_running = False
        self.exit_camera_feed_active = False
        self.exit_camera_source = 1  # Default: second local camera (1)
        
        # Legacy support (for backward compatibility)
        self.anpr = None
        self.cap = None
        self.detection_running = False
        self.camera_feed_active = False
        self.camera_source = 0
        
        # Initialize YOLO model for preview
        try:
            base_dir = os.path.abspath(os.path.dirname(__file__))
            project_dir = os.path.dirname(base_dir)
            model_path = os.path.join(project_dir, "yolov10", "runs", "detect", "train10", "weights", "best.pt")
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Could not load YOLO model: {e}")
            self.model = None
        
    def setup_database_connection(self):
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                database="gate_access"
            )
            return conn
        except Exception as e:
            ui.notify(f"Database connection failed: {e}", type='negative')
            return None

    def get_recent_logs(self):
        conn = self.setup_database_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute("SELECT plate_number, status, timestamp FROM access_log ORDER BY timestamp DESC LIMIT 20")
        logs = cursor.fetchall()
        conn.close()
        return logs

    def add_member(self, plate_number, name="Unknown"):
        conn = self.setup_database_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO member_list (plate_number, owner_name) VALUES (%s, %s)", (plate_number, name))
            conn.commit()
            ui.notify(f"Member {plate_number} added successfully", type='positive')
            return True
        except Exception as e:
            ui.notify(f"Failed to add member: {e}", type='negative')
            return False
        finally:
            cursor.close()
            conn.close()

    def check_detection_status(self):
        # Check if main.py detection process is running by looking for recent log entries
        import os
        log_file = os.path.join(os.getcwd(), "HasilDeteksi", "Captured License.txt")
        if os.path.exists(log_file):
            # Check if file was modified in last 30 seconds (detection active)
            import time
            mod_time = os.path.getmtime(log_file)
            if time.time() - mod_time < 30:
                return "ACTIVE"
        return "INACTIVE"
    
    def get_recent_detections(self):
        # Get recent detected plate images
        import glob
        image_dir = os.path.join(os.getcwd(), "Captured Image")
        if os.path.exists(image_dir):
            images = glob.glob(os.path.join(image_dir, "detected_plate_*.png"))
            # Return 5 most recent images
            images.sort(key=os.path.getmtime, reverse=True)
            return images[:5]
        return []

    def capture_frame_with_detection(self):
        """Capture a frame from camera with detection overlay"""
        if self.cap is None:
            try:
                # Try multiple URL formats for IP cameras
                if isinstance(self.camera_source, str) and self.camera_source.startswith('http'):
                    possible_urls = [
                        self.camera_source,
                        f"{self.camera_source}/video",
                        f"{self.camera_source}/videofeed",
                        f"{self.camera_source}/mjpg/video.mjpg"
                    ]
                    
                    for url in possible_urls:
                        self.cap = cv2.VideoCapture(url)
                        if self.cap.isOpened():
                            ret, test_frame = self.cap.read()
                            if ret:
                                break
                            else:
                                self.cap.release()
                                self.cap = None
                        else:
                            self.cap = None
                else:
                    self.cap = cv2.VideoCapture(self.camera_source)
                
                if self.cap is None or not self.cap.isOpened():
                    return None
                    
                # Set properties for better streaming and higher FPS
                if isinstance(self.camera_source, str):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
                    self.cap.set(cv2.CAP_PROP_FPS, 60)        # Request 60 FPS
                else:
                    # For local cameras, also set FPS
                    self.cap.set(cv2.CAP_PROP_FPS, 60)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
            except:
                return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Run YOLO detection if model is available
        if self.model is not None:
            try:
                results = self.model.predict(frame, conf=0.25, verbose=False)
                result = results[0]
                boxes = result.boxes.xyxy
                
                # Draw detection boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "License Plate", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Detection error: {e}")
        
        # Resize frame for web display
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to base64 for web display
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"
    
    def start_camera_feed(self):
        """Start the camera feed"""
        self.camera_feed_active = True
    
    def stop_camera_feed(self):
        """Stop the camera feed"""
        self.camera_feed_active = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    # ===== ENTRY CAMERA METHODS =====
    def start_entry_detection_process(self):
        """Start the entry camera detection process in a thread"""
        try:
            self.entry_anpr = EntryCameraANPR(camera_source=self.entry_camera_source)
            self.entry_detection_running = True
            
            # Run detection in a separate thread
            self.entry_detection_thread = threading.Thread(
                target=self.entry_anpr.detect_from_camera, 
                daemon=True
            )
            self.entry_detection_thread.start()
            print("✅ Entry detection started in thread")
        except Exception as e:
            print(f"Entry detection error: {e}")
            self.entry_detection_running = False
    
    def stop_entry_detection_process(self):
        """Stop the entry camera detection process"""
        self.entry_detection_running = False
        if hasattr(self, 'entry_anpr') and self.entry_anpr:
            self.entry_anpr.should_stop = True
        print("🛑 Entry detection stopped")
    
    # ===== EXIT CAMERA METHODS =====
    def start_exit_detection_process(self):
        """Start the exit camera detection process in a thread"""
        try:
            self.exit_anpr = ExitCameraANPR(camera_source=self.exit_camera_source)
            self.exit_detection_running = True
            
            # Run detection in a separate thread
            self.exit_detection_thread = threading.Thread(
                target=self.exit_anpr.detect_from_camera, 
                daemon=True
            )
            self.exit_detection_thread.start()
            print("✅ Exit detection started in thread")
        except Exception as e:
            print(f"Exit detection error: {e}")
            self.exit_detection_running = False
    
    def stop_exit_detection_process(self):
        """Stop the exit camera detection process"""
        self.exit_detection_running = False
        if hasattr(self, 'exit_anpr') and self.exit_anpr:
            self.exit_anpr.should_stop = True
        print("🛑 Exit detection stopped")
    
    # ===== LEGACY METHODS (for backward compatibility) =====
    def start_detection_process(self):
        """Legacy method - starts both entry and exit detection"""
        self.start_entry_detection_process()
        time.sleep(1)  # Small delay between camera starts
        self.start_exit_detection_process()
    
    def stop_detection_process(self):
        """Legacy method - stops both entry and exit detection"""
        self.stop_entry_detection_process()
        self.stop_exit_detection_process()
    
    # ===== ENTRY CAMERA METHODS =====
    def capture_entry_frame_with_detection(self):
        """Capture a frame from entry camera with detection overlay"""
        if self.entry_cap is None:
            try:
                # Try multiple URL formats for IP cameras
                if isinstance(self.entry_camera_source, str) and self.entry_camera_source.startswith('http'):
                    possible_urls = [
                        self.entry_camera_source,
                        f"{self.entry_camera_source}/video",
                        f"{self.entry_camera_source}/videofeed",
                        f"{self.entry_camera_source}/mjpg/video.mjpg"
                    ]
                    
                    for url in possible_urls:
                        self.entry_cap = cv2.VideoCapture(url)
                        if self.entry_cap.isOpened():
                            ret, test_frame = self.entry_cap.read()
                            if ret:
                                break
                            else:
                                self.entry_cap.release()
                                self.entry_cap = None
                        else:
                            self.entry_cap = None
                else:
                    self.entry_cap = cv2.VideoCapture(self.entry_camera_source)
                
                if self.entry_cap is None or not self.entry_cap.isOpened():
                    return None
                    
                # Set properties for better streaming and higher FPS
                if isinstance(self.entry_camera_source, str):
                    self.entry_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
                    self.entry_cap.set(cv2.CAP_PROP_FPS, 60)        # Request 60 FPS
                else:
                    # For local cameras, also set FPS
                    self.entry_cap.set(cv2.CAP_PROP_FPS, 60)
                    self.entry_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
            except:
                return None
        
        ret, frame = self.entry_cap.read()
        if not ret:
            return None
        
        # Run YOLO detection if model is available
        if self.model is not None:
            try:
                results = self.model.predict(frame, conf=0.25, verbose=False)
                result = results[0]
                boxes = result.boxes.xyxy
                
                # Draw detection boxes in green for entry
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, "ENTRY - License Plate", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except:
                pass
        
        # Convert frame to base64 for web display
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            import base64
            frame_data = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_data}"
        except:
            return None
    
    def start_entry_camera_feed(self):
        """Start the entry camera feed"""
        self.entry_camera_feed_active = True
    
    def stop_entry_camera_feed(self):
        """Stop the entry camera feed"""
        self.entry_camera_feed_active = False
        if self.entry_cap is not None:
            self.entry_cap.release()
            self.entry_cap = None
    
    def start_entry_detection_process(self):
        """Start the entry ANPR detection process"""
        try:
            from src.core.camera_anpr import CameraANPR
            self.entry_anpr = CameraANPR(camera_source=self.entry_camera_source)
            self.entry_detection_running = True
            self.entry_anpr.detect_from_camera()
        except Exception as e:
            print(f"Entry detection error: {e}")
            self.entry_detection_running = False
    
    def stop_entry_detection_process(self):
        """Stop the entry ANPR detection process"""
        self.entry_detection_running = False
        if hasattr(self, 'entry_anpr') and self.entry_anpr:
            self.entry_anpr.should_stop = True
    
    # ===== EXIT CAMERA METHODS =====
    def capture_exit_frame_with_detection(self):
        """Capture a frame from exit camera with detection overlay"""
        if self.exit_cap is None:
            try:
                # Try multiple URL formats for IP cameras
                if isinstance(self.exit_camera_source, str) and self.exit_camera_source.startswith('http'):
                    possible_urls = [
                        self.exit_camera_source,
                        f"{self.exit_camera_source}/video",
                        f"{self.exit_camera_source}/videofeed",
                        f"{self.exit_camera_source}/mjpg/video.mjpg"
                    ]
                    
                    for url in possible_urls:
                        self.exit_cap = cv2.VideoCapture(url)
                        if self.exit_cap.isOpened():
                            ret, test_frame = self.exit_cap.read()
                            if ret:
                                break
                            else:
                                self.exit_cap.release()
                                self.exit_cap = None
                        else:
                            self.exit_cap = None
                else:
                    self.exit_cap = cv2.VideoCapture(self.exit_camera_source)
                
                if self.exit_cap is None or not self.exit_cap.isOpened():
                    return None
                    
                # Set properties for better streaming and higher FPS
                if isinstance(self.exit_camera_source, str):
                    self.exit_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
                    self.exit_cap.set(cv2.CAP_PROP_FPS, 60)        # Request 60 FPS
                else:
                    # For local cameras, also set FPS
                    self.exit_cap.set(cv2.CAP_PROP_FPS, 60)
                    self.exit_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
            except:
                return None
        
        ret, frame = self.exit_cap.read()
        if not ret:
            return None
        
        # Run YOLO detection if model is available
        if self.model is not None:
            try:
                results = self.model.predict(frame, conf=0.25, verbose=False)
                result = results[0]
                boxes = result.boxes.xyxy
                
                # Draw detection boxes in red for exit
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "EXIT - License Plate", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except:
                pass
        
        # Convert frame to base64 for web display
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            import base64
            frame_data = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_data}"
        except:
            return None
    
    def start_exit_camera_feed(self):
        """Start the exit camera feed"""
        self.exit_camera_feed_active = True
    
    def stop_exit_camera_feed(self):
        """Stop the exit camera feed"""
        self.exit_camera_feed_active = False
        if self.exit_cap is not None:
            self.exit_cap.release()
            self.exit_cap = None
    
    def start_exit_detection_process(self):
        """Start the exit ANPR detection process"""
        try:
            from src.core.camera_anpr import CameraANPR
            self.exit_anpr = CameraANPR(camera_source=self.exit_camera_source)
            self.exit_detection_running = True
            self.exit_anpr.detect_from_camera()
        except Exception as e:
            print(f"Exit detection error: {e}")
            self.exit_detection_running = False
    
    def stop_exit_detection_process(self):
        """Stop the exit ANPR detection process"""
        self.exit_detection_running = False
        if hasattr(self, 'exit_anpr') and self.exit_anpr:
            self.exit_anpr.should_stop = True

app = ANPRApp()

# Main UI Layout
ui.page_title('ANPR Control System')

with ui.header():
    ui.label('ANPR Control System').classes('text-h4')

with ui.tabs() as tabs:
    monitor_tab = ui.tab('Monitor')
    database_tab = ui.tab('Database')
    logs_tab = ui.tab('Logs')

with ui.tab_panels(tabs, value=monitor_tab):
    with ui.tab_panel(monitor_tab):
        ui.label('Entry & Exit Monitoring').classes('text-h4 text-center mb-4')
        
        # Dual Camera Layout
        with ui.row().classes('w-full gap-4'):
            # Entry Camera Section
            with ui.column().classes('w-1/2'):
                ui.label('🚪 ENTRY MONITOR').classes('text-h5 text-green-600 text-center font-bold')
                
                # Entry camera preview
                entry_camera_image = ui.image().classes('w-full max-w-lg border-4 border-green-300 rounded-lg')
                
                # Entry camera configuration
                with ui.card().classes('w-full mt-2'):
                    ui.label('Entry Camera Configuration').classes('text-h6 font-bold')
                    with ui.row():
                        entry_camera_input = ui.input(
                            label='Entry Camera Source', 
                            value='0',
                            placeholder='0 for local, http://IP:8080/video for IP camera'
                        ).classes('flex-grow')
                        ui.button('Set Entry Source', on_click=lambda: set_entry_camera_source()).classes('bg-green-500')
                
                # Entry camera controls
                with ui.row().classes('w-full mt-2'):
                    entry_start_btn = ui.button('Start Entry Preview', on_click=lambda: start_entry_preview()).classes('bg-green-500')
                    entry_stop_btn = ui.button('Stop Entry Preview', on_click=lambda: stop_entry_preview()).classes('bg-red-500')
                
                # Entry detection controls
                with ui.card().classes('w-full mt-4'):
                    ui.label('Entry Detection Control').classes('text-h6 font-bold')
                    with ui.row():
                        entry_detect_start_btn = ui.button('Start Entry Detection', 
                                                          on_click=lambda: start_entry_detection()).classes('bg-green-600')
                        entry_detect_stop_btn = ui.button('Stop Entry Detection', 
                                                         on_click=lambda: stop_entry_detection()).classes('bg-red-600')
                    
                    entry_detection_status = ui.label('Entry Detection: STOPPED').classes('text-lg font-bold text-red-600')
            
            # Exit Camera Section  
            with ui.column().classes('w-1/2'):
                ui.label('🚪 EXIT MONITOR').classes('text-h5 text-red-600 text-center font-bold')
                
                # Exit camera preview
                exit_camera_image = ui.image().classes('w-full max-w-lg border-4 border-red-300 rounded-lg')
                
                # Exit camera configuration
                with ui.card().classes('w-full mt-2'):
                    ui.label('Exit Camera Configuration').classes('text-h6 font-bold')
                    with ui.row():
                        exit_camera_input = ui.input(
                            label='Exit Camera Source', 
                            value='1',
                            placeholder='1 for local, http://IP:8080/video for IP camera'
                        ).classes('flex-grow')
                        ui.button('Set Exit Source', on_click=lambda: set_exit_camera_source()).classes('bg-red-500')
                
                # Exit camera controls
                with ui.row().classes('w-full mt-2'):
                    exit_start_btn = ui.button('Start Exit Preview', on_click=lambda: start_exit_preview()).classes('bg-red-500')
                    exit_stop_btn = ui.button('Stop Exit Preview', on_click=lambda: stop_exit_preview()).classes('bg-red-500')
                
                # Exit detection controls
                with ui.card().classes('w-full mt-4'):
                    ui.label('Exit Detection Control').classes('text-h6 font-bold')
                    with ui.row():
                        exit_detect_start_btn = ui.button('Start Exit Detection', 
                                                        on_click=lambda: start_exit_detection()).classes('bg-red-600')
                        exit_detect_stop_btn = ui.button('Stop Exit Detection', 
                                                       on_click=lambda: stop_exit_detection()).classes('bg-red-600')
                    
                    exit_detection_status = ui.label('Exit Detection: STOPPED').classes('text-lg font-bold text-red-600')
        
        # ===== ENTRY CAMERA CALLBACK FUNCTIONS =====
        def set_entry_camera_source():
            try:
                if entry_camera_input.value.isdigit():
                    app.entry_camera_source = int(entry_camera_input.value)
                else:
                    app.entry_camera_source = entry_camera_input.value
                
                if app.entry_cap is not None:
                    app.entry_cap.release()
                    app.entry_cap = None
                
                ui.notify(f'Entry camera source set to: {app.entry_camera_source}', type='positive')
            except Exception as e:
                ui.notify(f'Error setting entry camera source: {e}', type='negative')
        
        def start_entry_preview():
            app.start_entry_camera_feed()
            entry_start_btn.disable()
            entry_stop_btn.enable()
            ui.notify('Entry camera preview started', type='positive')
        
        def stop_entry_preview():
            app.stop_entry_camera_feed()
            entry_start_btn.enable()
            entry_stop_btn.disable()
            entry_camera_image.set_source('')
            ui.notify('Entry camera preview stopped', type='info')
        
        def start_entry_detection():
            try:
                if app.entry_camera_source is None:
                    ui.notify('Please set entry camera source first!', type='warning')
                    return
                
                import threading
                app.entry_detection_thread = threading.Thread(target=app.start_entry_detection_process, daemon=True)
                app.entry_detection_thread.start()
                
                entry_detect_start_btn.disable()
                entry_detect_stop_btn.enable()
                entry_detection_status.text = 'Entry Detection: RUNNING'
                entry_detection_status.classes('text-lg font-bold text-green-600')
                ui.notify('Entry detection started!', type='positive')
            except Exception as e:
                ui.notify(f'Error starting entry detection: {e}', type='negative')
        
        def stop_entry_detection():
            try:
                app.stop_entry_detection_process()
                entry_detect_start_btn.enable()
                entry_detect_stop_btn.disable()
                entry_detection_status.text = 'Entry Detection: STOPPED'
                entry_detection_status.classes('text-lg font-bold text-red-600')
                ui.notify('Entry detection stopped!', type='info')
            except Exception as e:
                ui.notify(f'Error stopping entry detection: {e}', type='negative')
        
        # ===== EXIT CAMERA CALLBACK FUNCTIONS =====
        def set_exit_camera_source():
            try:
                if exit_camera_input.value.isdigit():
                    app.exit_camera_source = int(exit_camera_input.value)
                else:
                    app.exit_camera_source = exit_camera_input.value
                
                if app.exit_cap is not None:
                    app.exit_cap.release()
                    app.exit_cap = None
                
                ui.notify(f'Exit camera source set to: {app.exit_camera_source}', type='positive')
            except Exception as e:
                ui.notify(f'Error setting exit camera source: {e}', type='negative')
        
        def start_exit_preview():
            app.start_exit_camera_feed()
            exit_start_btn.disable()
            exit_stop_btn.enable()
            ui.notify('Exit camera preview started', type='positive')
        
        def stop_exit_preview():
            app.stop_exit_camera_feed()
            exit_start_btn.enable()
            exit_stop_btn.disable()
            exit_camera_image.set_source('')
            ui.notify('Exit camera preview stopped', type='info')
        
        def start_exit_detection():
            try:
                if app.exit_camera_source is None:
                    ui.notify('Please set exit camera source first!', type='warning')
                    return
                
                import threading
                app.exit_detection_thread = threading.Thread(target=app.start_exit_detection_process, daemon=True)
                app.exit_detection_thread.start()
                
                exit_detect_start_btn.disable()
                exit_detect_stop_btn.enable()
                exit_detection_status.text = 'Exit Detection: RUNNING'
                exit_detection_status.classes('text-lg font-bold text-red-600')
                ui.notify('Exit detection started!', type='positive')
            except Exception as e:
                ui.notify(f'Error starting exit detection: {e}', type='negative')
        
        def stop_exit_detection():
            try:
                app.stop_exit_detection_process()
                exit_detect_start_btn.enable()
                exit_detect_stop_btn.disable()
                exit_detection_status.text = 'Exit Detection: STOPPED'
                exit_detection_status.classes('text-lg font-bold text-red-600')
                ui.notify('Exit detection stopped!', type='info')
            except Exception as e:
                ui.notify(f'Error stopping exit detection: {e}', type='negative')
        
        # ===== AUTO-UPDATE CAMERA FEEDS =====
        async def update_entry_camera_feed():
            if app.entry_camera_feed_active:
                frame_data = app.capture_entry_frame_with_detection()
                if frame_data:
                    entry_camera_image.set_source(frame_data)
        
        async def update_exit_camera_feed():
            if app.exit_camera_feed_active:
                frame_data = app.capture_exit_frame_with_detection()
                if frame_data:
                    exit_camera_image.set_source(frame_data)
        
        # Set up timers for 60 FPS updates
        ui.timer(0.0167, update_entry_camera_feed)  # Entry camera 60 fps
        ui.timer(0.0167, update_exit_camera_feed)   # Exit camera 60 fps
        
        # Initialize buttons
        entry_stop_btn.disable()
        # Initialize buttons
        entry_stop_btn.disable()
        entry_detect_stop_btn.disable()
        exit_stop_btn.disable()
        exit_detect_stop_btn.disable()
        
        ui.separator()
        ui.label('Recent Detections').classes('text-h6')
        
        # Recent detections container
        detections_container = ui.column()
        
        def refresh_detections():
            detections_container.clear()
            recent_images = app.get_recent_detections()
            
            with detections_container:
                if recent_images:
                    ui.label(f'Last {len(recent_images)} detected plates:')
                    for img_path in recent_images:
                        import os
                        filename = os.path.basename(img_path)
                        timestamp = filename.replace('detected_plate_', '').replace('.png', '')
                        # Format timestamp for display
                        if len(timestamp) == 15:  # YYYYMMDD_HHMMSS
                            formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
                            ui.label(f"• {formatted_time}")
                else:
                    ui.label('No recent detections found')
        
        ui.button('Refresh Detections', on_click=refresh_detections).props('color=primary')
        ui.timer(10.0, refresh_detections)  # Auto-refresh every 10 seconds
        refresh_detections()  # Initial load
        
        ui.separator()
        ui.label('Instructions').classes('text-h6')
        ui.label('🚀 Simple start: python main.py (starts both detection and GUI)')
        ui.label('🔧 Manual mode: python WorkBaseline (detection only)')
        ui.label('📊 GUI only: python src/app.py (dashboard only)')
        ui.label('⏹️ Stop detection: Press "q" in camera window')
    
    with ui.tab_panel(database_tab):
        ui.label('Database Management').classes('text-h5')
        
        with ui.card():
            ui.label('Add New Member')
            plate_input = ui.input('License Plate Number').classes('w-64')
            name_input = ui.input('Member Name').classes('w-64')
            
            def add_member_action():
                if plate_input.value and name_input.value:
                    app.add_member(plate_input.value, name_input.value)
                    plate_input.value = ''
                    name_input.value = ''
                else:
                    ui.notify('Please enter both plate number and name', type='warning')
            
            ui.button('Add Member', on_click=add_member_action).props('color=primary')
    
    with ui.tab_panel(logs_tab):
        ui.label('Access Logs').classes('text-h5')
        
        def refresh_logs():
            logs_container.clear()
            logs = app.get_recent_logs()
            
            if logs:
                with logs_container:
                    columns = [
                        {'name': 'plate', 'label': 'Plate Number', 'field': 'plate'},
                        {'name': 'status', 'label': 'Status', 'field': 'status'},
                        {'name': 'timestamp', 'label': 'Timestamp', 'field': 'timestamp'}
                    ]
                    rows = [
                        {'plate': plate, 'status': status.upper(), 'timestamp': str(timestamp)}
                        for plate, status, timestamp in logs
                    ]
                    ui.table(columns=columns, rows=rows).classes('w-full')
            else:
                with logs_container:
                    ui.label('No logs found or database connection failed')
        
        ui.button('Refresh Logs', on_click=refresh_logs).props('color=primary')
        logs_container = ui.column()
        
        # Load logs on startup
        refresh_logs()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='ANPR Control System')
