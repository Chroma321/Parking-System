from nicegui import ui
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mysql.connector
import cv2
import base64
import asyncio
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import threading
import time

class DualCameraANPRApp:
    def __init__(self):
        # Camera sources
        self.entry_camera_source = 0
        self.exit_camera_source = 1
        
        # Camera feed states
        self.entry_camera_active = False
        self.exit_camera_active = False
        self.entry_cap = None
        self.exit_cap = None
        
        # Detection states
        self.entry_detection_running = False
        self.exit_detection_running = False
        self.entry_detection_thread = None
        self.exit_detection_thread = None
        
        # Detection instances
        self.entry_anpr = None
        self.exit_anpr = None
        
        # Initialize YOLO model for preview
        try:
            model_path = os.path.join(os.getcwd(), "yolov10", "runs", "detect", "train10", "weights", "best.pt")
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print("✅ YOLO model loaded for preview")
            else:
                self.model = None
                print("⚠️ YOLO model not found, preview without detection")
        except Exception as e:
            self.model = None
            print(f"⚠️ Could not load YOLO model: {e}")

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
            return True
        except Exception as e:
            print(f"Error adding member: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    # ===== CAMERA CAPTURE METHODS =====
    def capture_frame_from_camera(self, camera_type="entry"):
        """Capture a frame from specified camera with detection overlay"""
        
        # Select the appropriate camera
        if camera_type == "entry":
            cap = self.entry_cap
            camera_source = self.entry_camera_source
        else:  # exit
            cap = self.exit_cap
            camera_source = self.exit_camera_source
        
        # Initialize camera if not opened
        if cap is None:
            try:
                # Handle IP cameras with multiple URL formats
                if isinstance(camera_source, str) and camera_source.startswith('http'):
                    possible_urls = [
                        camera_source,
                        f"{camera_source}/video",
                        f"{camera_source}/videofeed"
                    ]
                    
                    for url in possible_urls:
                        cap = cv2.VideoCapture(url)
                        if cap.isOpened():
                            ret, test_frame = cap.read()
                            if ret:
                                break
                            else:
                                cap.release()
                                cap = None
                        else:
                            cap = None
                else:
                    cap = cv2.VideoCapture(camera_source)
                
                if cap is None or not cap.isOpened():
                    return None
                    
                # Set camera properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)  # Reduced from 60 to prevent conflicts
                
                # Store the camera reference
                if camera_type == "entry":
                    self.entry_cap = cap
                else:
                    self.exit_cap = cap
                    
            except Exception as e:
                print(f"Error opening {camera_type} camera: {e}")
                return None
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            return None
        
        # Add detection overlay if model is available
        if self.model is not None:
            try:
                results = self.model.predict(frame, conf=0.25, verbose=False)
                result = results[0]
                boxes = result.boxes.xyxy
                
                # Draw detection boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if camera_type == "entry" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{camera_type.upper()} - License Plate"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"Detection overlay error for {camera_type}: {e}")
        
        # Convert to base64 for web display
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_base64}"
        except Exception as e:
            print(f"Frame encoding error for {camera_type}: {e}")
            return None

    # ===== CAMERA FEED CONTROL =====
    def start_camera_feed(self, camera_type="entry"):
        """Start camera feed for specified camera"""
        if camera_type == "entry":
            self.entry_camera_active = True
        else:
            self.exit_camera_active = True
        print(f"✅ {camera_type.upper()} camera feed started")

    def stop_camera_feed(self, camera_type="entry"):
        """Stop camera feed for specified camera"""
        if camera_type == "entry":
            self.entry_camera_active = False
            if self.entry_cap is not None:
                self.entry_cap.release()
                self.entry_cap = None
        else:
            self.exit_camera_active = False
            if self.exit_cap is not None:
                self.exit_cap.release()
                self.exit_cap = None
        print(f"⏹️ {camera_type.upper()} camera feed stopped")

    # ===== DETECTION CONTROL =====
    def start_detection(self, camera_type="entry"):
        """Start detection for specified camera"""
        try:
            if camera_type == "entry":
                if self.entry_camera_source is None:
                    return False
                    
                # Import here to avoid circular imports
                from src.core.entry_camera_anpr import EntryCameraANPR
                self.entry_anpr = EntryCameraANPR(camera_source=self.entry_camera_source)
                self.entry_detection_running = True
                
                def entry_detection_runner():
                    try:
                        self.entry_anpr.detect_from_camera()
                    except Exception as e:
                        print(f"Entry detection error: {e}")
                        self.entry_detection_running = False
                
                self.entry_detection_thread = threading.Thread(
                    target=entry_detection_runner,
                    daemon=True,
                    name="EntryDetectionThread"
                )
                self.entry_detection_thread.start()
                
            else:  # exit
                if self.exit_camera_source is None:
                    return False
                    
                from src.core.exit_camera_anpr import ExitCameraANPR
                self.exit_anpr = ExitCameraANPR(camera_source=self.exit_camera_source)
                self.exit_detection_running = True
                
                def exit_detection_runner():
                    try:
                        self.exit_anpr.detect_from_camera()
                    except Exception as e:
                        print(f"Exit detection error: {e}")
                        self.exit_detection_running = False
                
                self.exit_detection_thread = threading.Thread(
                    target=exit_detection_runner,
                    daemon=True,
                    name="ExitDetectionThread"
                )
                self.exit_detection_thread.start()
            
            print(f"✅ {camera_type.upper()} detection started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {camera_type} detection: {e}")
            if camera_type == "entry":
                self.entry_detection_running = False
            else:
                self.exit_detection_running = False
            return False

    def stop_detection(self, camera_type="entry"):
        """Stop detection for specified camera"""
        try:
            if camera_type == "entry":
                self.entry_detection_running = False
                if hasattr(self, 'entry_anpr') and self.entry_anpr:
                    self.entry_anpr.should_stop = True
                self.entry_anpr = None
            else:  # exit
                self.exit_detection_running = False
                if hasattr(self, 'exit_anpr') and self.exit_anpr:
                    self.exit_anpr.should_stop = True
                self.exit_anpr = None
            
            print(f"⏹️ {camera_type.upper()} detection stopped")
            return True
            
        except Exception as e:
            print(f"❌ Error stopping {camera_type} detection: {e}")
            return False

# Create app instance
app = DualCameraANPRApp()

# Main UI Layout
ui.page_title('Dual Camera ANPR System')

with ui.header():
    ui.label('Dual Camera ANPR System').classes('text-h4')

with ui.tabs() as tabs:
    monitor_tab = ui.tab('Monitor')
    database_tab = ui.tab('Database')
    logs_tab = ui.tab('Logs')

with ui.tab_panels(tabs, value=monitor_tab):
    with ui.tab_panel(monitor_tab):
        ui.label('Dual Camera Monitoring').classes('text-h5')
        
        # Dual Camera Layout
        with ui.row().classes('w-full gap-4'):
            # ENTRY CAMERA SECTION
            with ui.column().classes('w-1/2'):
                ui.label('🚪 ENTRY CAMERA').classes('text-h6 text-green-600 font-bold')
                
                # Entry camera preview
                entry_image = ui.image().classes('w-full max-w-lg border-2 border-green-300')
                
                # Entry camera source
                with ui.row():
                    entry_input = ui.input(
                        label='Entry Camera Source',
                        value='0',
                        placeholder='0 or http://192.168.1.100:8080/video'
                    ).classes('flex-1')
                    ui.button('Set', on_click=lambda: set_entry_source())
                
                # Entry camera controls
                with ui.row():
                    entry_preview_start = ui.button('Start Preview', 
                                                   on_click=lambda: start_entry_preview()).classes('bg-blue-500')
                    entry_preview_stop = ui.button('Stop Preview', 
                                                  on_click=lambda: stop_entry_preview()).classes('bg-gray-500')
                
                with ui.row():
                    entry_detect_start = ui.button('Start Detection', 
                                                  on_click=lambda: start_entry_detection()).classes('bg-green-500')
                    entry_detect_stop = ui.button('Stop Detection', 
                                                 on_click=lambda: stop_entry_detection()).classes('bg-red-500')
                
                entry_status = ui.label('Entry Status: STOPPED').classes('text-sm font-bold text-gray-600')
            
            # EXIT CAMERA SECTION
            with ui.column().classes('w-1/2'):
                ui.label('🚪 EXIT CAMERA').classes('text-h6 text-red-600 font-bold')
                
                # Exit camera preview
                exit_image = ui.image().classes('w-full max-w-lg border-2 border-red-300')
                
                # Exit camera source
                with ui.row():
                    exit_input = ui.input(
                        label='Exit Camera Source',
                        value='1',
                        placeholder='1 or http://192.168.1.101:8080/video'
                    ).classes('flex-1')
                    ui.button('Set', on_click=lambda: set_exit_source())
                
                # Exit camera controls
                with ui.row():
                    exit_preview_start = ui.button('Start Preview', 
                                                  on_click=lambda: start_exit_preview()).classes('bg-blue-500')
                    exit_preview_stop = ui.button('Stop Preview', 
                                                 on_click=lambda: stop_exit_preview()).classes('bg-gray-500')
                
                with ui.row():
                    exit_detect_start = ui.button('Start Detection', 
                                                 on_click=lambda: start_exit_detection()).classes('bg-green-500')
                    exit_detect_stop = ui.button('Stop Detection', 
                                                on_click=lambda: stop_exit_detection()).classes('bg-red-500')
                
                exit_status = ui.label('Exit Status: STOPPED').classes('text-sm font-bold text-gray-600')

        # Camera Control Functions
        def set_entry_source():
            try:
                source = entry_input.value
                app.entry_camera_source = int(source) if source.isdigit() else source
                if app.entry_cap:
                    app.entry_cap.release()
                    app.entry_cap = None
                ui.notify(f'Entry camera source set to: {app.entry_camera_source}', type='positive')
            except Exception as e:
                ui.notify(f'Error setting entry source: {e}', type='negative')

        def set_exit_source():
            try:
                source = exit_input.value
                app.exit_camera_source = int(source) if source.isdigit() else source
                if app.exit_cap:
                    app.exit_cap.release()
                    app.exit_cap = None
                ui.notify(f'Exit camera source set to: {app.exit_camera_source}', type='positive')
            except Exception as e:
                ui.notify(f'Error setting exit source: {e}', type='negative')

        # Entry Camera Functions
        def start_entry_preview():
            app.start_camera_feed("entry")
            entry_preview_start.disable()
            entry_preview_stop.enable()
            ui.notify('Entry camera preview started', type='positive')

        def stop_entry_preview():
            app.stop_camera_feed("entry")
            entry_preview_start.enable()
            entry_preview_stop.disable()
            entry_image.set_source('')
            ui.notify('Entry camera preview stopped', type='info')

        def start_entry_detection():
            if app.start_detection("entry"):
                entry_detect_start.disable()
                entry_detect_stop.enable()
                entry_status.text = 'Entry Status: DETECTING'
                entry_status.classes('text-sm font-bold text-green-600')
                ui.notify('Entry detection started!', type='positive')
            else:
                ui.notify('Failed to start entry detection', type='negative')

        def stop_entry_detection():
            if app.stop_detection("entry"):
                entry_detect_start.enable()
                entry_detect_stop.disable()
                entry_status.text = 'Entry Status: STOPPED'
                entry_status.classes('text-sm font-bold text-red-600')
                ui.notify('Entry detection stopped', type='info')

        # Exit Camera Functions
        def start_exit_preview():
            app.start_camera_feed("exit")
            exit_preview_start.disable()
            exit_preview_stop.enable()
            ui.notify('Exit camera preview started', type='positive')

        def stop_exit_preview():
            app.stop_camera_feed("exit")
            exit_preview_start.enable()
            exit_preview_stop.disable()
            exit_image.set_source('')
            ui.notify('Exit camera preview stopped', type='info')

        def start_exit_detection():
            if app.start_detection("exit"):
                exit_detect_start.disable()
                exit_detect_stop.enable()
                exit_status.text = 'Exit Status: DETECTING'
                exit_status.classes('text-sm font-bold text-green-600')
                ui.notify('Exit detection started!', type='positive')
            else:
                ui.notify('Failed to start exit detection', type='negative')

        def stop_exit_detection():
            if app.stop_detection("exit"):
                exit_detect_start.enable()
                exit_detect_stop.disable()
                exit_status.text = 'Exit Status: STOPPED'
                exit_status.classes('text-sm font-bold text-red-600')
                ui.notify('Exit detection stopped', type='info')

        # Auto-update camera feeds
        async def update_entry_feed():
            if app.entry_camera_active:
                frame_data = app.capture_frame_from_camera("entry")
                if frame_data:
                    entry_image.set_source(frame_data)

        async def update_exit_feed():
            if app.exit_camera_active:
                frame_data = app.capture_frame_from_camera("exit")
                if frame_data:
                    exit_image.set_source(frame_data)

        # Timers for camera updates (reduced frequency to prevent conflicts)
        ui.timer(0.05, update_entry_feed)   # 20 FPS for entry
        ui.timer(0.05, update_exit_feed)    # 20 FPS for exit

        # Initialize button states
        entry_preview_stop.disable()
        entry_detect_stop.disable()
        exit_preview_stop.disable()
        exit_detect_stop.disable()

    # Database tab (simplified for now)
    with ui.tab_panel(database_tab):
        ui.label('Member Database').classes('text-h5')
        
        with ui.row():
            new_plate = ui.input(label='Plate Number', placeholder='ABC1234')
            new_name = ui.input(label='Owner Name', placeholder='John Doe')
            
        async def add_member():
            if not new_plate.value or not new_name.value:
                ui.notify('Please fill in all fields', type='warning')
                return
            
            if app.add_member(new_plate.value, new_name.value):
                ui.notify(f'Member {new_name.value} added successfully!', type='positive')
                new_plate.value = ''
                new_name.value = ''
            else:
                ui.notify('Failed to add member', type='negative')
        
        ui.button('Add Member', on_click=add_member).classes('bg-green-500')

    # Logs tab (simplified for now)
    with ui.tab_panel(logs_tab):
        ui.label('Access Logs').classes('text-h5')
        
        logs_table = ui.table(
            columns=[
                {'name': 'plate', 'label': 'Plate Number', 'field': 'plate'},
                {'name': 'status', 'label': 'Status', 'field': 'status'},
                {'name': 'timestamp', 'label': 'Timestamp', 'field': 'timestamp'},
            ],
            rows=[],
        )
        
        def refresh_logs():
            try:
                logs = app.get_recent_logs()
                logs_table.rows = [
                    {'plate': log[0], 'status': log[1], 'timestamp': str(log[2])}
                    for log in logs
                ]
                logs_table.update()
            except Exception as e:
                ui.notify(f'Error loading logs: {e}', type='negative')
        
        ui.button('Refresh Logs', on_click=refresh_logs).classes('bg-blue-500')
        
        # Auto-refresh logs
        ui.timer(10.0, refresh_logs)

# Run the app
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host='0.0.0.0', port=8080, reload=False, show=True)
