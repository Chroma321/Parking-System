
from nicegui import ui
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.camera_anpr import CameraANPR
import mysql.connector
import cv2
import base64
import asyncio
from datetime import datetime

class ANPRApp:
    def __init__(self):
        self.anpr = None
        self.cap = None
        self.detection_running = False
        
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
        ui.label('Detection Monitoring').classes('text-h5')
        
        # Detection status indicator
        status_label = ui.label('Detection Status: Checking...').classes('text-lg')
        
        def update_status():
            status = app.check_detection_status()
            color = 'green' if status == 'ACTIVE' else 'red'
            status_label.text = f'Detection Status: {status}'
            status_label.style(f'color: {color}')
        
        # Auto-refresh status every 5 seconds
        ui.timer(5.0, update_status)
        update_status()  # Initial check
        
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
