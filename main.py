import subprocess
import sys
import threading
import time
from src.core.camera_anpr import CameraANPR

def run_detection():
    """Run the modular CameraANPR detection"""
    print("🚀 Starting ANPR detection...")
    try:
        anpr = CameraANPR()
        anpr.detect_from_camera()
    except Exception as e:
        print(f"❌ Detection error: {e}")

def run_gui():
    """Run the GUI dashboard"""
    print("🖥️ Starting GUI dashboard...")
    try:
        subprocess.run([sys.executable, "src/app.py"])
    except Exception as e:
        print(f"❌ GUI error: {e}")

if __name__ == "__main__":
    print("=== ANPR System Starting ===")
    print("Starting GUI first, then detection system...")
    
    # Start GUI first in a separate thread
    gui_thread = threading.Thread(target=run_gui, daemon=True)
    gui_thread.start()
    
    # Give GUI more time to fully start and display URL
    print("⏳ Waiting for GUI to start...")
    time.sleep(8)
    
    print("🚀 Now starting detection system...")
    print("📱 Open your browser to the URL shown above for the dashboard")
    
    try:
        # Run detection in main thread (so camera window shows)
        run_detection()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ANPR system...")
        sys.exit(0)
