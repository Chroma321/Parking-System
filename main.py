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
    print("This will start both detection and GUI")
    print("Press Ctrl+C to stop everything")
    
    # Start GUI first in a separate thread
    gui_thread = threading.Thread(target=run_gui, daemon=True)
    gui_thread.start()
    
    # Give GUI a moment to start
    time.sleep(3)
    
    try:
        # Run detection in main thread
        run_detection()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ANPR system...")
        sys.exit(0)
