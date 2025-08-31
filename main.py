import subprocess
import sys

def run_gui():
    """Run the GUI dashboard only"""
    print("�️ Starting ANPR Dashboard...")
    print("📱 Configure camera source and start detection from the web interface")
    try:
        subprocess.run([sys.executable, "src/app.py"])
    except Exception as e:
        print(f"❌ GUI error: {e}")

if __name__ == "__main__":
    print("=== ANPR System ===")
    print("🌐 Starting web-based control panel...")
    print("👉 Use the web interface to:")
    print("   1. Configure camera source (local or IP camera)")
    print("   2. Start/stop detection")
    print("   3. Manage database and view logs")
    print()
    
    run_gui()
