#!/usr/bin/env python3
"""
IP Camera Connection Tester
Test various IP camera URL formats to find the working one
"""

import cv2
import sys

def test_ip_camera(base_url):
    """Test different URL formats for IP camera"""
    
    # Common URL formats to try
    url_formats = [
        base_url,
        f"{base_url}/video",
        f"{base_url}/videofeed", 
        f"{base_url}/mjpg/video.mjpg",
        f"{base_url}/axis-cgi/mjpg/video.cgi",
        f"{base_url}/videostream.cgi",
        f"{base_url}/cam_1.mjpg"
    ]
    
    print(f"Testing IP camera connection for: {base_url}")
    print("=" * 50)
    
    working_urls = []
    
    for url in url_formats:
        print(f"Testing: {url}")
        
        try:
            cap = cv2.VideoCapture(url)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ SUCCESS: {url}")
                    print(f"   Frame size: {frame.shape}")
                    working_urls.append(url)
                else:
                    print(f"❌ FAILED: Could not read frame from {url}")
                cap.release()
            else:
                print(f"❌ FAILED: Could not open {url}")
                
        except Exception as e:
            print(f"❌ ERROR: {url} - {e}")
        
        print()
    
    print("=" * 50)
    if working_urls:
        print("✅ WORKING URLs:")
        for url in working_urls:
            print(f"   {url}")
        print("\nUse any of these URLs in your ANPR system!")
    else:
        print("❌ No working URLs found")
        print("\nTroubleshooting tips:")
        print("1. Check if camera is on and connected to network")
        print("2. Verify IP address is correct")
        print("3. Test URL in web browser first")
        print("4. Check camera app settings for port number")
        print("5. Ensure computer and camera are on same network")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_ip_camera.py <base_url>")
        print("Example: python test_ip_camera.py http://192.168.1.100:8080")
        sys.exit(1)
    
    base_url = sys.argv[1]
    test_ip_camera(base_url)
