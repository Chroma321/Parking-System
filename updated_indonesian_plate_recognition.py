import cv2
import os
import pytesseract

# frameWidth = 640   #Frame Width
frameWidth = 1000   #Frame Width
frameHeight = 480   # Frame Height

# Update this to the correct cascade for Indonesian plates if available
cascade_path = "haarcascade_license_plate_rus_16stages.xml"  # Change this if you have a specific cascade
if not os.path.exists(cascade_path):
    print(f"Warning: Cascade file '{cascade_path}' not found. Using default Russian cascade.")
    cascade_path = "haarcascade_license_plate_rus_16stages.xml"

plateCascade = cv2.CascadeClassifier(cascade_path)
if plateCascade.empty():
    raise IOError(f"Failed to load cascade classifier from {cascade_path}")

minArea = 500

# Ensure the IMAGES directory exists
save_dir = "IMAGES"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    imgRoi = None
    detected_text = None
    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imgRoi = img[y:y + h, x:x + w]
            # OCR on the detected plate
            plate_text = pytesseract.image_to_string(imgRoi, config='--psm 8')
            detected_text = plate_text.strip()
            # Overlay detected text on the image
            cv2.putText(img, detected_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Number Plate", imgRoi)
    cv2.imshow("Result", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and imgRoi is not None:
        save_path = os.path.join(save_dir, f"{str(count)}.jpg")
        cv2.imwrite(save_path, imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
    elif key == ord('q'):
        break

    # Print detected text in real time (optional)
    if detected_text:
        print("Detected Plate Text:", detected_text)



cap.release()
cv2.destroyAllWindows()
