import cv2
import requests
import numpy as np

# Replace with your actual ngrok URL (with /predict)
SERVER_URL = "https://a40a9362e082.ngrok-free.app/predict"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to reduce upload + display size
    frame = cv2.resize(frame, (500, 300))

    # Encode image as JPG
    _, img_encoded = cv2.imencode('.jpg', frame)

    try:
        # Send to cloud server
        response = requests.post(SERVER_URL, files={"image": img_encoded.tobytes()})

        if response.ok:
            # Decode depth map
            depth_map = cv2.imdecode(
                np.frombuffer(response.content, np.uint8),
                cv2.IMREAD_COLOR
            )

            # Resize depth to match webcam size
            depth_map = cv2.resize(depth_map, (500 , 320))

            # Stack vertically (webcam on top, depth below)
            combined = cv2.vconcat([frame, depth_map])

            # Show the stacked result
            cv2.imshow("Depth Viewer", combined)

        else:
            print("Server response error:", response.status_code)

    except Exception as e:
        print("Request failed:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
