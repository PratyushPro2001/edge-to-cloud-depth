import cv2
import requests
import numpy as np

SERVER_URL = "https://9f2516dc7e7b.ngrok-free.app/predict"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (500, 300))
    _, img_encoded = cv2.imencode('.jpg', frame)

    try:
        response = requests.post(SERVER_URL, files={"image": img_encoded.tobytes()})

        if response.ok:
            depth_map = cv2.imdecode(
                np.frombuffer(response.content, np.uint8),
                cv2.IMREAD_COLOR
            )

            depth_map = cv2.resize(depth_map, (500 , 320))
            combined = cv2.vconcat([frame, depth_map])
            cv2.imshow("Depth Viewer", combined)

        else:
            print("Server response error:", response.status_code)

    except Exception as e:
        print("Request failed:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
