# 🌐 Edge-to-Cloud Real-Time Depth Estimation using MiDaS

Imagine deploying a fleet of warehouse robots to navigate dynamic environments identifying obstacles, gauging shelf distances, or avoiding collisions. But there's a catch: each robot is equipped with only a simple monocular camera and lacks the computational power to run heavy deep learning models locally. So, how do we enable these low-power edge devices to perform reliable real-time depth estimation? This project presents an elegant and scalable solution: streaming live video frames from the robot (edge device) to a cloud-hosted GPU server that runs MiDaS a state-of-the-art depth estimation model. The server processes each frame and returns the corresponding depth map in real time, which is then visualized back on the edge device. This approach eliminates the need for costly onboard GPUs, leverages the power of the cloud, and makes high quality depth perception feasible for lightweight robotic systems operating in the field.


> Designed for low-power edge devices by offloading computation-heavy MiDaS inference to the cloud.

---

## Demo

![Depth Demo](assets/demo_vertical.gif)  
*Webcam on top, real-time depth map below (computed using MiDaS in the cloud)*

---

## How It Works

1. **Webcam** captures a frame using OpenCV  
2. **Client** encodes and sends the frame to a **cloud server** via HTTP POST  
3. **Colab Server** runs the MiDaS depth estimation model  
4. **Response** is the predicted depth map, which is displayed alongside the original frame

```
Edge Device (Webcam)  →  Cloud (Colab + GPU)  →  Returns Depth Map (MiDaS)
         |                                 |
     client.py                       FastAPI + MiDaS Model
```

---

## Features

- Uses MiDaS (a powerful, accurate depth estimation model)
- Offloads heavy inference to Colab’s GPU using FastAPI + ngrok
- Real-time frame exchange between edge and cloud
- Lightweight Python client that runs even on resource-constrained devices

---

## Installation

### Requirements

- Python 3.8+
- OpenCV
- NumPy
- Requests
- ngrok account
- Google Colab (for server)

### Repo Structure

```
edge-to-cloud-depth/
│
├── src/
│   └── client.py              # Streams webcam frames and receives depth map
├── edge_to_cloud_depth.ipynb  # Colab notebook (runs the FastAPI server + MiDaS)
├── requirements.txt
├── .gitignore
├── README.md
└── assets/
    └── demo_vertical.gif      # (Optional) GIF/Image for README
```

---

## How to Run

### Step 1: Run the Cloud Server on Colab

1. Open `cloud_depth_est.ipynb` in [Google Colab](https://colab.research.google.com)
2. Install dependencies and start the FastAPI server
3. Copy the `ngrok` public URL (should end in `/predict`)

### Step 2: Run the Local Client

In your terminal:

```bash
cd edge-to-cloud-depth/src
python client.py
```

Make sure to update `SERVER_URL` in `client.py` to your Colab-hosted `ngrok` URL.


### ✅ Expected Outcomes

Once both the **Colab server** and **local client** are running, you should see:

#### 1) On the **Colab (cloud_depth_est.ipynb) console**:
- Server startup lines from Uvicorn
- POST requests hitting the `/predict` endpoint
- Occasional ngrok keepalive logs

**Example:**
```plaintext
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
INFO:     127.0.0.1:xxxxx - "POST /predict HTTP/1.1" 200 OK
INFO:     127.0.0.1:xxxxx - "POST /predict HTTP/1.1" 200 OK
```


---

## Why MiDaS?

MiDaS is one of the most accurate monocular depth estimation models but is computationally expensive. Running it directly on low-power edge systems like Raspberry Pi or drones is impractical.

This project solves that by using:
- **Colab GPU** for MiDaS inference
- **ngrok** to expose the API securely over the internet
- **Real-time streaming** between client and server

---

## Tips for Better Presentation

- 📷 Include an inference screenshot: `assets/demo_vertical.png`
- 📹 Include a screen recording GIF under the project title to attract attention

---

## License

MIT License

---

## Acknowledgements

- [Intel Labs - MiDaS](https://github.com/isl-org/MiDaS)
- [pyngrok](https://github.com/alexdlaird/pyngrok)
- [OpenCV](https://opencv.org/)