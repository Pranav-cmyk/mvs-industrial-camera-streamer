# MVS Real-Time Industrial Camera Streaming API

A thread-safe **FastAPI** streaming microservice that bridges industrial **Hikrobot MVS (Machine Vision) GigE & USB3 cameras** to web clients and downstream robotics pipelines over HTTP.

## 🚀 Architectural Highlights

- **Stateful `CameraManager` Wrapper:** Encapsulates the pre-built `MvCameraControl` SDK methods into a clean, object-oriented Python service.
- **Non-Blocking Daemon Threading (`_grab_loop`):** Isolates blocking hardware frame acquisition (`MV_CC_GetImageBuffer`) and `BGR8` pixel format conversion into a dedicated background thread so the FastAPI event loop remains responsive.
- **Mutex-Locked Frame Buffer (`threading.Lock`):** Prevents race conditions between high-frequency camera buffer writes (`np.frombuffer` / OpenCV) and concurrent HTTP `/frame` reads.
- **FastAPI Lifespan Management (`asynccontextmanager`):** Automates SDK initialization, GigE packet-size negotiation, thread teardown, and hardware handle release on server startup and shutdown.
- **Zero-Dependency Web Viewer:** Built-in HTML5/JS live stream viewer using `requestAnimationFrame` and Base64 JPEG encoding (`cv2.imencode`).

## 🛠️ Tech Stack
- **Backend:** FastAPI, Uvicorn, Python 3.x
- **Concurrency:** `threading.Thread` (Daemon), `threading.Lock`, `asynccontextmanager`
- **Vision & Buffers:** Hikrobot MVS SDK (`MvCameraControl_class`), OpenCV (`cv2`), NumPy, Base64

## ⚙️ Running the Server
1. Ensure the Hikrobot MVS SDK (`MvImport`) is installed and an MVS-compatible GigE or USB camera is connected.
2. Start the FastAPI server:
   ```bash
   python main.py
   ```


3. View the live stream at http://localhost:8000 or poll raw Base64 frames at GET http://localhost:8000/frame.
