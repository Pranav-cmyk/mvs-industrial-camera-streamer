
import base64
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from ctypes import *

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.samples.MvImport.MvCameraControl_class import *


# --- MVS Helper Functions ---
def decoding_char(ctypes_char_array):
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b"\x00")
    if null_index != -1:
        byte_str = byte_str[:null_index]
    for encoding in ["gbk", "utf-8", "latin-1"]:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_str.decode("latin-1", errors="replace")


class CameraManager:
    def __init__(self):
        self.cam = MvCamera()
        self.is_running = False
        self.latest_frame = None
        self.thread = None
        self.lock = threading.Lock()
        self.dst_buf = None
        self.dst_buf_size = 0

    def initialize(self):
        MvCamera.MV_CC_Initialize()

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (
            MV_GIGE_DEVICE
            | MV_USB_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE
            | MV_GENTL_CXP_DEVICE
            | MV_GENTL_XOF_DEVICE
        )

        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            raise Exception("No MVS devices found!")

        # Default to first device
        stDeviceList = cast(
            deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)
        ).contents
        print(
            f"Connecting to: {decoding_char(stDeviceList.SpecialInfo.stGigEInfo.chModelName) if stDeviceList.nTLayerType == MV_GIGE_DEVICE else 'USB Device'}"
        )

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            raise Exception(f"Create handle fail! ret[0x{ret:x}]")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception(f"Open device fail! ret[0x{ret:x}]")

        # GigE Packet Size optimization
        if (
            stDeviceList.nTLayerType == MV_GIGE_DEVICE
            or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE
        ):
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 2)
        self.cam.MV_CC_SetEnumValue("GainAuto", 2)

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception(f"Start grabbing fail! ret[0x{ret:x}]")

        self.is_running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        while self.is_running:
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret == 0:
                nHeight = stOutFrame.stFrameInfo.nHeight
                nWidth = stOutFrame.stFrameInfo.nWidth

                required_dst_size = nWidth * nHeight * 3
                if self.dst_buf is None or self.dst_buf_size < required_dst_size:
                    self.dst_buf_size = required_dst_size
                    self.dst_buf = (c_ubyte * self.dst_buf_size)()

                stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
                memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                stConvertParam.nWidth = nWidth
                stConvertParam.nHeight = nHeight
                stConvertParam.pSrcData = stOutFrame.pBufAddr
                stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                stConvertParam.pDstBuffer = cast(self.dst_buf, POINTER(c_ubyte))
                stConvertParam.nDstBufferSize = self.dst_buf_size

                ret_cvt = self.cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
                if ret_cvt == 0:
                    img = np.frombuffer(
                        self.dst_buf, dtype=np.uint8, count=stConvertParam.nDstLen
                    ).reshape(nHeight, nWidth, 3)
                    with self.lock:
                        self.latest_frame = img.copy()

                self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                time.sleep(0.01)

    def get_latest_frame_b64(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            # Encode to JPEG
            _, buffer = cv2.imencode(".jpg", self.latest_frame)
            return base64.b64encode(buffer).decode("utf-8")

    def shutdown(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()


camera_manager = CameraManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        camera_manager.initialize()
        print("Camera initialized successfully.")
    except Exception as e:
        print(f"Camera initialization failed: {e}")
    yield
    # Shutdown
    camera_manager.shutdown()
    print("Camera shutdown completed.")


app = FastAPI(lifespan=lifespan)


@app.get("/frame")
async def get_frame():
    b64_str = camera_manager.get_latest_frame_b64()
    if b64_str is None:
        raise HTTPException(status_code=503, detail="Frame not available yet")
    return {"image": b64_str}


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head>
            <title>MVS Camera Stream</title>
            <style>
                h1 { font-size: 2em; margin-top: 1em; }
                body { font-family: sans-serif; text-align: center ; background: #222; color: #fff; }
                #stream { max-width: 90%; margin-top: 20px; border: 2px solid #444; }
            </style>
        </head>
        <body>
            <h1>MVS Camera Live Feed</h1>
            <img id="stream" src="" />
            <script>
                const img = document.getElementById('stream');
                async function updateFrame() {
                    try {
                        const response = await fetch('/frame');
                        const data = await response.json();
                        img.src = 'data:image/jpeg;base64,' + data.image;
                    } catch (e) {
                        console.error("Failed to fetch frame", e);
                    }
                    requestAnimationFrame(updateFrame);
                }
                updateFrame();
            </script>
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
