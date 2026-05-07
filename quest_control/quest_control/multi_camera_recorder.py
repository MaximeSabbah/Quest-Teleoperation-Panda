import cv2
import datetime
import os

class MultiCameraRecorder:

    def __init__(self, base_dir):

        self.devices = [
            "/dev/video0",
            "/dev/video2",
            "/dev/video4",
            "/dev/video5",
            "/dev/video6",
            "/dev/video7",
            "/dev/video8",
            "/dev/video9"
            
        ]

        self.caps = []
        self.writers = []

        self.currently_recording = False
        self.initialized = False

        self.width = 256
        self.height = 256
        self.fps = 20

        self.base_dir = base_dir #+ "/recordings" #os.path.expanduser("~/recordings")
        os.makedirs(self.base_dir, exist_ok=True)


    def initialize_cameras(self):

        self.caps = []

        for dev in self.devices:

            #cap = cv2.VideoCapture(dev)
            #cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_V4L2)
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {dev}")

            self.caps.append(cap)

        self.initialized = True


    def start_recording(self, timestamp):
        self.writers = []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        for i in range(len(self.caps)):

            filename = f"{self.base_dir}/{timestamp}/camera_{i}.mp4"
            os.makedirs(f"{self.base_dir}/{timestamp}", exist_ok=True)
            writer = cv2.VideoWriter(
                filename,
                fourcc,
                self.fps,
                (self.width, self.height)
            )

            if not writer.isOpened():
                raise RuntimeError(f"VideoWriter failed for {filename}")

            self.writers.append(writer)

            print(f"Recording started: {filename}")
        self.currently_recording  = True

    def stop_recording(self):

        for writer in self.writers:
            writer.release()

        self.writers = []

        print("Recording stopped")
        self.currently_recording  = False

    def capture_step(self):
        #print(f"Capturing image? {self.currently_recording}")

        frames = []

        for cap in self.caps:

            ret, frame = cap.read()

            if not ret:
                print("Frame capture failed")
                return

            # add timestamp overlay
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            #cv2.putText(
            #    frame,
            #    timestamp,
            #    (10, 30),
            #    cv2.FONT_HERSHEY_SIMPLEX,
            #    0.7,
            #    (0,255,0),
            #    2
            #)

            frames.append(frame)

        if self.currently_recording:

            for writer, frame in zip(self.writers, frames):
                #print(frame.shape)
                frame = cv2.resize(frame, (self.width, self.height))
                writer.write(frame)