
import cv2
import threading
import time
from queue import Queue, Empty
from src.utils.logger import logger as log

class VideoLoader:
    def __init__(self, source, buffer_size=2, reconnect_interval=5):
        self.source = source
        self.buffer_size = buffer_size
        self.reconnect_interval = reconnect_interval
        
        self.cap = None
        self.frame_queue = Queue(maxsize=buffer_size)
        self.stopped = False
        self.connected = False
        
        # Stats
        self.frames_read = 0
        self.start_time = None

        self._start_capture()
        
        # Start background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        log.info(f"VideoLoader started for source: {source}")

    def _start_capture(self):
        """Initializes the cv2.VideoCapture object."""
        try:
            if isinstance(self.source, str) and self.source.isdigit():
                src = int(self.source) # Webcam index
            else:
                src = self.source # File or RTSP
            
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                log.error(f"Failed to open video source: {self.source}")
                self.connected = False
            else:
                self.connected = True
                self.start_time = time.time()
                log.info("Video source opened successfully.")
        except Exception as e:
            log.error(f"Error opening video source: {e}")
            self.connected = False

    def _update(self):
        """Thread worker function to read frames."""
        while not self.stopped:
            if not self.connected:
                log.warning(f"Video source disconnected. Reconnecting in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)
                self._start_capture()
                continue

            ret, frame = self.cap.read()
            if not ret:
                # If file, we might stop. If stream, we treat as disconnect
                # For now, treat as disconnect/end
                log.warning("Failed to read frame (stream end or drop).")
                self.connected = False
                self.cap.release()
                continue
            
            # Keep queue size small (latest frames only)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            
            self.frame_queue.put(frame)
            self.frames_read += 1

    def read(self):
        """Returns the latest frame from the queue."""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        """Stops the thread and releases resources."""
        self.stopped = True
        self.thread.join()
        if self.cap:
            self.cap.release()
        log.info("VideoLoader stopped and resources released.")

    def get_fps(self):
        """Returns approximate read FPS."""
        if self.start_time is None or self.frames_read == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.frames_read / elapsed if elapsed > 0 else 0.0
