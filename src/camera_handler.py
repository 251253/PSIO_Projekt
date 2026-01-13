import cv2
import threading
import time


class ThreadedIPCamera:
    """
    Pomocnicza klasa do obsługi kamery IP w oddzielnym wątku.
    Zapobiega buforowaniu klatek i redukuje opóźnienia (lag).
    """

    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.status, self.frame = self.capture.read()
        self.stopped = False

        # Uruchomienie wątku czytającego
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        """Pętla działająca w tle - czyta klatki tak szybko jak się da"""
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    self.frame = frame
                    self.status = status
            else:
                time.sleep(0.1)

    def read(self):
        """Zwraca zawsze najnowszą dostępną klatkę"""
        return self.status, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()


class CameraHandler:
    def __init__(self, ip_url=None, laptop_source=0):
        self.ip_url = ip_url

        # 1. Kamera Laptopa
        self.cap_laptop = cv2.VideoCapture(laptop_source)

        # 2. Kamera IP (z użyciem wątków)
        self.cam_ip_threaded = None
        if self.ip_url:
            try:
                self.cam_ip_threaded = ThreadedIPCamera(self.ip_url)
            except Exception as e:
                print(f"Błąd połączenia z kamerą IP: {e}")

    def get_frames(self):
        frames = {}

        # Odczyt z laptopa
        if self.cap_laptop.isOpened():
            ret_laptop, frame_laptop = self.cap_laptop.read()
            frames['laptop'] = frame_laptop if ret_laptop else None
        else:
            frames['laptop'] = None

        # Odczyt z IP Webcam (przez wątek)
        if self.cam_ip_threaded:
            ret_ip, frame_ip = self.cam_ip_threaded.read()
            frames['ip_cam'] = frame_ip if ret_ip else None
        else:
            frames['ip_cam'] = None

        return frames

    def release(self):
        if self.cap_laptop.isOpened():
            self.cap_laptop.release()

        if self.cam_ip_threaded:
            self.cam_ip_threaded.stop()