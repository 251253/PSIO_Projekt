import cv2
import threading
import time


class ThreadedCamera:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        # Optymalizacja bufora
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.status = False
        self.stopped = False

        # Czytamy pierwszą klatkę, żeby upewnić się, że działa
        self.status, self.frame = self.capture.read()

        # Uruchamiamy wątek
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True  # Wątek zginie razem z programem
        self.thread.start()

    def update(self):
        """Pętla działająca w tle, pobierająca ciągle najnowszą klatkę."""
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    self.frame = frame
                    self.status = status
                else:
                    # Jeśli zgubiliśmy połączenie, czekamy chwilę
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def get_frame(self):
        """Zwraca ostatnią pobraną klatkę natychmiastowo."""
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()


class CameraHandler:
    def __init__(self, ip_url=None):
        # 0 = kamera laptopa
        self.cam_laptop = ThreadedCamera(0)

        self.cam_ip = None
        if ip_url:
            self.cam_ip = ThreadedCamera(ip_url)

    def get_frames(self):
        """Zwraca słownik z najnowszymi klatkami z wątków."""
        frames = {
            'laptop': self.cam_laptop.get_frame(),
            'ip_cam': None
        }

        if self.cam_ip:
            frames['ip_cam'] = self.cam_ip.get_frame()

        return frames

    def release(self):
        self.cam_laptop.stop()
        if self.cam_ip:
            self.cam_ip.stop()