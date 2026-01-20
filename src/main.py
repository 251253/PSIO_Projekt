import cv2
import sys
from camera_handler import CameraHandler
from pose_analysis import PoseAnalyzer


def main():
    # KONFIGURACJA
    IP_WEBCAM_URL = "http://10.94.63.158:8080/video"

    # 1. Inicjalizacja kamer
    try:
        cam_handler = CameraHandler(ip_url=IP_WEBCAM_URL)
    except Exception as e:
        print(f"Błąd inicjalizacji kamer: {e}")
        return

    # 2. Inicjalizacja Analizatorów Pozy - UWAGA: DWA OSOBNE OBIEKTY!
    print("Ładowanie modeli AI...")
    # Ten "mózg" pamięta tylko ruch z przodu
    detector_front = PoseAnalyzer()
    # Ten "mózg" pamięta tylko ruch z boku
    detector_side = PoseAnalyzer()

    print("System gotowy. Wciśnij 'q', aby zakończyć.")

    while True:
        # Pobranie klatek
        frames = cam_handler.get_frames()
        frame_laptop = frames.get('laptop')
        frame_ip = frames.get('ip_cam')

        # --- KAMERA LAPTOPA (Używamy detector_front) ---
        if frame_laptop is not None:
            # Używamy instancji FRONT
            results_front = detector_front.find_pose(frame_laptop)
            frame_laptop = detector_front.draw_styled_landmarks(frame_laptop, results_front)
            cv2.imshow('Kamera Laptopa (Front)', frame_laptop)

        # --- KAMERA TELEFONU (Używamy detector_side) ---
        if frame_ip is not None:
            # Obrót i skalowanie (dostosuj, jeśli trzeba)
            frame_ip = cv2.rotate(frame_ip, cv2.ROTATE_90_CLOCKWISE)

            h, w = frame_ip.shape[:2]
            new_h = 600
            scale_factor = new_h / h
            new_w = int(w * scale_factor)
            frame_ip_resized = cv2.resize(frame_ip, (new_w, new_h))

            # Używamy instancji SIDE (kluczowa zmiana!)
            results_side = detector_side.find_pose(frame_ip_resized)
            frame_ip_resized = detector_side.draw_styled_landmarks(frame_ip_resized, results_side)

            cv2.imshow('Kamera IP (Boczna)', frame_ip_resized)

        # Wyjście
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_handler.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()