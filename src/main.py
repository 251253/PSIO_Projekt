# Główny plik uruchamiający system
import cv2
import sys
from camera_handler import CameraHandler
from person_detector import YOLOPersonDetector

def main():
    # KONFIGURACJA
    # Adres IP z kamery telefonu
    IP_WEBCAM_URL = "https://192.168.1.101:8080/video"

    # Inicjalizacja handlera kamer
    try:
        cam_handler = CameraHandler(ip_url=IP_WEBCAM_URL)
    except Exception as e:
        print(f"Błąd inicjalizacji kamer: {e}")
        return

    # Inicjalizacja detektora YOLO (rozpoznawanie ludzi)
    detector = YOLOPersonDetector(
        model_name="yolov8n.pt",
        conf=0.35,
        iou=0.45
    )

    print("System gotowy. Wciśnij 'q', aby zakończyć.")

    while True:
        # 1. Pobierz klatki
        frames = cam_handler.get_frames()
        frame_laptop = frames.get('laptop')
        frame_ip = frames.get('ip_cam')

        # 2. Kamera laptopa – detekcja człowieka + wyświetlenie
        if frame_laptop is not None:
            vis_laptop, _ = detector.detect_and_draw(
                frame_laptop,
                window_label="Person"
            )
            cv2.imshow('Kamera Laptopa (Front)', vis_laptop)

        # 3. Kamera telefonu – detekcja człowieka i wyświetlenie
        if frame_ip is not None:
            frame_ip_resized = cv2.resize(frame_ip, (640, 480))
            vis_ip, _ = detector.detect_and_draw(
                frame_ip_resized,
                window_label="Person"
            )
            cv2.imshow('Kamera IP (Boczna)', vis_ip)
        else:
            # Informacja, jeśli IP Webcam nie odpowiada
            pass

            # 4. Obsługa wyjścia (klawisz 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Sprzątanie
    cam_handler.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()