# Główny plik uruchamiający system
import cv2
import sys
from camera_handler import CameraHandler


def main():
    # KONFIGURACJA
    # Adres IP z kamery telefonu
    IP_WEBCAM_URL = "https://192.168.55.102:8080/video"

    # Inicjalizacja handlera kamer
    try:
        cam_handler = CameraHandler(ip_url=IP_WEBCAM_URL)
    except Exception as e:
        print(f"Błąd inicjalizacji kamer: {e}")
        return

    print("System gotowy. Wciśnij 'q', aby zakończyć.")

    while True:
        # 1. Pobierz klatki
        frames = cam_handler.get_frames()
        frame_laptop = frames.get('laptop')
        frame_ip = frames.get('ip_cam')

        # 2. Wyświetl obraz z laptopa
        if frame_laptop is not None:
            cv2.imshow('Kamera Laptopa (Front)', frame_laptop)

        # 3. Wyświetl obraz z telefonu (profil boczny)
        if frame_ip is not None:
            frame_ip_resized = cv2.resize(frame_ip, (640, 480))
            cv2.imshow('Kamera IP (Boczna)', frame_ip_resized)
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