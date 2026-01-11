import cv2
import numpy as np

# Adresy z aplikacji IP Webcam (zmień na swoje!)
URL_FRONT = "http://192.168.1.15:8080/video"
URL_SIDE = "http://192.168.1.20:8080/video"

def main():
    # Inicjalizacja przechwytywania
    cap_front = cv2.VideoCapture(URL_FRONT)
    cap_side = cv2.VideoCapture(URL_SIDE)

    while True:
        ret1, frame_front = cap_front.read()
        ret2, frame_side = cap_side.read()

        if not ret1 or not ret2:
            print("Problem z połączeniem z kamerami.")
            break

        # Tutaj w przyszłości dodacie analizę MediaPipe/YOLO 
        
        # Wyświetlanie dwóch widoków obok siebie
        cv2.imshow('Kamera z przodu', frame_front)
        cv2.imshow('Lewa kamera', frame_side)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_front.release()
    cap_side.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
