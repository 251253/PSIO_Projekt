import cv2
import time
import numpy as np
from collections import deque  # Do przechowywania historii toru sztangi

from camera_handler import CameraHandler
from pose_analysis import PoseAnalyzer
from exercise_logic import ExerciseLogic
from ui_display import UIDisplay
from barbell_detector import BarbellDetector


def main():
    # --- KONFIGURACJA ---
    IP_WEBCAM_URL = None
    MODEL_PATH = "../models/barbell.pt"

    # 1. Inicjalizacja Obiektów
    try:
        cam_handler = CameraHandler(ip_url=IP_WEBCAM_URL)
        detector_front = PoseAnalyzer()
        detector_side = PoseAnalyzer()
        ohp_logic = ExerciseLogic()
        ui = UIDisplay()

        # Inicjalizacja YOLO (z obsługą błędu)
        try:
            barbell_detector = BarbellDetector(MODEL_PATH)
            print("[INFO] Model YOLO załadowany pomyślnie.")
        except Exception as e:
            print(f"[OSTRZEŻENIE] Nie udało się załadować YOLO: {e}")
            barbell_detector = None

    except Exception as e:
        print(f"Błąd krytyczny inicjalizacji: {e}")
        return

    # Zmienne stanu
    state = "MENU"  # MENU, COUNTDOWN, WORKOUT
    countdown_start = 0

    # Zmienne treningowe
    cur_reps = 0
    cur_stage = "START"
    avg_angle = 90

    # Historia toru sztangi (ostatnie 40 klatek)
    bar_path = deque(maxlen=40)

    # Ustawienia Okna
    win_name = "CyberTrener OHP"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Obsługa myszy (przekazywanie do UI)
    cv2.setMouseCallback(win_name, lambda e, x, y, f, p: setattr(ui, 'm_event', (e, x, y)))

    print("System gotowy. Start pętli głównej.")

    while True:
        # 1. Pobranie klatek
        frames = cam_handler.get_frames()
        f_laptop = frames.get('laptop')
        f_ip = frames.get('ip_cam')

        # Zabezpieczenie przed brakiem obrazu (czarne tło jeśli brak kamery)
        if f_laptop is None:
            f_laptop = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            f_laptop = cv2.resize(f_laptop, (1280, 720))

        if f_ip is None:
            f_ip = np.zeros((720, 405, 3), dtype=np.uint8)  # Wertykalny format
        else:
            # Obrót i formatowanie kamery bocznej
            f_ip = cv2.rotate(f_ip, cv2.ROTATE_90_CLOCKWISE)
            f_ip = cv2.resize(f_ip, (405, 720))  # Dopasowanie wysokości do laptopa

        # --- PRZETWARZANIE LOGIKI (AI) ---

        # A. Widok z przodu (Laptop) - MediaPipe
        results_front = detector_front.find_pose(f_laptop)
        f_laptop = detector_front.draw_styled_landmarks(f_laptop, results_front)

        if results_front.pose_landmarks:
            landmarks = results_front.pose_landmarks.landmark
            # Aktualizacja logiki tylko w trakcie treningu, ale detekcja zawsze
            if state == "WORKOUT":
                cur_reps, cur_stage, ang_l, ang_r = ohp_logic.process_front_view(landmarks)
                avg_angle = int((ang_l + ang_r) / 2)
            else:
                # Tylko odczyt kątów dla menu
                l_shoulder = [landmarks[11].x, landmarks[11].y]
                l_elbow = [landmarks[13].x, landmarks[13].y]
                l_wrist = [landmarks[15].x, landmarks[15].y]
                avg_angle = int(ohp_logic.calculate_angle(l_shoulder, l_elbow, l_wrist))

        # B. Widok z boku (Telefon) - YOLO + MediaPipe
        # 1. Detekcja Sztangi (YOLO)
        if barbell_detector:
            center, box = barbell_detector.detect(f_ip)

            if center:
                bar_path.append(center)
                # Rysowanie ramki sztangi
                x1, y1, x2, y2 = box
                cv2.rectangle(f_ip, (x1, y1), (x2, y2), (255, 0, 255), 2)
            else:
                # Przerwa w linii jeśli zgubiono sztangę
                if len(bar_path) > 0 and bar_path[-1] is not None:
                    bar_path.append(None)

            # 2. Rysowanie Toru Ruchu (Bar Path)
            for i in range(1, len(bar_path)):
                if bar_path[i - 1] is None or bar_path[i] is None:
                    continue
                # Efekt zanikania linii
                thickness = int(np.sqrt(64 / float(len(bar_path) - i + 1)) * 2)
                cv2.line(f_ip, bar_path[i - 1], bar_path[i], (0, 255, 255), thickness)

        # 3. Detekcja Ciała z boku (MediaPipe)
        results_side = detector_side.find_pose(f_ip)
        f_ip = detector_side.draw_styled_landmarks(f_ip, results_side)

        if results_side.pose_landmarks and state == "WORKOUT":
            ohp_logic.check_side_errors(results_side.pose_landmarks.landmark)

        # --- BUDOWANIE INTERFEJSU (UI) ---

        # Łączenie obrazów (Laptop po lewej, Telefon po prawej)
        # Upewniamy się, że mają tę samą wysokość (720)
        combined = np.hstack((f_laptop, f_ip))

        # Obsługa Stanów Aplikacji
        key = cv2.waitKey(1) & 0xFF

        # Logika przycisków myszy
        if hasattr(ui, 'm_event'):
            me, ex, ey = ui.m_event
            if me == cv2.EVENT_LBUTTONDOWN:
                if state == "MENU":
                    # Przycisk START
                    if ui.btn_start_rect[0] < ex < ui.btn_start_rect[2] and ui.btn_start_rect[1] < ey < \
                            ui.btn_start_rect[3]:
                        state = "COUNTDOWN"
                        countdown_start = time.time()
                        # Reset zmiennych treningowych przy starcie
                        ohp_logic.reps = 0
                        bar_path.clear()

                    # Przycisk QUIT
                    if ui.btn_quit_rect[0] < ex < ui.btn_quit_rect[2] and ui.btn_quit_rect[1] < ey < ui.btn_quit_rect[
                        3]:
                        break
            # Czyścimy zdarzenie po obsłużeniu
            del ui.m_event

        # Rysowanie w zależności od stanu
        if state == "MENU":
            combined = ui.draw_advanced_menu(combined)
            if key == ord(' '):
                state = "COUNTDOWN"
                countdown_start = time.time()
                ohp_logic.reps = 0
                bar_path.clear()

        elif state == "COUNTDOWN":
            rem = int(10 - (time.time() - countdown_start))
            combined = ui.draw_countdown(combined, max(0, rem))
            if rem <= 0:
                state = "WORKOUT"

        elif state == "WORKOUT":
            # Pobieramy błędy z logiki
            err_front = getattr(ohp_logic, 'feedback_front', "")
            err_side = getattr(ohp_logic, 'feedback_side', "")

            # Debug values (bezpieczne pobieranie)
            back_angle = getattr(ohp_logic, 'back_angle', 0)
            knee_angle = getattr(ohp_logic, 'knee_angle', 0)

            combined = ui.draw_workout_ui(
                combined,
                cur_reps,
                cur_stage,
                err_front,
                err_side,
                avg_angle,
                {"Plecy": back_angle, "Kolana": knee_angle}  # Słownik debugowania
            )

            # Wyjście do menu po wciśnięciu 'm'
            if key == ord('m'):
                state = "MENU"

        # Wyświetlanie
        cv2.imshow(win_name, combined)

        if key == ord('q'):
            break

    cam_handler.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()