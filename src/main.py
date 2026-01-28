import cv2
import time
import math
import numpy as np
from camera_handler import CameraHandler
from pose_analysis import PoseAnalyzer
from exercise_logic import ExerciseLogic
from ui_display import UIDisplay
from barbell_detector import BarbellDetector


def main():
    IP_WEBCAM_URL = None
    MODEL_PATH = "../models/barbell.pt"

    print("[INIT] Startowanie systemu...")
    cam_handler = CameraHandler(ip_url=IP_WEBCAM_URL)

    detector_front = PoseAnalyzer()
    detector_side = PoseAnalyzer()
    ohp_logic = ExerciseLogic()
    ui = UIDisplay()

    barbell_detector = None
    try:
        barbell_detector = BarbellDetector(MODEL_PATH)
    except Exception as e:
        print(f"[WARNING] YOLO nie działa: {e}")

    state = "MENU"
    countdown_start = 0
    cur_reps, cur_stage, avg_angle = 0, "START", 90
    SCALE_FACTOR = 0.5

    # --- ZMIENNE DO PŁYNNOŚCI (TRACKING) ---
    missed_frames = 0
    MAX_MISSED_FRAMES = 15  # Jak długo "udawać" sztangę (ok. 0.5 sek)

    win_name = "CyberTrener OHP"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win_name, lambda e, x, y, f, p: setattr(ui, 'm_event', (e, x, y)))

    while True:
        frames = cam_handler.get_frames()
        f_laptop = frames.get('laptop')
        f_ip = frames.get('ip_cam')

        if f_laptop is None and f_ip is None:
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        f_laptop = f_laptop if f_laptop is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        f_laptop = cv2.resize(f_laptop, (1280, 720))

        f_ip = cv2.rotate(f_ip, cv2.ROTATE_90_CLOCKWISE) if f_ip is not None else np.zeros((640, 480, 3),
                                                                                           dtype=np.uint8)
        f_ip = cv2.resize(f_ip, (405, 720))

        if state == "WORKOUT":
            f_small_laptop = cv2.resize(f_laptop, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            f_small_ip = cv2.resize(f_ip, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

            # A. YOLO (W TLE)
            yolo_center = None
            yolo_box = None

            if barbell_detector:
                barbell_detector.update_frame(f_small_laptop)

                yolo_center_small, yolo_box_small = barbell_detector.get_result()

                if yolo_center_small:
                    yolo_center = (int(yolo_center_small[0] / SCALE_FACTOR), int(yolo_center_small[1] / SCALE_FACTOR))
                    bx1, by1, bx2, by2 = yolo_box_small
                    yolo_box = (
                    int(bx1 / SCALE_FACTOR), int(by1 / SCALE_FACTOR), int(bx2 / SCALE_FACTOR), int(by2 / SCALE_FACTOR))

            # B. MEDIAPIPE FRONT
            res_f = detector_front.find_pose(f_small_laptop)
            f_laptop = detector_front.draw_styled_landmarks(f_laptop, res_f)

            # C. MEDIAPIPE SIDE
            res_s = detector_side.find_pose(f_small_ip)
            f_ip = detector_side.draw_styled_landmarks(f_ip, res_s)

            # --- SYSTEM HYBRYDOWEGO ŚLEDZENIA ---
            valid_bar_center = None
            tracking_mode = False  # Czy używamy trybu awaryjnego (pomarańczowy)

            if res_f.pose_landmarks:
                landmarks = res_f.pose_landmarks.landmark
                h, w, c = f_laptop.shape

                # Obliczamy środek dłoni (Baza dla trybu awaryjnego)
                wrist_l = landmarks[15]
                wrist_r = landmarks[16]
                wl_x, wl_y = wrist_l.x * w, wrist_l.y * h
                wr_x, wr_y = wrist_r.x * w, wrist_r.y * h
                hands_center = (int((wl_x + wr_x) / 2), int((wl_y + wr_y) / 2))

                # SCENARIUSZ 1: YOLO widzi sztangę
                if yolo_center:
                    # Sprawdzamy czy to nie półka (Filtr odległości)
                    dist_to_hands = math.hypot(hands_center[0] - yolo_center[0], hands_center[1] - yolo_center[1])
                    LIMIT_DISTANCE = w * 0.20

                    if dist_to_hands < LIMIT_DISTANCE:
                        valid_bar_center = yolo_center
                        missed_frames = 0  # Reset licznika błędów
                        tracking_mode = False

                # SCENARIUSZ 2: YOLO zgubiło sztangę (szybki ruch)
                else:
                    if missed_frames < MAX_MISSED_FRAMES:
                        # Używamy środka dłoni jako pozycji sztangi
                        valid_bar_center = hands_center
                        missed_frames += 1
                        tracking_mode = True  # Tryb awaryjny

            # Jeśli nie ma człowieka, to reset
            elif yolo_center:
                valid_bar_center = None

                # D. RYSOWANIE I LOGIKA
            if valid_bar_center:
                # Wybór koloru: Żółty (YOLO) lub Pomarańczowy (Tracking dłoni)
                box_color = (0, 255, 255) if not tracking_mode else (0, 165, 255)

                # Rysowanie kropki środka
                cv2.circle(f_laptop, valid_bar_center, 8, box_color, -1)

                # Jeśli mamy ramkę z YOLO, to ją rysujemy, jeśli tracking - rysujemy kółko wokół dłoni
                if yolo_box and not tracking_mode:
                    x1, y1, x2, y2 = yolo_box
                    cv2.rectangle(f_laptop, (x1, y1), (x2, y2), box_color, 2)
                elif tracking_mode:
                    # W trybie awaryjnym rysujemy mniejsze kółko sygnalizujące estymację
                    cv2.circle(f_laptop, valid_bar_center, 20, box_color, 2)

                # Logika "Środek" (Linia do nosa)
                if res_f.pose_landmarks:
                    nose = res_f.pose_landmarks.landmark[0]
                    nose_x, nose_y = int(nose.x * w), int(nose.y * h)

                    line_color = (0, 255, 0)
                    # Tolerancja błędów
                    if abs(nose_x - valid_bar_center[0]) > (w * 0.08):
                        line_color = (0, 0, 255)
                    cv2.line(f_laptop, (nose_x, nose_y), valid_bar_center, line_color, 2)

            # Logika biznesowa (Liczenie powtórzeń)
            if res_f.pose_landmarks:
                bar_x = valid_bar_center[0] if valid_bar_center else None

                reps, stage, la, ra = ohp_logic.process_front_view(
                    res_f.pose_landmarks.landmark,
                    bar_center_x=bar_x,
                    frame_width=f_laptop.shape[1]
                )
                cur_reps, cur_stage, avg_angle = reps, stage, (la + ra) / 2

            if res_s.pose_landmarks:
                ohp_logic.check_side_errors(res_s.pose_landmarks.landmark)

        # UI
        combined = ui.combine_and_scale(f_laptop, f_ip, target_width=1920)
        key = cv2.waitKey(1) & 0xFF

        if hasattr(ui, 'm_event') and ui.m_event[0] == cv2.EVENT_LBUTTONDOWN:
            ex, ey = ui.m_event[1], ui.m_event[2]
            if state == "MENU":
                if ui.btn_start_rect[0] < ex < ui.btn_start_rect[2] and ui.btn_start_rect[1] < ey < ui.btn_start_rect[
                    3]:
                    state = "COUNTDOWN";
                    countdown_start = time.time()
            if ui.btn_quit_rect[0] < ex < ui.btn_quit_rect[2] and ui.btn_quit_rect[1] < ey < ui.btn_quit_rect[3]:
                break
            del ui.m_event

        if state == "MENU":
            combined = ui.draw_advanced_menu(combined)
            if key == ord(' '): state = "COUNTDOWN"; countdown_start = time.time()
        elif state == "COUNTDOWN":
            rem = int(10 - (time.time() - countdown_start))
            combined = ui.draw_countdown(combined, max(0, rem))
            if rem <= 0: state = "WORKOUT"
        elif state == "WORKOUT":
            combined = ui.draw_workout_ui(combined, cur_reps, cur_stage,
                                          getattr(ohp_logic, 'feedback_front', ""),
                                          getattr(ohp_logic, 'feedback_side', ""),
                                          avg_angle,
                                          {"Plecy": getattr(ohp_logic, 'back_angle', 0),
                                           "Kolana": getattr(ohp_logic, 'knee_angle', 0)})

        cv2.imshow(win_name, combined)
        if key == 27 or key == ord('q'): break

    if barbell_detector:
        barbell_detector.stop()
    cam_handler.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()