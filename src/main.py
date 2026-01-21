import cv2
import sys
from camera_handler import CameraHandler
from pose_analysis import PoseAnalyzer
from exercise_logic import ExerciseLogic
from ui_display import UIDisplay


def draw_hud(frame, reps, stage, feedback):
    """Rysuje licznik i komunikaty (Twoja oryginalna funkcja)"""
    color = (0, 0, 255) if feedback else (245, 117, 16)
    cv2.rectangle(frame, (0, 0), (300, 100), color, -1)

    cv2.putText(frame, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(reps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, 'STAGE', (100, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, stage, (95, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    if feedback:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, feedback, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    IP_WEBCAM_URL = "http://192.168.55.102:8080/video"

    try:
        cam_handler = CameraHandler(ip_url=IP_WEBCAM_URL)
    except Exception as e:
        print(f"Błąd inicjalizacji kamer: {e}")
        return

    detector_front = PoseAnalyzer()
    detector_side = PoseAnalyzer()
    ohp_logic = ExerciseLogic()
    ui = UIDisplay()  # <--- DODANO

    print("System gotowy. Start treningu.")

    while True:
        frames = cam_handler.get_frames()
        frame_laptop = frames.get('laptop')
        frame_ip = frames.get('ip_cam')

        # --- WIDOK Z PRZODU ---
        if frame_laptop is not None:
            results_front = detector_front.find_pose(frame_laptop)
            frame_laptop = detector_front.draw_styled_landmarks(frame_laptop, results_front)

            if results_front.pose_landmarks:
                landmarks = results_front.pose_landmarks.landmark
                reps, stage, l_angle, r_angle = ohp_logic.process_front_view(landmarks)
                draw_hud(frame_laptop, reps, stage, ohp_logic.feedback_front)

                avg_angle = (l_angle + r_angle) / 2
                frame_laptop = ui.draw_progress_bar(frame_laptop, avg_angle)

            cv2.imshow('Kamera Laptopa (Front)', frame_laptop)

        # --- WIDOK Z BOKU ---
        if frame_ip is not None:
            # Obrót
            frame_ip = cv2.rotate(frame_ip, cv2.ROTATE_90_CLOCKWISE)

            h, w = frame_ip.shape[:2]
            new_h = 600
            scale = new_h / h
            frame_ip_resized = cv2.resize(frame_ip, (int(w * scale), new_h))

            results_side = detector_side.find_pose(frame_ip_resized)
            frame_ip_resized = detector_side.draw_styled_landmarks(frame_ip_resized, results_side)

            if results_side.pose_landmarks:
                landmarks = results_side.pose_landmarks.landmark
                is_error = ohp_logic.check_side_errors(landmarks)

                # Rysowanie linii pomocniczej (Idealny tor sztangi)
                shoulder_x = int(landmarks[12].x * frame_ip_resized.shape[1])
                cv2.line(frame_ip_resized, (shoulder_x, 0), (shoulder_x, new_h), (255, 255, 0), 1)

                # --- NOWE DODATKI (BOK) ---
                # Wyświetlamy panel z kątami pleców i kolan w rogu
                angles = {"Plecy": ohp_logic.back_angle, "Kolana": ohp_logic.knee_angle}
                frame_ip_resized = ui.draw_angle_dashboard(frame_ip_resized, angles)

                if is_error:
                    cv2.putText(frame_ip_resized, ohp_logic.feedback_side, (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            cv2.imshow('Kamera IP (Boczna)', frame_ip_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_handler.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()