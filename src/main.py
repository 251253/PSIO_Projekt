import cv2
import time
import numpy as np
from camera_handler import CameraHandler
from pose_analysis import PoseAnalyzer
from exercise_logic import ExerciseLogic
from ui_display import UIDisplay


def main():
    IP_WEBCAM_URL = "http://192.168.18.32:8080/video"
    cam_handler = CameraHandler(ip_url=IP_WEBCAM_URL)
    detector_front = PoseAnalyzer()
    detector_side = PoseAnalyzer()
    ohp_logic = ExerciseLogic()
    ui = UIDisplay()

    state = "MENU"
    countdown_start = 0
    cur_reps, cur_stage, avg_angle = 0, "START", 90

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
        f_ip = cv2.rotate(f_ip, cv2.ROTATE_90_CLOCKWISE) if f_ip is not None else np.zeros((640, 480, 3),
                                                                                           dtype=np.uint8)

        if state == "WORKOUT":
            res_f = detector_front.find_pose(f_laptop)
            f_laptop = detector_front.draw_styled_landmarks(f_laptop, res_f)
            if res_f.pose_landmarks:
                reps, stage, la, ra = ohp_logic.process_front_view(res_f.pose_landmarks.landmark)
                cur_reps, cur_stage, avg_angle = reps, stage, (la + ra) / 2

            res_s = detector_side.find_pose(f_ip)
            f_ip = detector_side.draw_styled_landmarks(f_ip, res_s)
            if res_s.pose_landmarks:
                ohp_logic.check_side_errors(res_s.pose_landmarks.landmark)

        combined = ui.combine_and_scale(f_laptop, f_ip, target_width=1920)
        key = cv2.waitKey(1) & 0xFF

        # Obsługa myszy przez obiekt UI
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

    cam_handler.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()