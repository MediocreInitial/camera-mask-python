import os
import sys

# Фикс для корректной работы камеры внутри скомпилированного EXE
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_ASYNC"] = "0"

import cv2
import numpy as np
import math
import ctypes
import urllib.request
import time

# --- КОНСОЛЬНЫЙ ЭКРАН ЗАГРУЗКИ С ЗЕЛЕНЫМ ПРОГРЕСС-БАРОМ И ПРОЦЕНТАМИ ---
def show_console_loader():
    steps = [
        ("Initializing system environment & dependencies...", 20),
        ("Loading MediaPipe vision tasks & XNNPACK delegate...", 40),
        ("Verifying neural face landmarker model asset...", 60),
        ("Configuring hardware streams & 16:9 widescreen...", 80),
        ("Starting Camera Mask engine successfully...", 100)
    ]
    
    print("\n" + "="*50)
    print("      CAMERA MASK ENGINE - INITIALIZING      ")
    print("="*50 + "\n")
    
    bar_len = 30
    for msg, percent in steps:
        print(f"[STATUS] {msg}")
        filled_len = int(bar_len * percent // 100)
        bar = '\033[92m' + '█' * filled_len + '\033[0m' + '░' * (bar_len - filled_len)
        sys.stdout.write(f"\rProgress: [{bar}] {percent}%")
        sys.stdout.flush()
        time.sleep(0.35)
        print("\n")
    
    print("="*50)
    print("[OK] All systems operational. Launching webcam window...")
    print("="*50 + "\n")
    time.sleep(0.5)

# Запускаем лоадер в консоли
show_console_loader()

# --- СТРОГИЙ ПОИСК РЕАЛЬНОЙ КАМЕРЫ (С ИГНОРИРОВАНИЕМ ОБХОДА НА OBS) ---
def init_camera():
    # Главный приоритет — индекс 0 (физическая камера). Индекс 1 (OBS) ставим в самый конец.
    indices_to_try = [0, 2, 3, 4, 1]
    backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for index in indices_to_try:
        for backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    time.sleep(0.6)
                    
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        # Убедимся, что кадр не пустой/черный
                        if np.mean(test_frame) > 2.0:
                            print(f"[OK] Physical camera connected successfully! (Index: {index})")
                            return cap
                    cap.release()
            except Exception:
                pass
    return None

cap = init_camera()
if cap is None or not cap.isOpened():
    print("[ERROR] Failed to find any working physical webcam! Make sure OBS or other apps are not locking index 0.")
    sys.exit()

# АВТОМАТИЧЕСКОЕ СКРЫТИЕ КОНСОЛИ ПЕРЕД ПОЯВЛЕНИЕМ ОКНА КАМЕРЫ
try:
    kernel32 = ctypes.WinDLL('kernel32')
    user32 = ctypes.WinDLL('user32')
    hWnd = kernel32.GetConsoleWindow()
    if hWnd:
        user32.ShowWindow(hWnd, 0)  # SW_HIDE = 0
except Exception:
    pass

# --- НАСТРОЙКИ ---
PIXEL_SIZE = 36          # Размер пикселей Bodycam

# Автоматическая загрузка модели Face Landmarker
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(SCRIPT_DIR, 'face_landmarker.task')
if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
    try:
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 
            model_path
        )
    except Exception:
        pass

# --- СЛОЙ 1: ИНИЦИАЛИЗАЦИЯ MEDIAPIPE С БЛЕНДШЕЙПАМИ ЭМОЦИЙ ---
use_mediapipe = False
landmarker = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.05,
        min_face_presence_confidence=0.05
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    use_mediapipe = True
except Exception:
    pass

# --- ПИКСЕЛИЗАЦИЯ (РЕЖИМ BODYCAM) ---
def apply_pixelate_bulletproof(frame, cx, cy, box_w, box_h, angle_deg, pixel_size):
    try:
        h_img, w_img = frame.shape[:2]
        box_h = max(40, int(box_h))
        box_w = max(40, int(box_w))
        angle_deg = max(-45, min(45, angle_deg))

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (box_w, box_h), angle_deg, 0, 360, 255, -1)
        
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        M_inv = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
        
        rotated_frame = cv2.warpAffine(frame, M, (w_img, h_img), flags=cv2.INTER_LINEAR)
        
        x1 = max(0, cx - box_w)
        x2 = min(w_img, cx + box_w)
        y1 = max(0, cy - box_h)
        y2 = min(h_img, cy + box_h)
        
        roi = rotated_frame[y1:y2, x1:x2]
        if roi.size > 0:
            fh, fw = roi.shape[:2]
            small = cv2.resize(roi, (max(1, fw // pixel_size), max(1, fh // pixel_size)), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
            rotated_frame[y1:y2, x1:x2] = pixelated
            
        unrotated_pixelated = cv2.warpAffine(rotated_frame, M_inv, (w_img, h_img), flags=cv2.INTER_LINEAR)
        
        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 5)
        alpha = np.stack([mask_blurred.astype(float) / 255.0] * 3, axis=-1)
        
        frame[:] = (frame * (1.0 - alpha) + unrotated_pixelated * alpha).astype(np.uint8)
    except Exception:
        pass
    return frame

# --- ОТРИСОВКА МАСКИ С СИЛЬНЫМ РАЗМЫТИЕМ ФОНА И ЭМОЦИЯМИ ---
def apply_dream_mask_bulletproof(frame, cx, cy, box_w, box_h, angle_deg, lx, ly, rx, ry, jaw_open, pucker_val, smile_val, frown_val, blink_l, blink_r, wide_l, wide_r, squint_l, squint_r):
    try:
        h_img, w_img = frame.shape[:2]
        box_h = max(45, int(box_h))
        box_w = max(45, int(box_w))
        angle_deg = max(-45, min(45, angle_deg))

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (box_w, box_h), angle_deg, 0, 360, 255, -1)
        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 5)
        alpha = np.stack([mask_blurred.astype(float) / 255.0] * 3, axis=-1)
        
        # Сильное размытие лица (эффект матового стекла)
        face_layer = cv2.GaussianBlur(frame, (101, 101), 50)
        
        face_size = max(20, np.hypot(rx - lx, ry - ly))
        base_eye_radius = max(6, int(face_size * 0.14))
        thickness = max(4, int(face_size * 0.07))
        cos_a, sin_a = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))

        # --- ТРЕХЭТАПНАЯ ОБРАБОТКА ГЛАЗ ---
        eyes_data = [
            (lx, ly, blink_l, wide_l, squint_l),
            (rx, ry, blink_r, wide_r, squint_r)
        ]
        for ex, ey, b_val, w_val, s_val in eyes_data:
            if b_val > 0.65:
                line_len = base_eye_radius * 1.2
                p1_x = int(ex - line_len * cos_a)
                p1_y = int(ey - line_len * sin_a)
                p2_x = int(ex + line_len * cos_a)
                p2_y = int(ey + line_len * sin_a)
                cv2.line(face_layer, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 0), thickness)
            else:
                scale = 1.0 - (b_val * 0.7) + (w_val * 0.4) - (s_val * 0.25)
                scale = max(0.25, min(1.5, scale))
                current_radius = max(3, int(base_eye_radius * scale))
                cv2.circle(face_layer, (int(ex), int(ey)), current_radius, (0, 0, 0), -1)

        # --- ДИНАМИЧЕСКИЙ РЕЧЕВОЙ КОНВЕЙЕР ДЛЯ РТА ---
        mx = int((lx + rx) / 2)
        my = int((ly + ry) / 2 + face_size * 0.55)

        if jaw_open > 0.03:
            if pucker_val > 0.12:
                mouth_w = int(face_size * 0.22)
                mouth_h = int(face_size * (0.20 + jaw_open * 0.6))
                cv2.ellipse(face_layer, (mx, my), (mouth_w, mouth_h), angle_deg, 0, 360, (0, 0, 0), -1)
            elif smile_val > 0.12:
                mouth_w = int(face_size * (0.42 + smile_val * 0.2))
                mouth_h = int(face_size * (0.25 + jaw_open * 0.5))
                cv2.ellipse(face_layer, (mx, my), (mouth_w, mouth_h), angle_deg, 0, 360, (0, 0, 0), -1)
            else:
                mouth_w = int(face_size * 0.32)
                mouth_h = int(face_size * (0.22 + jaw_open * 0.8))
                cv2.ellipse(face_layer, (mx, my), (mouth_w, mouth_h), angle_deg, 0, 360, (0, 0, 0), -1)
        elif frown_val > 0.08 and frown_val >= smile_val:
            mouth_w = int(face_size * 0.38)
            mouth_h = int(face_size * 0.22 * min(2.0, frown_val * 3.0))
            cv2.ellipse(face_layer, (mx, my + int(face_size * 0.06)), (mouth_w, mouth_h), angle_deg, 180, 360, (0, 0, 0), thickness)
        elif smile_val > 0.08:
            mouth_w = int(face_size * (0.38 + smile_val * 0.15))
            mouth_h = int(face_size * (0.24 + smile_val * 0.25))
            cv2.ellipse(face_layer, (mx, my), (mouth_w, mouth_h), angle_deg, 0, 180, (0, 0, 0), thickness)
            p1_x = int(mx - mouth_w * cos_a)
            p1_y = int(my - mouth_w * sin_a)
            p2_x = int(mx + mouth_w * cos_a)
            p2_y = int(my + mouth_w * sin_a)
            cv2.line(face_layer, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 0), thickness)
        else:
            mouth_w = int(face_size * 0.36)
            p1_x = int(mx - mouth_w * cos_a)
            p1_y = int(my - mouth_w * sin_a)
            p2_x = int(mx + mouth_w * cos_a)
            p2_y = int(my + mouth_w * sin_a)
            cv2.line(face_layer, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 0), thickness)
        
        frame[:] = (frame * (1.0 - alpha) + face_layer * alpha).astype(np.uint8)
    except Exception:
        pass
    return frame

window_name = 'Camera Mask'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

def toggle_window_visibility(name, hide):
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, name)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0 if hide else 5)
    except Exception:
        pass

window_hidden = False
mode = 'sticker'

stable_cx, stable_cy, stable_w, stable_h, stable_angle = None, None, None, None, 0.0
stable_lx, stable_ly, stable_rx, stable_ry = None, None, None, None
stable_jaw, stable_pucker, stable_smile, stable_frown = 0.0, 0.0, 0.0, 0.0
stable_blink_l, stable_blink_r = 0.0, 0.0
stable_wide_l, stable_wide_r = 0.0, 0.0
stable_squint_l, stable_squint_r = 0.0, 0.0

prev_gray = None
prev_pts_lk = None
lost_frames_counter = 0
MAX_LOST_FRAMES = 450

start_time = time.time()

while cap.isOpened():
    ret, frame = False, None
    try:
        ret, frame = cap.read()
    except Exception:
        pass
        
    if not ret or frame is None or frame.size == 0:
        raw_key = cv2.waitKey(1)
        if (raw_key & 0xFF) == 27:
            break
        time.sleep(0.01)
        continue
        
    frame = cv2.flip(frame, 1)
    h_img, w_img, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected_cx, detected_cy, detected_w, detected_h, detected_angle = None, None, None, None, 0.0
    d_lx, d_ly, d_rx, d_ry = None, None, None, None
    d_jaw, d_pucker, d_smile, d_frown = 0.0, 0.0, 0.0, 0.0
    d_blink_l, d_blink_r = 0.0, 0.0
    d_wide_l, d_wide_r = 0.0, 0.0
    d_squint_l, d_squint_r = 0.0, 0.0
    success = False

    # СЛОЙ 1: СЧИТЫВАНИЕ ЧЕРЕЗ MEDIAPIPE
    if use_mediapipe and landmarker is not None:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            if results and results.face_landmarks:
                landmarks = results.face_landmarks[0]
                pts = np.array([(int(lm.x * w_img), int(lm.y * h_img)) for lm in landmarks], np.float32)
                
                forehead = pts[10]
                chin = pts[152]
                bridge = pts[168]
                d_lx, d_ly = pts[33][0], pts[33][1]
                d_rx, d_ry = pts[263][0], pts[263][1]
                
                detected_cx = int(bridge[0])
                detected_cy = int((forehead[1] + chin[1]) / 2)
                face_h = abs(chin[1] - forehead[1])
                detected_h = int(face_h * 0.75)
                detected_w = int(face_h * 0.60)
                
                angle_rad = math.atan2(d_ry - d_ly, d_rx - d_lx)
                detected_angle = math.degrees(angle_rad)
                
                if results.face_blendshapes and len(results.face_blendshapes) > 0:
                    categories = results.face_blendshapes[0]
                    for cat in categories:
                        name = cat.category_name
                        score = cat.score
                        if name == 'jawOpen':
                            d_jaw = score
                        elif name == 'mouthPucker':
                            d_pucker = score
                        elif name in ['mouthSmileLeft', 'mouthSmileRight']:
                            d_smile = max(d_smile, score)
                        elif name in ['mouthFrownLeft', 'mouthFrownRight']:
                            d_frown = max(d_frown, score)
                        elif name == 'eyeBlinkLeft':
                            d_blink_l = score
                        elif name == 'eyeBlinkRight':
                            d_blink_r = score
                        elif name == 'eyeWideLeft':
                            d_wide_l = score
                        elif name == 'eyeWideRight':
                            d_wide_r = score
                        elif name == 'eyeSquintLeft':
                            d_squint_l = score
                        elif name == 'eyeSquintRight':
                            d_squint_r = score

                success = True
                prev_pts_lk = np.array([[detected_cx, detected_cy]], dtype=np.float32)
        except Exception:
            pass

    # СЛОЙ 2: ОПТИЧЕСКИЙ ПОТОК (СТРАХОВКА)
    if not success and prev_gray is not None and prev_pts_lk is not None:
        try:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_pts_lk, None)
            if status[0][0] == 1 and stable_cx is not None:
                flow_dx = next_pts[0][0] - prev_pts_lk[0][0]
                flow_dy = next_pts[0][1] - prev_pts_lk[0][1]
                
                detected_cx = int(stable_cx + flow_dx)
                detected_cy = int(stable_cy + flow_dy)
                detected_w = stable_w
                detected_h = stable_h
                detected_angle = stable_angle
                d_lx, d_ly, d_rx, d_ry = stable_lx, stable_ly, stable_rx, stable_ry
                d_jaw, d_pucker, d_smile, d_frown = stable_jaw, stable_pucker, stable_smile, stable_frown
                d_blink_l, d_blink_r = stable_blink_l, stable_blink_r
                d_wide_l, d_wide_r = stable_wide_l, stable_wide_r
                d_squint_l, d_squint_r = stable_squint_l, stable_squint_r
                prev_pts_lk = next_pts
                success = True
        except Exception:
            pass

    prev_gray = gray_frame.copy()

    # СЛОЙ 3: ФИЛЬТРАЦИЯ И EMA-СГЛАЖИВАНИЕ
    if success and detected_cx is not None:
        lost_frames_counter = 0
        if stable_cx is None:
            stable_cx, stable_cy, stable_w, stable_h = detected_cx, detected_cy, detected_w, detected_h
            stable_angle = detected_angle
            stable_lx, stable_ly, stable_rx, stable_ry = d_lx, d_ly, d_rx, d_ry
            stable_jaw, stable_pucker, stable_smile, stable_frown = d_jaw, d_pucker, d_smile, d_frown
            stable_blink_l, stable_blink_r = d_blink_l, d_blink_r
            stable_wide_l, stable_wide_r = d_wide_l, d_wide_r
            stable_squint_l, stable_squint_r = d_squint_l, d_squint_r
        else:
            distance = math.hypot(detected_cx - stable_cx, detected_cy - stable_cy)
            if distance < 250:
                alpha = 0.55
                stable_cx = int(stable_cx * (1 - alpha) + detected_cx * alpha)
                stable_cy = int(stable_cy * (1 - alpha) + detected_cy * alpha)
                stable_w = int(stable_w * (1 - alpha) + detected_w * alpha)
                stable_h = int(stable_h * (1 - alpha) + detected_h * alpha)
                stable_angle = stable_angle * (1 - alpha) + detected_angle * alpha
                if d_lx is not None:
                    stable_lx = stable_lx * (1 - alpha) + d_lx * alpha
                    stable_ly = stable_ly * (1 - alpha) + d_ly * alpha
                    stable_rx = stable_rx * (1 - alpha) + d_rx * alpha
                    stable_ry = stable_ry * (1 - alpha) + d_ry * alpha
                
                em_alpha = 0.85
                stable_jaw = stable_jaw * (1 - em_alpha) + d_jaw * em_alpha
                stable_pucker = stable_pucker * (1 - em_alpha) + d_pucker * em_alpha
                stable_smile = stable_smile * (1 - em_alpha) + d_smile * em_alpha
                stable_frown = stable_frown * (1 - em_alpha) + d_frown * em_alpha
                stable_blink_l = stable_blink_l * (1 - em_alpha) + d_blink_l * em_alpha
                stable_blink_r = stable_blink_r * (1 - em_alpha) + d_blink_r * em_alpha
                stable_wide_l = stable_wide_l * (1 - em_alpha) + d_wide_l * em_alpha
                stable_wide_r = stable_wide_r * (1 - em_alpha) + d_wide_r * em_alpha
                stable_squint_l = stable_squint_l * (1 - em_alpha) + d_squint_l * em_alpha
                stable_squint_r = stable_squint_r * (1 - em_alpha) + d_squint_r * em_alpha
    else:
        lost_frames_counter += 1

    if lost_frames_counter > MAX_LOST_FRAMES:
        stable_cx = None

    # СЛОЙ 4: ПОСТОБРАБОТКА И ОТРИСОВКА
    if stable_cx is not None and stable_w is not None:
        if mode == 'sticker':
            frame = apply_dream_mask_bulletproof(
                frame, stable_cx, stable_cy, stable_w, stable_h, stable_angle,
                stable_lx, stable_ly, stable_rx, stable_ry,
                stable_jaw, stable_pucker, stable_smile, stable_frown,
                stable_blink_l, stable_blink_r, stable_wide_l, stable_wide_r, stable_squint_l, stable_squint_r
            )
        elif mode == 'pixel':
            frame = apply_pixelate_bulletproof(
                frame, stable_cx, stable_cy, stable_w, stable_h, stable_angle, PIXEL_SIZE
            )

    cv2.imshow(window_name, frame)
    
    raw_key = cv2.waitKey(1)
    if raw_key != -1:
        key_code = raw_key & 0xFF
        if key_code == 27:
            break
        elif key_code in (ord('b'), ord('B'), 232, 200) or raw_key in (1080, 1048, ord('и'), ord('И')):
            mode = 'sticker' if mode == 'pixel' else 'pixel'
        elif key_code in (ord('h'), ord('H'), 13, 203) or raw_key in (1088, 1056, ord('р'), ord('Р')):
            window_hidden = not window_hidden
            toggle_window_visibility(window_name, window_hidden)

cap.release()
cv2.destroyAllWindows()