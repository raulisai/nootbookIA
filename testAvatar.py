import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

avatar_image = cv2.imread('avatar.png')

pose_detection = mp_pose.Pose(static_image_mode=False, model_complexity=2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose_detection.process(frame_rgb)

    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

        avatar_height, avatar_width, _ = avatar_image.shape

        body_top = int(keypoints[11, 1] * frame.shape[0])
        body_bottom = int(keypoints[24, 1] * frame.shape[0])
        body_left = int(keypoints[11, 0] * frame.shape[1]) - avatar_width // 2
        body_right = int(keypoints[12, 0] * frame.shape[1]) + avatar_width // 2

        body_top = max(0, body_top)
        body_bottom = min(frame.shape[0], body_bottom)
        body_left = max(0, body_left)
        body_right = min(frame.shape[1], body_right)

        if body_right > body_left:  # Verificar que el ancho de la región del cuerpo sea mayor que 0
            body_region = frame[body_top:body_bottom, body_left:body_right]
            resized_avatar = cv2.resize(avatar_image, (body_right - body_left, body_bottom - body_top))
            frame[body_top:body_bottom, body_left:body_right] = resized_avatar

    cv2.imshow('Avatar', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
