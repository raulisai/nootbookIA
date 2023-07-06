import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

left_hand_image = cv2.imread('avatar.png')
right_hand_image = cv2.imread('avatar.png')
pose_image = cv2.imread('avatar.png')

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1) as holistic:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.left_hand_landmarks is not None:
            for landmark in results.left_hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                if x > 0 and y > 0:
                    top = y - left_hand_image.shape[0] // 2
                    bottom = top + left_hand_image.shape[0]
                    left = x - left_hand_image.shape[1] // 2
                    right = left + left_hand_image.shape[1]
                    if top >= 0 and bottom < frame.shape[0] and left >= 0 and right < frame.shape[1]:
                        frame[top:bottom, left:right] = left_hand_image

        if results.right_hand_landmarks is not None:
            for landmark in results.right_hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                if x > 0 and y > 0:
                    top = y - right_hand_image.shape[0] // 2
                    bottom = top + right_hand_image.shape[0]
                    left = x - right_hand_image.shape[1] // 2
                    right = left + right_hand_image.shape[1]
                    if top >= 0 and bottom < frame.shape[0] and left >= 0 and right < frame.shape[1]:
                        frame[top:bottom, left:right] = right_hand_image

        if results.pose_landmarks is not None:
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                if x > 0 and y > 0:
                    top = y - pose_image.shape[0] // 2
                    bottom = top + pose_image.shape[0]
                    left = x - pose_image.shape[1] // 2
                    right = left + pose_image.shape[1]
                    if top >= 0 and bottom < frame.shape[0] and left >= 0 and right < frame.shape[1]:
                        frame[top:bottom, left:right] = pose_image

        frame = cv2.flip(frame, 1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
