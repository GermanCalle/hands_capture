import cv2
import mediapipe as mp

index = [4,8,12,16,20]
peace = [8,12]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands = 1,
    min_detection_confidence = 0.5) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame,1)

        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                for i,points in enumerate(hand_landmarks.landmark):
                    if i in peace:
                        x = int(points.x*width)
                        y = int(points.y*height)
                        print(i,x,y)

        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
