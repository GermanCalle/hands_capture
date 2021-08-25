import math
import cv2
import mediapipe as mp
import time

class HandDetector():
    # Inicializamos los parametros de mediapipe
    def __init__(self, mode=False,max_hands=1,min_detection_confidence=0.5) -> None:
        self.mode = mode,
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
    
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,self.max_hands,self.min_detection_confidence)
        self.draw = mp.solutions.drawing_utils
        self.tips = [4,8,12,16,20]
        self.mcps = [1,5,9,13,17]

    def search_hands(self, frame, drawing=True):
        img_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_color)

        if self.results.multi_hand_landmarks:
            for land_mark in self.results.multi_hand_landmarks:
                if drawing:
                    self.draw.draw_landmarks(frame,land_mark,self.mp_hands.HAND_CONNECTIONS)
        
        return frame

    def search_position(self, frame, hand_num=0,drawing=True):
        x_list=[]
        y_list=[]
        bbox = []
        self.own_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(my_hand.landmark):
                height, width, _ = frame.shape
                cx,cy = int(lm.x*width), int(lm.y*height)
                x_list.append(cx)
                y_list.append(cy)
                self.own_list.append([id,cx,cy])
                if drawing:
                    cv2.circle(frame,(cx,cy),5,(0,0,0),cv2.FILLED)
            
            x_min,x_max = min(x_list), max(x_list)
            y_min,y_max = min(y_list), max(y_list)
            bbox = x_min, y_min, x_max, y_max
            if drawing:
                cv2.rectangle(frame,(x_min-20,y_min-20),(x_max+20,y_max+20), (0,255,0),2)
        return self.own_list, bbox
    
    def finger_positions(self):
        fingers = []
        if self.own_list[self.tips[0]][1] > self.own_list[self.tips[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for i in range(5):
            if self.own_list[self.tips[i]][2] < self.own_list[self.tips[i]-1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def distance(self,p1,p2,frame, drawing=True, r=15,t=3):
        x1, y1 = self.own_list[p1][1:]
        x2, y2 = self.own_list[p2][1:]
        cx, cy = (x1+x2)//2,(y1+y2)//2

        if drawing:
            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), t)
            cv2.circle(frame, (x1,y1), r, (x2,y2), (0,0,255), cv2.FILLED)
            cv2.circle(frame, (x2,y2), r, (x2,y2), (0,0,255), cv2.FILLED)
            cv2.circle(frame, (cx,cy), r, (x2,y2), (0,0,255), cv2.FILLED)
        length = math.hypot(x2-x1,y2-y1)

        return length, frame, [x1,y1,x2,y2,cx,cy]
    
def main():

    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        ret, frame = cap.read()
        frame = detector.search_hands(frame)
        my_list, bbox = detector.search_position(frame)
        
        finger_list = detector.finger_positions()
        if finger_list:
            print(finger_list)
        
        if my_list:
            print(my_list)

        cv2.imshow("hands",frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()