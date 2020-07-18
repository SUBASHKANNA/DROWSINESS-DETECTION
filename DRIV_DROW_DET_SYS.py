import cv2
import numpy as np
import dlib
from math import hypot
import winsound

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN
eye_closed_dur=0

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 1)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 0, 255), 1)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (255, 255, 0), 1)

        landmarks = predictor(gray, face)
        

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

        print("eye aspect ratio",eye_ratio)
       
            
        if eye_closed_dur>5 and eye_ratio>4.9:
            cv2.putText(frame, "SLEEPY",(450,150), font, 3, (0, 0, 255))
            winsound.Beep(2000,500)

        elif eye_ratio > 4.9:
            cv2.putText(frame, "BLINKING", (10, 150), font, 3, (0, 0, 255))
            eye_closed_dur+=1
            x="Frames="+str(eye_closed_dur)
            cv2.putText(frame,x ,(180,50), font, 3, (0, 0, 255))
        else:
            eye_closed_dur=0
            x="Frames="+str(eye_closed_dur)
            cv2.putText(frame,x ,(180,50), font, 3, (0, 0, 255))



    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
