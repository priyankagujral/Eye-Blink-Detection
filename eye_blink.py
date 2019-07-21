from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict

facial_landmark_indexes = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
    	("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 35)),
        ("jaw", (0, 17))
        ])

def shape_to_numpy(shape, dtype="int"):
    np_array = np.zeros((68,2), dtype = dtype)
    for i in range(0, 68):
        np_array[i] = (shape.part(i).x, shape.part(i).y)
    return np_array

def eye_aspect_ratio(eye):
    a_dist = dist.euclidean(eye[1], eye[5])
    b_dist = dist.euclidean(eye[2], eye[4])
    c_dist = dist.euclidean(eye[0], eye[3])
    ear = (a_dist + b_dist)/(2.0 * c_dist)
    return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

eye_thresh = 0.3
cons_frames = 1
counter_frames = 0
total_blinks = 0

cap = cv2.VideoCapture(0)
if(cap.isOpened() == False):
    print("can't Play")
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    (l_start,l_end) = facial_landmark_indexes["left_eye"]
    (r_start, r_end) = facial_landmark_indexes["right_eye"]
    for (i, face) in enumerate(faces):
        face_align = predictor(gray, face)
        face_align_numpy = shape_to_numpy(face_align)
        left_eye_pts = face_align_numpy[l_start:l_end]
        right_eye_pts = face_align_numpy[r_start:r_end]
        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        for (i, (x, y)) in enumerate(left_eye_pts):
            cv2.circle(frame, (x,y), 1, (0, 255, 0), -1)
        for (i, (x, y)) in enumerate(right_eye_pts):
            cv2.circle(frame, (x,y), 1, (0, 255, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        ear = (left_ear + right_ear)/2.0
        if ear < eye_thresh:
            counter_frames += 1
        else:
            if counter_frames > cons_frames:
                total_blinks += 1
            counter_frames = 0  
        cv2.putText(frame, "Blinks: "+str(total_blinks), (10, 30), font, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "ear: "+str(ear), (300, 30), font, 0.5, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
      
cap.release()
cv2.destroyAllWindows()
    
    







