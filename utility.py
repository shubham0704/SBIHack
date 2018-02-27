from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/master/Desktop/GIT/digitize_sbi_backend/controllers/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def num_blinks(video):

	vs = FileVideoStream(video).start()
	fileStream = True
	COUNTER = 0
	TOTAL = 0
	time.sleep(1.0)
	best_frame = None
	max_ear = -1
	while True:

		if fileStream and not vs.more():
			break
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		rects = detector(gray, 0)
		
		
		for rect in rects:

			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			ear = (leftEAR + rightEAR) / 2.0
			print ear
			if ear > max_ear:
				best_frame = gray
				max_ear = ear
			
			if ear < EYE_AR_THRESH:
				COUNTER += 1
			else:

				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1

				COUNTER = 0
	vs.stop()
	return TOTAL, best_frame

if __name__ == '__main__':
    
    #TOTAL, best_frame = num_blinks("/home/master/Desktop/GIT/digitize_sbi_backend/2017-11-22-225553.webm")
    TOTAL, best_frame = num_blinks("/home/master/Videos/Webcam/2017-11-23-010207.webm")
    
    #print type(best_frame)
    #exit()
    plt.imshow(best_frame)
    plt.show()
