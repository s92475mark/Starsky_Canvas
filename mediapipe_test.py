import cv2
import time
import mediapipe as mp
import numpy as np


mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose  = mpPose.Pose()

path = 0 #本地端可以改成這個，用筆電的視訊鏡頭
# path = "./mediapipe_test.mp4" #這影片是我隨便抓YT上的short的

cap = cv2.VideoCapture(path)
pTime = 0


while(True):
	ret,frame = cap.read()
	if not ret:
		print("沒有畫面")
		break

	imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	results = pose.process(imgRGB)
	print(results.pose_landmarks)
	if results.pose_landmarks:
		mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)


	cTime = time.time()
	fps =  1/(cTime-pTime)
	pTime = cTime
	cv2.putText(frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255), 1, cv2.LINE_AA)
	cv2.imshow("live",frame)
	# time.sleep(0.5)	#跑影片要記得設time.sleep，跑視訊鏡頭要記得關  我花了40分鐘在debug為甚麼我的fps不到1
	if cv2.waitKey(1) == ord('q'):
		break
