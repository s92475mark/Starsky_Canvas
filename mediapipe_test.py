import cv2
import time
import mediapipe as mp
import numpy as np

# mpPose = mp.solutions.pose	抓身體
# pose  = mpPose.Pose()
newblack = np.full((10,10,3),(0,0,0),np.uint8)
# cv2.imshow("black",newblack)
mpHand = mp.solutions.hands	#抓手
hands = mpHand.Hands()

mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness = 5 )	#設定點的參數
handConStyle = mpDraw.DrawingSpec(color=(255,255,255),thickness = 2 )	#設定線的參數
path = 0 #本地端可以改成這個，用筆電的視訊鏡頭
# path = "./mediapipe_test.mp4" #這影片是我隨便抓YT上的short的

cap = cv2.VideoCapture(path)
pTime = 0
f_round = True


while(True):
	ret,frame = cap.read()
	if not ret:
		print("沒有畫面")
		break

	if f_round:
		newblack = cv2.resize(newblack, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
		f_round = False
		# print(frame.shape[0])

	frame = cv2.flip(frame, 1)
	imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)
	HANDS = results.multi_hand_landmarks
	if results.multi_hand_landmarks:
		hands_Pose = [] #紀錄雙手食指座標
		# print("1",hands_Pose)
		HowMuchHands = 0
		for handLms in results.multi_hand_landmarks:
			forefinger = [int(handLms.landmark[8].x * frame.shape[1]),int(handLms.landmark[8].y * frame.shape[0])]
			HowMuchHands = HowMuchHands + 1
			frame = cv2.circle(frame, forefinger, 10, (0,0,225), -1)
			print("1",forefinger)
			# hands_Pose = hands_Pose.append(test)
			# print("2",hands_Pose)
		# for handLms in results.multi_hand_landmarks:
			# mpDraw.draw_landmarks(frame,handLms,mpHand.HAND_CONNECTIONS,handLmsStyle,handConStyle)


	cTime = time.time()
	fps =  1/(cTime-pTime)
	pTime = cTime
	cv2.putText(frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255), 1, cv2.LINE_AA)
	cv2.imshow("live",frame)
	cv2.imshow("newblack",newblack)
	# time.sleep(0.5)	#跑影片要記得設time.sleep，跑視訊鏡頭要記得關  我花了40分鐘在debug為甚麼我的fps不到1
	if cv2.waitKey(1) == ord('q'):
		break
