import cv2
import time
import mediapipe as mp
import numpy as np


Main_hand = "Left" #設定主手 Left/Right
newblack = np.full((10,10,3),(0,0,0),np.uint8)	#產生10x10黑色的圖
mpHand = mp.solutions.hands	#抓手	001
hands = mpHand.Hands()		#001
path = 0 #本地端可以改成這個，用筆電的視訊鏡頭
cap = cv2.VideoCapture(path)	#攝影機變數
pTime = 0 	#起始時間
f_round = True 	#第一次跑
# mpDraw = mp.solutions.drawing_utils
# handLmsStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness = 5 )	#設定點的參數
# handConStyle = mpDraw.DrawingSpec(color=(255,255,255),thickness = 2 )	#設定線的參數


def PointPprocessing(hands_Pose,hands_LR):	#分別處理左右手座標之副程式
	global frame,Main_hand

	Hand_Mark_blue = (255,0,0)	#顏色藍色
	Hand_Mark_red = (0,0,255)	#顏色紅色
	for i in range(len(hands_LR)):
		Pose = (hands_Pose[hands_LR.index(hands_LR[i])])
		Pose1 = [int(Pose.landmark[8].x * frame.shape[1]),int(Pose.landmark[8].y * frame.shape[0])]
		if hands_LR[i] == Main_hand:	#主手
			frame = cv2.circle(frame, Pose1, 10, Hand_Mark_blue, -1)
		else:	
			frame = cv2.circle(frame, Pose1, 10, Hand_Mark_red, -1)
		

def HandsIdentify(imgRGB):		#副程式處理"手部座標"、"左右手順序"

	hands_Pose = [] #紀錄雙手食指座標
	results = hands.process(imgRGB)			#手部辨識001
	hands_LR = []	#紀錄左手或右手
	hands_LR1 = results.multi_handedness	#medipi辨識左右手前置變數
	if results.multi_hand_landmarks:		#判斷有沒有抓到手
		for i in range(len(results.multi_hand_landmarks)):	#001 用迴圈一次處理一隻手的座標
			hands_Pose.append(results.multi_hand_landmarks[i])
			hands_LR.append(hands_LR1[i].classification[0].label)
	return hands_Pose,hands_LR

while(True):
	ret,frame = cap.read()
	if not ret:
		print("沒有畫面")
		break
	if f_round:	#判斷是不是第一次跑，是:把黑色畫布放大成跟鏡頭解析度一樣大
		newblack = cv2.resize(newblack, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
		f_round = False
	frame = cv2.flip(frame, 1)	#畫面左右翻轉
	imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)	#將影像通道從BGR轉成RGB
	hands_Pose1,hands_LR = HandsIdentify(imgRGB)	#副程式處理"手部座標"、"左右手順序"
	results1 = PointPprocessing(hands_Pose1,hands_LR)	#分別處理左右手座標之副程式
	cTime = time.time()
	fps =  1/(cTime-pTime)
	pTime = cTime
	cv2.putText(frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255,0), 1, cv2.LINE_AA)
	# print(fps)
	cv2.imshow("live",frame)
	cv2.imshow("newblack",newblack)
	# time.sleep(0.5)	#跑影片要記得設time.sleep，跑視訊鏡頭要記得關  我花了40分鐘在debug為甚麼我的fps不到1
	if cv2.waitKey(1) == ord('q'):
		break
