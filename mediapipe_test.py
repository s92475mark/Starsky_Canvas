import cv2
import time
import mediapipe as mp
import numpy as np

draw_hand = "Left" #工具手 "Right"為右手 "Left"為右手
newblack = np.full((10,10,3),(0,0,0),np.uint8)	#產生10x10黑色的圖
mpHand = mp.solutions.hands	#抓手	001
hands = mpHand.Hands()		#001
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness = 5 )	#設定點的參數
handConStyle = mpDraw.DrawingSpec(color=(255,255,255),thickness = 2 )	#設定線的參數
path = 0 #本地端可以改成這個，用筆電的視訊鏡頭
Hand_Mark_blue = (255,0,0)	#顏色藍色
Hand_Mark_red = (0,0,255)	#顏色紅色
# path = "./mediapipe_test.mp4" #這影片是我隨便抓YT上的short的

cap = cv2.VideoCapture(path)	#攝影機變數
pTime = 0 	#起始時間
f_round = True 	#第一次跑

def PointPprocessing(hands_Pose,hands_LR):	#固定繪圖手在第一位
	hands_LR1 = []	#固定左右手預定地
	if len(hands_Pose) == 2:	#兩隻手的狀況
		if hands_LR == draw_hand:	#座標調整，將繪圖手的座標固定在第一位
			hands_LR1 = [hands_Pose[0],hands_Pose[1]]
		else:
			hands_LR1 = [hands_Pose[1],hands_Pose[0]]
		# print("1",hands_LR1[0])
		# frame = cv2.circle(frame, (hands_LR1[0][0],hands_LR1[0][1]), 10, Hand_Mark_black, -1)
	elif len(hands_Pose) == 1:		#一隻手的狀況
		# print(hands_Pose[0])
		hands_LR1 = [hands_Pose[0]]
		# print(hands_LR1)
	else:
		pass
	return	hands_LR1

def HandsIdentify(imgRGB):		#此副程式會回傳食指指頭座標，與第一筆為左/右手
	results1 = []	#固定左右手預定地
	hands_Pose = [] #紀錄雙手食指座標
	hands_Pose1 = []
	results = hands.process(imgRGB)			#手部辨識001
	hands_LR = ""	#紀錄左手或右手
	if results.multi_hand_landmarks:		#判斷有沒有抓到手
		for handLms in results.multi_hand_landmarks:	#001 用迴圈一次處理一隻手的座標
			# forefinger = [int(handLms.landmark[8].x * frame.shape[1]),int(handLms.landmark[8].y * frame.shape[0])]
			hands_Pose1.append(handLms)
		hands_LR1 = results.multi_handedness	#medipi辨識左右手前置變數
		hands_LR = hands_LR1[0].classification[0].label 	#記錄第一筆座標是左手或右手
		results1 = PointPprocessing(hands_Pose1,hands_LR)	#副程式 固定繪圖手座標為第一位
		# print("固定後:",results1)

	return results1


while(True):
	ret,frame = cap.read()
	if not ret:
		print("沒有畫面")
		break
	if f_round:
		newblack = cv2.resize(newblack, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
		f_round = False
	frame = cv2.flip(frame, 1)	#畫面左右翻轉
	imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)	#將影像通道從BGR轉成RGB
	results1 = HandsIdentify(imgRGB)	#副程式處理"手部座標"、"第一個座標點是左手或右手"
	if len(results1) == 1:
		forefinger = [int(results1[0].landmark[8].x * frame.shape[1]),int(results1[0].landmark[8].y * frame.shape[0])]
		frame = cv2.circle(frame, forefinger, 10, Hand_Mark_blue, -1)
	elif len(results1) == 2:
		print(results1[0])
		forefinger = [int(results1[0].landmark[8].x * frame.shape[1]),int(results1[0].landmark[8].y * frame.shape[0])]
		frame = cv2.circle(frame, forefinger, 10, Hand_Mark_blue, -1)
		forefinger1 = [int(results1[1].landmark[8].x * frame.shape[1]),int(results1[1].landmark[8].y * frame.shape[0])]
		frame = cv2.circle(frame, forefinger1, 10, Hand_Mark_red, -1)

	cTime = time.time()
	fps =  1/(cTime-pTime)
	pTime = cTime
	# cv2.putText(frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255,0), 1, cv2.LINE_AA)
	# print(fps)
	cv2.imshow("live",frame)
	cv2.imshow("newblack",newblack)
	# time.sleep(0.5)	#跑影片要記得設time.sleep，跑視訊鏡頭要記得關  我花了40分鐘在debug為甚麼我的fps不到1
	if cv2.waitKey(1) == ord('q'):
		break
