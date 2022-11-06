import cv2
import time
import mediapipe as mp
import numpy as np
import math



newblack = np.full((10,10,3),(0,0,0),np.uint8)	#產生10x10黑色的圖
mpHand = mp.solutions.hands	#抓手	001
hands = mpHand.Hands()		#001
path = 0 #本地端可以改成這個，用筆電的視訊鏡頭
cap = cv2.VideoCapture(path)	#攝影機變數
pTime = 0 	#起始時間
f_round = True 	#第一次跑
color = (0,0,255)
lost_pix = 60 	#縮小禎數
offset = [0,0]	#偏移(x為正往右偏移，y為正往下偏移)

# mpDraw = mp.solutions.drawing_utils
# handLmsStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness = 5 )	#設定點的參數
# handConStyle = mpDraw.DrawingSpec(color=(255,255,255),thickness = 2 )	#設定線的參數
def Mouse(Canvas,CanvasSize,MousePose):	#鼠標層覆蓋上畫布
	MouseLevel = np.full((Canvas.shape[0],Canvas.shape[1],3),(0,0,0),np.uint8)	#產生與newblack大小相同黑色的圖
	MouseLevel = cv2.circle(MouseLevel, MousePose, 10, (255,255,255), -1)	#在這層上面點上白色鼠標
	# cv2.imshow("MouseLevel",MouseLevel)
	TrueCanvas = cv2.add(Canvas,MouseLevel)
	return TrueCanvas
# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
	v1_x = v1[0]
	v1_y = v1[1]
	v2_x = v2[0]
	v2_y = v2[1]
	try:
		angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
	except:
		angle_ = 180
	
	return angle_

def hand_angle(hand_):#計算五隻手指的角度函式
	angle_list = []
	# thumb 大拇指角度
	# print(hand_)
	angle_ = vector_2d_angle(
		((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
		((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
		)
	# print(angle_)
	angle_list.append(angle_)
	# index 食指角度
	angle_ = vector_2d_angle(
		((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
		((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
		)
	angle_list.append(angle_)
	# middle 中指角度
	angle_ = vector_2d_angle(
		((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
		((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
		)
	angle_list.append(angle_)
	# ring 無名指角度
	angle_ = vector_2d_angle(
		((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
		((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
		)
	angle_list.append(angle_)
	# pink 小拇指角度
	angle_ = vector_2d_angle(
		((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
		((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
		)
	angle_list.append(angle_)
	# print(angle_list)
	return angle_list

def ScalingDisplacementTophalf(newblack,lost_pix):	#畫布的縮放位移
	# smailblack = newblack.copy()	#複製
	# smailblack1 = smailblack[lost_pix:(newblack.shape[0]-lost_pix),lost_pix:(newblack.shape[1]-lost_pix)]	#
	# smailblack1 = cv2.resize(smailblack1, (newblack.shape[1],newblack.shape[0]), interpolation=cv2.INTER_AREA)
	
	# newblack3 = cv2.resize(newblack2, ((newblack.shape[1]-(lost_pix * 2)),(newblack.shape[0]-(lost_pix * 2))), interpolation=cv2.INTER_AREA)
	# newblack[lost_pix:(newblack.shape[0]-lost_pix),lost_pix:(newblack.shape[1]-lost_pix)] = newblack3
	# return smailblack1
	pass

def PointPprocessing(hands_Pose,hands_LR):	#分別處理左右手座標之副程式	(左手要做什麼，右手要做什麼 分別計算)
	global frame,color,smailblack1
	Main_hand = "Left" #設定主手 Left/Right
	finger_points = []			# 記錄手指節點座標的串列
	Hand_Mark_blue = (255,0,0)	#顏色藍色
	Hand_Mark_red = (0,0,255)	#顏色紅色
	Pose2 = [-10,-10] 	#鼠標座標
	for i in range(len(hands_LR)):
		Pose = (hands_Pose[hands_LR.index(hands_LR[i])])
		Pose1 = [int(Pose.landmark[8].x * frame.shape[1]),int(Pose.landmark[8].y * frame.shape[0])]
		###################################雙手處理###############################################
		if hands_LR[i] == Main_hand:	#主手
			Pose2 = [int(Pose.landmark[8].x * frame.shape[1]),int(Pose.landmark[8].y * frame.shape[0])]	#主手食指 給鼠標用
			frame = cv2.circle(frame, Pose1, 10, Hand_Mark_blue, -1)
			forefinger0 = [int(Pose.landmark[8].x * frame.shape[1]), int(Pose.landmark[8].y * frame.shape[0])]  # 食指頂端座標
			frame = cv2.circle(frame, forefinger0, 10, (0, 0, 225), -1)
			# 將 21 個節點換算成座標，記錄到 finger_points
			for i in Pose.landmark:
				x = i.x*frame.shape[1]
				y = i.y*frame.shape[0]
				finger_points.append((x,y))
			if finger_points:
				finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
				# 小於 50 表示手指伸直，大於等於 50 表示手指捲縮 ,finger_angle[0]大拇指角度,finger_angle[1]食指角度,finger_angle[2]中指角度,finger_angle[3]無名指角度,finger_angle[4]小拇指角度
				if finger_angle[1]<50 and finger_angle[0]>50 and finger_angle[2]>50 and finger_angle[3]>50 and finger_angle[4]>50:#當食指伸直後進行畫圖
					frame = cv2.circle(frame, (forefinger0[0], forefinger0[1]), 10, color, -1)#color是指顏色上面有預設值紅色
					smailblack1 = cv2.circle(smailblack1, (forefinger0[0], forefinger0[1]), 5, color, -1)
				# if finger_angle[1]<50 and finger_angle[0]<50 and finger_angle[2]<50 and finger_angle[3]<50 and finger_angle[4]<50 and forefinger0[0]<=20 and  forefinger0[1]<=20:#手指全部張開並且移到綠框時畫筆顏色變為綠
				# 	color=(0, 255, 0)#顏色為綠色
				# if finger_angle[1]<50 and finger_angle[0]<50 and finger_angle[2]<50 and finger_angle[3]<50 and finger_angle[4]<50 and forefinger0[0]<=20 and  20<=forefinger0[1]<=40:#手指全部張開並且食指移到紅色框格內畫筆顏色變為紅
				# 	color=(0, 0, 255)#顏色為紅色
		else:	#副手運算
			frame = cv2.circle(frame, Pose1, 10, Hand_Mark_blue, -1)
	return Pose2

def HandsIdentify(imgRGB):		#副程式處理"手部座標"、"左右手順序"

	hands_Pose = [] #紀錄雙手食指座標
	results = hands.process(imgRGB)			#手部辨識001
	hands_LR = []	#紀錄左手或右手
	hands_LR1 = results.multi_handedness	#medipi辨識左右手前置變數
	if results.multi_hand_landmarks:		#判斷有沒有抓到手
		for i in range(len(results.multi_hand_landmarks)):	#001 用迴圈一次處理一隻手的座標
			hands_Pose.append(results.multi_hand_landmarks[i])
			hands_LR.append(hands_LR1[i].classification[0].label)
	# print(hands_LR)
	return hands_Pose,hands_LR

if __name__ == '__main__':
	while(True):
		ret,frame = cap.read()
		if not ret:		#判定有沒有畫面存在
			print("沒有畫面")
			break
		CanvasSize = (frame.shape[1], frame.shape[0])	#畫布大小
		if f_round:	#判斷是不是第一次跑，是:把黑色畫布放大成跟鏡頭解析度一樣大
			newblack = cv2.resize(newblack, CanvasSize, interpolation=cv2.INTER_AREA)
			f_round = False
		cv2.imshow("newblack",newblack)
		frame = cv2.flip(frame, 1)	#畫面左右翻轉
		imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)	#將影像通道從BGR轉成RGB
		# newblack1 = ScalingDisplacementTophalf(newblack,lost_pix)	#畫布處理上半
		smailblack = newblack.copy()	#複製
		smailblack1 = smailblack[lost_pix:(newblack.shape[0]-lost_pix),lost_pix:(newblack.shape[1]-lost_pix)]	#
		smailblack1 = cv2.resize(smailblack1, (newblack.shape[1],newblack.shape[0]), interpolation=cv2.INTER_AREA)

		# cv2.imshow("newblack1",newblack1)
		hands_Pose1,hands_LR = HandsIdentify(imgRGB)	#副程式處理"手部座標"、"左右手順序"
		MousePose = PointPprocessing(hands_Pose1,hands_LR)	#分別處理左右手座標之副程式

		newblack3 = cv2.resize(smailblack1, ((newblack.shape[1]-(lost_pix * 2)),(newblack.shape[0]-(lost_pix * 2))), interpolation=cv2.INTER_AREA)
		newblack[lost_pix:(newblack.shape[0]-lost_pix),lost_pix:(newblack.shape[1]-lost_pix)] = newblack3

		TrueCanvas = Mouse(smailblack1,CanvasSize,MousePose)	#加入鼠標 回傳最終畫布
		cTime = time.time()
		fps =  1/(cTime-pTime)
		pTime = cTime
		cv2.putText(frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255,0), 1, cv2.LINE_AA)
		# print(fps)
		cv2.imshow("live",frame)
		cv2.imshow("TrueCanvas",TrueCanvas)
		# time.sleep(0.5)	#跑影片要記得設time.sleep，跑視訊鏡頭要記得關  我花了40分鐘在debug為甚麼我的fps不到1
		if cv2.waitKey(1) == ord('q'):
			break
