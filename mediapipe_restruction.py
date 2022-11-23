import cv2
import time
import mediapipe as mp
import numpy as np
import math

#語音套件
import librosa
from keras.models import load_model
from keras.utils import to_categorical
import playsound #play wav用  pip install playsound
import os
import threading
import wave
import pyaudio
max_pad_len = 120
voice_on = "on" # on/off

# 值， 址id
newblack = np.full((10, 10, 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
mpHand = mp.solutions.hands  # 抓手	001
hands = mpHand.Hands()  # 001
path = 0  # 本地端可以改成這個，用筆電的視訊鏡頭
cap = cv2.VideoCapture(path)  # 攝影機變數
pTime = 0  # 起始時間
f_round = True  # 第一次跑
color = (0, 0, 255)
lost_pix = 0.3  # 縮小比例0~1之間
offset = [0, 0]  # 偏移(x為正往右偏移，y為正往下偏移)
dots = []
Mode = 'Draw'  # 'Draw'為作畫模式/ 'Func' 為功能板模式
mod = 1
Hand_Mark_blue = (255, 0, 0)  # 顏色藍色
Hand_Mark_red = (0, 0, 255)  # 顏色紅色
colorx = 255
colorz = 255
colory = 255
Main_hand = "Right"  # 設定主手 Left/Right
sub_Pose2 = []
main_Pose2 = []
r_standard = 0  # 縮放用-五指平均半徑
middle_standard = [-20, -20]  # 縮放用-中心點判定
time_standard = 0
distance = []
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)  # 設定點的參數
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)  # 設定線的參數


def graphics_menu():
    graphics_menu = np.full((int(frame.shape[0]), int(frame.shape[1] / 5), 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
    cv2.rectangle(graphics_menu, (10, 10), (40, 40), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(graphics_menu, "square", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(graphics_menu, (10, 70), (40, 100), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(graphics_menu, 'round', (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return graphics_menu


def Mouse_Pos(Pos):  # 轉換成鼠標層座標
    global offset, lost_pix
    # Pos = Pos
    Pos = (int((Pos[0] - offset[0]) / lost_pix), int((Pos[1] - offset[1]) / lost_pix))
    # main_MousePose = (int((main_MousePose[0] - offset[0]) / lost_pix), int((main_MousePose[1] - offset[1]) / lost_pix))
    # sub_MousePose = (int((sub_MousePose[0] - offset[0]) / lost_pix), int((sub_MousePose[1] - offset[1]) / lost_pix))
    return Pos


def Mouse(Canvas, main_MousePose, sub_MousePose, mod):
    MouseLevel = np.full((Canvas.shape[0], Canvas.shape[1], 3), (0, 0, 0), np.uint8)  # 產生與newblack大小相同黑色的圖
    if mod != 4:
        MouseLevel = cv2.circle(MouseLevel, main_MousePose, 10, (255, 255, 255), -1)  # 在這層上面點上主手白色鼠標
        MouseLevel = cv2.circle(MouseLevel, sub_MousePose, 10, (0, 255, 0), -1)  # 在這層上面點上副手綠色鼠標
    TrueCanvas = cv2.add(Canvas, MouseLevel)
    return TrueCanvas


# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 180

    return angle_


def Hand_Text(finger_angle):  # 根據手指角度的串列內容，返回對應的手勢名稱
    f0 = finger_angle[0]  # 大拇指角度
    f1 = finger_angle[1]  # 食指角度
    f2 = finger_angle[2]  # 中指角度
    f3 = finger_angle[3]  # 無名指角度
    f4 = finger_angle[4]  # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f0 >= 50 and f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50:
        return '2'  # 比ya
    elif f0 >= 50 and f1 < 50 and f2 > 50 and f3 >= 50 and f4 >= 50:
        return '1'  # 伸出食指
    elif f0 < 50 and f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50:
        return '5'  # 張開手掌
    elif f0 > 50 and f1 > 50 and f2 > 50 and f3 > 50 and f4 > 50:
        return '0'  # 握拳
    elif f0 < 50 and f1 > 50 and f2 > 50 and f3 > 50 and f4 > 50:
        return '4'  # 比讚
    else:
        return ''


def hand_angle(hand_):  # 計算五隻手指的角度函式
    angle_list = []
    # thumb 大拇指角度
    # print("xy",hand_[0])
    # print("x",hand_[0][0])

    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    # print(angle_)
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    # print(angle_list)
    return angle_list


def ScalingDisplacement(newblack, lost_pix, offset):  # 畫布的縮放位移
    smailblack = newblack.copy()  # 複製
    # smailblack1 = smailblack[int(lost_pix+offset[1]):int(newblack.shape[0]-lost_pix+offset[1]),
    # int(lost_pix + offset[0]):int(newblack.shape[1]-lost_pix + offset[0])]	#
    # print(offset[0],(int(newblack.shape[1] * lost_pix) + offset[0]))
    smailblack1 = smailblack[(offset[1]):(int(newblack.shape[0] * lost_pix) + offset[1]),
                  offset[0]:(int(newblack.shape[1] * lost_pix) + offset[0])]

    newblack1 = newblack.copy()
    cv2.rectangle(newblack1, ((offset[0]), (offset[1])),
                  (int(newblack.shape[1] * lost_pix + offset[0]), int(newblack.shape[0] * lost_pix + offset[1])),
                  (255, 255, 0), 3)
    # print(newblack.shape)
    smailblack1 = cv2.resize(smailblack1, (newblack.shape[1], newblack.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imshow("newblack1", newblack1)

    # newblack3 = cv2.resize(smailblack1, ((newblack.shape[1]-(lost_pix * 2)),(newblack.shape[0]-(lost_pix * 2))), interpolation=cv2.INTER_AREA)
    # newblack[lost_pix:(newblack.shape[0]-lost_pix),lost_pix:(newblack.shape[1]-lost_pix)] = newblack3

    return smailblack1


""""
讀取到手->
雙手座標->
雙手顯示鼠標->
判斷角度->手勢 -> 
	"狀態" 切換為作畫模式或功能版模式 -> 
		1. 作畫模式：主手食指判讀與否
		2. 功能版模式：切換功能 直到 副手為"5"，切換回作畫模式			  
"""


def PointPprocessing(hands_Pose, hands_LR, menu, Main_hand, colormain):  # 分別處理左右手座標之副程式	(左手要做什麼，右手要做什麼 分別計算)
    global frame, color

    # 若手不再畫面內，重設參數(??)
    main_finger_points = []  # 記錄主手指節點座標的串列
    sub_finger_points = []
    main_mouse_pos = [-10, -10]  # 主手鼠標座標
    sub_mouse_pos = [-10, -10]  # 副手鼠標座標
    main_hand_text = ""
    sub_hand_text = ""
    main_Pose = {}
    sub_Pose = {}
    main_Pose1 = []
    sub_Pose1 = []

    # 當讀取到雙手序列
    for i in range(len(hands_LR)):
        # 讀取到主手
        if hands_LR[i] == Main_hand:
            main_Pose = (hands_Pose[i])  # 當前抓取主手手部21個座標
            # print(main_Pose.landmark[8])
            main_Pose1 = [int(main_Pose.landmark[8].x * frame.shape[1]),
                          int(main_Pose.landmark[8].y * frame.shape[0])]  # 當前抓取到的手的食指座標
            # print(main_Pose1)
            # 顯示主手藍色鼠標於監視器上
            main_mouse_pos = [int(main_Pose.landmark[8].x * frame.shape[1]),
                              int(main_Pose.landmark[8].y * frame.shape[0])]  # 主手食指 給鼠標用
            frame = cv2.circle(frame, main_Pose1, 10, Hand_Mark_blue, -1)  # 鼠標藍色 顯示於 監視器上

            ### 將主手 21 個節點換算成座標，記錄到 finger_points
            # print(main_Pose.landmark[8].z)
            for i in main_Pose.landmark:
                x = i.x * frame.shape[1]
                y = i.y * frame.shape[0]
                main_finger_points.append((x, y))
            # print(i)
            # print(main_finger_points)
            main_finger_angle = hand_angle(main_finger_points)  # 計算手指角度，回傳長度為 5 的串列
            # 判斷手勢
            main_hand_text = Hand_Text(main_finger_angle)  # 取得手勢所回傳的內容

        # return main_hand_text, main_finger_points, main_Pose, main_Pose1
        else:  # 讀取到副手
            sub_Pose = (hands_Pose[hands_LR.index(hands_LR[i])])  # 當前抓取到的手的全部座標
            sub_Pose1 = [int(sub_Pose.landmark[8].x * frame.shape[1]),
                         int(sub_Pose.landmark[8].y * frame.shape[0])]  # 當前抓取到的手的食指座標
            # print("1111",sub_Pose1)
            # print("shape1",frame.shape[1])
            # 副手鼠標顯示在監視器上
            sub_mouse_pos = [int(sub_Pose.landmark[8].x * frame.shape[1]),
                             int(sub_Pose.landmark[8].y * frame.shape[0])]  # 副手食指位置給鼠標用
            frame = cv2.circle(frame, sub_mouse_pos, 10, Hand_Mark_blue, -1)  # 鼠標藍色 顯示於 監視器上

            ### 將 21 個節點換算成座標，記錄到 finger_points
            for i in sub_Pose.landmark:
                x = i.x * frame.shape[1]
                y = i.y * frame.shape[0]
                sub_finger_points.append((int(x), int(y)))
            # print(sub_finger_points)
            ###判斷手勢
            sub_finger_angle = hand_angle(sub_finger_points)  # 計算手指角度，回傳長度為 5 的串列
            sub_hand_text = Hand_Text(sub_finger_angle)  # 取得手勢所回傳的內容
        # return sub_hand_text, sub_finger_points, sub_Pose, sub_Pose1, Mode
        Function_Select(main_hand_text, sub_hand_text, main_finger_points, sub_finger_points, main_Pose, sub_Pose,
                        main_Pose1, sub_Pose1, menu, frame, colormain)

    return main_mouse_pos, sub_mouse_pos


def Function_Select(main_hand_text, sub_hand_text, main_finger_points, sub_finger_points, main_Pose, sub_Pose,
                    main_Pose1, sub_Pose1, menu, frame, colormain):
    # 主手執行作畫
    global lost_pix, dots, color, Mode, colorx, colory, colorz, mod, smailblack1, fingertip, r_standard, middle_standard, time_standard, sub_Pose2, main_Pose2,distance, newblack
    # print(Mode)
    # print(sub_hand_text,Mode,mod)
    if Mode == 'Draw' and main_hand_text == '1':

        # 轉為"紅色鼠標"於監視器上
        frame = cv2.circle(frame, main_Pose1, 10, Hand_Mark_red, -1)  # 鼠標藍色 顯示於 監視器上
        fx = int(main_finger_points[8][0])  # 如果手勢為 1，記錄食指末端的座標
        fy = int(main_finger_points[8][1])
        dots.append([fx, fy])  # 記錄食指座標
        # print(dots)
        dl = len(dots)
        if dl > 1:
            dx1 = dots[dl - 2][0]
            dy1 = dots[dl - 2][1]  # 上一刻的食指xy座標
            dx2 = dots[dl - 1][0]
            dy2 = dots[dl - 1][1]  # 這一刻的食指xy座標
            cv2.line(newblack, (dx1, dy1), (dx2, dy2), color, 5)  # 取兩個時間差的點畫線，在黑色畫布上
        # print(dots)
        if dl >= 100:  ###當dots累積超過50組座標，將上上一刻與上衣刻的座標記錄起來，並刷新整組座標紀錄
            dots = [(dots[dl - 2]), (dots[dl - 1])]
    # print(dots)
    # 若副手伸出食中指 : 1. 伸出"副手食中指"，則停止作畫功能 -> 進入功能選擇階段 -> 直到"副手全張開" 則關閉功能選擇階段， 可以繼續作畫
    elif sub_hand_text == '1' and mod == 1 and Mode != 'zoon_move':
        Mode = 'Func'  # 停止主手迴圈，進入副手迴圈
        fx = int(sub_finger_points[8][0] / 2)  # 如果手勢為 1，記錄食指末端的座標
        fy = int(sub_finger_points[8][1])
        menu = cv2.circle(menu, (int(sub_Pose1[0] / 2), int(sub_Pose1[1])), 10, (255, 255, 255), -1)  # 製作副手鼠標 並繪製於功能版上
        cv2.imshow("menu", menu)  # 顯示副手鼠標+功能版
        # 若副手食指座標移動到以下位置，則切換顏色
        if 10 <= fy <= 40 and 10 <= fx <= 40:
            mod = 2
        elif 70 <= fy <= 100 and 10 <= fx <= 40:
            Mode = 'zoon_move'
        elif 130 <= fy <= 160 and 10 <= fx <= 40:
            mod = 'graphics'
        elif 190 <= fy <= 220 and 10 <= fx <= 40:
            pass
        elif 250 <= fy <= 280 and 10 <= fx <= 40:
            pass
        else:
            dots.clear()
    elif Mode == 'Func' and sub_Pose1 != [] and sub_hand_text != '2' and mod == 1:
        menu = cv2.circle(menu, (int(sub_Pose1[0] / 2), int(sub_Pose1[1])), 10, (255, 255, 255), -1)  # 製作副手鼠標 並繪製於功能版上
        cv2.imshow("menu", menu)  # 顯示副手鼠標+功能版
    elif Mode == 'zoon_move':
        fingertip = []  # 縮放用-尖座標
        fingertip_R = 0
        middle = 0
        move_range = 30
        cv2.putText(smailblack1, "zoon_move", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if sub_finger_points:
            try:
                cv2.destroyWindow("menu")
            except:
                pass
            for i in range(4, 21, 4):  # 抓出副手五指
                fivefingerpos = Mouse_Pos(sub_finger_points[i])  # 轉換成鼠標層座標
                a = [int(sub_finger_points[i][0]), int(sub_finger_points[i][1])]
                fingertip.append(a)
            middle = int((fingertip[0][0] + fingertip[3][0]) / 2), int(
                (fingertip[0][1] + fingertip[3][1]) / 2)  # 計算中心點座標
            Dmiddle = (((middle[0] - middle_standard[0]) ** 2) + (
                    (middle[1] - middle_standard[1]) ** 2)) ** 0.5  # 座標偏移量
            for j in range(len(fingertip)):
                fingertip_R1 = (((fingertip[j][0] - middle[0]) ** 2 + (fingertip[j][1] - middle[1]) ** 2) ** 0.5)
                fingertip_R = fingertip_R + fingertip_R1
            fingertip_R = int(fingertip_R / 5)  # 平均長度
            cv2.circle(smailblack1, (middle), int(fingertip_R), (0, 255, 0), 1)
            middle1 = Mouse_Pos(middle)
            smailblack1 = cv2.circle(smailblack1, (middle1), 10, color, -1)
            if mod != 4:
                if Dmiddle >= 20:  # 如果中心點偏移20pix以上 重置時間與中心點位置
                    middle_standard = [middle[0], middle[1]]
                    time_standard = time.time()
                if abs(fingertip_R - r_standard) > 6:
                    r_standard = fingertip_R
                    time_standard = time.time()
                if (time.time() - time_standard) > 1.5:
                    mod = 4
            elif mod == 4:
                cv2.circle(smailblack1, (middle), (int(r_standard + r_standard / 5)), (255, 255, 0), 2)  # 放大縮小的範圍
                cv2.circle(smailblack1, (middle), int(r_standard - r_standard / 7), (255, 255, 0), 2)  # 放大縮小的範圍
                cv2.line(smailblack1, (middle), (middle_standard), color, 5)
                if fingertip_R > (r_standard + r_standard / 5) and lost_pix < 1:  # 此 為 用
                    if (offset[1] >= frame.shape[0] * (1 - lost_pix)) or (offset[0] >= frame.shape[1] * (1 - lost_pix)):
                        pass
                    else:
                        lost_pix = lost_pix + 0.01  # 判 縮 。
                elif fingertip_R < (r_standard - 3) and lost_pix > 0.3:  # 斷 放
                    lost_pix = lost_pix - 0.01  # 式 使
                if abs(middle[0] - middle_standard[0]) > move_range:
                    offset[0] = int(offset[0] + ((middle[0] - middle_standard[0]) / 20))
                if abs(middle[1] - middle_standard[1]) > move_range:
                    offset[1] = int(offset[1] + ((middle[1] - middle_standard[1]) / 20))
                if offset[0] < 0:
                    offset[0] = 0
                elif offset[0] > frame.shape[1] * (1 - lost_pix):
                    offset[0] = int(frame.shape[1] * (1 - lost_pix))
                if offset[1] < 0:
                    offset[1] = 0
                elif offset[1] > frame.shape[0] * (1 - lost_pix):
                    offset[1] = int(frame.shape[0] * (1 - lost_pix))
            if main_hand_text == '5':
                Mode = 'Func'
                mod = 1
    elif mod == 'graphics' and sub_Pose1 != []:
        if sub_Pose1 != []:
            menu = graphics_menu()
            fx = int(sub_Pose1[0])  # 如果手勢為 1，記錄食指末端的座標
            fy = int(sub_Pose1[1])
            if Mode == 'Func':
                menu = cv2.circle(menu, sub_Pose1, 10, (255, 255, 255), -1)
                cv2.imshow("menu", menu)
        if 10 <= fy <= 40 and 10 <= fx <= 40 and sub_hand_text == '1' and Mode == 'Func':
            Mode = 'square'
            try:
                cv2.destroyWindow("menu")
            except:
                pass
        elif 70 <= fy <= 100 and 10 <= fx <= 40 and sub_hand_text == '1' and Mode == 'Func':
            Mode = 'round'
            try:
                cv2.destroyWindow("menu")
            except:
                pass
        if main_hand_text == "1" and sub_hand_text == "1" and Mode == 'square':
            # main_Pose1 sub_Pose1
            main_Pose2 = main_Pose1
            sub_Pose2 = sub_Pose1
            main_Pose1 = Mouse_Pos(main_Pose1)
            sub_Pose1 = Mouse_Pos(sub_Pose1)
            smailblack1 = cv2.rectangle(smailblack1, main_Pose1, sub_Pose1, (colorx, colory, colorz), 7)
        elif main_hand_text == "2" and main_Pose2 != [] and sub_Pose2 != [] and Mode == 'square':
            cv2.rectangle(newblack, main_Pose2, sub_Pose2, (colorx, colory, colorz), 5)
            main_Pose2 = []
            sub_Pose2 = []
        elif main_hand_text == "1" and sub_hand_text == "1" and Mode == 'round':
            main_Pose2 = main_Pose1
            sub_Pose2 = sub_Pose1
            main_Pose1 = Mouse_Pos(main_Pose1)
            sub_Pose1 = Mouse_Pos(sub_Pose1)
            distance = int(math.sqrt((main_Pose1[0] - sub_Pose1[0]) ** 2 + (main_Pose1[1] - sub_Pose1[1]) ** 2))
            smailblack1 = cv2.circle(smailblack1, main_Pose1, distance, (0, 0, 225), -1)
        elif main_hand_text == "2" and Mode == 'round' and distance != []:
            distance=int(math.sqrt((main_Pose2[0] - sub_Pose2[0]) ** 2 + (main_Pose2[1] - sub_Pose2[1]) ** 2))
            cv2.circle(newblack, main_Pose2, distance, (0, 0, 225), -1)
            #main_Pose2 = []
        if main_hand_text == "5" and Mode != []:
            Mode = 'Func'
            mod = 1
        #print(fx, fy)
        #print(Mode)
        #print(main_hand_text)
    # 副手全張：關閉功能版，轉回繪畫模式
    elif sub_hand_text == '2' and Mode == 'Func' and mod == 1:
        Mode = 'Draw'
        try:
            cv2.destroyWindow("menu")
        except:
            pass

    if mod == 2 and Mode != 'zoon_move':
        if sub_hand_text == '1' and 70 <= int(sub_Pose1[0]) <= 580 and 70 <= int(sub_Pose1[1]) <= 120:
            colorx = int((sub_Pose1[0] - 70) / 2)
        elif sub_hand_text == '1' and 70 <= int(sub_Pose1[0]) <= 580 and 150 <= int(sub_Pose1[1]) <= 200:
            colory = int((sub_Pose1[0] - 70) / 2)
        elif sub_hand_text == '1' and 70 <= int(sub_Pose1[0]) <= 580 and 230 <= int(sub_Pose1[1]) <= 280:
            colorz = int((sub_Pose1[0] - 70) / 2)
        elif sub_hand_text == '5':
            color = (colorz, colory, colorx)
            try:
                cv2.destroyWindow("menu")
            except:
                pass
            mod = 1
        else:
            dots.clear()
        if sub_Pose1:
            colormain = cv2.circle(colormain, (sub_Pose1[0], sub_Pose1[1]), 10, (255, 255, 255), -1)
        # print(sub_Pose1)
        cv2.putText(colormain, str(int(colorx)), (600, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(colormain, str(int(colory)), (600, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(colormain, str(int(colorz)), (600, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(colormain, (70, 400), (520, 420), (colorz, colory, colorx), -1)  # 在畫面上方放入紅色正方形
        cv2.rectangle(colormain, (70, 420), (90, 460), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
        cv2.imshow("menu", colormain)


    elif Mode == 'Draw' and main_hand_text == '5' and sub_hand_text == '5':
        newblack = np.full((frame.shape[1], frame.shape[0], 3), (0, 0, 0), np.uint8)   


    return Mode
# 若主手不伸出食指作畫，則清除主手座標紀錄
# print(mod)
# return Mode


def func_window():  ###準備功能視窗 -> menu
    # collor = [(0, 0, 255),(0, 255, 255),(255, 0, 255),(135,23,126)]
    # x = 20
    # y = int(((int(frame.shape[1] / 4)) - (len(collor) * x))/(len(collor)+1))
    # print(y)
    menu = np.full((10, 10, 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
    menu = cv2.resize(menu, (int(frame.shape[1] / 4), frame.shape[0]), interpolation=cv2.INTER_AREA)  # 依照讀取到的畫面調整功能版大小
    # smailblack2 = ScalingDisplacement(menu, lost_pix, offset)  # 縮小畫布
    # for i in range(len(collor)):
    # 	cv2.rectangle(menu, (int(y*(i+1) + x * i),30), (int((i+1)*(y+x)), (30+x)), collor[i], -1)  # 在畫面上方放入紅色正方形
    cv2.rectangle(menu, (10, 10), (40, 40), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, "color", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 70), (40, 100), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'zoom', (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 130), (40, 160), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'graphics', (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 190), (40, 220), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'else', (50, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 250), (40, 280), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'else', (50, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return menu


def func_color():
    colormain = np.full(((frame.shape[0]), int(frame.shape[1] + 120), 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
    # colormain = cv2.resize(colormain, (int(frame.shape[1] + 120), frame.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.rectangle(colormain, (70, 70), (580, 120), (0, 0, 255, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.rectangle(colormain, (70, 150), (580, 200), (0, 255, 0, 255), -1)  # 在畫面上方放入綠色正方形
    cv2.rectangle(colormain, (70, 230), (580, 280), (255, 0, 0, 255), -1)  # 在畫面上方放入藍色正方形

    return colormain


def HandsIdentify(img):  # 副程式處理"手部座標"、"左右手順序"
    hands_Pose = []  # 紀錄雙手食指座標
    results = hands.process(img)  # 手部辨識001
    hands_LR = []  # 紀錄左手或右手
    hands_LR1 = results.multi_handedness  # medipi辨識左右手前置變數
    if results.multi_hand_landmarks:  # 判斷有沒有抓到手
        for i in range(len(results.multi_hand_landmarks)):  # 001 用迴圈一次處理一隻手的座標
            hands_Pose.append(results.multi_hand_landmarks[i])
            hands_LR.append(hands_LR1[i].classification[0].label)
    return hands_Pose, hands_LR

def wav2mfcc(file_path, max_pad_len=max_pad_len): #音頻預處理
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    # print(wave.shape) #(112014,)
    wave = wave[::3] 
    # print("wave[::3].shape:",wave.shape) #(37338,) (除3 ??)
    mfcc = librosa.feature.mfcc(wave, sr=16000) #SR 採樣頻率
    # print("mfcc.shape in wav2mfcc before padding:",mfcc.shape) #(20 ,73)
    pad_width = max_pad_len - mfcc.shape[1] #11 -73
    # pad_width =  mfcc.shape[1] - max_pad_len  #差距73-11 = 62
    if pad_width <0:
      mfcc = mfcc[:,0:max_pad_len]
    #   print("mfcc.shape pad_width>11",mfcc.shape)
    else:
      mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # print("mfcc.shape in wav2mfcc after padding:",mfcc.shape) #(20 ,73)
    return mfcc

#用Pyaudio錄製音頻
def audio_record(out_file, rec_time):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16 #16bit编码格式
    CHANNELS = 2 #单声道
    RATE = 16000 #16000采样频率
    p = pyaudio.PyAudio()
    # 创建音频流
    stream = p.open(format=FORMAT, # 音频流wav格式
                    channels=CHANNELS, # 单声道
                    rate=RATE, # 采样率16000
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Start Recording...")
    frames = [] # 录制的音频流
    # 录制音频数据
    for i in range(0, int(RATE / CHUNK * rec_time)):
        data = stream.read(CHUNK)
        frames.append(data)
    # 录制完成
    #print(frames)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存音频文件
    with wave.open(out_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# 把函式放到改寫到類的run方法中，便可以通過呼叫類方法，實現執行緒的終止
class VoiceStoppableThread(threading.Thread):
	def __init__(self,daemon=None):
		super(VoiceStoppableThread,self).__init__(daemon=daemon)
		self.__is_running = True
		self.daemon = daemon
		self.func = ""
		#讀取語音模型
		self.model = load_model('./models/best_hier.h5')
	def terminate(self):
		self.__is_running = False
	def run(self):
		pid = os.getpid() # 當前進程組 ID
		
		while self.__is_running:
			if voice_on == 'on':
				print("喚醒功能開啟...開始錄音")
				#語音存檔路徑與檔名
				audio_call_path = "./record_wav/calling.wav"
				# 錄製語音指令 ，秒數
				audio_record(audio_call_path, 2.5) 
				print("開始喚醒語音識別...")
				
				#預測
				mfcc = wav2mfcc(audio_call_path)  #這裡放上要判讀的語音檔
				mfcc_reshaped = mfcc.reshape(1, 20, max_pad_len, 1)
				# print("labels= ['mark_pen.npy', 'eraser.npy', 'call_func.npy']") # label 要針對訓練時的label來定
				# print("labels= ['mark_pen.npy', 'eraser.npy', 'call_func.npy', 'hey_julia.npy', 'hey_star.npy']") #vgg16 labels
				label_list = ['mark_pen.npy', 'eraser.npy', 'call_func.npy', 'hey_julia.npy', 'hey_star.npy', 'others.npy']
				label_idx = np.argmax(self.model.predict(mfcc_reshaped))
				# print("predict=", label_list[label_idx]) # 印出最高機率項目
				# print("prob=", self.model.predict(mfcc_reshaped))			# 印出個項目的機率
				prob_list = self.model.predict(mfcc_reshaped)
				# print(prob_list[0])
				print("predict={} prob={}".format(label_list[label_idx], prob_list[0][label_idx]))

				#喚醒程式
				voice_pre = str(np.argmax(self.model.predict(mfcc_reshaped)))
				#當聽到 hey julia時
				if voice_pre == "hey_julia":
					#play julia.mp3
					
					#進入語音錄製 與 AI 判讀一次， 輸出 功能項目
					# print("開始功能語音錄音")
					# audio_func_path = "./record_wav/func.wav"
					# audio_record(audio_func_path, 2.5)
					# print("開始功能語音識別")
					# mfcc = wav2mfcc('./func.wav')
					# mfcc_reshaped = mfcc.reshape(1, 20, max_pad_len, 1)
					# print("labels= ['mark_pen.npy', 'eraser.npy', 'call_func.npy']")
					# print("predict=", np.argmax(model.predict(mfcc_reshaped)))
					# self.func = str(np.argmax(model.predict(mfcc_reshaped)))

					pass
				# 當聽到hey 星空時
				elif voice_pre == "hey_star":
					#playsound.playsound('./meowx2.wav') # meow meow~
					#進入語音錄製 與 AI 判讀一次， 輸出 功能項目
					# print("開始功能語音錄音")
					# audio_func_path = "./record_wav/func.wav"
					# audio_record(audio_func_path, 2.5)
					# print("開始功能語音識別")
					# mfcc = wav2mfcc('./func.wav')
					# mfcc_reshaped = mfcc.reshape(1, 20, max_pad_len, 1)
					# print("labels= ['mark_pen.npy', 'eraser.npy', 'call_func.npy']")
					# print("predict=", np.argmax(model.predict(mfcc_reshaped)))
					# self.func = str(np.argmax(model.predict(mfcc_reshaped)))
					pass
				else:
					pass
				
				time.sleep(0.1) #每0.1秒重新跑一次thread
			elif voice_on == 'off':
				time.sleep(0.5) #每0.5秒判斷一次是否重新開啟麥克風

if __name__ == '__main__':
    #執行 語音thread
    voice_thread = VoiceStoppableThread() #創建一個可終止程序的語音thread
    voice_thread.daemon = True			  #thread True, 判定開啟
    voice_thread.start()

    # voice_thread.terminate() #終止thread 用
    # pid = os.getpid() #可以查process ID
    # print("start pid:", pid)
    
    while (True):
        ret, frame = cap.read()
        if not ret:  # 判定有沒有畫面存在
            print("沒有畫面")
            break

        CanvasSize = (frame.shape[1], frame.shape[0])  # 畫布大小
        blur = cv2.GaussianBlur(frame, (7, 7), cv2.BORDER_DEFAULT)
        # blur = cv2.Canny(blur, 125, 175)
        blur = cv2.dilate(blur, (7, 7), iterations=1)
        if f_round:  # 判斷是不是第一次跑，是:把黑色畫布放大成跟鏡頭解析度一樣大
            newblack = cv2.resize(newblack, CanvasSize, interpolation=cv2.INTER_AREA)
            f_round = False
        smailblack1 = ScalingDisplacement(newblack, lost_pix, offset)  # 縮小畫布
        frame = cv2.flip(frame, 1)  # 畫面左右翻轉，放回畫面frame
        blur = cv2.flip(blur, 1)

        imgRGB = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)  # 將影像通道從BGR轉成RGB

        menu = func_window()  # 初始化功能版
        colormain = func_color()
        hands_Pose1, hands_LR = HandsIdentify(imgRGB)  # 副程式處理"手部座標"、"左右手順序"
        main_MousePose, sub_MousePose = PointPprocessing(hands_Pose1, hands_LR, menu, Main_hand,
                                                         colormain)  # 分別處理左右手座標之副程式
        # Function_Select(main_hand_text, sub_hand_text, main_finger_points, sub_finger_points,main_Pose, sub_Pose,main_Pose1, sub_Pose1)
        main_MousePose = Mouse_Pos(main_MousePose)
        sub_MousePose = Mouse_Pos(sub_MousePose)
        TrueCanvas = Mouse(smailblack1, main_MousePose, sub_MousePose, mod)  # 加入鼠標 回傳最終畫布
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        # print("newblack",newblack.shape)
        cv2.imshow("live", frame)
        cv2.imshow("liv", blur)
        # TrueCanvas = cv2.resize(TrueCanvas, (1920,1920), interpolation=cv2.INTER_AREA)	#resize 指令用於調整畫布大小
        cv2.imshow("TrueCanvas", TrueCanvas)
        # time.sleep(0.5)	#跑影片要記得設time.sleep，跑視訊鏡頭要記得關  我花了40分鐘在debug為甚麼我的fps不到1
        if cv2.waitKey(1) == ord('q'):
            break
