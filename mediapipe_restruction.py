import csv
import math
import os
import threading
import time
import wave
import cv2

# play wav用  pip install playsound==1.2.2
# 語音套件
import librosa
import mediapipe as mp
import numpy as np
import pyaudio
import requests
from keras.models import load_model
from playsound import playsound  # pip install playsound==1.2.2
from pydub import AudioSegment  # 載入 pydub 的 AudioSegment 模組
from pydub.playback import play  # 載入 pydub.playback 的 play 模組

# 語音參數
# 1.5second pad:65
# 2second pad:87
# 2.5second pad:107
# 3second pad: 129
max_pad_len = 107
max_pad_len2 = 87
voice_on = "on"  # on/off 語音功能開或關
sr_set = 22050
n_mfcc = 60
voice_pre_func = ""  # ['mark_pen.npy', 'eraser.npy', 'call_func.npy'] "0", "1", "2"
voice_check_func = ""  # "eraser"畫面清洗 "menu"開啟功能版 "draw" 轉回繪畫模式

count_long = 0
count_short = 0

# 值， 址id
newblack = np.full((10, 10, 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
mpHand = mp.solutions.hands  # 抓手	001
hands = mpHand.Hands()  # 001
path = 0  # 本地端可以改成這個，用筆電的視訊鏡頭

pTime = 0  # 起始時間
f_round = True  # 第一次跑
color = (0, 0, 255)
lost_pix = 0.3  # 縮小比例0~1之間
offset = [0, 0]  # 偏移(x為正往右偏移，y為正往下偏移)
dots = []
Mode = 'Draw'  # 'Draw'為作畫模式/ 'Func' 為功能板模式
mod = 1
pic_change = 0
Hand_Mark_blue = (255, 0, 0)  # 顏色藍色
Hand_Mark_red = (0, 0, 255)  # 顏色紅色
colorx = 255
colorz = 255
colory = 255
Main_hand = "Right"  # 設定主手 Left/Right
sub_Pose2 = []
main_Pose2 = []
r_standard = 0  # 縮放用-五指平均半徑
time_standard_long = 1.5
middle_standard = [-20, -20]  # 縮放用-中心點判定
time_standard = 0
distance = []
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)  # 設定點的參數
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)  # 設定線的參數
token = ' '
thickness = 5
pics = ['pig', 'smile', 'money', 'heart', 'plans']
sub_hand_text = '-1'
text=1

def graphics_menu():
    graphics_menu = np.full((int(frame.shape[0]), int(frame.shape[1] / 4), 3), (0, 0, 0), np.uint8)  # 產生視訊畫面大小的黑色的圖
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
    elif f0 < 50 and f1 < 50 and f2 >= 50 and f3 >= 50 and f4 < 50:
        return '6'  # disco
    elif f0 < 50 and f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50:
        return '7'  # child
    elif f0 >= 50 and f1 < 50 and f2 < 50 and f3 < 50 and f4 >= 50:
        return '3'  # sunglass
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

            main_finger_angle = hand_angle(main_finger_points)  # 計算手指角度，回傳長度為 5 的串列
            # 判斷手勢
            main_hand_text = Hand_Text(main_finger_angle)  # 取得手勢所回傳的內容

        # return main_hand_text, main_finger_points, main_Pose, main_Pose1
        elif hands_LR[i] != Main_hand:  # 讀取到副手
            sub_Pose = (hands_Pose[hands_LR.index(hands_LR[i])])  # 當前抓取到的手的全部座標
            sub_Pose1 = [int(sub_Pose.landmark[8].x * frame.shape[1]),
                         int(sub_Pose.landmark[8].y * frame.shape[0])]  # 當前抓取到的手的食指座標
            # 副手鼠標顯示在監視器上
            sub_mouse_pos = [int(sub_Pose.landmark[8].x * frame.shape[1]),
                             int(sub_Pose.landmark[8].y * frame.shape[0])]  # 副手食指位置給鼠標用
            frame = cv2.circle(frame, sub_mouse_pos, 10, Hand_Mark_blue, -1)  # 鼠標藍色 顯示於 監視器上

            ### 將 21 個節點換算成座標，記錄到 finger_points
            for i in sub_Pose.landmark:
                x = i.x * frame.shape[1]
                y = i.y * frame.shape[0]
                sub_finger_points.append((int(x), int(y)))

            ###判斷手勢
            sub_finger_angle = hand_angle(sub_finger_points)  # 計算手指角度，回傳長度為 5 的串列
            sub_hand_text = Hand_Text(sub_finger_angle)  # 取得手勢所回傳的內容
        # return sub_hand_text, sub_finger_points, sub_Pose, sub_Pose1, Mode
    Function_Select(main_hand_text, sub_hand_text, main_finger_points, sub_finger_points, main_Pose, sub_Pose,
                    main_Pose1, sub_Pose1, menu, frame, colormain)

    return main_mouse_pos, sub_mouse_pos, sub_hand_text


def Function_Select(main_hand_text, sub_hand_text, main_finger_points, sub_finger_points, main_Pose, sub_Pose,
                    main_Pose1, sub_Pose1, menu, frame, colormain):
    # 主手執行作畫
    global lost_pix, dots, color, Mode, colorx, colory, colorz, mod, smailblack1, fingertip, r_standard, middle_standard, time_standard_long, time_standard, sub_Pose2, main_Pose2, distance, newblack, token, pic_change, Mode, voice_check_func, voice_on, cover_pics,text
    print(Mode)
    print(mod)
    if Mode == 'Draw' and sub_hand_text == '7':
        mod = '4'
        Mode = 'Draw'
    if Mode == 'Draw' and main_hand_text == '1' and sub_hand_text != '7':
        # 轉為"紅色鼠標"於監視器上
        frame = cv2.circle(frame, main_Pose1, 10, Hand_Mark_red, -1)  # 鼠標藍色 顯示於 監視器上
        fx = int(main_finger_points[8][0])  # 如果手勢為 1，記錄食指末端的座標
        fy = int(main_finger_points[8][1])
        dots.append([fx, fy])  # 記錄食指座標
        dl = len(dots)
        if dl > 1:
            dx1 = dots[dl - 2][0]
            dy1 = dots[dl - 2][1]  # 上一刻的食指xy座標
            dx2 = dots[dl - 1][0]
            dy2 = dots[dl - 1][1]  # 這一刻的食指xy座標
            cv2.line(newblack, (dx1, dy1), (dx2, dy2), color, 5)  # 取兩個時間差的點畫線，在黑色畫布上
        if dl >= 100:  ###當dots累積超過50組座標，將上上一刻與上衣刻的座標記錄起來，並刷新整組座標紀錄
            dots = [(dots[dl - 2]), (dots[dl - 1])]
    else:
        dots=[]
    # 若副手伸出食中指 : 1. 伸出"副手食中指"，則停止作畫功能 -> 進入功能選擇階段 -> 直到"副手全張開" 則關閉功能選擇階段， 可以繼續作畫
    if sub_hand_text == '1' and mod == 1 and Mode != 'zoon_move' or voice_check_func == "menu":
        # print("function show up menu, voice_check_func = ", voice_check_func)
        voice_check_func = ""  # reset voice check func

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
            Mode = "Draw"
            mod = '4'
            try:
                cv2.destroyWindow("menu")
            except:
                pass

        elif 250 <= fy <= 280 and 10 <= fx <= 40:
            cv2.imwrite('./test.png', newblack)
            sendLineNotify(token)
        elif 310 <= fy <= 340 and 10 <= fx <= 40:
            exit()
        else:
            dots.clear()
    elif Mode == 'Func' and sub_Pose1 != [] and sub_hand_text != '2' and mod == 1:
        menu = cv2.circle(menu, (int(sub_Pose1[0] / 2), int(sub_Pose1[1])), 10, (255, 255, 255), -1)  # 製作副手鼠標 並繪製於功能版上
        cv2.imshow("menu", menu)  # 顯示副手鼠標+功能版

    if Mode == 'zoon_move':
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
                if mod != 4:
                    smailblack1 = cv2.circle(smailblack1, a, 10, (170, 0, 170), -1)
            middle = int((fingertip[0][0] + fingertip[3][0]) / 2), int(
                (fingertip[0][1] + fingertip[3][1]) / 2)  # 計算中心點座標
            Dmiddle = (((middle[0] - middle_standard[0]) ** 2) + (
                    (middle[1] - middle_standard[1]) ** 2)) ** 0.5  # 座標偏移量
            for j in range(len(fingertip)):
                fingertip_R1 = (((fingertip[j][0] - middle[0]) ** 2 + (fingertip[j][1] - middle[1]) ** 2) ** 0.5)
                fingertip_R = fingertip_R + fingertip_R1
            time1 = (int((time.time() - time_standard) / (time_standard_long / 5)))
            if time1 <= 5 and mod != 4:
                for k in range(time1):
                    try:
                        cv2.line(smailblack1, (fingertip[k]), (fingertip[k + 1]), color, 5)
                    except:
                        cv2.line(smailblack1, (fingertip[4]), (fingertip[0]), color, 5)
            fingertip_R = int(fingertip_R / 5)  # 平均長度
            if mod == 4:
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
                if (time.time() - time_standard) > time_standard_long:
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
    elif mod == 'graphics' and sub_hand_text:

        menu = graphics_menu()
        fx = int(sub_finger_points[8][0])  # 如果手勢為 1，記錄食指末端的座標
        fy = int(sub_finger_points[8][1])
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
        print("main_hand_text", main_hand_text)
        print("sub_hand_text", sub_hand_text)
        if main_hand_text == "1" and sub_hand_text == "1" and Mode == 'square':
            text = '0'
            sub = sub_Pose1
            main = main_Pose1
            submouse = Mouse_Pos(sub)
            mainmouse = Mouse_Pos(main)
            smailblack1 = cv2.rectangle(smailblack1, submouse, mainmouse, (colorz, colory, colorx),
                                        int(thickness * lost_pix))
        elif sub_hand_text == "1" and main_hand_text == "2" and Mode == 'square' and text == '0':
            sub = sub_Pose1
            main = main_Pose1
            cv2.rectangle(newblack, sub, main, (colorz, colory, colorx), int(thickness * lost_pix))
            text='1'
        if main_hand_text == "1" and sub_hand_text == "1" and Mode == 'round':
            text = '0'
            sub = sub_Pose1
            main = main_Pose1
            submouse = Mouse_Pos(sub)
            mainmouse = Mouse_Pos(main)
            distance = int(math.sqrt((mainmouse[0] - submouse[0]) ** 2 + (mainmouse[1] - submouse[1]) ** 2))
            smailblack1 = cv2.circle(smailblack1, mainmouse, distance, (colorz, colory, colorx), -1)
        elif main_hand_text == "2" and Mode == 'round' and distance != [] and text=='0':
            sub = sub_Pose1
            main = main_Pose1
            submouse = Mouse_Pos(sub)
            mainmouse = Mouse_Pos(main)
            distance = int(math.sqrt((main[0] - sub[0]) ** 2 + (main[1] - sub[1]) ** 2))
            cv2.circle(newblack, main, distance, (0, 0, 225), -1)
            main_Pose2 = []
            text='1'
    elif Mode == "Draw" and mod == '4':
        key = cv2.waitKey(1)
        print(pic_change)
        if pic_change >= 5:
            # print("2345")
            pic_change = 0

        elif key & 0xFF == 83 or key & 0xFF == ord('e'):
            cover_pics = cv2.imread('./child_pic/' + pics[pic_change] + '.png')
            pic_change += 1
        child_black = np.full((int(frame.shape[0]), int(frame.shape[1]), 3), (0, 0, 0), np.uint8)  # 產生黑色的圖
        child_black, Mode, smailblack1 = child_Mode(child_black, cover_pics,
                                                    smailblack1)  # 呼叫child Mode
        if sub_hand_text == '2':
            mod = 1
    if main_hand_text == "5" and mod == 'graphics':
        Mode = 'Func'
        mod = 1
    # print(fx, fy)
    # print(Mode)
    # print(main_hand_text)

    # 副手全張：關閉功能版，轉回繪畫模式
    elif sub_hand_text == '2' and Mode == 'Func' and mod == 1 or voice_check_func == "draw":
        # print("關閉功能版，轉回繪圖模式。voice_check_func=", voice_check_func)
        voice_check_func = ""  # 重設語音功能
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

    # 繪圖模式下，雙手比五：清洗畫面
    if Mode == 'Draw' and main_hand_text == '5' and sub_hand_text == '5' or voice_check_func == "eraser":
        # print("清除畫面, voice_check_func = ",voice_check_func)
        voice_check_func = ""
        newblack = np.full((frame.shape[0], frame.shape[1], 3), (0, 0, 0), np.uint8)


    elif Mode == 'Draw' and main_hand_text == '6':
        # 起始座標
        ix, iy = 0, 0

        def mouse(event, x, y):
            global ix, iy
            # 如果為滑鼠點擊事件
            if event == cv2.EVENT_LBUTTONDOWN:
                ix, iy = x, y
                return ix, iy

        # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        # 帶入子畫面影片
        subcap = cv2.VideoCapture("disco.mp4")
        subcap.set(cv2.CAP_PROP_POS_FRAMES, 100)

        # 設定回傳
        cv2.setMouseCallback('newblack1', mouse)
        # 讀取影片
        while cap.isOpened():
            ret, frame = cap.read()
            subret, subframe = subcap.read()

            # 子畫面寬/高縮放
            subframe = cv2.resize(subframe, (subframe.shape[1] // 10, subframe.shape[0] // 10))
            subw, subh = subframe.shape[:2]
            # 將子畫面放在指定位置，(x,y)是左上角的坐标
            if ix > frame.shape[1] - subw or iy > frame.shape[0] - subh:
                ix, iy = 0, 0
            frame[iy:iy + subframe.shape[0], ix:ix + subframe.shape[1]] = subframe

            cv2.imshow('newblack1', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('newblack1', cv2.WND_PROP_AUTOSIZE) < 1:
                break
            else:
                return Mode
    # 若為繪畫模式, 右手比三的時候
    elif Mode == 'Draw' and main_hand_text == '3':
        # 建立偵測方法
        mp_face_detection = mp.solutions.face_detection
        # 建立繪圖方法
        mp_drawing = mp.solutions.drawing_utils

        # cap = cv2.VideoCapture(0)
        # pTime = 0
        # cTime = 0

        # 開始偵測人臉
        with mp_face_detection.FaceDetection(
                min_detection_confidence=0.7) as face_detection:

            while cap.isOpened():
                success, image = cap.read()
                imgFront = cv2.imread("canvas.png", cv2.IMREAD_UNCHANGED)
                s_h, s_w, _ = imgFront.shape

                imageHeight, imageWidth, _ = image.shape
                # 將BGR轉換成RGB, 並使用Mediapipe人臉偵測進行處理
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # 繪製每張人臉的臉部偵測
                if results.detections:
                    for detection in results.detections:
                        # 鼻子
                        normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                             mp_face_detection.FaceKeyPoint.NOSE_TIP)
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                               normalizedLandmark.y,
                                                                                               imageWidth, imageHeight)
                        Nose_tip_x = pixelCoordinatesLandmark[0]
                        Nose_tip_y = pixelCoordinatesLandmark[1]
                        # 左耳
                        normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                             mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                               normalizedLandmark.y,
                                                                                               imageWidth, imageHeight)
                        Left_Ear_x = pixelCoordinatesLandmark[0]
                        Left_Ear_y = pixelCoordinatesLandmark[1]
                        # 右耳
                        normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                             mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                               normalizedLandmark.y,
                                                                                               imageWidth, imageHeight)
                        Right_Ear_x = pixelCoordinatesLandmark[0]
                        Right_Ear_y = pixelCoordinatesLandmark[1]
                        # 左眼
                        normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                             mp_face_detection.FaceKeyPoint.LEFT_EYE)
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                               normalizedLandmark.y,
                                                                                               imageWidth, imageHeight)
                        Left_EYE_x = pixelCoordinatesLandmark[0]
                        Left_EYE_y = pixelCoordinatesLandmark[1]
                        # 右眼
                        normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                             mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                               normalizedLandmark.y,
                                                                                               imageWidth, imageHeight)
                        Right_EYE_x = pixelCoordinatesLandmark[0]
                        Right_EYE_y = pixelCoordinatesLandmark[1]

                        sunglass_width = Left_Ear_x - Right_Ear_x + 60
                        sunglass_height = int((s_h / s_w) * sunglass_width)

                        imgFront = cv2.resize(imgFront, (sunglass_width, sunglass_height), None, 0.3, 0.3)

                        hf, wf, cf = imgFront.shape
                        hb, wb, cb = image.shape

                        # 調整太陽眼鏡位置
                        y_adjust = int((sunglass_height / 90) * 90)
                        x_adjust = int((sunglass_width / 194) * 100)

                        pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]

                        hf, wf, cf = imgFront.shape
                        hb, wb, cb = image.shape
                        *_, mask = cv2.split(imgFront)
                        maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                        maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
                        imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

                        imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
                        imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
                        imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
                        maskBGRInv = cv2.bitwise_not(maskBGR)
                        imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

                        image = cv2.bitwise_and(image, imgMaskFull2)
                        image = cv2.bitwise_or(image, imgMaskFull)

                # cv2.namedWindow("Sunglass Effect",cv2.WINDOW_NORMAL)

                def get_video_info(video_cap):
                    numFrames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
                    return numFrames, fps

                # cTime = time.time()
                # fps = 1 / (cTime - pTime)
                # pTime = cTime

                # 顯示FPS
                cv2.putText(image, "FPS: {}".format(fps), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 193, 27),
                            1, cv2.LINE_AA)
                # cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('newblack1', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                else:
                    return Mode
        # 語音功能開關
        # elif Mode == "Func" and sub_hand_text =="1":
        # 	if 10<=int(sub_Pose1[0])<=40 and 370<=int(sub_Pose1[1])<=400 and voice_on == "on":
        # 		voice_on = "off"
        # 		Mode = "Draw"
        # 		print("語音辨識功能關閉")
        # 		try:
        # 			cv2.destroyWindow("menu")
        # 		except:
        # 			pass
        # 	elif 10<=int(sub_Pose1[0])<=40 and 370<=int(sub_Pose1[1])<=400 and voice_on == "off":
        # 		voice_on = "on"
        # 		Mode = "Draw"
        # 		print("語音辨識功能開啟")
        # 		try:
        # 			cv2.destroyWindow("menu")
        # 		except:
        # 			pass

        cv2.rectangle(menu, (10, 370), (40, 400), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'Voice on/off', (50, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return Mode


# 若主手不伸出食指作畫，則清除主手座標紀錄
# print(mod)
# return Mode
def child_Mode(child_black, cover_pics, smailblack1):
    global pics
    dst = cv2.resize(cover_pics, (child_black.shape[0], child_black.shape[0]))
    # print((child_black.shape[1] - dst.shape[1])/2,((child_black.shape[1] - dst.shape[1])/2)+child_black.shape[0])
    child_black[0:child_black.shape[0], int((child_black.shape[1] - dst.shape[1]) / 2):int(
        ((child_black.shape[1] - dst.shape[1]) / 2) + child_black.shape[0])] = dst  # 圖片至中
    smailblack1 = cv2.addWeighted(smailblack1, 0.5, child_black, 0.3, 70)
    return child_black, Mode, smailblack1


def func_window():  ###準備功能視窗 -> menu
    # collor = [(0, 0, 255),(0, 255, 255),(255, 0, 255),(135,23,126)]
    # x = 20
    # y = int(((int(frame.shape[1] / 4)) - (len(collor) * x))/(len(collor)+1))
    # print(y)
    menu = np.full((10, 10, 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
    menu = cv2.resize(menu, (int(frame.shape[1] / 4), frame.shape[0]), interpolation=cv2.INTER_AREA)  # 依照讀取到的畫面調整功能版大小
    # smailblack2 = ScalingDisplacement(menu, lost_pix, offset)  # 縮小畫布
    # for i in range(len(collor)):
    #	 cv2.rectangle(menu, (int(y*(i+1) + x * i),30), (int((i+1)*(y+x)), (30+x)), collor[i], -1)  # 在畫面上方放入紅色正方形
    cv2.rectangle(menu, (10, 10), (40, 40), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, "color", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 70), (40, 100), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'screen adjustment', (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 130), (40, 160), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'graphics', (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 190), (40, 220), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'else', (50, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 250), (40, 280), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'Save and line', (50, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 310), (40, 340), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.putText(menu, 'Exit', (50, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(menu, (10, 370), (40, 400), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形 語音開關按鈕
    cv2.putText(menu, 'Voice on/off', (50, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

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


def sendLineNotify(token):
    msg = '圖片來瞜'
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': 'Bearer ' + token}
    data = {'message': msg}
    image = open('./test.png', 'rb')
    imageFile = {'imageFile': image}
    data = requests.post(url, headers=headers, data=data, files=imageFile)


# data = requests.post(url, headers=headers, files=imageFile)
def readconfig():
    global path, lost_pix, Main_hand, token
    with open('./config.csv', mode='r') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[0]: rows[1] for rows in reader}
        path = int(dict_from_csv.get("path"))
        lost_pix = int(dict_from_csv.get("lost_pix"))
        Main_hand = str(dict_from_csv.get("Main_hand"))
        token = str(dict_from_csv.get("token"))


def wav2mfcc2(file_path):
    global n_mfcc, sr_set, max_pad_len2
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    # print(wave.shape) #(112014,)
    # wave = wave[::3]
    # print("wave[::3].shape:",wave.shape) #(37338,) (除3 ??)
    mfcc = librosa.feature.mfcc(wave, n_mfcc=n_mfcc, sr=sr_set)  # SR 採樣頻率
    # print("mfcc.shape in wav2mfcc before padding:",mfcc.shape) #(20 ,73)
    pad_width = max_pad_len2 - mfcc.shape[1]  # 設定的長度-抓到音訊的長度=需要補足的長度
    # pad_width =  mfcc.shape[1] - max_pad_len  #差距73-11 = 62
    # 若抓到的音訊長度大於設定長度，取全部資訊
    if pad_width < 0:
        mfcc = mfcc[:, 0:max_pad_len2]
        print("mfcc.shape 抓到的音訊大於設定長度", mfcc.shape)

    # 若抓到的音訊長度小於設定長度，補足0
    else:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        print("mfcc.shape in wav2mfcc after padding:", mfcc.shape)

    # print("count long and short:",count_long, " ",count_short)
    return mfcc


def wav2mfcc(file_path):
    global n_mfcc, sr_set, max_pad_len
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    # print(wave.shape) #(112014,)
    # wave = wave[::3]
    # print("wave[::3].shape:",wave.shape) #(37338,) (除3 ??)
    mfcc = librosa.feature.mfcc(wave, n_mfcc=n_mfcc, sr=sr_set)  # SR 採樣頻率
    # print("mfcc.shape in wav2mfcc before padding:",mfcc.shape) #(20 ,73)
    pad_width = max_pad_len - mfcc.shape[1]  # 設定的長度-抓到音訊的長度=需要補足的長度
    # pad_width =  mfcc.shape[1] - max_pad_len  #差距73-11 = 62
    # 若抓到的音訊長度大於設定長度，取全部資訊
    if pad_width < 0:
        mfcc = mfcc[:, 0:max_pad_len]
    #   print("mfcc.shape 抓到的音訊大於設定長度",mfcc.shape)

    # 若抓到的音訊長度小於設定長度，補足0
    else:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    #   print("mfcc.shape in wav2mfcc after padding:",mfcc.shape)

    # print("count long and short:",count_long, " ",count_short)
    return mfcc


# 用Pyaudio錄製音頻
def audio_record(out_file, rec_time):
    global sr_set

    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # 16bit编码格式
    CHANNELS = 2  # 单声道
    RATE = sr_set  # 16000采样频率
    p = pyaudio.PyAudio()
    # 创建音频流
    stream = p.open(format=FORMAT,  # 音频流wav格式
                    channels=CHANNELS,  # 单声道
                    rate=RATE,  # 采样率16000
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Start Recording...")
    frames = []  # 录制的音频流
    # 录制音频数据
    for i in range(0, int(RATE / CHUNK * rec_time)):
        data = stream.read(CHUNK)
        frames.append(data)
    # 录制完成
    # print(frames)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 保存音频文件
    with wave.open(out_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


# def voice_yn(yes_no):
# 	global voice_pre_func, Mode, max_pad_len2
# 	#進入語音錄製 與 AI 判讀一次， 輸出 功能項目
# 	# yn = AudioSegment.from_wav("") # 播放確認語音
# 	#play(yn)
# 	# time.sleep()
# 	print("確認語音開啟...開始錄音")
# 	audio_func_path = "./record_wav/yn.wav"
# 	audio_record(audio_func_path, 2)
# 	mfcc = wav2mfcc2(audio_func_path)
# 	mfcc_reshaped = mfcc.reshape(1, n_mfcc, max_pad_len2, 1)

# 	label_list_yn = ['yes', 'no', 'others'] # 答案標籤集list 0,1,2
# 	label_idx_yn = np.argmax(voice_thread.model_yn.predict(mfcc_reshaped))  # 預測的答案index
# 	prob_list_yn = voice_thread.model_yn.predict(mfcc_reshaped) # 個個答案的機率list

# 	print("predict={} prob={}".format(label_list_yn[label_idx_yn], prob_list_yn[0][label_idx_yn])) #印出 預測的答案 與 機率
# 	voice_pre_func = str(label_idx_yn) # ['yes', 'no', 'others'] "0", "1", "2"

# 	if label_idx_yn == "0": # 若判讀為yes
# 		yes_no = "yes"
# 	elif label_idx_yn == "1" or "2": #若判讀為no 或 others
# 		yes_no = "no"
# 	return yes_no

def voice_func():
    global voice_pre_func, Mode, voice_check_func
    # 進入語音錄製 與 AI 判讀一次， 輸出 功能項目
    print("功能語音開啟...開始錄音")
    audio_func_path = "./record_wav/func.wav"
    audio_record(audio_func_path, 2.5)
    # print("開始功能語音識別")
    mfcc = wav2mfcc(audio_func_path)
    mfcc_reshaped = mfcc.reshape(1, n_mfcc, max_pad_len, 1)

    label_list_func = ['mark_pen.npy', 'eraser.npy', 'call_func.npy']  # 答案標籤集list 0,1,2
    label_idx_func = np.argmax(voice_thread.model.predict(mfcc_reshaped))  # 預測的答案index
    prob_list_func = voice_thread.model.predict(mfcc_reshaped)  # 個個答案的機率list

    print("predict={} prob={}".format(label_list_func[label_idx_func],
                                      prob_list_func[0][label_idx_func]))  # 印出 預測的答案 與 機率
    voice_pre_func = str(label_idx_func)  # ['mark_pen.npy', 'eraser.npy', 'call_func.npy'] "0", "1", "2"
    yes_no = ""
    if voice_pre_func == "0":  # 判斷為 麥克筆
        playsound("./Respond/re/pens.mp3")
        voice_check_func = "draw"
        return voice_check_func
    # playsound("./Respond/re/pens.mp3") #playsound是否開啟麥克筆？
    # time.sleep(2) #播放的語音長度
    # playsound("./Respond/re/cat1b.mp3") # 喵?
    # time.sleep(1) #播放的語音長度
    # yes_no = voice_yn(yes_no) # 開啟 是否 的語音判讀
    # if yes_no == "yes":
    # 	playsound("./Respond/re/pens.mp3")
    # 	Mode = "Draw"
    # 	print("轉為繪畫模式")
    # elif yes_no == "no":
    # 	playsound("./Respond/re/crow1.mp3") # 播放烏鴉叫聲

    elif voice_pre_func == "1":  # 判斷為 橡皮擦
        playsound("./Respond/re/tissue.mp3")
        voice_check_func = 'eraser'
        # print("轉為橡皮擦模式")
        return voice_check_func
    # playsound("./Respond/re/tissue.mp3") # 是否開啟橡皮擦？
    # time.sleep(0.5) #播放的語音長度
    # playsound("./Respond/re/cat1b.mp3") # 喵?
    # time.sleep(1) #播放的語音長度
    # yes_no = voice_yn(yes_no) # 開啟 是否 的語音判讀
    # if yes_no == "yes":
    # 	playsound("./Respond/re/tissue.mp3")
    # 	voice_check_func = 'eraser'
    # 	print("轉為橡皮擦模式")
    # 	return voice_check_func
    # elif yes_no == "no":
    # 	playsound("./Respond/re/crow1.mp3") # 播放烏鴉叫聲
    else:
        playsound("./Respond/re/pulling_back_a_chair.mp3")  # 再撥放一次功能語音
        voice_check_func = "menu"
        # print("轉為功能版模式")
        return voice_check_func  # 判斷為 功能版


# playsound("./Respond/re/pulling_back_a_chair.mp3") # 是否開啟功能版
# time.sleep(1) #播放的語音長度
# playsound("./Respond/re/cat1b.mp3") # 喵?
# time.sleep(1) #播放的語音長度
# yes_no = voice_yn(yes_no) # 開啟 是否 的語音判讀
# if yes_no == "yes":
# 	playsound("./Respond/re/pulling_back_a_chair.mp3") # 再撥放一次功能語音
# 	voice_check_func = "menu"
# 	print("轉為功能版模式")
# 	return voice_check_func
# elif yes_no == "no":
# 	playsound("./Respond/re/crow1.mp3") # 播放烏鴉叫聲

class VoiceStoppableThread(threading.Thread):
    global n_mfcc, max_pad_len, voice_pre_func

    def __init__(self, daemon=None):
        super(VoiceStoppableThread, self).__init__(daemon=daemon)
        self.__is_running = True
        self.daemon = daemon
        # 讀取語音模型
        self.model_call = load_model('./models/VGG16_mfcc60_pad107_call_mix_best.h5')
        self.model = load_model('./models/VGG16_mfcc60_pad107_3func_best.h5')
        self.model_yn = load_model("./models/VGG16_mfcc60_padf87_yn_best.h5")

    def terminate(self):
        self.__is_running = False

    def run(self):
        pid = os.getpid()  # 當前進程組 ID

        while self.__is_running:
            if voice_on == 'on':
                print("喚醒功能開啟...開始錄音")
                # 語音存檔路徑與檔名
                audio_call_path = "./record_wav/call.wav"
                # 錄製語音指令 ，秒數
                audio_record(audio_call_path, 2.5)
                # print("開始喚醒語音識別...")

                # 預測
                mfcc = wav2mfcc(audio_call_path)  # 這裡放上要判讀的語音檔
                mfcc_reshaped = mfcc.reshape(1, n_mfcc, max_pad_len, 1)
                label_list = ['hey_julia.npy', 'others.npy', 'hey_star.npy']  # 答案標籤集list 0,1,2
                label_idx = np.argmax(self.model_call.predict(mfcc_reshaped))  # 預測的答案index
                prob_list = self.model_call.predict(mfcc_reshaped)  # 個個答案的機率list

                print("predict={} prob={}".format(label_list[label_idx], prob_list[0][label_idx]))  # 印出 預測的答案 與 機率

                # 喚醒程式
                voice_pre = str(label_idx)
                # 當聽到 hey julia時
                if voice_pre == "0":
                    playsound("./Respond/re/cat1a.mp3")
                    time.sleep(0.7)  # 等待播放的語音長度(聽完語音)
                    voice_func()  # 錄製語音，執行功能
                    time.sleep(2)  # 功能等候兩秒，讓程式運作
                # pass
                # 當聽到hey 星空時
                elif voice_pre == "2":
                    playsound('./meowx2.wav')  # meow meow~
                    time.sleep(1)
                    voice_func()  # 錄製語音，執行功能
                    time.sleep(2)  # 功能等候兩秒，讓程式運作
                # pass
                else:
                    pass

                voice_pre_func = ""  # reset voice pre func

                time.sleep(0.1)  # 每0.1秒重新跑一次thread
            elif voice_on == 'off':
                time.sleep(0.5)  # 每0.5秒判斷一次是否重新開啟麥克風


if __name__ == '__main__':
    # 把函式放到改寫到類的run方法中，便可以通過呼叫類方法，實現執行緒的終止
    # 執行 語音thread
    voice_thread = VoiceStoppableThread()  # 創建一個可終止程序的語音thread
    # print("terminated")
    # voice_thread.terminate()
    voice_thread.daemon = True  # thread True, 判定開啟
    voice_thread.start()

    # voice_thread.terminate() #終止thread 用
    # pid = os.getpid() #可以查process ID
    # print("start pid:", pid)

    readconfig()
    cap = cv2.VideoCapture(path)  # 攝影機變數
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
            cover_pics = cv2.imread('./child_pic/' + pics[pic_change] + '.png')
        smailblack1 = ScalingDisplacement(newblack, lost_pix, offset)  # 縮小畫布

        # print(child,sub_hand_text)

        frame = cv2.flip(frame, 1)  # 畫面左右翻轉，放回畫面frame
        blur = cv2.flip(blur, 1)
        imgRGB = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)  # 將影像通道從BGR轉成RGB
        menu = func_window()  # 初始化功能版
        colormain = func_color()
        hands_Pose1, hands_LR = HandsIdentify(imgRGB)  # 副程式處理"手部座標"、"左右手順序"
        main_MousePose, sub_MousePose, sub_hand_text = PointPprocessing(hands_Pose1, hands_LR, menu, Main_hand,
                                                                        colormain)  # 分別處理左右手座標之副程式
        # Function_Select(main_hand_text, sub_hand_text, main_finger_points, sub_finger_points,main_Pose, sub_Pose,main_Pose1, sub_Pose1)
        main_MousePose = Mouse_Pos(main_MousePose)
        sub_MousePose = Mouse_Pos(sub_MousePose)
        TrueCanvas = Mouse(smailblack1, main_MousePose, sub_MousePose, mod)  # 加入鼠標 回傳最終畫布
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("live", frame)
        cv2.imshow("TrueCanvas", TrueCanvas)
        # time.sleep(0.5)	#跑影片要記得設time.sleep，跑視訊鏡頭要記得關  我花了40分鐘在debug為甚麼我的fps不到1
        if cv2.waitKey(1) == ord('q'):
            break
