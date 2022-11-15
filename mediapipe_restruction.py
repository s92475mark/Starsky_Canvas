import cv2
import time
import mediapipe as mp
import numpy as np
import math

newblack = np.full((10, 10, 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
mpHand = mp.solutions.hands  # 抓手	001
hands = mpHand.Hands()  # 001
path = 1  # 本地端可以改成這個，用筆電的視訊鏡頭
cap = cv2.VideoCapture(path)  # 攝影機變數
pTime = 0  # 起始時間
f_round = True  # 第一次跑
color = (0, 0, 255)
lost_pix = 0.8  # 縮小比例0~1之間
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


# mpDraw = mp.solutions.drawing_utils
# handLmsStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness = 5 )	#設定點的參數
# handConStyle = mpDraw.DrawingSpec(color=(255,255,255),thickness = 2 )	#設定線的參數
def Mouse(Canvas, CanvasSize, main_MousePose, sub_MousePose):  # 鼠標層覆蓋上畫布
    global offset, lost_pix
    # cv2.imshow("newblack11",newblack11)
    MouseLevel = np.full((Canvas.shape[0], Canvas.shape[1], 3), (0, 0, 0), np.uint8)  # 產生與newblack大小相同黑色的圖
    main_MousePose = (int((main_MousePose[0] - offset[0]) / lost_pix), int((main_MousePose[1] - offset[1]) / lost_pix))
    sub_MousePose = (int((sub_MousePose[0] - offset[0]) / lost_pix), int((sub_MousePose[1] - offset[1]) / lost_pix))

    MouseLevel = cv2.circle(MouseLevel, main_MousePose, 10, (255, 255, 255), -1)  # 在這層上面點上主手白色鼠標
    MouseLevel = cv2.circle(MouseLevel, sub_MousePose, 10, (0, 255, 0), -1)  # 在這層上面點上副手綠色鼠標
    TrueCanvas = cv2.add(Canvas, MouseLevel)
    # print("TrueCanvas",TrueCanvas.shape)
    # print("TrueCanvas",TrueCanvas.shape)

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
    # print(hand_)
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
    smailblack1 = smailblack[(offset[1]):(int(newblack.shape[0] * lost_pix)) + offset[1],
                  offset[0]:(int(newblack.shape[1] * lost_pix)) + offset[0]]
    # print("smailblack1",smailblack1.shape)
    newblack1 = newblack.copy()
    cv2.rectangle(newblack1, ((offset[0]), (offset[1])),
                  (int(newblack.shape[1] * lost_pix + offset[0]), int(newblack.shape[0] * lost_pix + offset[1])),
                  (255, 255, 0), 3)
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
            main_Pose = (hands_Pose[hands_LR.index(hands_LR[i])])  # 當前抓取到的手的全部座標
            main_Pose1 = [int(main_Pose.landmark[8].x * frame.shape[1]),
                          int(main_Pose.landmark[8].y * frame.shape[0])]  # 當前抓取到的手的食指座標
            # 顯示主手藍色鼠標於監視器上
            main_mouse_pos = [int(main_Pose.landmark[8].x * frame.shape[1]),
                              int(main_Pose.landmark[8].y * frame.shape[0])]  # 主手食指 給鼠標用
            frame = cv2.circle(frame, main_Pose1, 10, Hand_Mark_blue, -1)  # 鼠標藍色 顯示於 監視器上

            ### 將主手 21 個節點換算成座標，記錄到 finger_points
            for i in main_Pose.landmark:
                x = i.x * frame.shape[1]
                y = i.y * frame.shape[0]
                main_finger_points.append((x, y))
            main_finger_angle = hand_angle(main_finger_points)  # 計算手指角度，回傳長度為 5 的串列
            # 判斷手勢
            main_hand_text = Hand_Text(main_finger_angle)  # 取得手勢所回傳的內容

        # return main_hand_text, main_finger_points, main_Pose, main_Pose1
        else:  # 讀取到副手
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
                sub_finger_points.append((x, y))

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
    global dots, color, Mode, colorx, colory, colorz, mod
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
    elif sub_hand_text == '1' and mod == 1:
        Mode = 'Func'  # 停止主手迴圈，進入副手迴圈

        menu = cv2.circle(menu, (int(sub_Pose1[0] / 2), int(sub_Pose1[1] / 2)), 10, (255, 255, 255),
                          -1)  # 製作副手鼠標 並繪製於功能版上
        cv2.imshow("menu", menu)  # 顯示副手鼠標+功能版

        # 紀錄副手食指座標
        fx = int(sub_finger_points[8][0])  # 如果手勢為 1，記錄食指末端的座標
        fy = int(sub_finger_points[8][1])
        # print(fx,fy)

        # 若副手食指座標移動到以下位置，則切換顏色
        if 20 <= fy <= 80 and 20 <= fx <= 80:
            mod = 2
        elif fy >= 10 and fy <= 40 and fx >= 45 and fx <= 75:
            mod = 2
        elif fy >= 10 and fy <= 40 and fx >= 80 and fx <= 110:
            color = (255, 0, 0, 255)  # 如果食指末端碰到藍色，顏色改成藍色
        else:
            dots.clear()
    # 副手全張：關閉功能版，轉回繪畫模式
    elif sub_hand_text == '5' and Mode == 'Func' and mod == 1:
        Mode = 'Draw'
        cv2.destroyWindow("menu")


    if mod == 2:
        if sub_hand_text == '1' and 70 <= int(sub_Pose1[0]) <= 580 and 70 <= int(sub_Pose1[1]) <= 120:
            colorx = int((sub_Pose1[0] - 70) / 2)
        elif sub_hand_text == '1' and 70 <= int(sub_Pose1[0]) <= 580 and 150 <= int(sub_Pose1[1]) <= 200:
            colory = int((sub_Pose1[0] - 70) / 2)
        elif sub_hand_text == '1' and 70 <= int(sub_Pose1[0]) <= 580 and 230 <= int(sub_Pose1[1]) <= 280:
            colorz = int((sub_Pose1[0] - 70) / 2)
        elif sub_hand_text == '5':
            color = (colorx, colory, colorz)
            cv2.destroyWindow("menu")
            mod = 1
        else:
            dots.clear()
        if sub_Pose1:
            colormain = cv2.circle(colormain, (sub_Pose1[0],sub_Pose1[1]), 10, (255, 255, 255), -1)
        #print(sub_Pose1)
        cv2.putText(colormain, str(int(colorx)), (600, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(colormain, str(int(colory)), (600, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(colormain, str(int(colorz)), (600, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(colormain, (70, 400), (520, 420), (colorx, colory, colorz), -1)  # 在畫面上方放入紅色正方形
        cv2.rectangle(colormain, (70, 420), (90, 460), (0, 0, 255), -1)  # 在畫面上方放入紅色正方形
        cv2.imshow("menu", colormain)

    # 若主手不伸出食指作畫，則清除主手座標紀錄
    # print(mod)
    return Mode


# print("subtext", sub_hand_text)
# print("Mode:", Mode)


def func_window():  ###準備功能視窗 -> menu
    menu = np.full((10, 10, 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
    menu = cv2.resize(menu, (int(frame.shape[1] / 5), frame.shape[0]), interpolation=cv2.INTER_AREA)  # 依照讀取到的畫面調整功能版大小
    # smailblack2 = ScalingDisplacement(menu, lost_pix, offset)  # 縮小畫布
    cv2.rectangle(menu, (10, 10), (40, 40), (0, 0, 255, 255), -1)  # 在畫面上方放入紅色正方形
    cv2.rectangle(menu, (45, 10), (75, 40), (0, 255, 0, 255), -1)  # 在畫面上方放入綠色正方形
    cv2.rectangle(menu, (80, 10), (110, 40), (255, 0, 0, 255), -1)  # 在畫面上方放入藍色正方形

    return menu


def func_color():
    colormain = np.full((10, 10, 3), (0, 0, 0), np.uint8)  # 產生10x10黑色的圖
    colormain = cv2.resize(colormain, (int(frame.shape[1] + 120), frame.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.rectangle(colormain, (70, 70), (580, 120), (0, 0, 255, 255), 5)  # 在畫面上方放入紅色正方形
    cv2.rectangle(colormain, (70, 150), (580, 200), (0, 255, 0, 255), 5)  # 在畫面上方放入綠色正方形
    cv2.rectangle(colormain, (70, 230), (580, 280), (255, 0, 0, 255), 5)  # 在畫面上方放入藍色正方形
    return colormain


def HandsIdentify(imgRGB):  # 副程式處理"手部座標"、"左右手順序"

    hands_Pose = []  # 紀錄雙手食指座標
    results = hands.process(imgRGB)  # 手部辨識001
    hands_LR = []  # 紀錄左手或右手
    hands_LR1 = results.multi_handedness  # medipi辨識左右手前置變數
    if results.multi_hand_landmarks:  # 判斷有沒有抓到手
        for i in range(len(results.multi_hand_landmarks)):  # 001 用迴圈一次處理一隻手的座標
            hands_Pose.append(results.multi_hand_landmarks[i])
            hands_LR.append(hands_LR1[i].classification[0].label)
    # print(hands_LR)
    return hands_Pose, hands_LR


if __name__ == '__main__':
    while (True):
        ret, frame = cap.read()
        if not ret:  # 判定有沒有畫面存在
            print("沒有畫面")
            break
        CanvasSize = (frame.shape[1], frame.shape[0])  # 畫布大小
        blur = cv2.GaussianBlur(frame, (7, 7), cv2.BORDER_DEFAULT)
        #blur = cv2.Canny(blur, 125, 175)
        blur = cv2.dilate(blur, (7, 7), iterations=1)
        if f_round:  # 判斷是不是第一次跑，是:把黑色畫布放大成跟鏡頭解析度一樣大
            newblack = cv2.resize(newblack, CanvasSize, interpolation=cv2.INTER_AREA)
            f_round = False

        frame = cv2.flip(frame, 1)  # 畫面左右翻轉，放回畫面frame
        blur = cv2.flip(blur, 1)
        # print(frame.shape)
        imgRGB = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)  # 將影像通道從BGR轉成RGB
        smailblack1 = ScalingDisplacement(newblack, lost_pix, offset)  # 縮小畫布
        menu = func_window()  # 初始化功能版
        colormain = func_color()
        hands_Pose1, hands_LR = HandsIdentify(imgRGB)  # 副程式處理"手部座標"、"左右手順序"
        main_MousePose, sub_MousePose = PointPprocessing(hands_Pose1, hands_LR, menu, Main_hand,
                                                         colormain)  # 分別處理左右手座標之副程式
        # Function_Select(main_hand_text, sub_hand_text, main_finger_points, sub_finger_points,main_Pose, sub_Pose,main_Pose1, sub_Pose1)
        TrueCanvas = Mouse(smailblack1, CanvasSize, main_MousePose, sub_MousePose)  # 加入鼠標 回傳最終畫布
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        # print("newblack",newblack.shape)
        cv2.imshow("live", frame)
        cv2.imshow("liv", blur)
        cv2.imshow("TrueCanvas", TrueCanvas)
        # time.sleep(0.5)	#跑影片要記得設time.sleep，跑視訊鏡頭要記得關  我花了40分鐘在debug為甚麼我的fps不到1
        if cv2.waitKey(1) == ord('q'):
            break
