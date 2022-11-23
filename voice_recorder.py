import pyaudio
import playsound
import glob
import os
import wave
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

### 設定 ###
########################  檔名，錄製次數，錄音秒數，資料夾路徑  ##################################
name = "star"  #檔案名稱
count = 1        #錄音次數
second = 2.5        #錄音秒數
#資料夾路徑
# folder_path = f"E:\\Users\\lwing\\coding projects\\wav_for_train\\record\\" #我的批次路徑
folder_path = ".\\record_wav\\"   #我的單一路徑
#####################################################################

#資料夾內檔案
files = glob.glob(folder_path+"*")


# 檢測資料夾內檔案，並且錄製音頻
n = 0
file_name = name + "_" + str(n) + ".wav"

# 設定名稱從 1 開始
################ 若檔名重複，n+1
for file in files:
    # print(file)
    file_name = name + "_"+ str(n) + ".wav"
    lenth = len(file_name)
    #若檔名重複
    if file[-lenth:] == file_name:
        n = n + 1
        
print("已有 {} 個檔案重複命名".format(n))
file_name = name + "_"+ str(n) + ".wav" 

#######################  設定錄製次數  ##########################
for i in range(0, count):
    file_name = name + "_"+ str(n+i) + ".wav" 
    i+=1
    #錄製音頻的檔案路徑
    file_path = folder_path + file_name
    print("存檔路徑： ",file_path)

    #錄製語音指令 ，秒數
    print("喚醒功能開啟...開始錄音")
    audio_record(file_path, second) 
    print("檔案名稱: ", file_name)
    # playsound.playsound(file_path)