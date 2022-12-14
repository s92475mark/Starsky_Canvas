# 發現一個bug, 訓練出的答案標籤是在訓練時決定的，這裡的標籤本地重標的，無法相匹配，正確的要看當時訓練時標上的標籤

import librosa
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import os

import soundfile as sf

count_long = 0
count_short = 0
n_mfcc = 60
sr_set = 22050

# 1.5second pad:65 
# 2second pad:87
# 2.5second pad:107
# 3second pad: 129

max_pad_len = 79
def wav2mfcc(file_path, max_pad_len=max_pad_len):
    global count_long, count_short, n_mfcc, sr_set
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    # print(wave.shape) #(112014,)
    # wave = wave[::3] 
    print("wave[::3].shape:",wave.shape) #(37338,) (除3 ??)
    mfcc = librosa.feature.mfcc(wave, n_mfcc=n_mfcc, sr=sr_set) #SR 採樣頻率
    print("mfcc.shape in wav2mfcc before padding:",mfcc.shape) #(20 ,73)
    pad_width = max_pad_len - mfcc.shape[1] # 設定的長度-抓到音訊的長度=需要補足的長度
    # pad_width =  mfcc.shape[1] - max_pad_len  #差距73-11 = 62
    # 若抓到的音訊長度大於設定長度，取全部資訊
    if pad_width <0:
      mfcc = mfcc[:,0:max_pad_len]
      print("mfcc.shape 抓到的音訊大於設定長度",mfcc.shape)
      count_long+=1
    # 若抓到的音訊長度小於設定長度，補足0
    else:
      mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
      print("mfcc.shape in wav2mfcc after padding:",mfcc.shape) 
      count_short+=1
    # print("count long and short:",count_long, " ",count_short)
    return mfcc

##########  這裡設定要測試的語音檔  ###########
# file_path = "./record_wav/long_0.wav"
file_path = "E:\\Users\\lwing\\coding projects\\wav_for_train\\record/3second_0.wav"

##########  設定語音模型  ######333
model = load_model('./models/m1125_VGG16_mfcc60_RN_best.h5')


#預測
mfcc = wav2mfcc(file_path)  
mfcc_reshaped = mfcc.reshape(1, n_mfcc, max_pad_len, 1)


# label_list = ['mark_pen.npy', 'eraser.npy', 'call_func.npy', 'hey_julia.npy', 'hey_star.npy']
# label_idx = np.argmax(model.predict(mfcc_reshaped))
# print("label_idx type", type(label_idx)) #int64
# prob_list = model.predict(mfcc_reshaped)
# # print(label_list)
# print("所有機率: \n")
# c = 0
# for i in prob_list[0]:
#     print(label_list[c],format(i, '.3f'))
#     c+=1
# print("\ninput = {}\npredict= {}  \nprob= {}".format(file_path,label_list[label_idx], prob_list[0][label_idx]))


# audio_ndarray = librosa.feature.inverse.mfcc_to_audio(mfcc, n_mels=128, dct_type=2, norm='ortho', ref=1.0, lifter=0)
# print(audio_ndarray.shape)
# sf.write('mfcc_audio.wav',audio_ndarray, 16000)