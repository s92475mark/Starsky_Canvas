# 發現一個bug, 訓練出的答案標籤是在訓練時決定的，這裡的標籤本地重標的，無法相匹配，正確的要看當時訓練時標上的標籤

import librosa
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import os

max_pad_len = 120
def wav2mfcc(file_path, max_pad_len=max_pad_len):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    # print(wave.shape) #(112014,)
    wave = wave[::3] 
    print("wave[::3].shape:",wave.shape) #(37338,) (除3 ??)
    mfcc = librosa.feature.mfcc(wave, sr=16000) #SR 採樣頻率
    print("mfcc.shape in wav2mfcc before padding:",mfcc.shape) #(20 ,73)
    pad_width = max_pad_len - mfcc.shape[1] #11 -73
    # pad_width =  mfcc.shape[1] - max_pad_len  #差距73-11 = 62
    if pad_width <0:
      mfcc = mfcc[:,0:max_pad_len]
      print("mfcc.shape pad_width>11",mfcc.shape)
    else:
      mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    print("mfcc.shape in wav2mfcc after padding:",mfcc.shape) #(20 ,73)
    return mfcc

##########  這裡設定要測試的語音檔  ###########
file_path = "./record_wav/calling.wav"

##########  設定語音模型  ######333
model = load_model('./models/best_hier.h5')


#預測
mfcc = wav2mfcc(file_path)  
mfcc_reshaped = mfcc.reshape(1, 20, max_pad_len, 1)
# print("labels= ['mark_pen.npy', 'eraser.npy', 'call_func.npy']")
# print("predict=", np.argmax(model.predict(mfcc_reshaped)))
# print("predict=", model.predict(mfcc_reshaped))

label_list = ['mark_pen.npy', 'eraser.npy', 'call_func.npy', 'hey_julia.npy', 'hey_star.npy', 'others.npy']
label_idx = np.argmax(model.predict(mfcc_reshaped))
prob_list = model.predict(mfcc_reshaped)
# print(label_list)
print("所有機率: \n")
c = 0
for i in prob_list[0]:
    print(label_list[c],format(i, '.3f'))
    c+=1
print("\ninput = {}\npredict= {}  \nprob= {}".format(file_path,label_list[label_idx], prob_list[0][label_idx]))