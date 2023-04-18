import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import sys




# openbci_datapath = '/Users/cm/Desktop/erp/test.txt'
openbci_datapath = '/Users/cm/Desktop/erp/analogueboi.txt'
sample_rate = 250


data = pd.read_csv(openbci_datapath, sep=", ", header=4, index_col=False, engine='python', usecols=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp'])
new_names = {'EXG Channel 0':'ch1', 'EXG Channel 1':'ch2', 'EXG Channel 2':'ch3', 'EXG Channel 3':'ch4', 'EXG Channel 4':'ch5', 'EXG Channel 5':'ch6', 'EXG Channel 6':'ch7', 'EXG Channel 7':'ch8', 'Analog Channel 1':'A6', 'Analog Channel 2':'A7', 'Timestamp':'TimeStamp'}
data = data.rename(columns=new_names)

print("OpenBCI data shape: ", data.shape)
print(data.columns)
Timestamps = (data['TimeStamp'].to_numpy() - data['TimeStamp'].to_numpy()[0])
print("Timestamps shape: ", Timestamps.shape)
print("Total length of data: ", Timestamps[-1], 's')
EEG_data = data[['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']].to_numpy()
# print(EEG_data)
avg_EEG_data = []
# for e in EEG_data[2000:2500]:
for e in EEG_data:
    # avg = sum(e) / len(e)
    avg = np.mean(e)
    avg_EEG_data.append(avg)
# plt.plot(avg_EEG_data)
# plt.xticks(ticks=np.arange(0, round(Timestamps[-1]+1)*sample_rate, round(Timestamps[-1]+1)*sample_rate/4), 
#            labels=np.arange(0, round(Timestamps[-1]+1),round(Timestamps[-1]+1)/4))
# plt.xlabel('Time (s)')
# plt.draw()
# plt.show()
# plt.pause(5)
# sys.exit()
photo_data = data['A6']
button_data = data['A7']
# plt.plot(analog_data)
# plt.xticks(ticks=np.arange(0, round(Timestamps[-1]+1)*sample_rate, round(Timestamps[-1]+1)*sample_rate/5), 
#            labels=np.arange(0, round(Timestamps[-1]+1),round(Timestamps[-1]+1)/5))
plt.tight_layout
photopeaks, _ = find_peaks(photo_data, height=(5, 20), distance=0.1*250) # distance is essential!
buttonpeaks, _ = find_peaks(button_data, height=1000, distance=0.1*250) # distance is essential!
# plt.plot(buttonpeaks, analog_data[buttonpeaks], "x")
# plt.xticks(ticks=np.arange(0, len(analog_data), len(analog_data)/4), labels=np.arange(0, len(analog_data)/250,len(analog_data)/250/4))
# plt.show()
def clean(item):
    i, val = item
    if i in buttonpeaks:
        val = 1
    return 0

fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[1].plot(photo_data)
# ax[1].plot(photopeaks, photo_data[photopeaks], "x")

# buttonClean = button_data
# buttonClean = button_data
# buttonClean = list(map(clean, enumerate(buttonClean)))
# for i, _ in enumerate(button_data):
#     if i in buttonpeaks:
#         buttonClean[i] = 1
#     else:
#         buttonClean[i] = 0
# buttonClean[buttonpeaks] = 1
# buttonClean[buttonClean not in buttonpeaks] = 0
# print(max(buttonClean))
# print(buttonClean)

# print(buttonClean)
# sys.exit()
# for i,x in enumerate(button_data):
#     # print("{} {}".format(i, x))
#     if x==1:
#         print(i)
    # break
# print(button_data)
# print(buttonClean)


ax[1].plot(photo_data[photopeaks], "x")
ax[1].set_xticks(ticks=np.arange(0, len(photo_data), len(photo_data)/4))
ax[1].set_xticklabels(labels=np.arange(0, len(photo_data)/250,len(photo_data)/250/4))
ax[1].set_title("Blinks")

# ax[2].plot(button_data)
# ax[2].plot(buttonpeaks, 1, "x")
ax[2].scatter(buttonpeaks, np.ones(buttonpeaks.shape), marker='x')
# ax[2].plot(buttonClean)
# ax[2].plot(button_data, buttonClean, "x")
ax[2].set_xticks(ticks=np.arange(0, len(button_data), len(button_data)/4))
ax[2].set_xticklabels(labels=np.arange(0, len(button_data)/250,len(button_data)/250/4))
ax[2].set_title("Button clicks")
plt.show()