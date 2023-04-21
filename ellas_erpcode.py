from logging import captureWarnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import argrelmin
import tkinter as tk
import datetime
import sys




# openbci_datapath = '/Users/cm/Desktop/erp/test.txt'
openbci_datapath = '/Users/ellaneurohr/Desktop/erp/raw.txt'
sample_rate = 250
number_of_stimuli = 92
key = [
    True,
    True,
    False,
    False,
    True,
    True,
    False,
    True,
    False,
    True,
    False,
    False,
    False,
    False,
    True,
    False,
    False,
    True,
    False,
    True,
    True,
    True,
    True,
    False,
    False,
    True,
    False,
    True,
    False,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    False,
    False,
    True,
    False,
    True,
    True,
    False,
    False,
    False,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    True,
    False,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    False,
    True,
    False,
    True,
    True,
    True,
    True,
    False,
    False,
    True,
    False,
    False,
    False,
    True,
    True,
    True,
    True,
    True]


data = pd.read_csv(openbci_datapath, sep=", ", header=4, index_col=False, engine='python', usecols=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp'])
new_names = {'EXG Channel 0':'ch1', 'EXG Channel 1':'ch2', 'EXG Channel 2':'ch3', 'EXG Channel 3':'ch4', 'EXG Channel 4':'ch5', 'EXG Channel 5':'ch6', 'EXG Channel 6':'ch7', 'EXG Channel 7':'ch8', 'Analog Channel 1':'A6', 'Analog Channel 2':'A7', 'Timestamp':'TimeStamp'}
data = data.rename(columns=new_names)


# Open BCI includes some irrelevant data from before you start recording, then inserts a row of zeros when recording actually began
for i, val in enumerate(data['ch1']):
    if val == 0:
        data = data.drop(index=range(0, i+1))
        break
data = data.reset_index(drop=True)

# Finds the first occurence of the minimum value, which means the screen is on the black of the video. Now the first 5 peaks will be the starting 5 flashes
min = np.argmin(data['A6'].to_numpy())
data = data.drop(index=range(0, min))
data = data.reset_index(drop=True)

# Now drop everything before the first flash
buttonpeaks, _ = find_peaks(data['A7'], height=1000, distance=0.1*250) # distance is essential!
photopeaks, _ = find_peaks(data['A6'], height=(10, 20), distance=0.1*250) # distance is essential!
lastFlash = photopeaks[4]
lastClick = buttonpeaks[-1]
# data = data.drop(index=range(0, photopeaks[0]-1))
data = data.loc[lastFlash:lastClick+1]
data = data.reset_index(drop=True)

# Adjust times now that all of that has been removed
Timestamps = (data['TimeStamp'].to_numpy() - data['TimeStamp'].to_numpy()[0])
data['Adjusted Timestamp'] = Timestamps
time = datetime.timedelta(seconds=Timestamps[-1])
print("Total length of data: ", time, 's')

# Find peaks again after slicing
photo_data = data['A6']
button_data = data['A7']
photopeaks, _ = find_peaks(photo_data, height=(10, 20), distance=0.1*250) # distance is essential!
buttonpeaks, _ = find_peaks(button_data, height=1000, distance=0.1*250) # distance is essential!

# Convert peaks to 1 everything else to 0
# The peak of an event occurs at the apex, not at the onset
data['A7'] = photodata = [1 if i in buttonpeaks else 0 for i,_ in enumerate(button_data)]
data['A6'] = button_data =  [1 if i in photopeaks else 0 for i,_ in enumerate(photo_data)]

'''
TODO: Filter data
https://erpinfo.org/order-of-steps
'''
EEG_data = data[['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']]
# EEG_data = data[['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8','A6']]


# Dump to CSV
data.to_csv('data.csv', index=True) 

'''
-.2 sec to .8 sec
TODO: somehthing is getting fucked up with time/sample rate
TODO: need to include the button peak - what range going out? 
'''
epochs = []
for i, peak in enumerate(photopeaks):
    back = int(-.2 * sample_rate + peak)
    ahead = int(.8 * sample_rate + peak)
    EEG = EEG_data.loc[back : ahead]
    button = False
    if len(buttonpeaks) > 0 and buttonpeaks[0] in range(back, ahead+1):
        button = True
    epoch = {'EEG': EEG, 'Button': button}
    epochs.append(epoch)

'''
TODO: average channels
'''

''' AVERAGE EPOCHS
TODO: Gotta make sure flashes match up to key length
'''
correctCat = None
correctDog = None
incorrectCat = None
incorrectDog = None
for i, epoch in enumerate(epochs):
    # for i, e in epoch['EEG'].iterrows():
    #     print(e)
        # break
    if epoch['Button']:
        if key[i]: # if the button was clicked and it was a dog
            category = correctDog
        else: # if button was clicked and it was a cat
            category = incorrectCat
    else:
        if key[i]: # if the button wasn't clicked and it was a dog
            category = incorrectDog
        else: # if the button wasn't clicked and it was a cat
            category = correctCat

    avg_EEG = None
    break

sys.exit()


# print(EEG_data)
# avg_EEG_data = []
# # print(data)
# # print(type(data))
# # data.reindex()
# # for e in EEG_data[2000:2500]:
# for e in EEG_data:
#     # avg = sum(e) / len(e)
#     avg = np.mean(e)
#     avg_EEG_data.append(avg)
# plt.plot(avg_EEG_data)
# plt.xticks(ticks=np.arange(0, round(Timestamps[-1]+1)*sample_rate, round(Timestamps[-1]+1)*sample_rate/4), 
#            labels=np.arange(0, round(Timestamps[-1]+1),round(Timestamps[-1]+1)/4))
# plt.xlabel('Time (s)')
# plt.draw()
# plt.show()
# plt.pause(5)

# print(type(photo_data))

# buttonpeaks, _ = find_peaks(button_data, height=1000, distance=0.1*250) # distance is essential!
# photopeaks, _ = find_peaks(photo_data, height=(10, 20), distance=0.1*250) # distance is essential!
# photopeaks, _ = find_peaks(photo_data, height=(5, 20), distance=0.1*250) # distance is essential!
# print(photo_data[photopeaks[0]])
# print(data['Adjusted Timestamp'][photopeaks[96:101]])






# photo_data = data['A6']
# print(photopeaks.size)
# print(photopeaks[0])
# print(data)



# sys.exit()

# plt.plot(analog_data)
# plt.xticks(ticks=np.arange(0, round(Timestamps[-1]+1)*sample_rate, round(Timestamps[-1]+1)*sample_rate/5), 
#            labels=np.arange(0, round(Timestamps[-1]+1),round(Timestamps[-1]+1)/5))
plt.tight_layout

# plt.plot(buttonpeaks, analog_data[buttonpeaks], "x")
# plt.xticks(ticks=np.arange(0, len(analog_data), len(analog_data)/4), labels=np.arange(0, len(analog_data)/250,len(analog_data)/250/4))
# plt.show()

# rows = 10
# cols = 10
# fig = plt.figure(figsize=(rows, cols), constrained_layout=True)
# gs = fig.add_gridspec(rows, cols)
# for r in range(rows):
#     for c in range(cols):
#         ax = fig.add_subplot(gs[r, c])
# # plt.show()

# window = tk.Tk()
# scroll = tk.Scrollbar(window, orient = 'vertical')
# scroll.pack(side='top')
# window.mainloop()



# sys.exit()
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
ax[1].plot(photo_data)
ax[1].plot(photopeaks, photo_data[photopeaks], "x")

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

# sub = data.iloc[photopeaks[5]-20:buttonpeaks[0]-20][['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']]
# ax[0].plot(sub)
# ax[0].set_xticks(ticks=np.arange(0, len(sub), len(sub)/4))
# ax[0].set_xticklabels(labels=np.arange(0, len(sub)/250,len(sub)/250/4))
epoch1 = epochs[0]
ax[0].plot(epoch1['EEG'])
# ax[0].scatter([buttonpeaks[0]], [1])
ax[0].set_title(epoch1['Button'])
ax[0].axvline(x=photopeaks[0], color='blue', linestyle='--')
ax[0].axvline(x=buttonpeaks[0], color='red', linestyle='--')

plt.show()