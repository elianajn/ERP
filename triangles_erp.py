import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences
import os
import pickle

class ERP:
    def __init__(self):
        """
        left = UP = Analog Ch 0 = A5
        right = DOWN = Analog Ch 1 = A6
        """
        self.openbcipath = '/Users/ellaneurohr/Desktop/erp/triangles.txt'
        self.sample_rate = 250
        # self.data = self.read_raw_erp(self.openbcipath) # TODO: read data
        self.data = None
        self.fig = None
        self.file = None
        self.up_peaks = None
        self.down_peaks = None
        self.up_epochs = []
        self.down_epochs = []
        # left flash - up triangle
        # right flash - down triangle


    def read_bci_text_file(self):
        """ Read in the name of the text file from the user. Save name so that output csv has the same name"""
        # self.file = input("Name of the text file (ex. demo.txt): ")
        self.file = 'triangles.txt'



    def read_raw_erp(self):
        """ Read into txt file that is the output of Open BCI
        First 4 lines need to be removed and 5 is header
        """
        cwd = os.getcwd()
        while(True):
            openbcipath = '{}/{}'.format(cwd, self.file)
            try:
                data = pd.read_csv(openbcipath, sep=", ", header=4, index_col=False, engine='python', usecols=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'Analog Channel 0', 'Analog Channel 1', 'Timestamp'])
                break
            except:
                self.file = input("Unable to open file. Please confirm the name of the BCI text file and reenter: \n")
        new_names = {'EXG Channel 0':'ch1', 'EXG Channel 1':'ch2', 'EXG Channel 2':'ch3', 'EXG Channel 3':'ch4', 'EXG Channel 4':'ch5', 'EXG Channel 5':'ch6', 'EXG Channel 6':'ch7', 'EXG Channel 7':'ch8', 'Analog Channel 0':'A5', 'Analog Channel 1':'A6', 'Timestamp':'TimeStamp'}
        data = data.rename(columns=new_names)
        self.data = data


    def serialize(self):
        """ Serialize the data object so you don't have to keep reading the same file in """
        pass


    def clean_raw_erp(self):
        """ Open BCI records some irrelevant data from before you start recording, then inserts a row of zeros when recording actually began """
        for i, val in enumerate(self.data['ch1']):
            if val == 0:
                self.data = self.data.drop(index=range(0, i+1))
                self.data = self.data.reset_index(drop=True)
                return
        


    def trim_to_video(self):
        """ Trim data down to revelant timeframe 
        Finds the first occurence of the minimum value, which means the screen is on the black of the video. Trim everything before this out so
        peak finding isn't muddled by other light on the screen.
        Reset indices after trim
        """
        a5 = self.data['A5']
        a6 = self.data['A6']
        common_min = min(set(a5) & set(a6))

        # Filter channels so that they only contain the minimum value but retain index values 
        a5 = a5[a5==common_min]
        a6 = a6[a6==common_min]
        a5.name='a5'
        a6.name='a6'
        
        # Join filtered channels so the resulting dataframe only contains indices where both were at the minimum (darkest)
        joined = pd.merge(a5, a6, left_index=True, right_index=True, how='inner')
        first_dark_index = joined.index[0]
        last_dark_index = joined.index[-1]
        self.data = self.data.loc[first_dark_index:last_dark_index]
        self.data = self.data.reset_index(drop=True)
        
    def trim_to_video_2(self):
        a5 = self.data['A5']
        a6 = self.data['A6']
        common_min = min(set(a5) & set(a6))
        uptroughs, _ = find_peaks(-a5, prominence=common_min, width=self.sample_rate*10)
        downtroughs, _ = find_peaks(-a6, prominence=common_min, width=self.sample_rate*10)
        common = np.intersect1d(uptroughs, downtroughs)[0]
        self.data = self.data.loc[common:]
        self.data = self.data.reset_index(drop=True)

    def trim_data(self):
        """ 
        First five peaks are intro signals
        Trim out intro and outro signal flashes and everything before or after them
        Adjust timestamps to restart at 0 after trim
        """
        a5 = self.data['A5']
        a6 = self.data['A6']
        """
        TODO: fix these so they're not hardcoded in
        a5max = max(A5)
        a6max = max(A6)
        """
        a5max = 20
        a6max = 30
        uppeaks, _ = find_peaks(a5, height=4)
        downpeaks, _ = find_peaks(a6, height=4)
        common = np.intersect1d(uppeaks, downpeaks)
        mask = np.isclose(uppeaks[:,None], downpeaks, atol=15)
        idx, _ = np.where(mask)
        last_first = uppeaks[idx[4]]
        first_last = uppeaks[idx[-5]]
        self.data = self.data.loc[last_first+1:first_last-1]
        self.data = self.data.reset_index(drop=True)
        Timestamps = (self.data['TimeStamp'].to_numpy() - self.data['TimeStamp'].to_numpy()[0])
        self.data['Adjusted Timestamp'] = Timestamps


    def find_peaks(self):
        """ Find peaks in photosensor data 
        TODO: don't think i need this? 
        """
        leftpeaks, _ = find_peaks(self.data['A7'], height=(10, 20), distance=0.1*250) # distance is essential!
        rightpeaks, _ = find_peaks(self.data['A6'], height=(10, 20), distance=0.1*250) # distance is essential!
        print(np.argmin(self.data['A6'], keepdims=True))
        print(np.argmin(self.data['A7']))
        # plt.plot(self.data['A6'])
        # plt.scatter(rightpeaks, np.ones(rightpeaks), marker='x')
        # plt.show()
        print(len(rightpeaks))
        print(len(leftpeaks))


    def clean_peaks(self):
        """ Convert peaks to 1 everything else to 0 (Channels A6 and A7) 
        The peak of an event occurs at the apex, not at the onset
        Need to find peaks again because indices have been reset
        """
        a5 = self.data['A5']
        a6 = self.data['A6']
        self.up_peaks = uppeaks, _ = find_peaks(a5, height=4, distance=0.1*self.sample_rate)
        self.down_peaks = downpeaks, _ = find_peaks(a6, height=4, distance=0.1*self.sample_rate)
        a5[~np.isin(a5, uppeaks)] = 0
        a5[uppeaks] = 1
        a6[~np.isin(a6, downpeaks)] = 0
        a6[downpeaks] = 1
        self.data['A5'] = a5
        self.data['A6'] = a6

    def dump_to_csv(self):
        """ Export EEG and peakdata with timestamps to CSV usable in MatLab
        """
        self.data.to_csv('data.csv', index=True) 

    def find_epochs(self):
        """ Find the start of the timeframes surrounding each flash and store in 2 separate lists 
        -0.2 - 0.8 seconds around flash
        """
        for peak in self.up_peaks[0]:
            back = int(-0.2 * self.sample_rate) + peak
            ahead = int(0.8 * self.sample_rate) + peak
            eeg = self.data.loc[back:ahead]
            eeg = eeg.reset_index(drop=True)
            eeg.index.name = 'Index'
            self.up_epochs.append(eeg)
        for peak in self.down_peaks[0]:
            back = int(-0.2 * self.sample_rate) + peak
            ahead = int(0.8 * self.sample_rate) + peak
            eeg = self.data.loc[back:ahead]
            eeg = eeg.reset_index(drop=True)
            eeg.index.name = 'Index'
            self.down_epochs.append(eeg)


    def average_epochs(self):
        """ 
        Average the epochs of each channel for up and down separately so there are 8 channels for each up and down epochs
        """
        merge_up = pd.concat(self.up_epochs, axis = 1)
        merge_down = pd.concat(self.down_epochs, axis=1)
        averaged_up = pd.DataFrame()
        averaged_down = pd.DataFrame()
        for i in range(1, 9):
            ch = 'ch{}'.format(i)
            up_filtered = merge_up.filter(like=ch)
            avg = up_filtered.mean(axis=1)
            averaged_up[ch] = avg
            down_filtered = merge_down.filter(like=ch)
            avg = down_filtered.mean(axis=1)
            averaged_down[ch] = avg

        self.up_epochs = averaged_up
        self.down_epochs = averaged_down
        print(averaged_up['ch1'])
        print(averaged_down['ch1'])
        



    def plot_data(self):
        """ Plot data """
        analog_data = self.data[['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']]
        plt.plot(analog_data)
        plt.show()

    def filter_EEG(self):
        """ Filter EEG data
        TODO: https://erpinfo.org/order-of-steps
        High pass filter at 0.1 Hz
        """
        pass


    def plot_channels(self):
        """ Plot each channel indivdually with an average of the epochs """
        self.fig, (top, bottom) = plt.subplots(nrows=2, ncols=4, figsize=(18,8))
        for i, ax in enumerate(top, start=1):
            title = 'Channel {}'.format(i)
            ax.set_title(title)
            ch = 'ch{}'.format(i)
            ax.plot(self.data[ch])
        for i, ax in enumerate(bottom, start=5):
            title = 'Channel {}'.format(i)
            ax.set_title(title)
            ch = 'ch{}'.format(i)
            ax.plot(self.data[ch])
        plt.show()


    def plot_channels_2(self):
        self.fig, (top, bottom) = plt.subplots(nrows=2, ncols=4, figsize=(18,8))
        for i, ax in enumerate(top, start=1):
            title = 'Channel {}'.format(i)
            ax.set_title(title)
            ch = 'ch{}'.format(i)
            ax.plot(self.up_epochs[ch])
            ax.plot(self.down_epochs[ch])
            ax.axvline(x=50, color='black', linestyle='--')
            # print('Channel {} max: {}\n Channel {} min: {}'.format(i, self.up_epochs[ch].max(), i, self.up_epochs[ch].min()))
            # ax.set_yticks(ticks=np.arange(self.up_epochs[ch].min()+1000, self.up_epochs.min()-1000))
        for i, ax in enumerate(bottom, start=5):
            title = 'Channel {}'.format(i)
            ax.set_title(title)
            ch = 'ch{}'.format(i)
            ax.plot(self.up_epochs[ch])
            ax.plot(self.down_epochs[ch])
            ax.axvline(x=50, color='black', linestyle='--')
        plt.show()



    def save_fig(self):
        """ Save the figure """
        pass
        


    def main(self):
        self.read_bci_text_file()
        self.read_raw_erp()
        self.clean_raw_erp()
        # self.trim_to_video()
        self.trim_to_video_2()
        self.trim_data()
        # self.find_peaks()
        self.clean_peaks()
        self.dump_to_csv()
        self.find_epochs()
        self.average_epochs()
        # self.plot_data()
        self.plot_channels_2()
        print("yuh")




erp = ERP()
erp.main()