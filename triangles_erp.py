import enum
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy import stats
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import os
import pickle

class ERP:
    def __init__(self):
        """
        left = UP = Analog Ch 0 = A5
        right = DOWN = Analog Ch 1 = A6
        """
        self.openbcipath = '/Users/ellaneurohr/Desktop/erp/firstone.txt'
        self.sample_rate = 250
        # self.data = self.read_raw_erp(self.openbcipath) # TODO: read data
        self.data = None
        self.fig = None
        self.file = None
        self.up_peaks = None
        self.down_peaks = None
        self.up_epochs = []
        self.down_epochs = []
        self.standard_errors = None
        """
        UP = left = A5 = Analog Channel 0
        DOWN = right = A6 = Analog Channel 1
        """


    def read_bci_text_file(self):
        """ Read in the name of the text file from the user. Save name so that output csv has the same name"""
        # self.file = input("Name of the text file (ex. demo.txt): ")
        self.file = 'triangles.txt'



    def read_raw_erp(self):
        """ Read into txt file that is the output of Open BCI
        First 4 lines need to be removed and 5 is header
        TODO: this doesn't work super well when it can't find it
        """
        cwd = os.getcwd()
        while(True):
            openbcipath = '{}/{}'.format(cwd, self.file)
            try:
                data = pd.read_csv(openbcipath, sep=", ", header=4, index_col=False, engine='python', usecols=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'Analog Channel 0', 'Analog Channel 1', 'Timestamp'])
                break
            except:
                self.file = input("Unable to open file. Please confirm the name of the BCI text file and reenter: ")
        new_names = {'EXG Channel 0':'ch1', 'EXG Channel 1':'ch2', 'EXG Channel 2':'ch3', 'EXG Channel 3':'ch4', 'EXG Channel 4':'ch5', 'EXG Channel 5':'ch6', 'EXG Channel 6':'ch7', 'EXG Channel 7':'ch8', 'Analog Channel 0':'A5', 'Analog Channel 1':'A6', 'Timestamp':'TimeStamp'}
        data = data.rename(columns=new_names)
        self.data = data


    def serialize(self):
        """ Serialize the data object so you don't have to keep reading the same file in """
        pickle.dump(self.data, open('data.p', 'wb'))


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
        TODO: delete??
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
        """ Trim data down to revelant timeframe 
        Finds the first occurence of the minimum value, which means the screen is on the black of the video. Trim everything before this out so
        peak finding isn't muddled by other light on the screen.
        Reset indices after trim
        """
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
        

    def plot_data(self):
        """ Dummy method used in testing/writing """
        analog_data = self.data[['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']]
        plt.plot(analog_data)
        plt.show()


    def bandpass_filter(self, data, lowcut, highcut, order=4):
        """
        TODO: order???? 
        """
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data


    def filter_EEG(self):
        """ Filter EEG data
        TODO: https://erpinfo.org/order-of-steps
        High pass filter at 0.1 Hz
        """
        lowcut = 0.5
        highcut = 30
        for i in range(1,9):
            ch = 'ch{}'.format(i)
            filtered_channel = self.bandpass_filter(self.data[ch], lowcut, highcut)
            self.data[ch] = filtered_channel

    def std_err_mean(self, data):
        return stats.sem(data)

    def compute_sem(self):
        """ 
        TODO: document
        """
        idx = self.up_epochs[0].index
        self.standard_errors = pd.DataFrame(index=idx)
        for i in range(1,9):
            ch = 'ch{}'.format(i)
            up = down = pd.DataFrame(index=idx)
            for epoch in self.up_epochs:
                col = epoch[ch]
                up = pd.concat([up, col], axis=1)
            stde = up.apply(self.std_err_mean, axis=1)
            header = 'up{}'.format(i)
            self.standard_errors[header] = stde
            for epoch in self.down_epochs:
                col = epoch[ch]
                down = pd.concat([down, col], axis=1)
            stde = down.apply(self.std_err_mean, axis=1)
            header = 'down{}'.format(i)
            self.standard_errors[header] = stde




    def plot_raw_channels(self):
        """ Plot each channel indivdually with an average of the epochs 
        """
        self.fig, (top, bottom) = plt.subplots(nrows=2, ncols=4, figsize=(18,8))
        # print(len(self.up_peaks[0]))
        # print(np.ones(len(self.up_peaks[0])))
        tm = len(self.data)
        print(tm)
        for i, ax in enumerate(top, start=1):
            title = 'Channel {}'.format(i)
            ax.set_title(title)
            ch = 'ch{}'.format(i)
            ax.plot(self.data[ch])
            # for up in self.up_peaks[0]:
                # ax.axvline(x=up, color='black', linestyle='--')
        for i, ax in enumerate(bottom, start=5):
            title = 'Channel {}'.format(i)
            ax.set_title(title)
            ch = 'ch{}'.format(i)
            ax.plot(self.data[ch])
        plt.show()


    def plot_avgepochs_channels(self):
        """ Plot the average of each epoch for all 8 channels
        Up: orange
        Down: blue
        TODO: legend
        TODO: standardize axes
        """
        self.fig, (top, bottom) = plt.subplots(nrows=2, ncols=4, figsize=(18,8))
        txt = 'Each of these figures represents the average of each epoch surrounding either an up or down triangle\nAn epoch contains the brainwave data from 0.2 seconds before the stimuli and 0.8 seconds after\nThe dotted line represents the appearance of the stimuli on the screen'
        txt = 'Number of up triangles: {}\nNumber of down triangles: {}'.format(len(self.up_peaks[0]), len(self.down_peaks[0]))
        self.fig.text(0.01,0.93,txt)
        legax = None
        # plt.ylim(-25, 7)
        xticks = [0, 50, 100, 150, 200, 250]
        labels = list(map(lambda x : (x-50)/self.sample_rate, xticks))

        for i, ax in enumerate(top, start=1):
            title = 'Channel {}'.format(i)
            ax.set_title(title)
            ch = 'ch{}'.format(i)
            ax.plot(self.up_epochs[ch], color='red')
            ax.plot(self.down_epochs[ch], color='blue')
            up = 'up{}'.format(i)
            uy1 = self.up_epochs[ch].add(self.standard_errors[up], fill_value=0)
            uy2 = self.up_epochs[ch].sub(self.standard_errors[up], fill_value=0)
            ax.fill_between(self.up_epochs[ch].index, uy1, uy2, color='orange', alpha=0.3)
            down = 'down{}'.format(i)
            dy1 = self.down_epochs[ch].add(self.standard_errors[down], fill_value=0)
            dy2 = self.down_epochs[ch].sub(self.standard_errors[down], fill_value=0)
            ax.fill_between(self.up_epochs[ch].index, dy1, dy2, color='blue', alpha=0.3)
            ax.axvline(x=50, color='black', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black')
            ax.set_ylim(-25, 15)
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels)

            legax = ax
            # print('Channel {} max: {}\n Channel {} min: {}'.format(i, self.up_epochs[ch].max(), i, self.up_epochs[ch].min()))
            # ax.set_yticks(ticks=np.arange(self.up_epochs[ch].min()+1000, self.up_epochs.min()-1000))
        # legax.legend()
        # handles, labels = legax.get_legend_handles_labels()
        # self.fig.legend(handles, labels, loc='upper center')
        # plt.legend([red, blue], ['Up', 'Down'])
        for i, ax in enumerate(bottom, start=5):
            title = 'Channel {}'.format(i)
            ax.set_title(title)
            ch = 'ch{}'.format(i)
            ax.plot(self.up_epochs[ch], color='red')
            ax.plot(self.down_epochs[ch], color='blue')
            up = 'up{}'.format(i)
            uy1 = self.up_epochs[ch].add(self.standard_errors[up], fill_value=0)
            uy2 = self.up_epochs[ch].sub(self.standard_errors[up], fill_value=0)
            ax.fill_between(self.up_epochs[ch].index, uy1, uy2, color='orange', alpha=0.3)
            down = 'down{}'.format(i)
            dy1 = self.down_epochs[ch].add(self.standard_errors[down], fill_value=0)
            dy2 = self.down_epochs[ch].sub(self.standard_errors[down], fill_value=0)
            ax.fill_between(self.up_epochs[ch].index, dy1, dy2, color='blue', alpha=0.3)
            ax.axvline(x=50, color='black', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black')
            ax.set_ylim(-25, 15)
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels)
        # plt.figlegend()
        red_patch = mpatches.Patch(color='red', label='Up Triangles')
        blue_patch = mpatches.Patch(color='blue', label='Down Triangles')
        self.fig.legend(handles=[red_patch, blue_patch],loc='outside upper right')
        plt.show()



    def save_fig(self):
        """ Save the figure """
        self.fig.savefig('triangles.png')
        


    def main(self):
        self.read_bci_text_file()
        self.read_raw_erp()
        self.clean_raw_erp()
        # self.trim_to_video()
        self.trim_to_video_2()
        self.trim_data()
        self.clean_peaks()
        self.filter_EEG()
        # self.serialize()
        self.dump_to_csv()
        self.find_epochs()
        self.compute_sem()
        self.average_epochs()
        # self.plot_raw_channels()
        self.plot_avgepochs_channels()
        # self.save_fig()
        print("yuh")




erp = ERP()
erp.main()