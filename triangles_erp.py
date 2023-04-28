import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

class ERP:
    def __init__(self):
        self.openbcipath = '/Users/ellaneurohr/Desktop/erp/changedgain.txt'
        self.sample_rate = 250
        # self.data = self.read_raw_erp(self.openbcipath) # TODO: read data
        self.data = None
        self.fig = None


    def read_raw_erp(self):
        """ Read into txt file that is the output of Open BCI
        First 4 lines need to be removed and 5 is header
        """
        data = pd.read_csv(self.openbcipath, sep=", ", header=4, index_col=False, engine='python', usecols=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'Analog Channel 0', 'Analog Channel 1', 'Timestamp'])
        new_names = {'EXG Channel 0':'ch1', 'EXG Channel 1':'ch2', 'EXG Channel 2':'ch3', 'EXG Channel 3':'ch4', 'EXG Channel 4':'ch5', 'EXG Channel 5':'ch6', 'EXG Channel 6':'ch7', 'EXG Channel 7':'ch8', 'Analog Channel 0':'A6', 'Analog Channel 1':'A7', 'Timestamp':'TimeStamp'}
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
        


    def trim_data(self):
        """ Trim data down to revelant timeframe 
        Finds the first occurence of the minimum value, which means the screen is on the black of the video. 
        Now the first 5 peaks will be the starting 5 flashes 

        Now trim out intro and outro signal flashes and everything before or after them

        Adjust timestamps to restart at 0 after trimp
        """
        pass


    def find_peaks(self):
        """ Find peaks in photosensor data 
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
        """
        pass

    def dump_to_csv(self):
        """ Export EEG and peakdata with timestamps to CSV usable in MatLab
        """
        self.data.to_csv('data.csv', index=True) 

    def find_epochs():
        """ Find the start of the timeframes surrounding each flash and store in 2 separate lists 
        -0.2 - 0.8 seconds around flash
        """
        pass


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


    def save_fig(self):
        """ Save the figure """
        pass
        


    def main(self):
        self.read_raw_erp()
        self.clean_raw_erp()
        # self.dump_to_csv()
        self.find_peaks()
        # self.plot_data()
        self.plot_channels()
        print("yuh")




erp = ERP()
erp.main()