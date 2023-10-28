from tabnanny import verbose
import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas
import os
import sys
import datetime

class ERP:
    def __init__(self):
        """
        UP = left = Analog Channel 0 = STI0
        DOWN = right = Analog Channel 1 = STI1
        """
        self.sample_rate = 250
        self.file = 'flowexperiment.csv'
        self.eeg_channels = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8']
        self.raw = None
        self.up_events = None
        self.down_events = None
        self.up_epochs = None
        self.down_epochs = None
        # self.raw.drop_channels(['Sample Index', 'EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'Timestamp'])

    def sandbox(self):
        """
        TODO: delete sample data from /Users/ellaneurohr/mne_data
        """
        sample_data_folder = mne.datasets.sample.data_path()
        sample_data_raw_file = (sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif")
        raw = mne.io.read_raw_fif(sample_data_raw_file)
        fig = raw.plot(show=True)
        print(type(fig))
        fig.show()
    
    def basic_plot(self):
        self.raw.plot(block=True)

    def serialize(self, data):
        """ Serialize the data object so you don't have to keep reading the same file in. 
        Used during testing/writing 
        """
        pickle.dump(data, open('data.p', 'wb'))

    def load_serialized(self):
        self.raw = pickle.load(open('data.p', 'rb'))
        print('Raw data loaded from serialized object\n')

    def read_txt_file(self):
        """
        Reads in the name of the Biosemi Data Format file
        This is isolated from reading the data so IO file input can be disabled during testing
        """
        self.file = input("Name of the txt file (ex. demo.txt): ")

    def read_raw_data(self):
        """
        Open and read the raw data file that is output from OpenBCI
        Pandas DataFrame -> np ndarray -> mne.Raw
        """
        ch_names = ['Sample Index', 'EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'Accel X', 'Accel Y', 'Accel Channel Z', '13', 'D11', 'D12', 'D13', 'D17', '18', 'D18', 'STI0', 'STI1', 'Analog Channel 2', 'Timestamp', 'Marker']
        while(True):
            cwd = os.getcwd()
            csv_path = '{}/{}'.format(cwd, self.file)
            try:
                data_pd = pandas.read_csv(csv_path, sep='\t', index_col=False, engine='python', header=0, names=ch_names)
                break
            except:
                self.file = input("Unable to open file. Please confirm the name of the BCI csv file and reenter: ")
        data_np = data_pd.to_numpy().transpose()
        info = mne.create_info(ch_names=ch_names, sfreq=self.sample_rate)
        self.raw = mne.io.RawArray(data_np, info, verbose='ERROR')
        # self.serialize(self.raw)
    
    def trim_raw_data(self):
        """
        There are five quick signal flashes at the beginning and end of the video
        Find all 'simultaneous' flashes in the stimulus channels then narrow that down to the 10 true signal flashes
        Trim the data from the last of the first five to the first of the last five
        First drop all irrelevant channels to reduce memory usage
        TODO: error if it cannot find both sets of five
        """
        self.raw.drop_channels(['Accel X', 'Accel Y', 'Accel Channel Z', '13', 'D11', 'D12', 'D13', 'D17', '18', 'D18', 'Analog Channel 2', 'Marker'])
        AC0 = mne.find_events(self.raw, stim_channel='STI0', consecutive=False, verbose='ERROR')
        AC1 = mne.find_events(self.raw, stim_channel='STI1', consecutive=False, verbose='ERROR')
        AC0_col0 = AC0[:,0]
        AC1_col0 = AC1[:,0]
        mask = np.isclose(AC0_col0[:,None], AC1_col0, atol=15)
        idx, _ = np.where(mask)
        simulFlashes = AC0[idx]
        i = 0
        mask = np.array([], dtype=np.int64)
        while i < len(simulFlashes)-4:
            flash = simulFlashes[i][0]
            fifth = simulFlashes[i+4][0]
            if fifth - flash <= 250:
                idx = np.array(list(range(i, i+5)))
                mask = np.concatenate((mask, idx))
                i += 5
            else:
                i += 1
        signalFlashes = simulFlashes[mask]
        if len(signalFlashes) != 10:
            print('Uh Oh! Didn\'t find the signal flahes. Exiting program.') #TODO: actually throw error here!
            sys.exit()
        lastFirstTime = (signalFlashes[4][0] + 1) / self.sample_rate
        firstLastTime = (signalFlashes[5][0] -1) / self.sample_rate
        self.raw.crop(lastFirstTime, firstLastTime)
        seconds = self.raw.n_times / self.sample_rate
        total = str(datetime.timedelta(seconds=seconds))
        self.serialize(self.raw)
        return 'Data trimmed to relevant timeframe.\nLength of analyzed data: {}\nExpected length of analyzed data: 03:46:2\n'.format(total)

    def filter_raw_data(self):
        """
        TODO: oh lordt
        """
        self.raw.filter(l_freq=2, h_freq=10, picks=self.eeg_channels, verbose='ERROR')
        
    def find_stimuli(self):
        """
        Find the events in the two stimuli channels that indicate that a triangle was displayed
        Zhang, G., Garrett, D. R., & Luck, S. J. Optimal Filters for ERP Research I: A General Approach for Selecting Filter Settings. BioRxiv.
        """
        self.up_events = mne.find_events(self.raw, stim_channel='STI0', consecutive=False, min_duration=1 / self.sample_rate, verbose='ERROR')
        print('{} up triangles found\n'.format(len(self.up_events)))
        self.down_events = mne.find_events(self.raw, stim_channel='STI1', consecutive=False, min_duration=1 / self.sample_rate, verbose='ERROR')
        print('{} down triangles found\n'.format(len(self.down_events)))

    def find_epochs(self):
        """
        Isolate the up and down triangle epochs (0.2 seconds before stim signal to 0.8 seconds after)
        TODO may be good to delete the raw data after getting this
        """
        self.up_epochs = mne.Epochs(raw=self.raw, events=self.up_events, picks=self.eeg_channels, tmin=-0.2, tmax=0.8, verbose='ERROR')
        self.down_epochs = mne.Epochs(raw=self.raw, events=self.down_events, picks=self.eeg_channels, tmin=-0.2, tmax=0.8, verbose='ERROR')

    def average_epochs(self):
        """
        Evoked data are obtained by averaging epochs
        """
        self.up_evoked = self.up_epochs.average(picks=self.eeg_channels)
        self.down_evoked = self.down_epochs.average(picks=self.eeg_channels)
        # evokeds = dict(
        #     up=list(self.up_epochs.iter_evoked()),
        #     down=list(self.down_epochs.iter_evoked()),
        # )
        # print(evokeds['up'])
        print(self.up_evoked.pick(['EEG 2']))
        # fig = mne.viz.plot_compare_evokeds(evokeds, combine="mean", picks=self.eeg_channels)

    def plot_data(self):
        """
        TODO montages
        """
        evokeds = dict(
            up=list(self.up_epochs.iter_evoked()),
            down=list(self.down_epochs.iter_evoked()),
        )
        fig = plt.figure(figsize=(18, 8))
        for i, channel in enumerate(self.eeg_channels, start=1):
            plot = mne.viz.plot_compare_evokeds(evokeds, picks=[channel], title=channel, show=False)
            canvas = FigureCanvas(plot[0])
            canvas.draw()
            img = np.asarray(canvas.buffer_rgba())
            subplot = fig.add_subplot(2, 4, i)
            subplot.imshow(img)
            plt.close(plot[0])

        plt.show()



    def main(self):
        # self.sandbox()
        # self.read_raw_data()
        self.load_serialized()
        # self.trim_raw_data()
        # print(self.trim_raw_data()) #TODO: uncomment for logging
        self.filter_raw_data()
        self.find_stimuli()
        self.find_epochs()
        # self.average_epochs()
        self.plot_data()


erp = ERP()
erp.main()