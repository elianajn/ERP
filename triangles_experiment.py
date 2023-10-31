import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.table as mtable
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas
import os
import sys
import datetime
from datetime import date

""" TODO
- Are the ends of the last flash getting caught in that? I know they're not getting considered but don't want the output to include that
    in case someone tries to use that
- delete sample data from /Users/ellaneurohr/mne_data
- Add file into output folder that describes what is being output
- Actually add montages in (ignored while creating mne.Info in read_raw_data and in plotting)
- Read in monitor sizing and use while creating figure
- Use Docker instead of venv
"""

class ERP:
    def __init__(self):
        """
        UP = left = Analog Channel 0 = STI0
        DOWN = right = Analog Channel 1 = STI1
        """
        self.sample_rate = 250
        # self.file = 'demo.csv'
        self.file = 'demo.csv'
        # self.eeg_channels = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8']
        # self.eeg_channels = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8']
        self.eeg_channels = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
        # self.locations = {'EEG1': 'Frontal Left', 'EEG2': 'Frontal Right', 'EEG3': 'Central Left', 'EEG4': 'Central Right', 'EEG5': 'Parietal Left', 'EEG6': 'Parietal Right', 'EEG7': 'Occipital Left', 'EEG8': 'Occipital Right'}
        # self.locations = {'EEG 1': 'Frontal Left', 'EEG 2': 'Frontal Right', 'EEG 3': 'Central Left', 'EEG 4': 'Central Right', 'EEG 5': 'Parietal Left', 'EEG 6': 'Parietal Right', 'EEG 7': 'Occipital Left', 'EEG 8': 'Occipital Right'}
        self.locations = {'Fp1': 'Frontal Left', 'Fp2': 'Frontal Right', 'C3': 'Central Left', 'C4': 'Central Right', 'P7': 'Parietal Left', 'P8': 'Parietal Right', 'O1': 'Occipital Left', 'O2': 'Occipital Right'}
        self.params = dict(
            l_freq = 1.5,
            h_freq = 8,
            t_min = -0.3,
            t_max = 0.7,
            eeg_drop = 225
        )
        self.raw = None
        self.up_events = None
        self.down_events = None
        self.up_epochs = None
        self.down_epochs = None
        self.epoch_info = {'Expected': [30, 173]}
        self.fig = None
        # self.raw.drop_channels(['Sample Index', 'EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'Timestamp'])

    def sandbox(self):
        """
        Used while developing and debugging; will be removed
        """
        up_evoked = self.up_epochs.average(self.eeg_channels)
        down_evoked = self.down_epochs.average(self.eeg_channels)
        print(type(up_evoked))


    def serialize(self, data):
        """ Serialize the data object so you don't have to keep reading the same file in. 
        Used during testing/writing 
        """
        pickle.dump(data, open('data.p', 'wb'))

    def load_serialized(self):
        self.raw = pickle.load(open('data.p', 'rb'))
        print('Raw data loaded from serialized object\n')

    def read_csv_file(self):
        """
        Reads in the name of the Biosemi Data Format file
        This is isolated from reading the data so IO file input can be disabled during testing
        """
        self.file = input("Name of the csv file (ex. demo.csv): ")

    def read_raw_data(self):
        """
        Open and read the raw data file that is output from OpenBCI
        Pandas DataFrame -> np ndarray -> mne.Raw
        """
        # ch_names = ['Sample Index', 'EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'Accel X', 'Accel Y', 'Accel Channel Z', '13', 'D11', 'D12', 'D13', 'D17', '18', 'D18', 'STI0', 'STI1', 'Analog Channel 2', 'Timestamp', 'Marker']
        # ch_names = ['Sample Index', 'EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'Accel X', 'Accel Y', 'Accel Channel Z', '13', 'D11', 'D12', 'D13', 'D17', '18', 'D18', 'STI0', 'STI1', 'Analog Channel 2', 'Timestamp', 'Marker']
        ch_names = ['Sample Index', 'Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'Accel X', 'Accel Y', 'Accel Channel Z', '13', 'D11', 'D12', 'D13', 'D17', '18', 'D18', 'STI0', 'STI1', 'Analog Channel 2', 'Timestamp', 'Marker']
        while(True):
            print('\nLoading input file...\n')
            cwd = os.getcwd()
            csv_path = '{}/{}'.format(cwd, self.file)
            try:
                data_pd = pandas.read_csv(csv_path, sep='\t', index_col=False, engine='python', header=0, names=ch_names)
                break
            except:
                self.file = input("Unable to open file. Please confirm the name of the BCI csv file and reenter: ")
        data_np = data_pd.to_numpy().transpose()
        info = mne.create_info(ch_names=ch_names, sfreq=self.sample_rate)
        info.set_montage(None, on_missing='ignore')
        # info.set_montage('standard_1020')
        self.raw = mne.io.RawArray(data_np, info, verbose='ERROR')
        channel_types = dict.fromkeys(self.eeg_channels, 'eeg')
        self.raw.set_channel_types(channel_types, on_unit_change='ignore') #TODO: make sure weird shit isnt happening
        total = str(datetime.timedelta(seconds=self.raw.n_times / self.sample_rate))
        return 'Input file loaded succesfully\nTotal length of recording: {}\n'.format(total)
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

    def find_stimuli(self):
        """
        Find the events in the two stimuli channels that indicate that a triangle was displayed
        Zhang, G., Garrett, D. R., & Luck, S. J. Optimal Filters for ERP Research I: A General Approach for Selecting Filter Settings. BioRxiv.
        """
        print('Finding stimuli signals...')
        self.up_events = mne.find_events(self.raw, stim_channel='STI0', consecutive=False, min_duration=1 / self.sample_rate, verbose='ERROR')
        self.down_events = mne.find_events(self.raw, stim_channel='STI1', consecutive=False, min_duration=1 / self.sample_rate, verbose='ERROR')
        self.epoch_info['Found'] = [len(self.up_events), len(self.down_events)]
        return '{} up triangles found\n{} down triangles found\n'.format(len(self.up_events), len(self.down_events))

    def find_epochs(self):
        """
        Isolate the up and down triangle epochs (0.3 seconds before stim signal to 0.7 seconds after)
        TODO may be good to delete the raw data after getting this
        """
        self.up_epochs = mne.Epochs(raw=self.raw, events=self.up_events, picks=self.eeg_channels, tmin=self.params['t_min'], tmax=self.params['t_max'], verbose='ERROR')
        self.down_epochs = mne.Epochs(raw=self.raw, events=self.down_events, picks=self.eeg_channels, tmin=self.params['t_min'], tmax=self.params['t_max'], verbose='ERROR')
        return 'Epochs isolated\n'

    def filter_raw_data(self):
        """
        Bandpass filter from 2 - 10 Hz
        """
        self.raw.filter(l_freq=self.params['l_freq'], h_freq=self.params['h_freq'], picks=self.eeg_channels, verbose='ERROR')
        # self.raw.filter(l_freq=self.params['l_freq'], h_freq=self.params['h_freq'], picks=self.eeg_channels)

    def artifact_rejection(self):
        reject_criteria = dict(eeg=self.params['eeg_drop'])
        self.up_epochs.drop_bad(reject=reject_criteria, verbose='ERROR')
        self.down_epochs.drop_bad(reject=reject_criteria, verbose='ERROR')
        self.epoch_info['Rejected\nEpochs\n'] = [30 - len(self.up_epochs), 173 - len(self.down_epochs)]

    def figure_description(self, subfig, colors):
        axs = subfig.subplots(1, 4, sharex=True, sharey=True)
        for ax in axs:
            ax.set_axis_off()
        up_patch = mpatches.Patch(color=colors['up'], label='Up Triangles')
        down_patch = mpatches.Patch(color=colors['down'], label='Down Triangles')
        axs[0].legend(handles=[up_patch, down_patch],loc='lower left')
        triangle_data = pandas.DataFrame(self.epoch_info)
        axs[2].table(cellText=triangle_data.values, colLabels=triangle_data.columns, rowLabels=['UP', 'DOWN'], rowLoc='right', loc='lower right', cellLoc='center', edges='open')
        params = pandas.DataFrame({'Column 1':[
            '{} Hz'.format(self.params['l_freq']),
            '{} Hz'.format(self.params['h_freq']),
            '{} µV'.format(self.params['eeg_drop'])]
        }, index=['Lower Cutoff', 'Upper Cutoff', 'Reject Threshold'])
        paramTable = axs[3].table(cellText=params.values, rowLabels=params.index, loc='lower center', rowLoc='right', cellLoc='center', edges='open')
        paramTable.auto_set_column_width(0)
        

    def plot_data(self):
        """
        TODO montages
        TODO adjust sizing
        """
        print('Plotting data...\n')
        evokeds = dict(
            up=list(self.up_epochs.iter_evoked()),
            down=list(self.down_epochs.iter_evoked()),
        )
        colors = {'up': 'tab:orange', 'down': 'tab:blue'}
        self.fig = plt.figure('Figures', figsize=(16, 7), layout='constrained')
        self.fig.suptitle(self.file.split('.')[0], fontweight='demibold')
        subfigs = self.fig.subfigures(nrows=2, ncols=1, height_ratios=[1, 8])
        subfigs[1].get_layout_engine().set(wspace=0.1, hspace=0.1)
        axs = subfigs[1].subplots(2, 4)
        self.figure_description(subfigs[0], colors)
        txt = 'Close this window to finish running the program.'
        closeText = self.fig.text(x=0.005, y=0.965,s=txt, fontstyle='italic')
        for i, channel in enumerate(self.eeg_channels):
            if i >= 4:
                subplot = axs[1][i-4]
            else:
                subplot = axs[0][i]
            mne.viz.plot_compare_evokeds(evokeds, picks=[channel], axes=subplot, show=False, legend=False, title=self.locations[channel], colors=colors, show_sensors=False)[0]
        plt.show()
        closeText.set_text('')

    def dump_data(self):
        """ Create an output folder and dump the csv and figure into it
        If another folder with that name exists this will create a folder with the date and time appended to the folder name
        TODO: he needed another kind of data dumped oout...
        """
        input_file = self.file.split('.')[0]
        cwd = os.getcwd()
        folder = 'output_{}'.format(input_file)
        path = os.path.join(cwd, folder)
        try:
            os.mkdir(path)
        except FileExistsError:
            day = date.today().strftime('%d-%m-%Y')
            time = datetime.datetime.now().strftime('%H-%M-%S')
            folder = 'output_{}_{}_{}'.format(input_file, day, time)
            path = os.path.join(cwd, folder)
            os.mkdir(path)
        print('Outputting raw data as a CSV and the figure to the folder {}\n'.format(folder))
        csv = '{}.csv'.format(input_file)
        png = '{}.png'.format(input_file)
        csv_path = os.path.join(path, csv)
        png_path = os.path.join(path, png)
        dataframe = self.raw.to_data_frame(picks= self.eeg_channels + ['STI0', 'STI1'])
        dataframe = dataframe.rename(columns={'time':'Time', 'STI0':'Up Stimulus', 'STI1':'Down Stimulus'})
        dataframe.to_csv(csv_path, index=True)
        self.fig.savefig(png_path, facecolor=self.fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
        return path

    def dump_plotted_data(self, path=None):
        mne.set_log_level('ERROR')
        averages = {}
        up = self.up_epochs.average(picks=self.eeg_channels).to_data_frame(time_format='ms', index='time')
        down = self.down_epochs.average(picks=self.eeg_channels).to_data_frame(time_format='ms', index='time')
        up_names = dict(map(lambda channel: (channel, 'Up {}'.format(channel)), self.eeg_channels))
        down_names = dict(map(lambda channel: (channel, 'Down {}'.format(channel)), self.eeg_channels))
        up.rename(columns=up_names, inplace=True)
        down.rename(columns=down_names, inplace=True)
        averages = pandas.concat([up, down], axis=1)
        averages.to_csv(os.path.join(path, 'Figure Data.csv'), index=True)



    def main(self):
        self.read_csv_file()
        print(self.read_raw_data())
        print(self.trim_raw_data())
        # self.load_serialized()
        self.filter_raw_data()
        print(self.find_stimuli())
        print(self.find_epochs())
        self.artifact_rejection()
        self.plot_data()
        folder = self.dump_data()
        self.dump_plotted_data(folder)

        # self.sandbox()

erp = ERP()
erp.main()