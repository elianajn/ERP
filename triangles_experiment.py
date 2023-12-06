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
import traceback
import datetime
from datetime import date
import yaml


class ERP:
    def __init__(self):
        """
        TODO set log level everywhere
        """
        self.sample_rate = None
        self.log_level = None
        self.output_results = None
        self.file = 'open.txt'
        # self.file = '7.txt'
        self.eeg_channels = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
        self.locations = {'Fp1': 'Frontal Left', 'Fp2': 'Frontal Right', 'C3': 'Central Left', 'C4': 'Central Right', 'P7': 'Parietal Left', 'P8': 'Parietal Right', 'O1': 'Occipital Left', 'O2': 'Occipital Right'}
        # self.params = dict(
        #     lower_passband_edge = 1.5,
        #     upper_passband_edge = 8,
        #     t_min = -0.3,
        #     t_max = 0.7,
        #     epoch_rejection_threshold = 225
        # )
        self.params = None
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
        up = self.up_epochs.average()
        # up.plot_joint(show=True, title='Up')
        self.up_epochs.average().plot_topomap(show=True, times=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        plt.show()
        # up_evoked = self.up_epochs.average(self.eeg_channels)
        # down_evoked = self.down_epochs.average(self.eeg_channels)
        # print(type(up_evoked))


    def serialize(self, data):
        """ Serialize the data object so you don't have to keep reading the same file in. 
        Used during testing/writing 
        """
        pickle.dump(data, open('data.p', 'wb'))

    def load_serialized(self):
        self.raw = pickle.load(open('data.p', 'rb'))
        print('Raw data loaded from serialized object\n')

    def load_config(self):
        """
        Read parameters from YAML file
        """
        print('Loading parameters from config.yml...')
        with open('config.yml', 'r') as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        default = {'sample_rate': 250, 't_min': -0.3, 't_max': 0.7, 'lower_passband_edge': 1.5, 'upper_passband_edge': 8.0, 'epoch_rejection_threshold': 225, 'rejection_log_level': 'ERROR', 'log_level': 'ERROR', 'output_results': True}
        if data == default:
            txt = 'Default parameters loaded from configuration file'
            self.params = default
        else:
            self.params = data
            txt = 'Using user defined parameters'
        col_width = 30
        table = [
            '',
            ''.center(col_width*2 + 3, '-'),
            '|' + txt.center(col_width*2 + 1) + '|',
            '|' + 'Edit these in config.yml'.center(col_width*2 + 1) + '|',
            ''.center(col_width*2 + 3, '-')
        ]
        units = {'sample_rate': 'Hz', 't_min': 'sec', 't_max': 'sec', 'lower_passband_edge': 'Hz', 'upper_passband_edge': 'Hz', 'epoch_rejection_threshold': 'µV', 'log_level': '', 'rejection_log_level': '', 'output_results': ''}
        try:
            for key in self.params:
                table.append(
                    '|' + '{}'.format(key.replace('_',' ')).center(col_width) +
                    '|' + '{} {}'.format(self.params[key], units[key]).center(col_width) + '|'
                )
        except KeyError as err:
            print(
                '\nUh Oh! The variable name {} is not valid. Please ensure that the variables in config.yml match these names:\n'.format(traceback.format_exception_only(err)[0].split('\'')[1]) +
                ('\t{}\n'*len(default)).format(*default.keys()) + 
                '\nProgram exiting.\n'
            )
            sys.exit()
        table.append(''.center(col_width*2 + 3, '-'))
        table.append('')
        self.sample_rate = self.params.pop('sample_rate')
        self.rejection_log_level = self.params.pop('rejection_log_level')
        self.log_level = self.params.pop('log_level')
        self.output_results = self.params.pop('output_results')
        return '\n'.join(table)


    def read_csv_file(self):
        """
        Reads in the name of the Biosemi Data Format file
        This is isolated from reading the data so IO file input can be disabled during testing
        """
        self.file = input("Enter the name of the txt file (ex. demo.txt): ")

    def read_raw_data(self):
        """
        Open and read the raw data file that is output from OpenBCI
        Pandas DataFrame -> np ndarray -> mne.Raw
        """
        new_names = ['Sample Index', 'Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2','STI0', 'STI1', 'Timestamp']
        old_names = ['Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7', 'Analog Channel 0', 'Analog Channel 1', 'Timestamp']
        ch_names = {}
        for i in range(len(old_names)):
            ch_names[old_names[i]] = new_names[i]
        while(True):
            print('\nLoading input file...\n')
            cwd = os.getcwd()
            csv_path = '{}/{}'.format(cwd, self.file)
            try:
                data_pd = pandas.read_csv(csv_path, sep=", ", header=4, index_col=False, engine='python', usecols=old_names)
                # data_pd = pandas.read_csv(csv_path, sep='\t', index_col=False, engine='python', header=0, names=ch_names)
                break
            except:
                self.file = input("Unable to open file. Please confirm the name of the OpenBCI text file and reenter: ")
        data_pd.rename(columns=ch_names, inplace=True)
        data_np = data_pd.to_numpy().transpose()
        info = mne.create_info(ch_names=new_names, sfreq=self.sample_rate)
        mne.set_log_level(self.log_level)
        # info.set_montage(None, on_missing='ignore')
        # info.set_montage('standard_1020')
        self.raw = mne.io.RawArray(data_np, info)
        channel_types = dict.fromkeys(self.eeg_channels, 'eeg')
        self.raw.set_channel_types(channel_types, on_unit_change='ignore') #TODO: make sure weird shit isnt happening
        self.raw.set_montage('standard_1020')
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
        # self.raw.drop_channels(['Accel X', 'Accel Y', 'Accel Channel Z', '13', 'D11', 'D12', 'D13', 'D17', '18', 'D18', 'Analog Channel 2', 'Marker'])
        AC0 = mne.find_events(self.raw, stim_channel='STI0', consecutive=False)
        AC1 = mne.find_events(self.raw, stim_channel='STI1', consecutive=False)
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
        justified = 'Length of analyzed data: '.ljust(35)
        return 'Data trimmed to relevant timeframe.\n{}{}\nExpected length of analyzed data:  03:46:2\n'.format(justified, total)

    def find_stimuli(self):
        """
        TODO: demo.txt finding extra down!!!
        Find the events in the two stimuli channels that indicate that a triangle was displayed
        Zhang, G., Garrett, D. R., & Luck, S. J. Optimal Filters for ERP Research I: A General Approach for Selecting Filter Settings. BioRxiv.
        """
        print('Finding stimuli signals...')
        STI0 = mne.find_events(self.raw, stim_channel='STI0', consecutive=False, min_duration=1 / self.sample_rate)
        STI1 = mne.find_events(self.raw, stim_channel='STI1', consecutive=False, min_duration=1 / self.sample_rate)
        self.up_events = min(STI0, STI1, key=len)
        self.down_events = max(STI0, STI1, key=len)
        self.epoch_info['Found'] = [len(self.up_events), len(self.down_events)]
        up_info = 'Up triangles found: {}'.format(len(self.up_events)).ljust(30)
        down_info = 'Down triangles found: {}'.format(len(self.down_events)).ljust(30)
        if len(self.up_events) != 30 or len(self.down_events) != 173:
            print('\nWARNING: Unexpected number of events found. Please ensure that the light sensors are well secured with dark tape')
            up_info = '\t{}Expected: 30'.format(up_info)
            down_info = '\t{}Expected: 173'.format(down_info)
        return '{}\n{}\n'.format(up_info, down_info)
        # return 'Up triangles found: {}{}\nDown triangles found: {}{}\n'.format(len(self.up_events), up_warn, len(self.down_events), down_warn)

    def find_epochs(self):
        """
        Isolate the up and down triangle epochs (0.3 seconds before stim signal to 0.7 seconds after)
        TODO may be good to delete the raw data after getting this
        """
        # self.up_epochs = mne.Epochs(raw=self.raw, events=self.up_events, picks=self.eeg_channels, tmin=self.params['t_min'], tmax=self.params['t_max'], verbose='ERROR')
        self.up_epochs = mne.Epochs(raw=self.raw, events=self.up_events, picks=self.eeg_channels, tmin=self.params['t_min'], tmax=self.params['t_max'], preload=True)
        # self.down_epochs = mne.Epochs(raw=self.raw, events=self.down_events, picks=self.eeg_channels, tmin=self.params['t_min'], tmax=self.params['t_max'], verbose='ERROR')
        self.down_epochs = mne.Epochs(raw=self.raw, events=self.down_events, picks=self.eeg_channels, tmin=self.params['t_min'], tmax=self.params['t_max'], preload=True)
        return 'Epochs isolated\n'

    def filter_raw_data(self):
        """
        Bandpass FIR filter
        """
        self.raw = self.raw.filter(l_freq=self.params['lower_passband_edge'], h_freq=self.params['upper_passband_edge'], picks=self.eeg_channels)
        # self.raw = self.raw.filter(l_freq=self.params['l_freq'], h_freq=self.params['h_freq'], picks=self.eeg_channels, verbose='ERROR')
        # self.raw.filter(l_freq=self.params['l_freq'], h_freq=self.params['h_freq'], picks=self.eeg_channels)

    def artifact_rejection(self):
        reject_criteria = dict(eeg=self.params['epoch_rejection_threshold'])
        up, down = self.epoch_info['Found']
        if self.rejection_log_level in ['DEBUG', 'INFO']:
            print('Rejecting bad up epochs')
            self.up_epochs.drop_bad(reject=reject_criteria, verbose=self.rejection_log_level)
            print('\nRejecting bad down epochs')
            self.down_epochs.drop_bad(reject=reject_criteria, verbose=self.rejection_log_level)
            print()
        else:
            self.up_epochs.drop_bad(reject=reject_criteria, verbose=self.rejection_log_level)
            self.down_epochs.drop_bad(reject=reject_criteria, verbose=self.rejection_log_level)
            print('{} / {} up epochs rejected'.format(up - len(self.up_epochs), up))
            print('{} / {} down epochs rejected\n'.format(down - len(self.down_epochs), down))
        self.epoch_info['Rejected\nEpochs\n'] = rejUp, rejDown= [up - len(self.up_epochs), down - len(self.down_epochs)]
        self.epoch_info['Percent\nRejected\n'] = [
            '{:.2f}%'.format(100 * (rejUp / up)),
            '{:.2f}%'.format(100 * (rejDown / down))
        ]

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
            '{} Hz'.format(self.params['lower_passband_edge']),
            '{} Hz'.format(self.params['upper_passband_edge']),
            '{} µV'.format(self.params['epoch_rejection_threshold']),
            '{} Hz'.format(self.sample_rate)]
        }, index=['Lower Cutoff', 'Upper Cutoff', 'Reject Threshold', 'Sample Rate'])
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
            # mne.viz.plot_compare_evokeds(dict(
            #     up = list(self.up_epochs.copy().pick(channel).iter_evoked()),
            #     down = list(self.down_epochs.copy().pick(channel).iter_evoked())
            # ), 
            # picks=channel, axes=subplot, show=False, legend=False, title=self.locations[channel], colors=colors, show_sensors=False)[0]
            mne.viz.plot_compare_evokeds(evokeds, picks=channel, axes=subplot, show=False, legend=False, title=self.locations[channel], colors=colors, show_sensors=False)[0]
        plt.show()
        closeText.set_text('')

    def dump_data(self):
        """ Create an output folder and dump the csv and figure into it
        If another folder with that name exists this will create a folder with the date and time appended to the folder name
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
        """
        Output the average (either up or down, same as the plotted data) of all all epochs for each channel into to a csv
        Useful for further analysis of average peaks in Excel
        """
        # mne.set_log_level('ERROR')
        averages = pandas.DataFrame()
        for channel in self.eeg_channels:
            up = self.up_epochs.average(picks=channel).to_data_frame(time_format='ms', index='time')
            down = self.down_epochs.average(picks=channel).to_data_frame(time_format='ms', index='time')
            up.rename(columns={channel : 'Up {}'.format(channel)}, inplace=True)
            down.rename(columns={channel : 'Down {}'.format(channel)}, inplace=True)
            ch_avg = pandas.concat([up, down], axis=1)
            averages = pandas.concat([averages, ch_avg], axis=1)
        averages.to_csv(os.path.join(path, 'Figure Data.csv'), index=True)

    def epoch_drop_fig(self):
        """
        TODO: confirm averaging done in dump plotted data doesn't do it in place
        """
        fig = plt.figure()




    def main(self):
        print(self.load_config())
        self.read_csv_file()
        print(self.read_raw_data())
        # fig = self.raw.plot_sensors(show_names=True, block=True)
        # cwd = os.getcwd()
        # fig_path = os.path.join(cwd, 'montage.png')
        # fig.savefig(fig_path)
        # return
        print(self.trim_raw_data())
        # self.load_serialized()
        self.filter_raw_data()
        print(self.find_stimuli())
        print(self.find_epochs())
        self.artifact_rejection()
        # self.sandbox()
        self.plot_data()
        if self.output_results:
            folder = self.dump_data()
            self.dump_plotted_data(folder)


erp = ERP()
erp.main()
