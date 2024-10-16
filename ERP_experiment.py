import os
import sys
import traceback
import datetime
from datetime import date
import yaml
from screeninfo import get_monitors
import numpy as np
import mne
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.table as mtable
import matplotlib.gridspec as gridspec
import pandas

class SignalFlashesNotFound(Exception):
    """
    Exception raised when all 10 signal flashes are not found
    """
    def __init__(self, found):
        self.message = "Raw data could not be trimmed because {} singal flashes were found but 10 were expected".format(found)
        super().__init__(self.message)

class ERP:
    def __init__(self):
        """
        """
        self.sample_rate = None
        self.log_level = None
        self.output_results = None
        self.file = 'demo.txt'
        self.eeg_channels = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
        self.locations = {'Fp1': 'Frontal Left', 'Fp2': 'Frontal Right', 'C3': 'Central Left', 'C4': 'Central Right', 'P7': 'Parietal Left', 'P8': 'Parietal Right', 'O1': 'Occipital Left', 'O2': 'Occipital Right'}
        self.params = None
        self.raw = None
        self.left_flash_events = None
        self.right_flash_events = None
        self.left_flash_epochs = None
        self.right_flash_epochs = None
        self.epoch_info = None
        self.video_length = None
        self.fig = None
        

    def load_config(self):
        """
        Read configuration parameters from YAML file
        """
        print('Loading parameters from config.yml...')
        with open('config.yml', 'r') as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        default = {'sample_rate': 250, 
                   't_min': -0.3, 
                   't_max': 0.7, 
                   'lower_passband_edge': 1.5, 
                   'upper_passband_edge': 8.0, 
                   'epoch_rejection_threshold': 225, 
                   'rejection_log_level': 'ERROR', 
                   'log_level': 'ERROR', 
                   'output_results': True,
                   'left_flash_condition_name': 'Up Triangles',
                   'right_flash_condition_name': 'Down Triangles',
                   'number_of_left_flashes': 30,
                   'number_of_right_flashes': 173,
                   'video_length': '0:03:46.2'}
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
        units = {'sample_rate': 'Hz', 't_min': 'sec', 't_max': 'sec', 'lower_passband_edge': 'Hz', 'upper_passband_edge': 'Hz', 'epoch_rejection_threshold': 'µV'}
        try:
            for key in self.params:
                if key in units:
                    unit = units[key]
                else:
                    unit = ''
                table.append(
                    '|' + '{}'.format(key.replace('_',' ')).center(col_width) +
                    '|' + '{} {}'.format(self.params[key], unit).center(col_width) + '|'
                )
            table.append(''.center(col_width*2 + 3, '-'))
            table.append('')
            self.sample_rate = self.params.pop('sample_rate')
            self.rejection_log_level = self.params.pop('rejection_log_level')
            self.log_level = self.params.pop('log_level')
            self.output_results = self.params.pop('output_results')
            self.epoch_info = {'Expected': [self.params.pop('number_of_left_flashes'), self.params.pop('number_of_right_flashes')]}
            self.video_length = str(datetime.timedelta(seconds=self.params.pop('video_length')))
        except KeyError as err:
            print(
                '\nUh Oh! There is an invalid variable name in config.yml. Please ensure that the variables in config.yml match these names:\n' +
                ('\t{}\n'*len(default)).format(*default.keys()) + 
                '\nProgram exiting.\n'
            )
            sys.exit()
        return '\n'.join(table)


    def read_csv_file(self):
        """
        Reads in the name of the Biosemi Data Format file
        This is isolated from reading the data so IO file input can be disabled during developing/testing
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
                break
            except:
                self.file = input("Unable to open file. Please confirm the name of the OpenBCI text file and reenter: ")
        data_pd.rename(columns=ch_names, inplace=True)
        data_pd[self.eeg_channels] *= -1
        data_np = data_pd.to_numpy().transpose()
        info = mne.create_info(ch_names=new_names, sfreq=self.sample_rate)
        mne.set_log_level(self.log_level)
        self.raw = mne.io.RawArray(data_np, info)
        channel_types = dict.fromkeys(self.eeg_channels, 'eeg')
        self.raw.set_channel_types(channel_types, on_unit_change='ignore')
        self.raw.set_montage('standard_1020')
        total = str(datetime.timedelta(seconds=self.raw.n_times / self.sample_rate))
        return 'Input file loaded succesfully\nTotal length of recording: {}\n'.format(total)
    

    def trim_raw_data(self):
        """
        There are five quick signal flashes at the beginning and end of the video
        Find all 'simultaneous' flashes in the stimulus channels then narrow that down to the 10 true signal flashes
        Trim the data from the first signal flash to the last signal flashes
        The data during the signal flashes themselves is not cropped out because it includes time that falls within the first and last epochs
        """
        AC0 = mne.find_events(self.raw, stim_channel='STI0', consecutive=False)
        AC1 = mne.find_events(self.raw, stim_channel='STI1', consecutive=False)
        AC0_col0 = AC0[:,0]
        AC1_col0 = AC1[:,0]
        mask = np.isclose(AC0_col0[:, None], AC1_col0, atol=15)
        idx, _ = np.where(mask)
        simulFlashes = AC0[idx]
        i = 0
        mask = np.array([], dtype=np.int64)
        while i < len(simulFlashes) - 4:
            flash = simulFlashes[i][0]
            fifth = simulFlashes[i + 4][0]
            if fifth - flash <= 250:
                idx = np.array(list(range(i, i + 5)))
                mask = np.concatenate((mask, idx))
                i += 5
            else:
                i += 1
        signalFlashes = simulFlashes[mask]
        if len(signalFlashes) != 10:
            raise SignalFlashesNotFound(len(signalFlashes))
        firstSignal = (signalFlashes[0][0]) / self.sample_rate
        secondSignal = (signalFlashes[-1][0]) / self.sample_rate
        self.raw.crop(firstSignal, secondSignal)
        seconds = self.raw.n_times / self.sample_rate
        total = str(datetime.timedelta(seconds=seconds))
        justified = 'Length of analyzed data: '.ljust(35)
        return 'Data trimmed to relevant timeframe.\n{}{}\nExpected length of analyzed data:  {}\n'.format(justified, total, self.video_length) # Change this if you keep the cropping to include signal flashes


    def filter_raw_data(self):
        """
        Bandpass FIR filter
        """
        self.raw = self.raw.filter(l_freq=self.params['lower_passband_edge'], h_freq=self.params['upper_passband_edge'], picks=self.eeg_channels)


    def find_stimuli(self):
        """
        Find the events in the two stimuli channels. 
        The first and last 5 stimuli flashes are not included as events, but the data during those 10 flashes is not cropped out
        because it includes EEG data for the first and last epochs
        """
        print('Finding stimuli signals...')
        STI0 = mne.find_events(self.raw, stim_channel='STI0', consecutive=False, min_duration=1 / self.sample_rate, initial_event=True)[5:-5]
        STI1 = mne.find_events(self.raw, stim_channel='STI1', consecutive=False, min_duration=1 / self.sample_rate, initial_event=True)[5:-5]
        self.left_flash_events, self.right_flash_events = STI0, STI1
        self.epoch_info['Found'] = [len(self.left_flash_events), len(self.right_flash_events)]
        up_info = '{} found: {}'.format(self.params['left_flash_condition_name'], len(self.left_flash_events)).ljust(30)
        down_info = '{} found: {}'.format(self.params['right_flash_condition_name'], len(self.right_flash_events)).ljust(30)
        if len(self.left_flash_events) != self.epoch_info['Expected'][0] or len(self.right_flash_events) != self.epoch_info['Expected'][1]:
            print('\nWARNING: Unexpected number of events found. Please ensure that the light sensors are well secured with dark tape')
            up_info = '\t{}Expected: {}'.format(up_info, self.epoch_info['Expected'][0])
            down_info = '\t{}Expected: {}'.format(down_info, self.epoch_info['Expected'][1])
        return '{}\n{}\n'.format(up_info, down_info)


    def find_epochs(self):
        """
        Isolate the epochs (t_mind seconds before stim signal to t_max seconds after)
        """
        self.left_flash_epochs = mne.Epochs(raw=self.raw, events=self.left_flash_events, picks=self.eeg_channels, tmin=self.params['t_min'], tmax=self.params['t_max'], preload=True)
        self.right_flash_epochs = mne.Epochs(raw=self.raw, events=self.right_flash_events, picks=self.eeg_channels, tmin=self.params['t_min'], tmax=self.params['t_max'], preload=True)
        return 'Epochs isolated\n'
    

    def artifact_rejection(self):
        """
        Use the epoch rejection threshold defined in the config file to drop bad epochs
        """
        reject_criteria = dict(eeg=self.params['epoch_rejection_threshold'])
        left, right = self.epoch_info['Found']
        if self.rejection_log_level in ['DEBUG', 'INFO']:
            print('Rejecting bad up epochs')
            self.left_flash_epochs.drop_bad(reject=reject_criteria, verbose=self.rejection_log_level)
            print('\nRejecting bad down epochs')
            self.right_flash_epochs.drop_bad(reject=reject_criteria, verbose=self.rejection_log_level)
            print()
        else:
            self.left_flash_epochs.drop_bad(reject=reject_criteria, verbose=self.rejection_log_level)
            self.right_flash_epochs.drop_bad(reject=reject_criteria, verbose=self.rejection_log_level)
            print('{} / {} of left condition epochs rejected'.format(left - len(self.left_flash_epochs), left))
            print('{} / {} of right condition epochs rejected\n'.format(right - len(self.right_flash_epochs), right))
        self.epoch_info['Rejected\nEpochs\n'] = rejLeft, rejRight= [left - len(self.left_flash_epochs), right - len(self.right_flash_epochs)]
        self.epoch_info['Percent\nRejected\n'] = [
            '{:.2f}%'.format(100 * (rejLeft / left)),
            '{:.2f}%'.format(100 * (rejRight / right))
        ]

    def figure_description(self, subfig, colors):
        axs = subfig.subplots(1, 3, sharex=True, sharey=False, width_ratios=[2, 1, 1])
        for ax in axs:
            ax.set_axis_off()
        left_condition_patch = mpatches.Patch(color=colors['up'], label=self.params['left_flash_condition_name'])
        right_condition_patch = mpatches.Patch(color=colors['down'], label=self.params['right_flash_condition_name'])
        subfig.legend(handles=[left_condition_patch, right_condition_patch],loc='lower left')
        triangle_data = pandas.DataFrame(self.epoch_info)
        axs[1].table(cellText=triangle_data.values, colLabels=triangle_data.columns, rowLabels=[self.params['left_flash_condition_name'].upper(), self.params['right_flash_condition_name'].upper()], rowLoc='right', loc='right', cellLoc='center', edges='open')
        params = pandas.DataFrame({'Column 1':[
            '{} Hz'.format(self.params['lower_passband_edge']),
            '{} Hz'.format(self.params['upper_passband_edge']),
            '{} µV'.format(self.params['epoch_rejection_threshold']),
            '{} Hz'.format(self.sample_rate)]
        }, index=['Lower Cutoff', 'Upper Cutoff', 'Reject Threshold', 'Sample Rate'])
        paramTable = axs[2].table(cellText=params.values, rowLabels=params.index, loc='center', rowLoc='right', cellLoc='center', edges='open')
        paramTable.auto_set_column_width(0)
        

    def plot_data(self):
        """
        GTAGG
        """
        print('Plotting data...\n')
        evokeds = dict(
            up=list(self.left_flash_epochs.iter_evoked()),
            down=list(self.right_flash_epochs.iter_evoked()),
        )
        colors = {'up': 'tab:orange', 'down': 'tab:blue'}
        primary_monitor = get_monitors()[0]
        screen_width, screen_height = primary_monitor.width, primary_monitor.height
        fig_width = (screen_width // 100) * 100
        fig_height = fig_width // 2
        diff_width, diff_height = (screen_width - fig_width) // 2, (screen_height - fig_height) // 2
        self.fig = plt.figure('Averaged Channels', layout='constrained')
        self.fig.suptitle(self.file.split('.')[0], fontweight='demibold')
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(diff_width, 0, fig_width, fig_height) # X, Y, width, height
        txt = 'Close this window to finish running the program.'
        closeText = self.fig.text(x=0.005, y=0.965,s=txt, fontstyle='italic')
        subfigs = self.fig.subfigures(nrows=2, ncols=1, height_ratios=[1, 8], hspace=0.05)
        subfigs[1].get_layout_engine().set(wspace=0.1, hspace=0.1)
        axs = subfigs[1].subplots(2, 4)
        self.figure_description(subfigs[0], colors)
        ticks = [x / 10 for x in range(int(self.params['t_min'] * 10) + 1, int(self.params['t_max'] * 10) + 1, 2)]
        for i, channel in enumerate(self.eeg_channels):
            if i >= 4:
                subplot = axs[1][i-4]
            else:
                subplot = axs[0][i]
            subplot.set_xticks(ticks)
            subplot.ticklabel_format(axis='y', style='scientific', scilimits=[0, 0])
            mne.viz.plot_compare_evokeds(evokeds, picks=channel, axes=subplot, show=False, legend=False, 
                                            title=self.locations[channel], colors=colors, show_sensors=False,
                                            truncate_xaxis=False, truncate_yaxis=False)[0]
        plt.show()
        closeText.set_text('')


    def dump_data(self):
        """ 
        Create an output folder and dump the csv and figure into it
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
        dataframe = dataframe.rename(columns={'time':'Time', 'STI0': '{} Stimulus'.format(self.params['left_flash_condition_name']), 'STI1':'{} Stimulus'.format(self.params['right_flash_condition_name'])})
        dataframe.to_csv(csv_path, index=True)
        self.fig.savefig(png_path, facecolor=self.fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
        return path

    def dump_plotted_data(self, path=None):
        """
        Output the average (either up or down, same as the plotted data) of all all epochs for each channel into to a csv
        Useful for further analysis of average peaks in Excel
        """
        averages = pandas.DataFrame()
        for channel in self.eeg_channels:
            left = self.left_flash_epochs.average(picks=channel).to_data_frame(time_format='ms', index='time')
            right = self.right_flash_epochs.average(picks=channel).to_data_frame(time_format='ms', index='time')
            left.rename(columns={channel : '{} {}'.format(self.params['left_flash_condition_name'], channel)}, inplace=True)
            right.rename(columns={channel : '{} {}'.format(self.params['right_flash_condition_name'], channel)}, inplace=True)
            ch_avg = pandas.concat([left, right], axis=1)
            averages = pandas.concat([averages, ch_avg], axis=1)
        averages.to_csv(os.path.join(path, 'Figure Data.csv'), index=True)


    def main(self):
        print(self.load_config())
        self.read_csv_file() # Comment out to use without prompting for file (uses demo.txt)
        print(self.read_raw_data())
        print(self.trim_raw_data())
        self.filter_raw_data()
        print(self.find_stimuli())
        print(self.find_epochs())
        self.artifact_rejection()
        self.plot_data()
        if self.output_results:
            folder = self.dump_data()
            self.dump_plotted_data(folder)


erp = ERP()
erp.main()
