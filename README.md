# Triangle ERP Experiment <!-- omit in toc -->
#### ***By Eliana Neurohr*** <!-- omit in toc -->  

> ***Note***: *This ReadMe is a work in progress, your patience is appreciated*
  
![](output_demo/demo.png)
<!-- no toc -->
## **Table of Contents** <!-- omit in toc -->
- [**Dependencies**](#dependencies)
- [**Neurophysiology Student Instructions**](#neurophysiology-student-instructions)
  - [**Downloading the Code**](#downloading-the-code)
  - [**Starting Virtual Environment**](#starting-virtual-environment)
  - [**Running the Program**](#running-the-program)
- [**EEG Filtering**](#eeg-filtering)
- [**Artifact Rejection**](#artifact-rejection)
- [**References**](#references)

## **Dependencies**
- [mne](https://mne.tools/stable/index.html); full citation below
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)
- [datetime](https://docs.python.org/3/library/datetime.html)
- [pickle](https://docs.python.org/3/library/pickle.html)
  - included in Python 3.9.* ; not used in main experiment but useful for testing
- [os](https://docs.python.org/3/library/os.html)


## **Neurophysiology Student Instructions**
This experiment requires Python 3 to be installed. If you do not have Python 3 please download [here](https://www.python.org/downloads/).

***Note**: The $ before a line which you type into the terminal is not actually typed out in the terminal. This is just to signifiy that the line is to be inputed into the terminal*

**If you are doing this experiment on one of the lab computers, skip to [*Starting Virtual Environment*](#starting-virtual-environment)**
****
### **Downloading the Code**
1. Click the green button that says "Code"
2. Click download zip and extract to a location like your Desktop
### **Starting Virtual Environment**
A virtual environment is an environment which can have modules that code needs without having those dependencies installed on your machine.  

Open the Terminal on your computer. Lab computer instructions:
   ```
   $ cd Desktop/ERP/
   $ source venv/bin/activate
   ```
   You should see (venv) appear at the far left of the command line now.
### **Running the Program**
First, make sure that the .csv file of the EEG data has been moved into the ERP folder. It is highly recommended that you rename the text file to something more succinct (like your student ID) with no spaces, but leave the .csv extension. 
```
(venv) $ python triangles_experiment.py
```
You will be prompted to enter the name of the .csv file with the EEG data. You may also try the demo.csv file.
The output CSV and image will be saved in a folder within the ERP folder once the window with the graphs that pops up is closed  

If you're getting import issues, first make sure the venv is active. If it is, try:
```
pip install <module causing issues> --force-reinstall
```
The modules that would most likely cause issues are: numpy, scipy, pandas, matplotlib, mne

## **EEG Filtering**
1. 0.5 - 30 Hz Bandpass filter before epochs are isolated
2. Perform artifact correction # NOT DONE
3. Isolate epochs and perform baseline correction (baseline correction not done)
4. Artifact rejection  
   *"Many systems require that you perform artifact after epoching, so I have put this step after epoching.  However, it works just as well to perform artifact rejection on the continuous EEG, prior to epoching, if your system allows it."*
5. Average the single-trial EEG epochs to create single-subject averaged ERP waveforms
6. Plot ERP waveforms  
    *"You may want to apply a low-pass filter (e.g., half amplitude cutoff = 30 Hz, slope = 12-24 dB/octave) before plotting so that you can see the data more clearly."*

## **Artifact Rejection**
Artifacts are parts of the recorded signal that arise from sources other than the source of interest (i.e., neuronal activity in the brain)
https://mne.tools/dev/auto_tutorials/preprocessing/10_preprocessing_overview.html


## **References**
Experiment inspired by https://docs.openbci.com/Examples/VideoExperiment/  
Code inspiration: https://github.com/OpenBCI/OpenBCI_Experiment
