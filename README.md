# **Triangle ERP Experiment** <!-- omit in toc -->
#### ***By Eliana Neurohr*** <!-- omit in toc -->  

> ***Note***: *This ReadMe is a work in progress, your patience is appreciated*
  
<!-- ![](output_demo/demo.png) -->
<!-- no toc -->
## **Table of Contents** <!-- omit in toc -->
- [**Dependencies**](#dependencies)
- [**Instructions**](#instructions)
  - [**Downloading the Code**](#downloading-the-code)
  - [**Creating and Starting the Conda Environment**](#creating-and-starting-the-conda-environment)
    - [**Mac Instructions**](#mac-instructions)
    - [**Windows Instructions**](#windows-instructions)
  - [**Running the Program**](#running-the-program)
  - [**CC Neurophysiology Student Instructions**](#cc-neurophysiology-student-instructions)
- [**EEG Filtering**](#eeg-filtering)
- [**Artifact Rejection**](#artifact-rejection)
- [**References**](#references)

## **Dependencies**
- [Anaconda](https://www.anaconda.com/download)
- Dependencies in Anaconda environment.yml (installation not required)
  - Python
  - [mne](https://mne.tools/stable/index.html); full citation below
  - [pandas](https://pandas.pydata.org/)
  - [numpy](https://numpy.org/)
  - [matplotlib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)
  - [datetime](https://docs.python.org/3/library/datetime.html)

## **Instructions**  
You need Anaconda downloaded and installed to run this program. Please [download](https://www.anaconda.com/download) if you do not have it already before proceeding. 
### **Downloading the Code**  
1. Click the green button that says "Code"
2. Click download zip and extract to a location like your Desktop
  
### **Creating and Starting the Conda Environment**  
***Note**: The $ before a line which you type into the terminal is not actually typed out in the terminal. This is just to signifiy that the line is to be inputed into the terminal*
#### **Mac Instructions**
1. Control/right click the ERP folder you just extracted. Hold Option and click "Copy ERP as Pathname"
2. Open a terminal window
3. 
  ```
  $ cd <paste pathname you just copied>
  $ conda env create -f environment.yml
  $ conda activate ERP_env
  ```
#### **Windows Instructions**
<mark>TODO</mark>

### **Running the Program**  

### **CC Neurophysiology Student Instructions**


## **EEG Filtering**
<mark>TODO</mark>
1. 0.5 - 30 Hz Bandpass filter before epochs are isolated
2. Perform artifact correction # NOT DONE
3. Isolate epochs and perform baseline correction (baseline correction not done)
4. Artifact rejection  
   *"Many systems require that you perform artifact after epoching, so I have put this step after epoching.  However, it works just as well to perform artifact rejection on the continuous EEG, prior to epoching, if your system allows it."*
5. Average the single-trial EEG epochs to create single-subject averaged ERP waveforms
6. Plot ERP waveforms  
    *"You may want to apply a low-pass filter (e.g., half amplitude cutoff = 30 Hz, slope = 12-24 dB/octave) before plotting so that you can see the data more clearly."*

## **Artifact Rejection**
<mark>TODO</mark>
Artifacts are parts of the recorded signal that arise from sources other than the source of interest (i.e., neuronal activity in the brain)
https://mne.tools/dev/auto_tutorials/preprocessing/10_preprocessing_overview.html


## **References**
<mark>TODO</mark>
Experiment inspired by https://docs.openbci.com/Examples/VideoExperiment/  
Code inspiration: https://github.com/OpenBCI/OpenBCI_Experiment
