<mark>TODO</mark>
1. 0.5 - 30 Hz Bandpass filter before epochs are isolated
2. Perform artifact correction # NOT DONE
3. Isolate epochs and perform baseline correction (baseline correction not done)
4. Artifact rejection  
   *"Many systems require that you perform artifact after epoching, so I have put this step after epoching.  However, it works just as well to perform artifact rejection on the continuous EEG, prior to epoching, if your system allows it."*
5. Average the single-trial EEG epochs to create single-subject averaged ERP waveforms
6. Plot ERP waveforms  
    *"You may want to apply a low-pass filter (e.g., half amplitude cutoff = 30 Hz, slope = 12-24 dB/octave) before plotting so that you can see the data more clearly."*