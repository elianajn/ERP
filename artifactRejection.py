import pickle 
import mne


class Artifact():
    def __init__(self):
        self.eeg = self.load_serialized()

    def load_serialized(self):
        return pickle.load(open('data.p', 'rb'))

    def main(self):
        pass


artifact = Artifact()
artifact.main()