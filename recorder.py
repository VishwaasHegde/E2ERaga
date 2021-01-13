import sounddevice as sd
# from scipy.io.wavfile import write

fs = 16000  # Sample rate


def record(seconds):
    print('Started recording for {} seconds'.format(seconds))
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print('Stopped recording')
    return myrecording[:,0]
