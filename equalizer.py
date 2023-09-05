from scipy.signal import butter, sosfilt, sosfreqz
from scipy.io import wavfile
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def parametric_eq(input_filename, output_filename, center_freq, bandwidth, gain, fs=44100):
    lowcut = center_freq - bandwidth / 2
    highcut = center_freq + bandwidth / 2

    # Read input file
    samplerate, data = wavfile.read(input_filename)

    # Ensure audio is mono
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = np.mean(data, axis=1)

    # Normalize to [-1, 1]
    data = data / 32768.0

    # Apply bandpass filter
    data = butter_bandpass_filter(data, lowcut, highcut, samplerate)

    # Apply gain
    data = data * gain

    # Ensure data is within [-1, 1]
    data = np.clip(data, -1.0, 1.0)

    # Convert back to int16
    data = (data * 32767).astype(np.int16)

    # Write output file
    wavfile.write(output_filename, samplerate, data)

if __name__ == "__main__":
    parametric_eq('clean_guitar.wav', 'eq_guitar.wav', 1000, 200, 2.0)
