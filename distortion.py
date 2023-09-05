import librosa
import numpy as np
from scipy.io.wavfile import write

def apply_distortion_gain(input_file, output_file, threshold, gain_factor):
    """
    Apply distortion and gain to an audio file.

    Parameters:
    input_file (str): The path to the audio file to distort.
    output_file (str): The path to save the distorted audio file.
    threshold (float): The absolute value above which the signal will be clipped.
    gain_factor (float): The factor by which to adjust the gain.
    """
    # Load the audio file
    sr, audio_signal = librosa.load(input_file, sr=None)

    # Apply hard clipping distortion
    audio_signal = np.clip(audio_signal, -threshold, threshold)

    # Adjust the gain
    audio_signal *= gain_factor

    # Save the distorted audio to a new file
    write(output_file, sr, audio_signal)

# Apply distortion and gain to the provided audio file
apply_distortion_gain('clean_guitar.wav', 'distorted_guitar.wav', threshold=0.05, gain_factor=1.0)
