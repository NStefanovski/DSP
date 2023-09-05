from scipy.io.wavfile import read, write
import numpy as np

def add_reverb(input_filename, output_filename, delay_samples=1600, decay_factor=0.6):
    # Read input file
    samplerate, data = read(input_filename)

    # Ensure audio is mono
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = np.mean(data, axis=1)

    # Normalize to [-1, 1]
    data = data / 32768.0

    # Create a delayed version of the data
    delayed_data = np.pad(data, (delay_samples, 0), 'constant')

    # Create the echo
    echo = 0.7 * (data + decay_factor * delayed_data[:len(data)])

    # Ensure echo is within [-1, 1]
    echo = np.clip(echo, -1.0, 1.0)

    # Convert back to int16
    echo = (echo * 32767).astype(np.int16)

    # Write output file
    write(output_filename, samplerate, echo)

if __name__ == "__main__":
    add_reverb('clean_guitar.wav', 'reverb_guitar.wav')
