import argparse
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft

class AudioFormatError(Exception): pass

# this function computes amplitude envelope of audio signal
def amplitude_envelope(audio, HOP_SIZE, FRAME_SIZE):
    return np.array([max(audio[i : (i + FRAME_SIZE)]) for i in range(0, len(audio), HOP_SIZE)])

# this function computes RMS energy of audio signal
def rms_energy(audio, HOP_SIZE, FRAME_SIZE):


# this function computes zero crossing rate of audio signal



fs = 44100  # Sample rate
FRAME_SIZE = 2048 
HOP_SIZE = 1024

parser = argparse.ArgumentParser()
parser.add_argument("--audio", help="Pass an audio file for analysis.")
args = parser.parse_args()
if (args.audio is None):
    while True:
        duration_str = input("Enter a duration of your voice audio file in seconds.")  # Duration of recording
        try:
            duration = float(duration_str)
            break
        except ValueError:
            print("You entered invalid duration. Please, try again.")
    audio_signal = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    print("Recording...")
    sd.wait()  # Wait until recording is finished
    #write('voice.wav', fs, voice)  # Save as WAV file

else:
    try:
        audio_signal, fs = librosa.load(args.audio)
    except AudioFormatError:
        print("Unsupported audio format passed.")



# visualize a loaded signal
plt.figure(num=1, figsize=(15, 17))
plt.subplot(3, 1, 1)
librosa.display.waveshow(audio_signal, alpha=1)
plt.ylim((-1, 1))
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Audio signal")

# extract amplitude envelope and plot it
ae_audio_signal = amplitude_envelope(audio_signal, HOP_SIZE, FRAME_SIZE)
frames = range(len(ae_audio_signal))
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)
librosa.display.waveshow(audio_signal, alpha=1)
plt.plot(t, amplitude_envelope, color = "r")
plt.ylim((-1, 1))
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Audio signal with amplitude envelope")

# extract rms energy and plot it
rms_audio = librosa.feature.rms(audio_signal, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
frames = range(len(rms_audio))
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)
librosa.display.waveshow(audio_signal, alpha=1)
plt.plot(t, rms_audio, color = "r")
plt.ylim((-1, 1))
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Audio signal with rms energy")

# extract zero-crossing rate 
plt.show() 


"""
# extract Fourier transform of a signal and plot it
audio_magnitude = np.abs(fft(voice))
freq = np.linspace(0, fs, len(audio_magnitude))
plt.subplot(3, 1, 2)
plt.plot(freq, audio_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.xlim([0, fs/5])

# plot spectrogram of an audio signal
S_scale = librosa.stft(audio_signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_scale = np.abs(S_scale) ** 2
Y_log_scale = librosa.power_to_db(Y_scale)
plt.subplot(3, 1, 3)
librosa.display.specshow(Y_log_scale, sr=fs, hop_length=HOP_SIZE, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.f")

"""











