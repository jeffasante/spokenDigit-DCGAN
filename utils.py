import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import os




## for graphs
def plot_time_domain(signal, sample_rate=None, title='Time Domain'):
    plt.figure(figsize=(15, 5))
    print(f'Shape of time domain: {signal.shape}')
    if sample_rate: print(f'Sample rate of time domain: {sample_rate}')
    plt.plot(np.linspace(0,  sample_rate, num=len(signal)), signal)
    plt.ylabel('amplitude')
    plt.xlabel('time (s)')
    plt.title(title)
    plt.grid(True)
    

    

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig



## for audio

def findFiles(path): return glob.glob(path)


def loadWav(full_path):
    sampling_rate, data = wavfile.read(full_path)
    
    return data, sampling_rate


def get_info(audio_data):
    
    # read audio
    sample_rate, audio = wavfile.read(audio_data)

    sample_rate = sample_rate
    print(f"number of channels = {audio.shape}")

    time = audio.shape[0] / sample_rate
    print(f"Audio length = {time:.2f}s")

    length = np.arange(0, audio.shape[0]) / sample_rate
    print('\nSampling rate', sample_rate)
    
    







