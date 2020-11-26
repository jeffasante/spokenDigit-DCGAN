import numpy as np
import scipy as scipy




# Pre-Emphasis
def normalizeSignal(signal, pre_emphasis=0.97):
    """    
    Pre-emphasis on the input signal
    :param signal: (time,)
    :param preemph:
    :return: (time,)

    """
    return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])




def saveMelSpectrograms(signal_data, numpy_save_directory, h):

    
    if not os.path.exists(numpy_save_directory):
        os.makedirs(numpy_save_directory)
        
    for i, au in enumerate(findFiles(signal_data)):
        
        
        name = au.split('\\')[-1].split('.')[0]

        print(f'\nProcessing {name} #{i+1}')
        
        sample_rate, signal = wavfile.read(au)
        
        signal = normalizeSignal(signal) 
        print('Actual signal size:', signal.size)
        
        if signal.size >= h.segment_size:
            
            max_signal_start = signal.size - h.segment_size
            signal_start = random.randint(0, max_signal_start)
            signal = signal[signal_start:signal_start+h.segment_size]
            
        else:
           # add silence
            signal = np.pad(signal, (0, abs(h.segment_size - signal.size)),
                            mode='constant')
    
       
        print('Altered signal size:', signal.size)
        
        
        melspec =  mel_spectrogram(signal, h.n_fft, h.num_mels, h.hop_size,
                         h.win_size, sample_rate=sample_rate,
                         fmin=h.fmin, fmax=h.fmax)
        
        name_dir = numpy_save_directory + name

        if i == 0:
            break
        
        np.save(name_dir, melspec.T)
        
        

def melspecToAudio(mel_spec, mel_basis, h, device=device):
    
    inverse_filter_banks = np.dot(mel_basis[str(h.fmax)+'_'+(device.type)].T,
                                  spectral_de_normalize(mel_spec))
    _, inverse_stft = scipy.signal.istft(inverse_filter_banks, fs=h.sample_rate,
                              nperseg=h.win_size,
                                      nfft=h.n_fft)
    
    return inverse_stft




def spectral_normalize(magnitudes, C=1, clip_val=1e-5):
    return  np.log(np.clip(magnitudes, a_min=clip_val, a_max=None) * C)

def spectral_de_normalize(x, C=1):
    return np.exp(x) / C




mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, hop_size,
                    win_size, sample_rate, fmin,
                   fmax, center=False):
    
     
    if np.min(y) < -1.:
        print('min value is', np.min(y))
    if np.max(y) > 1.:
        print('max value is', np.max(y))
        
    if fmax not in mel_basis:
        filter_banks = librosa_mel_fn(sample_rate, n_fft, num_mels)
        mel_basis[str(fmax)+'_'+(device.type)] = filter_banks.astype(float)
        hann_window[device.type] = np.hanning(win_size)
    
     
    y = np.pad(y, (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)) 
                   , mode='reflect')
        
    _, _, Zxx =  scipy.signal.stft(y,  fs=sample_rate, window=hann_window[device.type],
                                  nperseg=win_size, nfft=n_fft)
    
    spec = np.dot(mel_basis[str(fmax)+'_'+(device.type)], Zxx)

    spec = np.square(np.abs(spec)) + 1e-9
    
    spec = spectral_normalize(spec)
    

    return spec




class AudioFolder(Dataset):
    
    def __init__(self, data, h, base_mels_path=None,
                         transform=None):
        
        self.audio_files = data
        
        self.sample_rate = h.sample_rate
        
        self.n_fft = h.n_fft
        self.num_mels = h.num_mels
        self.hop_size = h.hop_size
        self.win_size = h.win_size
        self.fmin = h.fmin
        self.fmax = h.fmax
        self.segment_size = h.segment_size
        self.base_mels_path = base_mels_path
        
        self.transform = transform
        
    def __getitem__(self, index):
        
        filename = self.audio_files[index]
        
        sample_rate, audio = wavfile.read(filename)
        

        
        if sample_rate != self.sample_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                   sample_rate, self.sample_rate))
        
        
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        
        
        if self.transform:
            
            audio = normalizeSignal(audio) 

            if audio.size(1) >= self.segment_size:

                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start+self.segment_size]

            else:
              # add silence
                audio = np.pad(audio, (0, abs(self.segment_size - audio.size)),
                                mode='constant')

       

            melspec =  mel_spectrogram(audio.squeeze().numpy(, self.n_fft, self.num_mels, self.hop_size,
                             self.win_size, sample_rate=self.sample_rate,
                             fmin=self.fmin, fmax=self.fmax)

            melspec = torch.from_numpy(melspec).unsqueeze(0)
            
             
            
            
        else:
            
            mel = np.load(os.path.join(self.base_mels_path
                        ,os.path.splitext(os.path.split(filename)[-1])[0]  + '.npy'))

            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)


        fname = os.path.splitext(os.path.split(filename)[-1])[0]
        
        
        return mel[:,:,:80], fname

    
    def __len__(self):
        return len(self.audio_files)
        
        
            

        
        