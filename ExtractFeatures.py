
""" Chroma and Spectrogram Representations of given reconstructed and generated sounds"""
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
FRAME_SIZE = 2048
HOP_SIZE = 1024
SAMPLE_RATE = 22050

x_1, fs = librosa.load(r"C:\Users\dwill\Desktop\drum\original\18.wav")
plt.figure(figsize=(16, 4))
librosa.display.waveplot(x_1, sr=fs)
plt.title('Reconstructed Frequency Response over Time')
plt.tight_layout()
x_2, fs = librosa.load(r"C:\Users\dwill\Desktop\drum\generated\18.wav")
plt.figure(figsize=(16, 4))
librosa.display.waveplot(x_2, sr=fs)
plt.title('Generated Frequency Response over Time')
plt.tight_layout()

x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=SAMPLE_RATE, tuning=0, norm=2, hop_length=HOP_SIZE, n_fft=FRAME_SIZE)
x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=SAMPLE_RATE, tuning=0, norm=2, hop_length=HOP_SIZE, n_fft=FRAME_SIZE)

plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.title('Chroma Representation of Reconstucted Sound')
librosa.display.specshow(x_1_chroma, x_axis='time',
                         y_axis='chroma', cmap='gray_r', hop_length=HOP_SIZE)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.title('Chroma Representation of Generated Sound')
librosa.display.specshow(x_2_chroma, x_axis='time',
                         y_axis='chroma', cmap='gray_r', hop_length=HOP_SIZE)
plt.colorbar()
plt.tight_layout()
plt.savefig("DTWCompare.png")
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
x1 = librosa.amplitude_to_db(np.abs(librosa.stft(x_1, hop_length=HOP_SIZE)),
                            ref=np.max)
x2 = librosa.amplitude_to_db(np.abs(librosa.stft(x_2, hop_length=HOP_SIZE)),
                            ref=np.max)
img = librosa.display.specshow(x1, y_axis='log', sr=SAMPLE_RATE, hop_length=HOP_SIZE,
                         x_axis='time', ax=ax[0])
img = librosa.display.specshow(x2, y_axis='log', sr=SAMPLE_RATE, hop_length=HOP_SIZE,
                         x_axis='time', ax=ax[1])
ax[0].set(title='Reconstructed Log-frequency power spectrogram')
ax[0].label_outer()
ax[1].set(title='Generated Log-frequency power spectrogram')
ax[1].label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig("SpectrogramCompare.png")
plt.show()

