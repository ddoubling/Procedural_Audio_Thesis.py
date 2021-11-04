import librosa

from train_preprocess import MinMaxNormaliser


class SoundGenerator:
    """Used to generate audio from pregenerated spectrograms"""

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self.min_max_normaliser = MinMaxNormaliser(0,1)

    def generate(self, spectrograms, min_max_values):
        """Calls the model prediction and generation of audio"""
        generated_spectrograms, latent_representations =  self.vae.reconstruct(spectrograms)
        signals = self.spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def spectrograms_to_audio(self, spectrograms, min_max_values):
        """Inverse Short Time Fourier Transformation of spectrograms and matching minmax values to audio signal"""
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            log_spec = spectrogram[:, :, 0] #dropping greyscsale channel
            denorm_log_spec = self.min_max_normaliser.denormalise(log_spec, min_max_value["min"], min_max_value["max"])
            spec = librosa.db_to_amplitude(denorm_log_spec)
            signal = librosa.griffinlim(spec, hop_length=self.hop_length)# apply Griffin-Lim
            signals.append(signal)# append signal to "signals"
        return signals