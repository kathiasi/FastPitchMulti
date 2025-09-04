import glob
import os
import json
import torch
import sys
from .env import AttrDict
from .meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from .models import Generator
from .stft import TorchSTFT
from .Utils.JDC.model import JDCNet
from scipy.io.wavfile import write

class HiFTNet:
    """A class for HiFTNet inference."""
    def __init__(self, device="cpu"):
        self.device = device

        my_dir = os.path.dirname(os.path.abspath(__file__))

        checkpoint_path = os.path.join(my_dir, "libritts")


        # Load configuration
        config_file = os.path.join(checkpoint_path, 'config.json')
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)

        # Load models
        F0_model = JDCNet(num_class=1, seq_len=192)
        self.generator = Generator(self.h, F0_model).to(self.device)
        self.stft = TorchSTFT(filter_length=self.h.gen_istft_n_fft, 
                              hop_length=self.h.gen_istft_hop_size, 
                              win_length=self.h.gen_istft_n_fft).to(self.device)
        
        # Load checkpoint

        state_dict_g = self._load_checkpoint(checkpoint_path+"/g_00650000", self.device)
        self.generator.load_state_dict(state_dict_g['generator'])
        
        # Set to evaluation mode
        self.generator.remove_weight_norm()
        self.generator.eval()


    def _load_checkpoint(self, filepath, device):
        """Loads a checkpoint file."""
        assert os.path.isfile(filepath)
        print(f"Loading '{filepath}'")
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

    def _get_mel(self, x):
        """Computes a mel-spectrogram from a raw waveform."""
        return mel_spectrogram(x, self.h.n_fft, self.h.num_mels, self.h.sampling_rate, 
                               self.h.hop_size, self.h.win_size, self.h.fmin, self.h.fmax)

    def _infer_waveform(self, mel):
        """Private helper to run inference from a mel-spectrogram."""
        with torch.no_grad():
            # Run inference
            spec, phase = self.generator(mel)
            y_g_hat = self.stft.inverse(spec, phase)
            return y_g_hat
        
            audio = y_g_hat.squeeze()
            
            # Post-processing
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            
        return audio

    def analysis_synthesis(self, wav_path):
        """
        Synthesizes audio from a WAV file path.

        Args:
            wav_path (str): Path to the input WAV file.
        
        Returns:
            numpy.ndarray: The synthesized audio waveform as a 16-bit integer array.
        """
        # Load and pre-process audio
        wav, sr = load_wav(wav_path)
        print(f"Processing audio file: {wav_path}")
        wav_tensor = torch.FloatTensor(wav / MAX_WAV_VALUE).to(self.device)

        # Get mel-spectrogram
        mel_tensor = self._get_mel(wav_tensor.unsqueeze(0))
        print(mel_tensor.shape)
        # Synthesize and return audio
        return self._infer_waveform(mel_tensor)

    def synthesize_from_mel(self, mel_tensor):
        """
        Synthesizes audio from a pre-computed mel-spectrogram.

        Args:
            mel_tensor (torch.FloatTensor): A mel-spectrogram tensor of shape 
                                            [batch_size, num_mels, num_frames].
                                            Typically batch_size is 1.
        
        Returns:
            numpy.ndarray: The synthesized audio waveform as a 16-bit integer array.
        """
        print("Synthesizing from mel-spectrogram...")
        # Ensure tensor is on the correct device
        mel_tensor = mel_tensor.to(self.device)

        # Handle 2D input [num_mels, num_frames] by adding a batch dimension
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)

        # Synthesize and return audio
        return self._infer_waveform(mel_tensor)


if __name__ == '__main__':
    # Instantiate the vocoder. It loads the model automatically.
    vocoder = HiFTNet()

    # Get the input file path from the command line
    input_wav_path = sys.argv[1]

    # Synthesize the audio from the file
    audio_out = vocoder.analysis_synthesis(input_wav_path)

    # Define the output path
    output_wav_path = "/tmp/tmp_hift.wav"

    # Save the synthesized audio
    write(output_wav_path, vocoder.h.sampling_rate, audio_out)
    
    # Play the synthesized audio
    os.system(f"play -q {output_wav_path}")
