import argparse

import models
import time
import sys
import warnings
#from pathlib import Path

from nemo.collections.tts.models import UnivNetModel

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence
#import style_controller
from common.utils import load_wav_to_torch


from common import utils, layers

from common.text.text_processing import TextProcessing


import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""
device = "cuda:0"
#device = "cpu"
vocoder = "univnet"
vocoder1 = "hifigan"

from hifigan.data_function import MAX_WAV_VALUE, mel_spectrogram
from hifigan.models import Denoiser
import json
from scipy import ndimage

import os

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=False,
                        help='Full path to the input text (phareses separated by newlines)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--save-mels', action='store_true', help='')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')

    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')

    #parser.add_argument('--fastpitch', type=str, default='output_smj_sander/FastPitch_checkpoint_660.pt',
                        #help='Full path to the generator checkpoint file (skip to use ground truth mels)') #########

    parser.add_argument('--fastpitch', type=str, default='output_aj/FastPitch_checkpoint_830.pt',
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)') #########    

    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float,
                        help='WaveGlow denoising')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--amp', action='store_true',default=False,
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=1)
    
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    parser.add_argument('--p-arpabet', type=float, default=0.0, help='') ################

  
    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--text-cleaners', nargs='*',
                                 default=['basic_cleaners'], type=str,
                                 help='Type of text cleaners for input text')
    text_processing.add_argument('--symbol-set', type=str, default='sma_expanded', #################
                                 help='Define symbol set for input text')

    cond = parser.add_argument_group('conditioning on additional attributes')

    cond.add_argument('--n-speakers', type=int, default=2,
                      help='Number of speakers in the model.')

    return parser



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_model_from_ckpt(checkpoint_path, ema, model):

    
    checkpoint_data = torch.load(checkpoint_path,map_location = device)
    status = ''

    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')

        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k,v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')

    return model
    
def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):

    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    setattr(model_args, "energy_conditioning",True)
    model_config = models.get_model_config(model_name, model_args)
    # print(model_config)
    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)
    
    if checkpoint is not None:
        model = load_model_from_ckpt(checkpoint, ema, model)

    amp = False
    if amp:
        model.half()
    model.eval()
 
    return model.to(device)

class Synthesizer:

    def _load_pyt_or_ts_model(self, model_name, ckpt_path, format = 'pyt'):
        if format == 'ts':
          
            model = models.load_and_setup_ts_model(model_name, ckpt_path,
                                                   False, device)
            model_train_setup = {}
            return model, model_train_setup
          
        is_ts_based_infer = False
        model, _, model_train_setup = models.load_and_setup_model(
            model_name, self.parser, ckpt_path, False, device,
            unk_args=self.unk_args, forward_is_infer=True, jitable=is_ts_based_infer)

        if is_ts_based_infer:
            model = torch.jit.script(model)
        return model, model_train_setup



    def __init__(self): 
        parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
        self.parser = parse_args(parser)
        
        self.args, self.unk_args = self.parser.parse_known_args()
        self.generator = load_and_setup_model(
            'FastPitch', parser, self.args.fastpitch, self.args.amp, device,
            unk_args=self.unk_args, forward_is_infer=True, ema=self.args.ema,
            jitable=False)
        

        self.hifigan_model = "pretrained_models/hifigan/hifigan_gen_checkpoint_10000_ft.pt" # Better with Sander!
        #self.hifigan_model = "pretrained_models/hifigan/hifigan_gen_checkpoint_6500.pt"
        self.vocoder = UnivNetModel.from_pretrained(model_name="tts_en_libritts_univnet")
        self.vocoder1, voc_train_setup= self._load_pyt_or_ts_model('HiFi-GAN', self.hifigan_model)
        self.denoiser = Denoiser(self.vocoder1,device=device) #, win_length=self.args.win_length).to(device)
        self.tp = TextProcessing(self.args.symbol_set, self.args.text_cleaners, p_arpabet=0.0)
        
        
    # def speak(self, text): #, style_embedding, surprisals, output_file="/tmp/tmp", speaker_i = 0):
    def speak(self, text, output_file="/tmp/tmp"):

        # text = self.tp.encode_text(text)
        text = [9]+self.tp.encode_text(text)+[9]
        text = torch.LongTensor([text]).to(device)
        #probs = surprisals
        for p in [0]:
            #probs = np.array(surprisals)
           
            #probs = torch.Tensor([probs]).to(device)
                      
            #embedding = torch.from_numpy(style_embedding).to(device).float()
        
            with torch.no_grad():
               
                mel, mel_lens, *_ = self.generator(text, pace=0.95, max_duration=15, speaker=1) #, ref_vector=embedding, speaker=speaker_i) #, **gen_kw, speaker 0 = bad audio, speaker 1 = better audio   
            
            mel_np = mel.float().data.cpu().numpy()[0]
            blurred_f = ndimage.gaussian_filter(mel_np, 1.0) #3
            alpha = 0.2 #0.3 ta
            mel_np = mel_np + alpha * (mel_np - blurred_f)
            blurred_f = ndimage.gaussian_filter(mel_np, 3.0) #3
            alpha = 0.1 # 0.1 ta
            sharpened = mel_np + alpha * (mel_np - blurred_f)
            
            for i in range(0,80):
                sharpened[i, :]+=(i-40)*0.01 #0.01 ta
            mel[0] = torch.from_numpy(sharpened).float().to(device)
            
            
            with torch.no_grad():
                y_g_hat = self.vocoder(spec=mel).float()
                #y_g_hat = self.vocoder1(mel).float() ###########
                y_g_hat = self.denoiser(y_g_hat.squeeze(1), strength=0.01) #[:, 0]
                audio = y_g_hat.squeeze()* 32768.0
                audio = audio.cpu().numpy().astype('int16')
                   
                    
                write(output_file+".wav", 22050, audio)
            
            os.system("play -q "+output_file+".wav &")
            return audio
    

if __name__ == '__main__':
    syn = Synthesizer()
    hifigan = syn.hifigan_model
    hifigan_n = hifigan.replace(".pt", "")
    fastpitch = syn.args.fastpitch
    fastpitch_n = fastpitch.replace(".pt", "")
    print(hifigan_n + " " + fastpitch_n)

    hifigan_n_short = hifigan_n.split("/")
    hifigan_n_shorter = hifigan_n_short[2].split("_")
    hifigan_n_shortest = hifigan_n_shorter[3]

    fastpitch_n_short = fastpitch_n.split("/")
    fastpitch_n_shorter = fastpitch_n_short[1].split("_")
    fastpitch_n_shortest = fastpitch_n_shorter[2]
        
    #syn.speak("Gå lij riek mælggadav vádtsám, de bådij vijmak tjáppa vuobmáj.")
    i = 0
    
    while (1==1):
        
        text = input(">")
        text1 = text.split(" ")
        syn.speak(text, output_file="inf_output_exp_sma/"+str(i)+"_"+text1[0]+"_FP_"+fastpitch_n_shortest+"univnet")
        i += 1

        
