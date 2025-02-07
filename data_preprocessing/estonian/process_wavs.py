import os, glob
import locale
from tqdm import tqdm
import pitch_squeezer as ps
import numpy as np
import torch
# Set locale for Estonian sorting


def trim(in_fname, out_fname,threshold=20, sil_len=0.4):
    import librosa
    import soundfile as sf
    y, sr = librosa.load(in_fname, sr=22050) #, sr=sr)
    duration = len(y) / sr

    # Trim silence
    y_trimmed, index = librosa.effects.trim(y, top_db=threshold)  # Adjust `top_db` as needed

    min_silence_samples = int(sil_len * sr)

    # Ensure at least 0.5s silence remains if available
    index[0] = max(0, index[0] - min_silence_samples)
    index[1] = min(len(y), index[1] + min_silence_samples)

    # Extract the desired portion
    y_result = y[index[0]:index[1]]
    duration = len(y_result) / sr
    # do not use prompts less than second or longer than 20 seconds
    if duration < 0.8 or duration > 20:
        print(in_fname, duration)
        #os.system(f'play {in_fname}')
        return False
    #scale signal
    y_result = y_result / np.max(np.abs(y_result))

    sf.write(out_fname, y_result, sr)
    #os.system("play /tmp/tmp.wav &")
    return True


    
    
spkr_dirs = [d for d in os.listdir() if os.path.isdir(d)]

tmpfile = "/tmp/tmp.wav"
data_dir = "est_data"
meta_dir = "DATA/"+data_dir+"/"


lang = "est"

for input_dir in spkr_dirs:
    if input_dir == data_dir:
        continue
    print(input_dir, data_dir)
    spkr = input_dir
    output_dir = os.path.join(data_dir, spkr)

    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    max_per_speaker = 2000
    prompt_files = False
    i = 0

    
    for filename in tqdm(sorted(os.listdir(input_dir))):
        input_path = os.path.join(input_dir, filename)
        if i > max_per_speaker:
            break

        if filename.endswith(".wav"):
            root = filename[:-4]
            output_path = os.path.join(output_dir, filename)

            if not os.path.isfile(output_path):
                if not trim(input_path, tmpfile):
                    continue
                try:

                    f0, if0 = ps.track_pitch(tmpfile, min_hz=50, max_hz=500, frame_rate=86.1326) #, plot=True)
                except:
                    continue
            
                output_pitch_path = os.path.join(output_dir, root+".pt")
                torch.save(torch.from_numpy(f0).unsqueeze(0), output_pitch_path)
                sox_command = f'sox {tmpfile} -r 22050 -b 16 -c 1 {output_path}'
                os.system(sox_command)

            txtfile = os.path.join(input_dir,root+".txt")
            if os.path.isfile(txtfile):
                prompt_files = True
                meta_root = meta_dir+spkr+"/"+root
                txtfile = os.path.join(input_dir,root+".txt")
                prompt = open(txtfile, "r").read().strip()
                metadata.append(meta_root+".wav|"+meta_root+".pt|"+prompt+"|"+spkr+"|"+lang)


            i+=1
            
    print(f'writing {spkr} metadata..')
    with open(f'{data_dir}/metadata.{spkr}.txt', 'w', encoding='utf-8') as f:
        if prompt_files:
            print("prompt files")
            for line in metadata:
                f.write(line + "\n")

        else:
            print("reading korp.sisukord")
            try:
                prompts = open(f'{input_dir}/korp.sisukord','r').readlines()
            except:
                print(spkr+" prompts not found")
                
            for l in prompts:
                root, text = l.strip().split("|")
                if root.endswith('.wav'):
                    root = root[:-4]

                meta_root = meta_dir+spkr+"/"+root
                f.write(meta_root+".wav|"+meta_root+".pt|"+text+"|"+spkr+"|"+lang+"\n")
