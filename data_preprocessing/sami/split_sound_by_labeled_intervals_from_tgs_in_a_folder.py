# Katri Hiovain-Asikainen (Divvun) & Sebastién le Maguer (TCD IE), May 2021
# Input: a textgrid with 1 tier named ORT-MAU, word level segmented & corresponding/identically 
# named .wav
# output: split .wav and .TextGrid files according to the timestamps of a TextGrid sentence tier, 
# also the content of the sentence Tier saved as an individual .txt

# sentence = concatenation of words (joined with a space) with a fullstop appended at the end

# ------------------------------------------------------------
import os
import tgt
import librosa
import soundfile as sf

# Fill in
output_dir = "/home/hiovain/DATASETS/MaijuSaijetsProcess/splits"
os.makedirs(output_dir, exist_ok=True)

# 0.1 loop through all textgrids in a folder
tgt_dir = "/home/hiovain/DATASETS/MaijuSaijetsProcess"
file_list = os.listdir(tgt_dir)

# 0.2 load all .wav files in a folder, corresponding to the tgt:s

wav_dir = "/home/hiovain/DATASETS/MaijuSaijetsProcess"

for tg in file_list:
# for each path stored in the filelist; set the variable tg to the current path
    if not tg.endswith(".TextGrid"):
        continue

    # 1. load the textgrid
    the_tgt = tgt.io.read_textgrid(os.path.join(tgt_dir, tg), encoding="utf-8")

    # 2. Load the .wav file

    basename = os.path.basename(tg)
    basename = basename.replace(".TextGrid", ".wav")
    wav_data, sr = librosa.load(os.path.join(wav_dir, basename), sr=None)

    # 2. load the sentence tier
    tier_sents = the_tgt.get_tier_by_name("TRN")

    # create list of sentences (intervals) from sent_tier
    list_sentences = []
    start_sent = 0

    # Create a list to store the filenames and corresponding texts
    output_table = []

    # for all the sentences:
    for id_sent, sent_inter in enumerate(tier_sents.intervals):
        # Building filenames
        basename_wav = basename.replace(".wav", f"_{id_sent:03d}.wav") # XXX.wav => XXX_001.wav
        basename_tg = basename.replace(".wav", f"_{id_sent:03d}.TextGrid") # XXX.wav => XXX_001.TextGrid
        basename_txt = basename.replace(".wav", f"_{id_sent:03d}.txt") # XXX.wav => XXX_001.txt

        # Retrieve interval elements
        start_s = sent_inter.start_time
        end_s = sent_inter.end_time
        text = sent_inter.text

        # Ignore empty intervals
        if text == "" or text == "<p:>":
            continue

        # Convert timestamps in sample based timestamps
        start = int(start_s * sr)
        end = int(end_s * sr)
        sub_wav = wav_data[start:end]
        sf.write(os.path.join(output_dir, basename_wav), sub_wav, sr)

        # Extract the textgrid part of the sentence
        tgt_part = tgt.TextGrid()
        for tr in the_tgt:
            # Interval part
            intr_part = tr.get_annotations_between_timepoints(start_s, end_s)
            start = 0
            for intr in intr_part:
                dur = intr.end_time - intr.start_time
                intr.start_time = start
                intr.end_time = start + dur
                start = intr.end_time
            """
            # Tier part
            tier_part = tgt.IntervalTier(name=tr.name, start_time=0, end_time=end_s-start_s, objects=intr_part)
            tgt_part.add_tier(tier_part)

            tgt.write_to_file(tgt_part, os.path.join(output_dir, basename_tg), format="long", encoding="utf-8")
            """
            # Extract the text
            with open(os.path.join(output_dir, basename_txt), "w") as f_out:
                f_out.write(text)
        
        # Append the filename and text to the output table
        output_table.append((basename_wav, text))

    # Save the output table to a .txt file
    with open(os.path.join(output_dir, "output_table.txt"), "w") as f_out:
        for filename, text in output_table:
            f_out.write(f"{filename}|{text}\n")

    print(f"Processed all TextGrid files in {tgt_dir}")



