# Preprocessing of the Sámi datasets

## The Sámi dataset 

The Sámi dataset consists of recordings in three different Sámi language: North, Lule and South, abbreviated as sme, smj and sma, respectively. We have altogether 8 different speakers in the dataset: 3 for North Sámi, 4 for Lule Sámi and 2 for South Sámi.

### Language-independent processing pipeline

In this section, we describe the pipeline for file preprocessing that we have done for all Sámi datasets. Some datasets might have been additionally processed with more steps (depending on the initial quality of the data) which we describe in language-dependent subsections below.

For all speakers and subsets, we have processed raw recordings as follows:
* Converting all audio files to .wav and text files to .txt if needed.
* Checking that the texts match the audio recordings as accurately as possible and editing the text if it diverged from the audio. Normalizing (manually) all numbers, abbreviations and acronyms to match the audio.
* Force-aligning the texts and audio using the WebMAUS web service at LMU München: https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface/Pipeline. We use Pipeline without ASR -> Pipeline name: G2P->MAUS->Subtitle; Language: Finnish or Estonian works well for Sámi, at least for finding the correct sentence/utterance boundaries.
* After getting the automatically generated TextGrids from WebMAUS, we manually go through each sentence boundary (the TRN tier in Praat TextGrid) to fix any possible errors in segmentation/interval boundaries. We make sure that the sentence boundaries are not too tight and that they don't start or stop too abruptly or that the creaky-voiced ends are not cut off. For this, we always use the spectrogram and pitch contour in Praat to check it as well as the audio itself to decide on suitable boundaries.
* After checking all TextGrids for the sentence boundaries, we run a python script (split_sound_by_labeled_intervals_from_tgs_in_a_folder.py) to save all labeled and checked intervals into sentence-long individual files (.wav and .txt containing the individual text transcript).
* Then, we run a python script that collects all sound files and corresponding text transcripts in a single table (make_table_wavs_txts.py) 

* In FastPitch, all files need to be converted from 44.1 khz .wav to 22 khz. For this, we have used command line tool sox or SoundConverter Linux software.

* PitchSqueezer: https://asuni.github.io/PitchSqueezer/doc/pitch_squeezer.html was used to extract pitch from each individual speaker. For male speaker we used a range of 60-350 Hz and for female speakers 120-500 Hz. 

### North Sámi




Speaker Mapping:
aj0: 0
aj1: 1
am: 2
bi: 3
kd: 4
ln: 5
lo: 6
ms: 7
mu: 8
sa: 9
