# Preprocessing of the Sámi datasets

## The Sámi dataset 

The Sámi dataset consists of recordings in three different Sámi language: North, Lule and South, abbreviated as sme, smj and sma, respectively. We have altogether 8 different speakers in the dataset: 3 for North Sámi, 4 for Lule Sámi and 1 for South Sámi.

For all speakers and subsets, we have processed raw recordings as follows:
* Checking that the texts match the audio recordings as accurately as possible and editing the text if it diverged from the audio. Normalizing (manually) all numbers, abbreviations and acronyms to match the audio.
* Force-aligning the texts and audio using the WebMAUS web service at LMU München: https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface/Pipeline. We use Pipeline without ASR -> Pipeline name: G2P->MAUS->Subtitle; Language: Finnish or Estonian works well for Sámi, at least for finding the correct sentence/utterance boundaries.
* After getting the automatically generated TextGrids from WebMAUS, we manually go through each sentence boundary (the TRN tier in Praat TextGrid) to fix any possible errors in segmentation/interval boundaries. We make sure that the sentence boundaries are not too tight and that they don't start or stop too abruptly or that the creaky-voiced ends are not cut off. For this, we always use the spectrogram and pitch contour in Praat to check it as well as the audio itself to decide on good boundaries.
* 

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
