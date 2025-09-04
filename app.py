import gradio as gr
import os
import tempfile

import syn_hiftnet as syn
#import syn_vgan as syn
#import syn_k_univnet_multi as syn

description_text = """
# Multilingual TTS for Sámi languages (+ Finnish and Estonian)

Welcome! This is a demonstration of a multi-lingual and multi-speaker Text-to-Speech (TTS) model.
The demo is related to research on TTS for low-resource languages, and the effect of augmenting the training data with
areally close languages.


Disclaimers:
For convenience, the demo uses pretrained HiFi-GAN vocoder which doesn't work well with male voices. 
English does not well due to small dataset and orthographic transcriptions. Use the demo just for testing, not for frequent or commercial use.

 

"""
speakers = {
    "aj(sma)": 2,
    "am(sme)": 3,
    "ms(sme)": 4,
    "ln(sme)": 5,
    "mu(smj)": 7,
    "sa(smj)": 8,
    "bi(smj": 10,
    "css(fin)": 11,
    "ti(fin)": 13,
    "ta(fin)": 14,
    "liivika(est)": 15,
    "indek(est)": 16,
    "kylli(est)": 17,
    "andreas(est)": 18,
    "peeter(est)": 19,
    "kersti(est)": 20,
    "M6670(eng)": 21,
    "M6097(eng)": 22,
    "F92(eng)": 23,
    "F9136(eng)": 24
}

mean_pitch = {
    "aj0": 130,
    "aj1": 130,
    "am": 120,
    "ms": 120,
    "ln": 120,
    "lo": 120,
    "mu": 120,
    "sa": 120,
    "kd": 120,
    "bi": 120,
    "ti": 130,
    "ta": 115,
    "liivika": 120,
    "indek": 90,
    "kylli": 140,
    "andreas": 100,
    "peeter": 80,
    "kersti": 120
}

languages = {
    "guess": -1,
    "South Sámi": 0, #South
    "North Sámi": 1, #North
    "Lule Sámi": 2, #Lule
    "Finnish": 3,
    "Estonian": 4,
    "English": 5
}

# --- NEW: Add a dictionary for default prompts per language ---
default_prompts = {
    "guess": "Sáhtta go esso-burgera luohti, Koskenkorva dahje carpool karajoiki gádjut árgabeaivveluođi?",
    
    "North Sámi": "Riektačállinreaidduid lassin Divvun-joavkkus ovdanit dál maiddái hállanteknologiijareaidduid.",
    
    "South Sámi": " Buerie aerede gaajhkesh dovnesh jïh buerie båeteme dan bæjhkoehtæmman.", #Guktie datnine?",
    "Lule Sámi": "Sáme hållamsyntiesaj baktu máhttá adnegoahtet sáme gielajt ådå aktijvuodajn.",

    "Finnish": "Joka kuuseen kurkottaa, se katajaan kapsahtaa.",
    "Estonian": "Aprilli lõpp pani aiapidajate kannatuse jälle proovile – pärast mõnepäevast sooja saabub ootamatu külmalaine.",

    "English": "This obscure language is not supported by this model."
}


public = False

tempdir = tempfile.gettempdir()

tts = syn.Synthesizer()


def speak(text, language, speaker, l_weight, s_weight, pace, postfilter):  # pitch_shift,pitch_std):

    # text frontend not implemented...
    text = text.replace("...", "…")
    #print(speakers[speaker])
    #print(language)
    use_lid = False
    if language == "guess":
        use_lid = True

    audio = tts.speak(text, output_file=f'{tempdir}/tmp', lang=languages[language],
                      spkr=speakers[speaker], l_weight=l_weight, s_weight=s_weight,
                      pace=pace, clarity=postfilter, guess_lang=use_lid)  # , mean_pitch = mean_pitch[speaker])
    """
    if not public:
        try:
            os.system("play " + tempdir + "/tmp.wav &")
        except:
            pass
    """
    return (22050, audio)

# update the text box based on language selection 
def update_text_prompt(language):
    """
    Updates the text in the textbox to the default prompt for the selected language.
    """
    prompt = default_prompts.get(language, "") # Get the prompt, or an empty string if not found
    return gr.Textbox(value=prompt)


#
with gr.Blocks() as tts_gui:
    gr.Markdown(description_text) #"## Multilingual TTS for Sámi languages (+ Finnish and Estonian)")
    with gr.Row():
        with gr.Column(scale=2):
            # Define each component and assign it to a variable
            text_input = gr.Textbox(label="Text", value=default_prompts["North Sámi"])
            language_dd = gr.Dropdown(list(languages.keys()), label="Language", value="North Sámi")
            speaker_dd = gr.Dropdown(list(speakers.keys()), label="Voice", value="ms(sme)")
            
            with gr.Row():
                l_weight_slider = gr.Slider(minimum=0.5, maximum=1.5, step=0.05, value=1, label="Language Weight")
                s_weight_slider = gr.Slider(minimum=0.5, maximum=1.5, step=0.05, value=1, label="Speaker Weight")

            pace_slider = gr.Slider(minimum=0.5, maximum=1.5, step=0.05, value=1.0, label="Speech Rate")
            postfilter_slider = gr.Slider(minimum=0., maximum=2, step=0.05, value=1.0, label="Post-processing")
            
        with gr.Column(scale=1):
            # Add a button to trigger synthesis
            speak_button = gr.Button("Speak", variant="primary")
            audio_output = gr.Audio(label="Output")

    


    language_dd.change(
        fn=update_text_prompt,
        inputs=[language_dd],
        outputs=[text_input]
    )


    speak_button.click(
        fn=speak,
        inputs=[
            text_input,
            language_dd,
            speaker_dd,
            l_weight_slider,
            s_weight_slider,
            pace_slider,
            postfilter_slider
        ],
        outputs=[audio_output]
    )


if __name__ == "__main__":
    tts_gui.launch(share=public)
