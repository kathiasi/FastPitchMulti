import gradio as gr
#import syn_hifigan as syn
import syn_k_univnet_ural as syn
import os, tempfile
        
speakers = {
    "aj0": 0,
    "aj1": 1,
    "am": 2,
    "ms": 3,
    "ln": 4,
    "lo": 5,
    "mu": 6,
    "sa": 7,
    "kd": 8,
    "bi": 9,
    "ti": 10,
    "ta": 11,
    "liivika": 12,
    "indrek": 13,
    "kylli": 14,
    "andreas": 15,
    "peeter": 16,
    "kersti": 17
}

languages = {
    "guess": -1,
    "sma": 0,
    "sme": 1,
    "smj": 2,
    "fin": 3,
    "est": 4
    }
public=False

tempdir = tempfile.gettempdir()

tts = syn.Synthesizer()



def speak(text, language,speaker, l_weight, s_weight, pace, postfilter): #pitch_shift,pitch_std):


    
    # text frontend not implemented...
    text = text.replace("...", "…")
    print(speakers[speaker])
    print(language)
    use_lid = False
    if language == "guess":
        use_lid =True
    audio = tts.speak(text, output_file=f'{tempdir}/tmp', lang=languages[language],
                      spkr=speakers[speaker], l_weight=l_weight, s_weight=s_weight,
                      pace=pace, clarity=postfilter, guess_lang=use_lid)

    if not public:
        try:
            os.system("play "+tempdir+"/tmp.wav &")
        except:
            pass

    return (22050, audio)



controls = []
controls.append(gr.Textbox(label="text", value="Suohtas duinna deaivvadit."))
controls.append(gr.Dropdown(list(languages.keys()), label="language", value="guess"))
controls.append(gr.Dropdown(list(speakers.keys()), label="speaker", value="ms"))
controls.append(gr.Slider(minimum=0.5, maximum=1.5, step=0.05, value=1, label="language weight"))
controls.append(gr.Slider(minimum=0.5, maximum=1.5, step=0.05, value=1, label="speaker weight"))

controls.append(gr.Slider(minimum=0.5, maximum=1.5, step=0.05, value=1.0, label="speech rate"))
controls.append(gr.Slider(minimum=0., maximum=2, step=0.05, value=1.0, label="post-processing"))




tts_gui = gr.Interface(
    fn=speak,
    inputs=controls,
    outputs= gr.Audio(label="output"),
    live=False

)


if __name__ == "__main__":
    tts_gui.launch(share=public)
