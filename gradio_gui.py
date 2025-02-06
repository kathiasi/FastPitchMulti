import gradio as gr
#import syn_hifigan as syn
import syn_k_univnet_multi as syn
import os, tempfile
        
languages = {"South Sámi":0,
          "North Sámi":1,
          "Lule Sámi":2}

speakers={"aj0": 0,
          "aj1": 1,
          "am": 2,
          "bi": 3,
          "kd": 4,
          "ln": 5,
          "lo": 6,
          "ms": 7,
          "mu": 8,
          "sa": 9
}
public=True

tempdir = tempfile.gettempdir()

tts = syn.Synthesizer()



def speak(text, language,speaker,pace): #pitch_shift,pitch_std):



    # text frontend not implemented...
    text = text.replace("...", "…")
    print(speakers[speaker])
    audio = tts.speak(text, output_file=f'{tempdir}/tmp', lang=languages[language],
                      spkr=speakers[speaker],
                      pace=pace)

    if not public:
        try:
            os.system("play "+tempdir+"/tmp.wav &")
        except:
            pass

    return (22050, audio)



controls = []
controls.append(gr.Textbox(label="text", value="Suohtas duinna deaivvadit."))
controls.append(gr.Dropdown(list(languages.keys()), label="language", value="North Sámi"))
controls.append(gr.Dropdown(list(speakers.keys()), label="speaker", value="ms"))

#controls.append(gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1, label="Style strength"))
#controls.append(gr.Slider(minimum=-50.0, maximum=50, step=5, value=0, label="Pitch shift"))
#controls.append(gr.Slider(minimum=0.5, maximum=1.5, step=0.05, value=1.0, label="Pitch variance"))
controls.append(gr.Slider(minimum=0.5, maximum=1.5, step=0.05, value=1.0, label="speech rate"))




tts_gui = gr.Interface(
    fn=speak,
    inputs=controls,
    outputs= gr.Audio(label="output"),
    live=False

)


if __name__ == "__main__":
    tts_gui.launch(share=public)
