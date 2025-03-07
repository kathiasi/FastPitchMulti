import fasttext

model = fasttext.load_model("lang_id_model.bin")
model.quantize(input="lid_train.txt", retrain=True, cutoff=100000)  # Reduce unused vectors
model.save_model("lang_id_model_q.bin")
