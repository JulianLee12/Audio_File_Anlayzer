import streamlit as st
import numpy as np
import os
import librosa
import model
import time
from model import isSimilar

st.title("Upload two audio files")

audio1 = st.file_uploader("Upload audio first file",type = ["wav","mp3"])
audio2 = st.file_uploader("Upload audio second file",type = ["wav","mp3"])
progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()

st.button("Rerun")
save = st.button("Save Audio File")

os.makedirs("audio_data",exist_ok=True)

if save:
    if audio1 and audio2:
        a1, sr1 = librosa.load(audio1,sr=22050)
        a2, sr2 = librosa.load(audio2,sr=22050)

        min_len = (  min(len(a1),len(a2)))

        a1 = a1[:min_len]
        a2 = a2[:min_len]

        np.save("audio_data/a1.npy",a1)
        np.save("audio_data/a2.npy",a2)
        st.success("Audio file saved")


mse_threshold = st.number_input("MIN MSE",value = 0.0005,step= 0.0001,format="%.4f")
cos_threshold = st.number_input("Cos Threshold",value = 0.98)
similarity = st.button("Check Similarity")
if similarity:
    outputs = model.processModel()
    st.write(model.isSimilar(outputs[1],outputs[0],mse_threshold,cos_threshold))
