import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.metrics.pairwise import cosine_similarity

def processModel():
    audio1 = np.load("audio_data/a1.npy")
    audio2 = np.load("audio_data/a2.npy")

    mse = mean_squared_error(audio1, audio2)

    audio1_reshape = audio1.reshape(1,-1)
    audio2_reshape = audio2.reshape(1,-1)

    cos_sim = cosine_similarity(audio1_reshape, audio2_reshape)[0][0]


    return cos_sim, mse







def isSimilar(mse,cos_sim,mse_threshold,cos_threshold):
    if mse < mse_threshold and cos_sim > cos_threshold:
        return "Similar"
    else:
        return "Not Similar"


