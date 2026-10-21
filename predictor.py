"""Inference over the repository's original trained LSTM and tokenizer."""
import os
from pathlib import Path
import pickle

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parent


def load_resources():
    model = load_model(ROOT / "lstm_model (1).h5", compile=False)
    # Only load these version-controlled artifacts; never accept uploaded pickle files.
    with (ROOT / "tokenizer.pkl").open("rb") as handle:
        tokenizer = pickle.load(handle)
    with (ROOT / "max_len.pkl").open("rb") as handle:
        max_len = pickle.load(handle)
    return model, tokenizer, max_len


def predict_words(text, model, tokenizer, max_len, count=3):
    text = text.strip()
    if not text or len(text) > 1000:
        raise ValueError("Enter a phrase between 1 and 1,000 characters.")
    tokens = tokenizer.texts_to_sequences([text])[0]
    if not tokens:
        raise ValueError("None of these words are in the model's vocabulary. Try a common English phrase.")
    # The model's saved input shape is authoritative: the historical metadata
    # and original app used different off-by-one conventions.
    context_length = model.input_shape[1] or max_len - 1
    sequence = pad_sequences([tokens], maxlen=context_length, padding="pre", truncating="pre")
    probabilities = np.asarray(model(sequence, training=False))[0]
    if not np.isfinite(probabilities).all():
        raise ValueError("The model could not produce a valid prediction. Please try another phrase.")
    reverse = tokenizer.index_word
    candidates = np.argsort(-probabilities, kind="stable")
    return [(reverse[int(i)], float(probabilities[i])) for i in candidates if int(i) in reverse][:count]
