# Next Word Lab — LSTM next-word prediction

**[Open the live app](https://harsh-lstm-next-word.streamlit.app/)** · Hosted on Streamlit Community Cloud.

A browser-based neural language-model demo by Harsh Gavand. Enter an English phrase and explore the next word predicted by the repository's original trained LSTM. The interface shows three ranked suggestions and their model probabilities.

## How it works

1. The saved Keras tokenizer turns known words into integer IDs. Unknown words are skipped; an entirely unknown phrase is rejected.
2. The sequence is padded or truncated on the left to the saved context length (745 tokens).
3. The original model performs CPU inference: a 10,000-entry, 50-dimensional embedding → 128-unit LSTM → 10,000-output softmax.
4. Output IDs are mapped back to words and ranked. Padding IDs without a corresponding word are excluded.

The trained artifacts and quote dataset are preserved. The UI does not retrain the model, upload input to OpenAI, or require an API key. It demonstrates a small sequence model; it is not comparable to a modern general-purpose LLM. Predictions can be repetitive or grammatically imperfect. Probabilities are model outputs, **not measured prediction accuracy**. No benchmark score is claimed.

## Run locally

Use Python 3.12:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Try `the best way to`, `life is`, or `success is`. Input is limited to 1,000 characters. The tokenizer and model are loaded once per app process using Streamlit resource caching. Inference uses bounded CPU thread counts. Resources resolve relative to the source file, so launching from another working directory is supported.

## Free Streamlit Community Cloud deployment

Choose this repository, branch `main`, main file `app.py`, and Python **3.12** in Advanced settings. No secrets are needed. Keep the model and both tokenizer metadata files in the repository. First startup includes TensorFlow installation and model loading; subsequent predictions use the loaded model.

## Tests

```sh
pip install pytest==8.4.2
python -m pytest tests -q
```

The tests load the actual trained model, validate its dimensions, exercise ranked inference, and verify that empty, oversized and entirely unknown input is rejected before inference.

## Files and contributions

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit form, ranked results and model explanation |
| `predictor.py` | Cached-resource-compatible model loading, validation and inference |
| `lstm_model (1).h5` | Original trained Keras model; loaded with `compile=False` for inference |
| `tokenizer.pkl`, `max_len.pkl` | Original vocabulary and sequence metadata |
| `qoute_dataset.csv` | Included quote dataset |
| `codefile.ipynb`, `RNNimplementation.ipynb` | Original experimentation/training notebooks |
| `tests/test_predictor.py` | Real-artifact inference and input-validation tests |

The project brings together text preprocessing, recurrent neural network training artifacts, inference and a deployed web interface. Pickle files are trusted, version-controlled artifacts; the app never accepts uploaded model or pickle files. No user text is deliberately persisted by the application. Training provenance and evaluation should be reviewed before using this as a production autocomplete service.
