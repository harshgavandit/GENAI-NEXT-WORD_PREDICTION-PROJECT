"""Streamlit interface for the original quote-trained LSTM."""
import streamlit as st

st.set_page_config(page_title="Next Word Lab · Harsh Gavand", page_icon="🧠", layout="centered")
from predictor import load_resources, predict_words


@st.cache_resource
def resources():
    return load_resources()


st.caption("HARSH GAVAND / NEURAL LANGUAGE MODELLING")
st.title("One phrase. What comes next?")
st.write("Explore how a trained LSTM predicts the next word in a sentence — running directly on the server, without an external AI API.")
with st.expander("Inside the model"):
    st.write("The original model uses a 50-dimensional embedding, a 128-unit LSTM and a 10,000-output softmax layer. It uses the repository's quote dataset, saved tokenizer and fixed sequence length.")
    st.caption("This is a small educational language model. Suggestions can be repetitive or grammatically imperfect. Displayed probabilities are the model's distribution, not measured accuracy or confidence in factual correctness. Unknown words are skipped by the original tokenizer.")

with st.spinner("Loading the trained model…"):
    try:
        model, tokenizer, max_len = resources()
    except Exception:
        st.error("The saved model could not be loaded. Please contact the app owner.")
        st.stop()
with st.form("prediction"):
    text = st.text_input("Your opening words", placeholder="the best way to", max_chars=1000)
    submitted = st.form_submit_button("Predict next word", use_container_width=True, type="primary")
st.caption("Try: “the best way to”, “life is”, or “success is”.")
if submitted:
    try:
        results = predict_words(text, model, tokenizer, max_len)
        if not results:
            st.warning("No word was found. Try another phrase.")
        else:
            st.subheader("The model's next word")
            st.success(results[0][0])
            st.caption("Top three suggestions · model probability")
            for word, probability in results:
                st.write(f"**{word}** · {probability:.2%}")
                st.progress(min(1.0, max(0.0, probability)))
    except ValueError as error:
        st.warning(str(error))
st.divider()
st.caption("Built by Harsh Gavand · TensorFlow / Keras · LSTM · Streamlit")
