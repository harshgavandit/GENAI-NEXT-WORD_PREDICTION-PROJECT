import numpy as np
import pytest
from predictor import load_resources, predict_words


@pytest.fixture(scope="module")
def resources():
    return load_resources()


def test_saved_model_produces_ranked_vocabulary_predictions(resources):
    results = predict_words("the best way to", *resources)
    assert len(results) == 3
    assert all(word and 0 <= probability <= 1 for word, probability in results)
    assert results[0][1] >= results[1][1] >= results[2][1]
    assert len({word for word, _ in results}) == 3


@pytest.mark.parametrize("text", ["", "   ", "x" * 1001, "zzzzqqqqxxxx"], ids=["empty", "spaces", "long", "unknown"])
def test_bad_input_does_not_run_inference(resources, text):
    model, tokenizer, length = resources
    def forbidden(*args, **kwargs):
        pytest.fail("Inference must not run on invalid input")
    with pytest.raises(ValueError):
        predict_words(text, forbidden, tokenizer, length)


def test_model_loads_without_training_state_and_has_original_shape(resources):
    model, tokenizer, length = resources
    assert model.input_shape == (None, 745)
    assert model.output_shape[-1] == 10000
    assert "the" in tokenizer.word_index
