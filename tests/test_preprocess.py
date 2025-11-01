import pandas as pd
from src.preprocess import extract_features


def test_extract_features_basic():
    df = pd.DataFrame({
        "text": [
            "Hello world! Visit http://example.com",
            "FREE FREE $$$ Buy now!!!",
            "Normal message with numbers 12345",
        ],
        "label": [0, 1, 0],
    })
    X, wv, cv = extract_features(df, text_col="text", fit=True)
    # expect non-empty features
    assert X.shape[0] == 3
    assert X.shape[1] > 0
    # vectorizers should be fitted
    assert hasattr(wv, "vocabulary_")
    assert hasattr(cv, "vocabulary_")
