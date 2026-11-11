import pytest
from src.tokenizer import CharTokenizer


def test_vocab_size():
    sample_text = "hello world!"
    tokenizer = CharTokenizer(sample_text)

    # Number of unique characters in "hello world!"
    expected_unique = len(set(sample_text))
    assert tokenizer.vocab_size == expected_unique
    assert len(tokenizer.stoi) == expected_unique
    assert len(tokenizer.itos) == expected_unique


def test_encode_decode_roundtrip():
    sample_text = "abcdefghijklmnopqrstuvwxyz 1234567890 !?"
    tokenizer = CharTokenizer(sample_text)

    test_string = "hello world 123"
    encoded = tokenizer.encode(test_string)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert all(isinstance(t, int) for t in encoded)
    assert decoded == test_string


def test_token_indices_in_bounds():
    sample_text = "Transformer architecture in PyTorch"
    tokenizer = CharTokenizer(sample_text)

    test_string = "PyTorch Transformer"
    encoded = tokenizer.encode(test_string)

    # Every token ID must fall between 0 and vocab_size - 1
    for token_id in encoded:
        assert 0 <= token_id < tokenizer.vocab_size


def test_unknown_character_raises_error():
    training_text = "abcdefg"
    tokenizer = CharTokenizer(training_text)

    # 'z' is not in training text, should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        tokenizer.encode("abcz")

    assert "Unknown characters" in str(excinfo.value)


if __name__ == "__main__":
    test_vocab_size()
    test_encode_decode_roundtrip()
    test_token_indices_in_bounds()
    test_unknown_character_raises_error()
    print("All tokenizer tests passed successfully!")