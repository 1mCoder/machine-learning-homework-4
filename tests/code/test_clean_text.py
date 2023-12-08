import re
from src.code.clean_text import clean_text

# Test cases for clean_text function
def test_clean_text_removes_special_characters():
    # Test if special characters are removed
    text = "Hello! This is a test #text with @special characters."
    cleaned_text = clean_text(text)
    assert re.search(r"[^\w\s#@']", cleaned_text) is None

# Test case for handling lowercase conversion
def test_clean_text_lowercase_conversion():
    text = "HELLO World! This Is A TeSt."
    cleaned_text = clean_text(text)
    assert cleaned_text == "hello world test"

# Test case for handling special characters and punctuation
def test_clean_text_special_characters_and_punctuation():
    text = "Hey! This, is a test #text with @special characters."
    cleaned_text = clean_text(text)
    assert re.search(r"[^\w\s#@']", cleaned_text) is None

# Test case for handling consecutive whitespaces
def test_clean_text_consecutive_whitespaces():
    text = "This    has     multiple   whitespaces"
    cleaned_text = clean_text(text)
    assert "    " not in cleaned_text
    assert "  " not in cleaned_text
