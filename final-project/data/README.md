# About Dataset
https://www.kaggle.com/datasets/vishvapatel09/url-detection-dataset?resource=download

This dataset contains 822,010 URLs along with extracted features that can be used for phishing detection. The dataset includes various lexical and structural attributes of URLs, such as length, number of special characters, presence of HTTPS, subdomains, and whether the URL contains suspicious words. It is designed for researchers and practitioners in cybersecurity, machine learning, and fraud detection.

## Columns:
url: The full URL.
url_length: The length of the URL.
num_digits: Number of numeric characters in the URL.
digit_ratio: Ratio of digits in the URL.
special_char_ratio: Ratio of special characters in the URL.
num_hyphens: Number of hyphens (-) in the URL.
num_underscores: Number of underscores (_) in the URL.
num_slashes: Number of forward slashes (/) in the URL.
num_dots: Number of dots (.) in the URL.
num_question_marks: Number of question marks (?).
num_equals: Number of equal signs (=).
num_at_symbols: Number of at (@) symbols.
num_percent: Number of percentage (%) symbols.
num_hashes: Number of hash (#) symbols.
num_ampersands: Number of ampersands (&).
num_subdomains: Number of subdomains in the URL.
is_https: Whether the URL uses HTTPS (1 for Yes, 0 for No).
has_suspicious_word: Whether the URL contains known phishing-related words (True/False).
status: Label indicating whether the URL is phishing (1) or legitimate (0).

## Usage:
Cybersecurity research on phishing detection.
Machine learning model training for URL classification.
Feature engineering and analysis of URL structures