import re
import math
from collections import Counter

# Regular expression pattern for matching words
WORD_REGEX = re.compile(r'\w+')


def calculate_cosine_similarity(vector1, vector2):
    """
    Calculate the cosine similarity between two vectors.

    :param vector1: A dictionary representing the word frequencies in the first text
    :param vector2: A dictionary representing the word frequencies in the second text
    :return: The cosine similarity between the two input vectors
    """
    common_words = set(vector1.keys()) & set(vector2.keys())
    matched_word_weights = {}

    # Calculate matched word weights
    for word in common_words:
        if vector1[word] > vector2[word]:
            matched_word_weights[word] = vector2[word]
        else:
            matched_word_weights[word] = vector1[word]

    # Calculate the numerator of the cosine similarity formula
    numerator = sum([vector1[word] * matched_word_weights[word]
                     for word in common_words])

    # Calculate the sum of squares of vector1's word frequencies
    sum_of_squares1 = sum([vector1[word] ** 2 for word in vector1.keys()])

    # Calculate the sum of squares of matched word weights
    sum_of_squares2 = sum([matched_word_weights[word] ** 2
                          for word in matched_word_weights.keys()])

    # Calculate the denominator of the cosine similarity formula
    denominator = math.sqrt(sum_of_squares1) * math.sqrt(sum_of_squares2)

    return float(numerator) / denominator if denominator != 0 else 0.0


def text_to_vector(text):
    """
    Convert a text into a word frequency vector.

    :param text: The input text
    :return: A Counter dictionary representing word frequencies in the text
    """
    words = WORD_REGEX.findall(text)
    return Counter(words)


def cosine_similarity(text1, text2):
    """
    Calculate the cosine similarity between two texts.

    :param text1: The first input text
    :param text2: The second input text
    :return: The cosine similarity between the two input texts
    """
    normalized_text1 = text1.lower()
    normalized_text2 = text2.lower()

    vector1 = text_to_vector(normalized_text1)
    vector2 = text_to_vector(normalized_text2)

    similarity = calculate_cosine_similarity(vector1, vector2)
    return similarity
