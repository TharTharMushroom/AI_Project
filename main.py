from trigram_lm import *
from typing import List
import numpy as np
import random
from collections import Counter

TEMPERATURE = 0.8

BEGIN_SYMBOL = "<S>"
END_SYMBOL = "</S>"


def read_wikitext(path: str) -> List[List[str]]:
    """
    Reads a Wikitext file at the given path.
    :param path: The string path of the file to read
    :return: A nested List[List[str]]: The first List is of lines, and the second List is of string words on that line
    """
    print("Started reading from file " + path)
    f = open(path, encoding='utf-8')
    lines = []
    for line in f:
        # If it's a non-empty line
        if len(line.strip()) > 0:
            this_line = [BEGIN_SYMBOL, BEGIN_SYMBOL]
            split_line = line.split(" ")
            for word in split_line:
                if len(word.strip()) > 0:
                    this_line.append(word.strip())
            this_line.append(END_SYMBOL)
            lines.append(this_line)
    print("Read %i lines" % len(lines))
    return lines


def check_normalization(lm):
    """
    Check that the LM normalizes appropriately (probabilities sum to one) in several different contexts
    :param lm: The BigramLanguageModel to check
    """
    _check_normalization(lm, "the")
    _check_normalization(lm, "asked")
    _check_normalization(lm, "did")


def _check_normalization(lm, context: str) -> bool:
    """
    Checks the normalization of the LM in the given context
    :param lm:
    :param context:
    :return: True if the language model normalizes correctly for this context, false otherwise
    """
    total_prob = 0.0
    for word in lm.get_vocabulary():
        total_prob += lm.get_probability(context, word)
    if abs(total_prob - 1.0) > 1e-3:
        print("ERROR: normalization test failed: probabilities sum to " + repr(total_prob) + " rather than 1 for context \"" + context + "\"")
        return False
    else:
        print("Okay! Sums to " + repr(total_prob) + " after context \"" + context + "\"")
        return True


def query_lm(lm, context: str, num_words_to_print = 5):
    """
    Prints the top num_words_to_print most likely next words after context according to the lm
    :param lm:
    :param context:
    :param num_words_to_print
    """
    context_list = [word for word in context.strip().split(" ")]
    counter = Counter()
    for word in lm.get_vocabulary():
        counter[word] = lm.get_probability(context_list[-1], word)
    result = ""
    for (word, count) in counter.most_common(num_words_to_print):
        result += "(" + word + ", " + repr(counter[word]) + "), "
    print("Top 5 words and probabilities after \"" + context + "\": " + result[:-2])


def sample_word(lm, word1, word2):

    words = list(lm.get_vocabulary())

    probs = []

    for word in words:
        p = lm.get_probability(word1, word2, word)

        p = p ** (1 / TEMPERATURE)

        probs.append(p)

    total = sum(probs)

    if total == 0:
        return END_SYMBOL

    probs = [p / total for p in probs]

    r = random.random()

    cumulative = 0

    for i in range(len(words)):
        cumulative += probs[i]

        if r <= cumulative:
            return words[i]

    return words[-1]


def sample_sentence(lm, word1, word2):

    sentence = []

    #word1 = BEGIN_SYMBOL
    #word2 = BEGIN_SYMBOL

    for _ in range(100):

        next_word = sample_word(
            lm,
            word1,
            word2
        )

        if next_word == END_SYMBOL:
            break

        sentence.append(next_word)

        word1 = word2
        word2 = next_word

    return sentence


def get_best_word(lm, context_word: str):
    """
    :param lm:
    :param context_word:
    :return: The best word to follow context_word according to the probabilities from lm.get_probability
    """
    best_word = None
    max_prob = -1.0
    for word in lm.get_vocabulary():
        prob = lm.get_probability(context_word, word)
        if prob > max_prob:
            max_prob = prob
            best_word = word
            print(best_word)
    return best_word
    #raise Exception("IMPLEMENT ME")


def get_best_sentence(lm, context_word: str):
    """
    :param lm:
    :param context_word: An initial word to seed the sentence with
    :return: Up to 10 words as a continuation of context_word by repeatedly taking the best next word
    """
    sentence = [context_word]
    for i in range(0, 30):
        next_word = get_best_word(lm, context_word)
        context_word = next_word
        sentence.append(next_word)
        if next_word == END_SYMBOL:
            return sentence
    return sentence

def print_sentence(sentence: List[List[str]]):
    for i in range(len(sentence)):
        print(sentence[i], end="")
        if(i+1!=len(sentence) and sentence[i+1]!="." and sentence[i+1]!="," and sentence[i+1]!="?" and sentence[i+1]!="!" and sentence[i+1]!="'" and sentence[i+1]!="'s" and sentence[i+1]!=":"):
            print(" ", end="")


def read_data(path_to_wikitext: str = "./"):
    return (read_wikitext(path_to_wikitext + "/wiki.train.tokens"), read_wikitext(path_to_wikitext + "/wiki.valid.tokens"))


if __name__ == "__main__":
    (train, test) = read_data()

    lm = estimate_trigram_lm(train)

    print_sentence(sample_sentence(lm, BEGIN_SYMBOL, "We"))

