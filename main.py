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


def sample_sentence(lm, word1, word2, length):

    original_word1 = word1
    original_word2 = word2

    sentence = []

    if(word1 != BEGIN_SYMBOL):
        sentence.append(word1)
    if(word2 != BEGIN_SYMBOL):
        sentence.append(word2)

    for _ in range(length):

        next_word = sample_word(lm, word1, word2)

        if next_word == END_SYMBOL:
            sentence.append("\n")

            word1 = original_word1
            word2 = original_word2

            if(word1 != BEGIN_SYMBOL):
                sentence.append(word1)
            if(word2 != BEGIN_SYMBOL):
                sentence.append(word2)

        else:
            sentence.append(next_word)
            word1 = word2
            word2 = next_word

    return sentence



def print_sentence(sentence: List[List[str]]):
    for i in range(len(sentence)):
        if(sentence[i] != "="):
            print(sentence[i], end="")
        
        if(i+1!=len(sentence) and sentence[i+1]!="." and sentence[i+1]!="," and sentence[i+1]!="?" and sentence[i+1]!="!" and sentence[i+1]!="'" and sentence[i+1]!="'s" and sentence[i+1]!=":" and sentence[i]!="\n"):
            print(" ", end="")


def read_data(path_to_wikitext: str = "./"):
    return (read_wikitext(path_to_wikitext + "/wiki_slugs_train.tokens"), read_wikitext(path_to_wikitext + "/wiki_slugs_valid.tokens"))


if __name__ == "__main__":
    (train, test) = read_data()

    lm = estimate_trigram_lm(train)

    length = int(input("Response length: "))

    w1 = input("First word (type @ for blank): ")
    if(w1 == "@"):
        w1 = BEGIN_SYMBOL
    w2 = input("Second word (type @ for blank): ")
    if(w2 == "@"):
        w2 = BEGIN_SYMBOL

    print_sentence(sample_sentence(lm, w1, w2, length))

