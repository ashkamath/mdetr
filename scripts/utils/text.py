# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Provides various text related util function"""
import re
from typing import List, Tuple

import nltk
import spacy

nlp = spacy.load("en_core_web_sm")

nltk.download("stopwords")
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english")) - set(["above", "below", "between", "further", "he", "she", "they"])


def get_noun_phrase(root):
    queue = [root]
    all_toks = [root]
    while len(queue) > 0:
        curr = queue.pop()
        if curr.tag_ in ["NN", "NNS", "NNP", "NNPS"]:
            queue += curr.lefts
            all_toks += curr.lefts
    return all_toks


def get_root_and_nouns(text: str, lazy=True) -> Tuple[str, str, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Given a sentence, returns a tuple with the following items:
    -- root text:str  : the text associated with the root of the sentence
    -- negative_text:str: all the text that shouldn't be positively matched with a box other than the main one
    -- root_span: List[Tuple[int, int]] spans covering the root expressions, returned as a list of (beg, end) character spans
    -- negative_span: List[Tuple[int, int]] spans covering the negative expressions, returned as a list of (beg, end) character spans

    If lazy is False, then we try a bit harder to find the precise root of the sentence
    """
    sents = nlp(text)
    negative_text = []

    if len([x for x in sents if x.tag_ in ["NN", "NNS", "NNP", "NNPS", "PRP"]]) <= 1:
        if lazy or len([x for x in sents if x.tag_ in ["NN", "NNS", "NNP", "NNPS", "PRP"]]) == 0:
            return text, " ", [(0, len(text))], [(0, len(text))]

    root = None
    for token in sents:
        if token.dep_ == "ROOT":
            if token.tag_ == "UH":
                continue
            root = token
            break

    if root is None:
        return text, "", [(0, len(text))], [(0, len(text))]

    if (
        len([c for c in root.children if c.tag_ in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] and c.dep_ == "compound"])
        > 0
    ):
        return text, "", [(0, len(text))], [(0, len(text))]

    all_toks = []
    if root.tag_ in ["NN", "NNS", "NNP", "NNPS"]:
        all_toks = get_noun_phrase(root)
        root_text = " ".join([x.text for x in all_toks])
        root_spans = [(x.idx, x.idx + len(x.text)) for x in all_toks]
    else:
        root = [x for x in root.children if x.tag_ in ["NN", "NNS", "NNP", "NNPS", "PRP"]]
        if len(root) < 1:
            return text, "", [(0, len(text))], [(0, len(text))]
        else:
            root = root[0]
        all_toks = list(root.lefts) + [root]
        root_text = " ".join([x.text for x in all_toks])
        root_spans = [(x.idx, x.idx + len(x.text)) for x in all_toks]

    everything_else = set()
    for token in sents:
        if token.tag_ in ["NN", "NNS", "NNP", "NNPS"] and token.dep_ not in ["ROOT"] and token not in all_toks:
            everything_else = everything_else.union(set(get_noun_phrase(token)))

    negative_tokens = set(sents) - set(everything_else)
    negative_text = " ".join([x.text for x in negative_tokens])
    negative_spans = [(x.idx, x.idx + len(x.text)) for x in negative_tokens]

    return root_text, negative_text, root_spans, negative_spans


def normalize_sentence(sentence):
    """Returns a list of non stopwords for the sentence, obtained after cleaning ponctuation and spaces"""

    sent = sentence.lower()
    sent = remove_punctuation(sentence.lower())
    sent = normalize_whitespace(sent)
    tokens = nlp(sent)
    return " ".join(
        [
            tokens[i].lemma_ if tokens[i].lemma_[0] != "-" else w
            for i, w in enumerate(sent.split(" "))
            if w not in STOP_WORDS
        ]
    )


def remove_punctuation(text):
    """
    This function removes all ponctuation.
    """
    corrected = str(text)
    corrected = re.sub(r"([!?,;.:-])", r"", corrected)
    return corrected


def simplify_punctuation(text):
    """
    This function simplifies doubled or more complex punctuation. The exception is '...'.
    """
    corrected = str(text)
    corrected = re.sub(r"([!?,;:-])\1+", r"\1", corrected)
    corrected = re.sub(r"\.{2,}", r"...", corrected)
    corrected = re.sub(r"\s?-\s?", r"-", corrected)
    return corrected


def normalize_whitespace(text):
    """
    This function normalizes whitespaces, removing duplicates and converting all to standard spaces
    """
    corrected = str(text)
    corrected = re.sub(r"//t", r"\t", corrected)
    corrected = re.sub(r"\n", r" ", corrected)
    corrected = re.sub(r"_", r" ", corrected)
    corrected = re.sub(r"\r", r" ", corrected)
    corrected = re.sub(r"\t", r" ", corrected)
    corrected = re.sub(r"\s+", r" ", corrected)
    return corrected.strip(" ")
