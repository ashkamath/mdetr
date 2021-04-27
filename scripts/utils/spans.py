# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
""""""

from typing import List, Tuple

from .text import STOP_WORDS, nlp


class PreprocessError(Exception):
    pass


def span_intersect_span(span1: Tuple[int, int], span2: Tuple[int, int]):
    """Returns True if the given spans intersect"""
    return (span1[0] <= span2[0] < span1[1]) or (span2[0] <= span1[0] < span2[1])


def span_intersect_spanlist(span: Tuple[int, int], target_spans: List[Tuple[int, int]]):
    """Returns True if the given spans intersect with any in the given list"""
    for t in target_spans:
        if span_intersect_span(span, t):
            return True
    return False


def spanlist_intersect_spanlist(spans: List[Tuple[int, int]], target_spans: List[Tuple[int, int]]):
    """Returns True if the given spans intersect with any in the given list"""
    for s in spans:
        if span_intersect_spanlist(s, target_spans):
            return True
    return False


def consolidate_spans(spans: List[Tuple[int, int]], caption: str, rec=True):
    """Accepts a list of spans and the the corresponding caption.
    Returns a cleaned list of spans where:
        - Overlapping spans are merged
        - It is guaranteed that spans start and end on a word
    """
    sorted_spans = sorted(spans)
    cur_end = -1
    cur_beg = None
    final_spans: List[Tuple[int, int]] = []
    for s in sorted_spans:
        if s[0] >= cur_end:
            if cur_beg is not None:
                final_spans.append((cur_beg, cur_end))
            cur_beg = s[0]
        cur_end = max(cur_end, s[1])

    if cur_beg is not None:
        final_spans.append((cur_beg, cur_end))

    # Now clean the begining/end
    clean_spans: List[Tuple[int, int]] = []
    for s in final_spans:
        beg, end = s
        end = min(end, len(caption))
        while beg < len(caption) and not caption[beg].isalnum():
            beg += 1
        while end > 0 and not caption[end - 1].isalnum():
            end -= 1
        # Try to get hyphenated words
        if end < len(caption) and caption[end] == "-":
            # print("trigg")
            next_space = caption.find(" ", end)
            if next_space == -1:
                end = len(caption)
            else:
                end = next_space + 1
        if beg > 0 and caption[beg - 1] == "-":
            prev_space = caption.rfind(" ", 0, beg)
            if prev_space == -1:
                beg = 0
            else:
                beg = prev_space + 1
        if 0 <= beg < end <= len(caption):
            clean_spans.append((beg, end))
    if rec:
        return consolidate_spans(clean_spans, caption, False)
    return clean_spans


def get_canonical_spans(orig_spans: List[List[Tuple[int, int]]], orig_caption: str, whitespace_only=False):
    """This functions computes the spans after reduction of the caption to it's normalized version
    For example, if the caption is "There is a man wearing sneakers" and the span is [(11,14)] ("man"),
    then the normalized sentence is "man wearing sneakers" so the new span is [(0,3)]
    """
    # print("orig caption", orig_caption)
    # print("orig spans", [orig_caption[t[0]:t[1]] for span in orig_spans for t in span])
    new_spans = [sorted(spans) for spans in orig_spans]
    caption = orig_caption.lower()

    def remove_chars(pos, amount):
        for i in range(len(new_spans)):
            for j in range(len(new_spans[i])):
                if pos >= new_spans[i][j][1]:
                    continue
                beg, end = new_spans[i][j]
                if span_intersect_span(new_spans[i][j], (pos, pos + amount)):
                    # assert new_spans[i][j][0] == pos or amount == 1, "unexpected deletion from middle of span"
                    new_spans[i][j] = (beg, end - amount)
                else:
                    new_spans[i][j] = (beg - amount, end - amount)

    def change_chars(old_beg, old_end, delta):
        for i in range(len(new_spans)):
            for j in range(len(new_spans[i])):
                if old_beg >= new_spans[i][j][1]:
                    continue
                beg, end = new_spans[i][j]
                if span_intersect_span(new_spans[i][j], (old_beg, old_end)):
                    if not (new_spans[i][j][0] <= old_beg < old_end <= new_spans[i][j][1]):
                        raise PreprocessError(f"deleted spans should be contained in known span")
                    assert (
                        new_spans[i][j][0] <= old_beg < old_end <= new_spans[i][j][1]
                    ), "deleted spans should be contained in known span"
                    new_spans[i][j] = (beg, end + delta)
                else:
                    new_spans[i][j] = (beg + delta, end + delta)

    # Pre pass, removing double spaces and leading spaces
    # Check for leading spaces
    while caption[0] == " ":
        remove_chars(0, 1)
        caption = caption[1:]
    cur_start = 0
    pos = caption.find("  ", cur_start)
    while pos != -1:
        amount = 1
        # print("remvoing", removed, pos)
        remove_chars(pos, amount)
        caption = caption.replace("  ", " ", 1)
        pos = caption.find("  ", cur_start)
    # print("after whitespace caption", caption)
    # print("after whitespace spans", [caption[t[0]:t[1]] for span in new_spans for t in span])
    if whitespace_only:
        return new_spans, caption

    # First pass, removing punctuation
    for punct in [".", ",", "!", "?", ":"]:
        pos = caption.find(punct)
        while pos != -1:
            remove_chars(pos, len(punct))
            caption = caption.replace(punct, "", 1)
            pos = caption.find(punct)
    # print("after punct caption", caption)
    # print("after punct spans", [caption[t[0]:t[1]] for span in new_spans for t in span])

    # parsing needs to happen before stop words removal
    all_tokens = nlp(caption)
    tokens = []

    # Second pass, removing stop words
    ## Remove from tokenization
    for t in all_tokens:
        if str(t) not in STOP_WORDS:
            tokens.append(t)
    ## Remove from actual sentence
    for stop in STOP_WORDS:
        cur_start = 0
        pos = caption.find(stop, cur_start)
        while pos != -1:
            # Check that we are matching a full word
            if (pos == 0 or caption[pos - 1] == " ") and (
                pos + len(stop) == len(caption) or caption[pos + len(stop)] == " "
            ):
                removed = stop
                spaces = 0
                if pos + len(stop) < len(caption) and caption[pos + len(stop)] == " ":
                    removed += " "
                    spaces += 1
                if pos > 0 and caption[pos - 1] == " ":
                    removed = " " + removed
                    spaces += 1
                if spaces == 0:
                    raise PreprocessError(
                        f"No spaces found in '{caption}', position={pos}, stopword={stop}, len={len(stop)}"
                    )
                assert spaces > 0
                replaced = "" if spaces == 1 else " "
                amount = len(removed) - len(replaced)
                # print("remvoing", removed, pos)
                remove_chars(pos, amount)
                caption = caption.replace(removed, replaced, 1)
                # print("cur caption", caption)
                # print("cur spans", [caption[t[0]:t[1]] for span in new_spans for t in span if t[0] < t[1]])
            else:
                cur_start += 1
            pos = caption.find(stop, cur_start)

    # print("final caption", caption)
    # print("final spans", [caption[t[0]:t[1]] for span in new_spans for t in span if t[0] < t[1]])

    # Third pass, lemmatization
    final_caption = []
    if len(tokens) != len(caption.strip().split(" ")):
        raise PreprocessError(
            f"''{tokens}'', len={len(tokens)}, {caption.strip().split(' ')}, len={len(caption.strip().split(' '))}"
        )

    # tokens = nlp(caption)
    cur_beg = 0
    for i, w in enumerate(caption.strip().split(" ")):
        if tokens[i].lemma_[0] != "-":
            # print(w, "lemmatized to", tokens[i].lemma_)
            final_caption.append(tokens[i].lemma_)
            change_chars(cur_beg, cur_beg + len(w), len(tokens[i].lemma_) - len(w))
        else:
            # print(w, "skipped lemmatized to", tokens[i].lemma_)
            final_caption.append(w)
        cur_beg += 1 + len(final_caption[-1])
        # print("cur_beg", cur_beg)
        # print("cur spans", [caption[t[0]:t[1]] for span in new_spans for t in span if t[0] < t[1]], new_spans)

    clean_caption = " ".join(final_caption)
    # Cleanup empty spans
    clean_spans = []
    for spans in new_spans:
        cur = []
        for s in spans:
            if 0 <= s[0] < s[1]:
                cur.append(s)
        clean_spans.append(cur)

    # print("clean caption", clean_caption)
    # print("clean spans", [clean_caption[t[0]:t[1]] for span in clean_spans for t in span])
    return clean_spans, clean_caption


def shift_spans(spans: List[Tuple[int, int]], offset: int) -> List[Tuple[int, int]]:
    final_spans = []
    for beg, end in spans:
        final_spans.append((beg + offset, end + offset))
    return final_spans
