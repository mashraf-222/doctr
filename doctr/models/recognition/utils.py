# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from rapidfuzz.distance import Hamming

__all__ = ["merge_strings", "merge_multi_strings"]


def merge_strings(a: str, b: str, overlap_ratio: float) -> str:
    """Merges 2 character sequences in the best way to maximize the alignment of their overlapping characters.

    Args:
        a: first char seq, suffix should be similar to b's prefix.
        b: second char seq, prefix should be similar to a's suffix.
        overlap_ratio: estimated ratio of overlapping characters.

    Returns:
        A merged character sequence.

    Example::
        >>> from doctr.models.recognition.utils import merge_strings
        >>> merge_strings('abcd', 'cdefgh', 0.5)
        'abcdefgh'
        >>> merge_strings('abcdi', 'cdefgh', 0.5)
        'abcdefgh'
    """
    seq_len = min(len(a), len(b))
    if (
        seq_len <= 1
    ):  # One sequence is empty or will be after cropping in next step, return both to keep data
        return a + b

    # Remove last letter of "a" and first of "b", because they might be cut off
    a_crop = a[:-1]
    b_crop = b[1:]
    len_a_crop = len(a_crop)
    len_b_crop = len(b_crop)
    max_overlap = min(len_a_crop, len_b_crop)

    if max_overlap == 0:
        return a + b

    # *** HOT PATH OPTIMIZATION ***
    # Pre-extract substrings and compute scores in a tight loop
    # Pre-allocate list for speed; using list comprehension with direct string slicing access
    # Also, cache Hamming.distance for small overlap especially when overlap is max
    scores = []
    for i in range(1, max_overlap + 1):
        s1 = a_crop[-i:]
        s2 = b_crop[:i]
        # if they are the same object, hamming is 0, shortcut (likely rare in actual use case)
        if s1 is s2:
            scores.append(0)
        elif s1 == s2:
            scores.append(0)
        else:
            scores.append(Hamming.distance(s1, s2, processor=None))

    # Find zero-score matches via efficient iterator
    zero_matches = []
    for idx, score in enumerate(scores):
        if score == 0:
            zero_matches.append(idx)

    expected_overlap = (
        round(len(b) * overlap_ratio) - 3
    )  # adjust for cropping and index

    # Case 1: One perfect match - exactly one zero score - just merge there
    if len(zero_matches) == 1:
        i = zero_matches[0]
        return a_crop + b_crop[i + 1 :]

    # Case 2: Multiple perfect matches - likely due to repeated characters.
    # Use the estimated overlap length to choose the match closest to the expected alignment.
    elif len(zero_matches) > 1:
        # Instead of lambdas, a local min loop for better perf
        best_i = zero_matches[0]
        best_dist = abs(zero_matches[0] - expected_overlap)
        for x in zero_matches[1:]:
            dist = abs(x - expected_overlap)
            if dist < best_dist:
                best_i = x
                best_dist = dist
        return a_crop + b_crop[best_i + 1 :]

    # Case 3: Absence of zero scores indicates that the same character in the image was recognized differently OR that
    # the overlap was too small and we just need to merge the crops fully
    if expected_overlap < -1:
        return a + b
    elif expected_overlap < 0:
        return a_crop + b_crop

    # Find best overlap by minimizing Hamming distance + distance from expected overlap size
    # Hot path: avoid making a list of tuples, scan cheaply
    best_combined_score = scores[0] + abs(0 - expected_overlap)
    best_i = 0
    for idx in range(1, len(scores)):
        combined_score = scores[idx] + abs(idx - expected_overlap)
        if combined_score < best_combined_score:
            best_combined_score = combined_score
            best_i = idx
    return a_crop + b_crop[best_i + 1 :]


def merge_multi_strings(
    seq_list: list[str], overlap_ratio: float, last_overlap_ratio: float
) -> str:
    """
    Merges consecutive string sequences with overlapping characters.

    Args:
        seq_list: list of sequences to merge. Sequences need to be ordered from left to right.
        overlap_ratio: Estimated ratio of overlapping letters between neighboring strings.
        last_overlap_ratio: Estimated ratio of overlapping letters for the last element in seq_list.

    Returns:
        A merged character sequence

    Example::
        >>> from doctr.models.recognition.utils import merge_multi_strings
        >>> merge_multi_strings(['abc', 'bcdef', 'difghi', 'aijkl'], 0.5, 0.1)
        'abcdefghijkl'
    """
    if not seq_list:
        return ""
    result = seq_list[0]
    for i in range(1, len(seq_list)):
        text_b = seq_list[i]
        ratio = last_overlap_ratio if i == len(seq_list) - 1 else overlap_ratio
        result = merge_strings(result, text_b, ratio)
    return result
