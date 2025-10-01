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
    if seq_len <= 1:
        return a + b

    a_crop = a[:-1]
    b_crop = b[1:]
    max_overlap = min(len(a_crop), len(b_crop))

    # Precompute expected_overlap outside of branching:
    expected_overlap = round(len(b) * overlap_ratio) - 3

    # Preallocate scores and zero_matches, single loop
    scores = []
    zero_matches = []
    for i in range(1, max_overlap + 1):
        score = Hamming.distance(a_crop[-i:], b_crop[:i], processor=None)
        scores.append(score)
        if score == 0:
            zero_matches.append(i - 1)

    if len(zero_matches) == 1:
        i = zero_matches[0]
        return a_crop + b_crop[i + 1 :]

    elif len(zero_matches) > 1:
        # Use generator to avoid list allocation inside min()
        best_i = min(zero_matches, key=lambda x: abs(x - expected_overlap))
        return a_crop + b_crop[best_i + 1 :]

    if expected_overlap < -1:
        return a + b
    elif expected_overlap < 0:
        return a_crop + b_crop

    # Avoid enumerating twice by combining in one pass
    min_score = None
    min_idx = -1
    for i, score in enumerate(scores):
        combined = score + abs(i - expected_overlap)
        if (min_score is None) or (combined < min_score):
            min_score = combined
            min_idx = i

    return a_crop + b_crop[min_idx + 1 :]


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
