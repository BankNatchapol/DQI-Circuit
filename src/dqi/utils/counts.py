# src/dqi/utils/counts.py

import logging
from typing import Dict

logger = logging.getLogger("dqi")

def combine_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Consolidate counts by combining keys that share the same second part.

    Each key should be in the format 'first_part second_part'. The function returns a 
    dictionary with keys as the second part and values as the summed counts.
    """
    new_counts: Dict[str, int] = {}
    for key, value in counts.items():
        try:
            _, second_part = key.split()
        except ValueError:
            logger.error(f"Key '{key}' does not match the expected format.")
            continue
        new_counts[second_part] = new_counts.get(second_part, 0) + value
    return new_counts


def post_selection_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Sort all (key, count) pairs by count descending, then keep only those
    whose first_part (the substring before the space) has digits summing to 0.

    Each key is expected in the format 'first_part second_part', where first_part
    is a string of digits. Non-conforming keys are skipped with an error log.

    Args:
        counts: Dictionary mapping 'first_part second_part' → count.

    Returns:
        A dict of the filtered and sorted (key → count) pairs.
    """
    filtered: Dict[str, int] = {}
    # sort by count descending
    for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        parts = key.split()
        if len(parts) != 2:
            logger.error(f"Key '{key}' does not match the expected format.")
            continue

        first_part, second_part = parts
        try:
            total = sum(int(bit) for bit in first_part)
        except ValueError:
            logger.error(f"Non-digit found in first_part '{first_part}' of key '{key}'.")
            continue

        if total == 0:
            filtered[second_part] = value

    return filtered
