import re

import numpy as np


def neighbor_indices(cluster_labels: np.array, w: int) -> list[int]:
    """
    Take the last 'w' cluster labels and return all matching previous occurrences of this pattern in the rest of the
    list of cluster labels (so-called neighbors). The is done by first converting the list of ints
    to a list of chars and subsequently converting the latter to string. Finally, the 'finditer' function provided in
    python package "re" is used.

    Args:
        cluster_labels (np.array):
            The data in which to find neighbours.
        w (int):
            Size of window.

    Returns (list[int]):
        List of neighbor indices in the list of cluster labels
    """
    t = ''.join(to_char(i) for i in cluster_labels[:-1])
    pattern = cluster_labels[-w:]
    p = ''.join(to_char(i) for i in pattern)
    p = re.compile(p)
    matches = re.finditer(p, t)
    return [match.end() for match in matches]


def to_char(num: int) -> str:
    """
    Converts a given integer into a char by using the default 'chr' function, unless the length of the escaped
    representation of that char would be more than 1, in which case a fallback char is returned. The fallback chars
    consist of the char conversion of the numbers in the interval [1114088, 1114111], corresponding to the 24 largest
    numbers that can be converted using 'chr'.

    Args:
        num (int):
            The integer to convert to a Unicode string character

    Returns (str):
        Unicode string character
    """
    return bad_char_dict.get(num, chr(num))


bad_char_dict = {9: '\U0010ffff',
                 10: '\U0010fffe',
                 11: '\U0010fffd',
                 12: '\U0010fffc',
                 13: '\U0010fffb',
                 32: '\U0010fffa',
                 35: '\U0010fff9',
                 36: '\U0010fff8',
                 38: '\U0010fff7',
                 40: '\U0010fff6',
                 41: '\U0010fff5',
                 42: '\U0010fff4',
                 43: '\U0010fff3',
                 45: '\U0010fff2',
                 46: '\U0010fff1',
                 63: '\U0010fff0',
                 91: '\U0010ffef',
                 92: '\U0010ffee',
                 93: '\U0010ffed',
                 94: '\U0010ffec',
                 123: '\U0010ffeb',
                 124: '\U0010ffea',
                 125: '\U0010ffe9',
                 126: '\U0010ffe8'}
