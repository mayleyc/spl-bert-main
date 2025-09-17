import re
from typing import Callable

import pandas as pd


def regex_fun(regex: str, value: str) -> Callable:
    return lambda x: re.sub(regex, value, x)


def remove_linux_garbage(data):
    """
    Taken from: https://arxiv.org/abs/1807.02892
    Linux data contains lots of garbage, e.g. memory addresses - 0000f800
    """

    def is_garbage(w):
        return len(w) >= 7 and sum(c.isdigit() for c in w) >= 2

    if isinstance(data, pd.Series):
        data = data.map(lambda s: ' '.join(map(lambda w: w if not is_garbage(w) else ' ', s.split())))
    elif isinstance(data, list):
        data = map(lambda s: ' '.join(map(lambda w: w if not is_garbage(w) else ' ', s.split())), data)
    return data


def text_cleanup(msg, remove_garbage: bool = False):
    print("Performing text cleanup...")
    # Note: We now generally use lists, therefore changed .map to map()
    msg = map(regex_fun("\r", " "), msg)
    msg = map(regex_fun("\n", " "), msg)
    # 2. Remove URLs
    msg = map(regex_fun(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ""), msg)
    # 3. Remove Stack Trace
    # msg = msg.map(lambda s: (s + " ")[:s.find("Stack trace:")])  # WARN return -1 on failure
    # 4. Remove hex code
    msg = map(regex_fun(r"(\w+)0x\w+", ""), msg)
    # 5. Remove linux garbage
    if remove_garbage:
        msg = remove_linux_garbage(msg)
    # 6. Change to lower case
    return list(map(str.lower, msg))


def text_cleanup_old(msg, remove_garbage: bool = False):
    print("Performing text cleanup...")
    # Note: We now generally use lists, therefore changed .map to map()
    msg = msg.map(regex_fun("\r", " "))
    msg = msg.map(regex_fun("\n", " "))
    # 2. Remove URLs
    msg = msg.map(regex_fun(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ""))
    # 3. Remove Stack Trace
    # msg = msg.map(lambda s: (s + " ")[:s.find("Stack trace:")])  # WARN return -1 on failure
    # 4. Remove hex code
    msg = msg.map(regex_fun(r"(\w+)0x\w+", ""))
    # 5. Remove linux garbage
    if remove_garbage:
        msg = remove_linux_garbage(msg)
    # 6. Change to lower case
    return msg.map(str.lower)
