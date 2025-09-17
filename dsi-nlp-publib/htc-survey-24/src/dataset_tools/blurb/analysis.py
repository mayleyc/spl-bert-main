from collections import Counter
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def analyze_blurb(data: List[Dict[str, Any]]) -> None:
    summaries = [d["text"] for d in data]
    avg_len = np.mean(list(map(str.__len__, summaries)))

    print(f"Average len of book summaries: {avg_len}")

    for level in range(4):
        # Setup
        title = f"BLURB L{level}-topic distribution"
        sns.set_theme(style="whitegrid")
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(80, 30), sharex=False)
        plt.suptitle(f"{title}\n", fontsize=30)

        l_topics = [d[f"d{level}_topics"] for d in data]
        c_label = Counter([a for t in l_topics for a in t])
        c_book = Counter([len(t) for t in l_topics])

        pd.DataFrame(c_label.most_common()).to_csv(f"freq_bgc-{level + 1}.csv")

        # Label frequency
        y_label, x_count = zip(*c_label.most_common())
        sns.barplot(x=list(x_count), y=list(y_label), palette="dark", ax=ax1, orient="h")
        ax1.set_ylabel(f"Label (#: {len(c_label.keys())})", fontsize=24)
        ax1.set_xlabel(f"# of books ({len(l_topics)})", fontsize=24)
        for bars in ax1.containers:
            ax1.bar_label(bars)

        # Label number per book
        sns.barplot(x=list(c_book.values()), y=list(c_book.keys()), palette="dark", ax=ax2, orient="h")
        ax2.set_ylabel(f"# of labels", fontsize=24)
        ax2.set_xlabel(f"Book count", fontsize=24)
        for bars in ax2.containers:
            ax2.bar_label(bars)

        plt.tight_layout()
        plt.subplots_adjust(
            left=0.1,
            wspace=0.3
        )
        plt.show()
