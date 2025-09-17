import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# import src.dataset_tools.data_preparation.prepare_financial_dataset as financial
import src.dataset_tools.linux_bugs.prepare_linux_dataset as linux


def labels_barplots(df: pd.DataFrame, title: str = ""):
    """
    Bar plots for label counts

    :param df: dataframe containing the columns "label", "sub_label" and flattened_label
    :param title: Title for the plot
    """
    # Setup
    sns.set_theme(style="whitegrid")
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(80, 20), sharex=True)
    plt.suptitle(f"{title}\n", fontsize=30)

    # Label count
    labels = df["label"].value_counts()
    sns.barplot(x=labels, y=labels.index, palette="dark", ax=ax1, orient="h")
    ax1.set_ylabel(f"# of tickets ({len(df)})", fontsize=24)
    ax1.set_xlabel(f"Label (#: {len(labels.index)})", fontsize=24)
    for bars in ax1.containers:
        ax1.bar_label(bars)

    # Sub-label count
    sub_labels = df["sub_label"].value_counts()
    sns.barplot(x=sub_labels, y=sub_labels.index, palette="dark", ax=ax2, orient="h")
    ax2.set_ylabel(f"# of tickets ({len(df)})", fontsize=24)
    ax2.set_xlabel(f"Sub-label (#: {len(sub_labels.index)})", fontsize=24)
    for bars in ax2.containers:
        ax2.bar_label(bars)

    # Flattened label count
    flattened_labels = df["flattened_label"].value_counts()
    sns.barplot(x=flattened_labels, y=flattened_labels.index, palette="dark", ax=ax3, orient="h")
    ax3.set_ylabel(f"# of tickets ({len(df)})", fontsize=24)
    ax3.set_xlabel(f"Flattened-label (#: {len(flattened_labels.index)})", fontsize=24)
    for bars in ax3.containers:
        ax3.bar_label(bars)

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.1,
        wspace=0.3
    )

    plt.show()


def main():
    linux_df = linux.read_dataset(100)
    labels_barplots(linux_df, "Linux Bugs")


if __name__ == "__main__":
    main()
