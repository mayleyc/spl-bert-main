import shutil
from pathlib import Path

folder = Path("dumps")


def clear():
    shutil.rmtree(folder)
    folder.mkdir()


if __name__ == "__main__":
    clear()
