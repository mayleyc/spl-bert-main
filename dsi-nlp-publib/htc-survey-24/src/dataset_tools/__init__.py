from .amazon.generate_multidomain import get_amazon
from .blurb.generate_hierarchy import get_bgc_split_jsonl
from .linux_bugs.prepare_linux_dataset import read_dataset as read_bugs
from .rcv1.generate_splits import get_rcv1_split
from .wos.generation import read_wos_dataset

bgc_ohe_csv = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/spl-bert/spl/C-HMCNN/bgc_tax_one_hot.csv"
amz_ohe_csv = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/spl-bert/spl/C-HMCNN/amazon_tax_one_hot.csv"
wos_ohe_csv = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/spl-bert/spl/C-HMCNN/wos_tax_one_hot.csv"

