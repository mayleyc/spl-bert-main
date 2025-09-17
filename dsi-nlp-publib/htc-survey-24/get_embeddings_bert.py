import os
import sys
import datetime
import pickle
from pathlib import Path
import datetime as dt
from typing import List, Dict, Callable, Optional
from operator import itemgetter

import torch

from transformers import BertTokenizer, BertForSequenceClassification

from src.utils.generic_functions import load_yaml
from src.dataset_tools.dataset_manager import DatasetManager
from src.models.BERT_flat.torch_dataset import TransformerDatasetFlat


from tqdm import tqdm

os.environ['TORCH_HOME'] = '/mnt/cimec-storage6/users/nguyenanhthu.tran/torch'
sys.path.append("/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/dsi-nlp-publib/htc-survey-24")


#Input: CUB raw dataset with each folder as class name
#Output: the same folder tree in a different location, each folder containing an .npy embedding file
#using ResNet50 pretrained on ImageNet

#CUB modified to take string labels

def find_files_by_name(substring, folder_path):
    matching_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if substring in file:
                matching_files.append(os.path.join(root, file))
    
    return matching_files

def collate_batch(tokenizer, batch, ml: bool = False):
    x = [t for t, *_ in batch]
    max_len = tokenizer.model_max_length if tokenizer.model_max_length <= 2048 else 512
    encoded_x = tokenizer(x, truncation=True, max_length=max_len, padding=True)
    item_x = {key: torch.tensor(val) for key, val in encoded_x.items()}
    y = [t for _, t in batch]
    if ml:
        # multilabel: y is a list of tensors
        y_tensor = torch.stack(y, dim=0).long()
    else:
        # single label: convert list of values to long tensor
        y_tensor = torch.stack(y).long()
    return item_x, y_tensor

def _setup_training_BERT_SC(train_config, tokenizer, workers: int, data_train, labels_train, data_test, labels_test, data_val, labels_val):
    # -------------------------------


    train_data = TransformerDatasetFlat(data_train, labels_train)
    val_data = TransformerDatasetFlat(data_val, labels_val)
    test_data = TransformerDatasetFlat(data_test, labels_test)
    # Initialize model

    # Initialize Optimizer and loss
    # -------------------------------
    # Prepare dataset
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_config["BATCH_SIZE"],
                                                  num_workers=workers, shuffle=True,
                                                  collate_fn=lambda x: collate_batch(tokenizer, x, ml=False))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_config["TEST_BATCH_SIZE"],
                                                    num_workers=workers, shuffle=True,
                                                    collate_fn=lambda x: collate_batch(tokenizer, x, ml=False))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=train_config["TEST_BATCH_SIZE"],
                                                    num_workers=workers, shuffle=True,
                                                    collate_fn=lambda x: collate_batch(tokenizer, x, ml=False))
    
    # -------------------------------
 
    return train_loader, test_loader, val_loader

def _setup_training_BERT_SC_old(train_config, workers: int, data_train, labels_train, data_test, labels_test, data_val, labels_val):
    # -------------------------------


    train_data = TransformerDatasetFlat(data_train, labels_train)
    val_data = TransformerDatasetFlat(data_val, labels_val)
    test_data = TransformerDatasetFlat(data_test, labels_test)
    # Initialize model

    # Initialize Optimizer and loss
    # -------------------------------
    # Prepare dataset
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_config["BATCH_SIZE"],
                                                  num_workers=workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_config["TEST_BATCH_SIZE"],
                                                    num_workers=workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=train_config["TEST_BATCH_SIZE"],
                                                    num_workers=workers, shuffle=True)
    
    # -------------------------------
 
    return train_loader, test_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} for inference')

model_name = "bert-base-uncased"  # or "bert-large-uncased"

bbu_path = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/dsi-nlp-publib/htc-survey-24/bert-base-uncased-local"
data_path = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/dsi-nlp-publib/htc-survey-24/data"
model = BertForSequenceClassification.from_pretrained(bbu_path)
tokenizer = BertTokenizer.from_pretrained(bbu_path)

model.eval()
model.to(device)

config_base_path: Path = Path("config") / "BERT"
output_path: Path = Path("dumps") / "BERT"
config_list: List = ["bert_amz.yml", "bert_bgc.yml", "bert_wos.yml"] 



for c in config_list:
    # Prepare configuration
    config_path: Path = (config_base_path / c)
    config: Dict = load_yaml(config_path)
    print(f"Dataset: {config['dataset']}")
    dataset_config=config["dataset"]
    dataset_dict = {"amazon": "Amazon", "bgc": "BGC", "wos": "WebOfScience"}
    
    jsonl_folder=os.path.join(data_path, dataset_dict.get(dataset_config))
    # Prepare output
    out_folder = output_path / model_name / f"run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


    ds_manager = DatasetManager(dataset_name=dataset_config, training_config=config)

    for (x_train, y_train), (x_test, y_test), (x_val, y_val) in ds_manager.get_split(): #labels are already OHE? Okay though
        batch_no = 0
        training_loader, test_loader, val_loader = _setup_training_BERT_SC(train_config=config, tokenizer=tokenizer,
                                                    workers=0,
                                                    data_train=x_train, labels_train=y_train, data_test=x_test, labels_test=y_test, data_val=x_val, labels_val=y_val)
        #print(f"Training loader size: {len(training_loader)}")
        #print(f"Training example: {training_loader.dataset[0]}")
        #inputs, labels = next(iter(training_loader))
        # Print shapes
        '''print("input_ids shape:", inputs['input_ids'].shape)
        print("attention_mask shape:", inputs['attention_mask'].shape)
        print("labels shape:", labels.shape)

        # Print the first example in the batch
        print("\n--- First example ---")
        print("input_ids[0]:", inputs['input_ids'][0])
        print("attention_mask[0]:", inputs['attention_mask'][0])
        print("label[0]:", labels[0])

        quit()'''
        for loader in [training_loader, test_loader, val_loader]:
            if loader == training_loader:
                emb_type = "train"
                print("Extracting training embeddings...")
            elif loader == test_loader:
                emb_type = "test"
                print("Extracting test embeddings...")
            elif loader == val_loader:
                emb_type = "val"
                print("Extracting validation embeddings...")
            all_texts = []
            all_embeddings = []
            all_labels = []

            for (inputs, labels) in tqdm(loader, desc="Extracting embeddings"):
                #print(type(inputs))
                #print(inputs)
                original_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True) #batch of 8
                all_texts.extend(original_texts)

                inputs = {k: v.to(device) for k, v in inputs.items()} # 'dict' object has no attribute 'to'
                #first_input_ids = inputs["input_ids"][0]  # Take the first input_id tensor
                #first_input_text = tokenizer.decode(first_input_ids, skip_special_tokens=True)
                '''for i in range(3,8):
                    print("First input text:", original_texts[i])
                    print("first label:", labels[i])
                quit()
                '''
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)  # data must be a dict like {'input_ids': ..., 'attention_mask': ...}
                    cls_embedding = torch.cat([outputs.hidden_states[-i][:, 0, :] for i in range(1, 4)], dim=-1) #like them: concat 3 layers 
                    # cls only: outputs.hidden_states[-1][:, 0, :]
                    all_embeddings.extend(cls_embedding.tolist())
                    all_labels.extend(labels)
    
            print(f"Extracted {len(all_embeddings)} embeddings for {emb_type} set.")
            print('Embeddings extracted.')
            print('Saving dictionary to pickle...')
            #create new directory tree at /embeddings

            date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            emb_root = "/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/spl-bert/spl/C-HMCNN/embeddings"
            filename = f"emb_{model_name}_{dataset_config}_{date_string}_{emb_type}_batch{batch_no}.pickle"
            fp_dest = os.path.join(emb_root, filename)
            if Path(emb_root).exists():
                pass
            else:
                os.makedirs(emb_root)

            # option 1: 1 file with embeddings & 1 with labels
            # option 2: use a pickle to save it as a dictionary
            with open(fp_dest, 'wb') as f:
                pickle.dump([all_texts, all_embeddings, all_labels], f, protocol=pickle.HIGHEST_PROTOCOL)
                print('Checking length of lists:')
                print(len(all_embeddings), len(all_labels))
                print(len(all_embeddings[0]))
               
        break


#for path, _, label in zip(all_paths, all_embeddings, all_labels):

    

'''
counter = 0
for i, label in enumerate(labels):

    class_name = name_to_class[label]  # Use class name directly
    if class_name not in all_embeddings:
        all_embeddings[class_name] = []
    print(embeddings[i])
    all_embeddings[class_name].append(embeddings[i])
    counter += 1

for label, embs in all_embeddings.items():
    class_dir = os.path.join(fp_dest, str(label))
    os.makedirs(class_dir, exist_ok=True)  # Ensure directory exists
    print(label)
    np.save(os.path.join(class_dir, f"embeddings.npy"), np.vstack(embs))  # Save as np.vstack per class

print("Embeddings saved per class.")

'''