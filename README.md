# SPL-BERT: ENFORCING SOUND PREDICTIONS IN MULTILABEL HIERARCHICAL TEXT CLASSIFICATION

SPL-BERT is built upon Semantic Probabilistic Layers (Ahmed et al., 2022) and bert-base-uncased (Devlin et al., 2019) to tackle Hierarchical Text Classification (HTC) tasks, considering how the reproduced results from SOTA methods up to January 2024 (reproduced from Zangari et al., 2024) still display different types of hierarchical inconsistency.

The study utilizes the existing codebase from the **htc-survey** (Zangari et al., 2024) and the SPL implementation from Ahmed et al. (2022). Particularly, the original base model from Ahmed et al. (2022) has been replaced with an unfrozen bert-base-uncased, and the constraint circuit has seen significant changes to enforce exactly one prediction per hierarchy level, thereby significantly reducing the number of satisfying configurations. With a completely annotated dataset (all levels are annotated with exactly one class) and the constraint layers applied to all levels of the hierarchy, the constraint circuit will result in a model count equal to the number of classes (leaf nodes) in the dataset.

## Components
### dsi-nlp-publib/htc-survey-24
This folder contains the main experiments with SPL-BERT and its competitor BERT-MATCH, along with the baseline BERT-Naive.
- `/config`* contains the configurations needed to run the base models, as well as specifications to select the SPL pipeline.
- `/constraints`** is where the generated .sdd and .vtree files are stored as a result of running the constraint circuit without a previous copy.
- `/csv`** contains the multi-hot encoding dictionaries and the ancestor matrices, both are generated from `hierarchy_dict_gen.py`
- `/src`* contains the base models, losses, circuits and training scripts for SPL-BERT, BERT-Naive and BERT-MATCH, along with other preprocessing tools for **htc-survey** models

`*` indicate folders adapted from **htc-survey**

`**` are folders belonging to SPL

## How to run
1. Install the environment by running
`
cd dsi-nlp-publib/htc-survey-24
conda env create -f environment.yml
`
2. Download and prepare the dataset (e.g. generate_hierarchy.py for BGC) according to **htc-survey** instructions.
Lowercase the `.tax` files using the script `lowercase.py`

For BERT-Naive and BERT-MATCH: modify the config files if needed, then run the corresponding script using

`
python -m src.training_script.flat.bert
`
or 
`
python -m src.training_script.flat.bert_match
`

For SPL-BERT:
3. First generate the level map using 
`
python levels.py
`
Then, generate the multi-hot encoding dictionaries and matrices using 
`
python hierarchy_dict_gen.py
`
4. Delete the corresponding .sdd & .vtree files in `/constraints` (if previously created). Then run the SPL-BERT script:
`
python -m src.training_script.flat.spl-bert
`
The script will automatically output the HTC results files and raw predictions at `/dumps`.

5. Compute the hierarchy violation counts with `python check_pred_y.py` for a single prediction `.csv`, or `python check_pred_y_all.py --dataset [insert] --model [insert]` for each dataset-model combination. Compile the results with `python compile_results.py`. Export them into .txt files by any preferred category via `python load_file.py > [insert_name].txt`

## References

Ahmed, K., Teso, S., Chang, K. W., Van den Broeck, G., & Vergari, A. (2022). Semantic probabilistic layers for neuro-symbolic learning. Advances in Neural Information Processing Systems, 35, 29944-29959.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) (pp. 4171-4186).

Zangari, A., Marcuzzo, M., Rizzo, M., Giudice, L., Albarelli, A., & Gasparetto, A. (2024). Hierarchical text classification and its foundations: A review of current research. Electronics, 13(7), 1199.

**MATCH Loss**: Zhang, Y., Shen, Z., Dong, Y., Wang, K., & Han, J. (2021, April). Match: Metadata-aware text classification in a large hierarchy. In Proceedings of the Web Conference 2021 (pp. 3246-3257).
