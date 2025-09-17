import copy
import json
import os
from collections import defaultdict

from src.models.HiAGM.models.structure_model.tree import Tree

ROOT_LABEL = 'root'


class DatasetStatistic(object):
    def __init__(self, config):
        """
        class for prior probability
        :param config: helper.configure, Configure object
        """
        super(DatasetStatistic, self).__init__()
        self.config = config
        self.root = Tree('root')
        # label_tree => root
        self.label_trees = dict()
        self.label_trees[ROOT_LABEL] = self.root
        self.hierarchical_label_dict, self.label_vocab = self.get_hierar_relations_with_name(
            os.path.join(config.data.data_dir, config.data.hierarchy))
        self.level = 0
        self.level_dict = dict()
        self.init_prior_prob_dict = dict()

        # build tree structure for treelstm
        for parent in list(self.hierarchical_label_dict.keys()):
            assert parent in self.label_trees.keys()  # Check parent has been added to label trees (initially it's only root)
            parent_tree = self.label_trees[parent]  # Retrieve the tree
            self.init_prior_prob_dict[parent] = dict()  # set prior prob of parent to empty dict
            # Go through children of this parent
            for child in self.hierarchical_label_dict[parent]:
                assert child not in self.label_trees.keys()  # Check children has not a tree yet
                self.init_prior_prob_dict[parent][child] = 0  # Set prior prob of children to 0
                child_tree = Tree(child)  # Create tree
                parent_tree.add_child(child_tree)  # Add child tree to parent tree
                self.label_trees[child] = child_tree  # Assign label tree
        self.prior_prob_dict = copy.deepcopy(self.init_prior_prob_dict)
        self.total_train_prob_dict = copy.deepcopy(self.init_prior_prob_dict)

        # for label in self.hierarchical_label_dict.keys():
        #     print(label, len(self.hierarchical_label_dict[label]))
        #     print(self.hierarchical_label_dict[label])
        for label in self.label_vocab:
            label_depth = self.label_trees[label].depth()
            if label_depth not in self.level_dict.keys():
                self.level_dict[label_depth] = [label]
            else:
                self.level_dict[label_depth].append(label)

        # print(self.level)
        # print(self.level_dict)
        # for i in self.level_dict.keys():
        #     print(i, len(set(self.level_dict[i])))

    def get_taxonomy_file(self):
        """

        :return:
        """
        with open(os.path.join(self.config.data.data_dir, self.config.data.hierarchy), 'r') as file:
            data = file.readlines()
        hierarcy_dict = dict()
        for line in data:
            line = line.rstrip('\n')
            p = line.split('parent: ')[1].split(' ')[0]
            c = line.split('child: ')[1].split(' ')[0]
            if p not in hierarcy_dict.keys():
                hierarcy_dict[p] = [c]
            else:
                hierarcy_dict[p].append(c)
        print(hierarcy_dict)
        known_label = ['root']
        output_lines = []
        while len(known_label):
            output_lines.append([known_label[0]] + hierarcy_dict[known_label[0]])
            for i in hierarcy_dict[known_label[0]]:
                if i in hierarcy_dict.keys():
                    known_label.append(i)
            known_label = known_label[1:]
        print(output_lines)
        file = open(os.path.join(self.config.data.data_dir, self.config.data.hierarchy), 'w')
        for i in output_lines:
            file.write('\t'.join(i) + '\n')
        file.close()

    @staticmethod
    def get_hierar_relations_with_name(taxo_file_dir):
        parent_child_dict = dict()
        label_vocab = []
        with open(taxo_file_dir, 'r') as f:
            relation_data = f.readlines()
        for relation in relation_data:
            # relation_list = relation.split()
            relation_list = relation.rstrip('\n').split()
            parent, *children = relation_list
            assert parent not in parent_child_dict.keys()
            parent_child_dict[parent] = children
            # label_vocab.extend(children)
            # label_vocab.append(parent)
            label_vocab.extend(relation_list)
        # print(parent_child_dict)
        return parent_child_dict, set(label_vocab)

    # def get_data_statistic(self, file_name):
    #     # Number of all labels
    #     overall_label_number: int = 0
    #     label_num_dict: Dict = dict()
    #
    #     with open(file_name, 'r') as f:
    #         data = f.readlines()
    #
    #     data_length: int = len(data)
    #     sample_count_not_to_end: int = 0
    #     path_count_not_to_end: int = 0
    #     doc_length_all: int = 0
    #     label_doc_len_dict = dict()
    #     level_num_dict = defaultdict(int)
    #     for i in range(self.level + 1):
    #         level_num_dict[i] = 0
    #     prob_dict = copy.deepcopy(self.init_prior_prob_dict)
    #
    #     for sample in data:
    #         sample_flag: bool = False
    #         sample: Dict = json.loads(sample)
    #         sample_label_list: List[str] = sample['label']
    #         # Add length of label list to overall number
    #         overall_label_number += len(sample_label_list)
    #         # Add number of tokens to overall document length
    #         doc_length_all += len(sample['token'])
    #         # sample label : list of labels
    #         for label in sample_label_list:
    #             path_flag: bool = False
    #             # print("----------")
    #             print(label)
    #             print(self.label_vocab)
    #             assert label in self.label_vocab
    #             level_num_dict[self.label_trees[label]._depth] += 1
    #             if label in self.init_prior_prob_dict.keys():
    #                 # raise NotImplementedError
    #                 if label in self.hierarchical_label_dict[ROOT_LABEL]:
    #                     prob_dict[ROOT_LABEL][label] += 1
    #                     self.prior_prob_dict[ROOT_LABEL][label] += 1
    #                     if 'train' in file_name or 'val' in file_name:
    #                         self.total_train_prob_dict[ROOT_LABEL][label] += 1
    #
    #                 for c in self.init_prior_prob_dict[label].keys():
    #                     if c in sample_label_list:
    #                         prob_dict[label][c] += 1
    #                         self.prior_prob_dict[label][c] += 1
    #                         if 'train' in file_name or 'val' in file_name:
    #                             self.total_train_prob_dict[label][c] += 1
    #
    #             if label not in label_num_dict:
    #                 label_num_dict[label] = 1
    #                 label_doc_len_dict[label] = len(sample['token'])
    #             else:
    #                 label_num_dict[label] += 1
    #                 label_doc_len_dict[label] += len(sample['token'])
    #
    #             if self.label_trees[label].num_children > 0 and not (sample_flag and path_flag):
    #                 # flag = False
    #                 for child in self.label_trees[label].children:
    #                     if child.idx in sample_label_list:
    #                         sample_flag = True
    #                         path_flag = True
    #
    #                 if not path_flag:
    #                     path_count_not_to_end += 1
    #                 if not sample_flag:
    #                     sample_count_not_to_end += 1
    #                     # print(sample)
    #     avg_label_num = float(overall_label_number) / data_length
    #     avg_doc_len = float(doc_length_all) / data_length
    #
    #     for label in self.label_vocab:
    #         if label not in label_doc_len_dict.keys():
    #             label_doc_len_dict[label] = 0.0
    #         else:
    #             label_doc_len_dict[label] = float(label_doc_len_dict[label]) / label_num_dict[label]
    #
    #     return {
    #         'num_of_samples': data_length,
    #         'average_label_num_per_sample': avg_label_num,
    #         'average_doc_length_per_sample': avg_doc_len,
    #         'label_num_dict': label_num_dict,
    #         'average_doc_length_per_label': label_doc_len_dict,
    #         'sample_end_before_leaf_nodes': sample_count_not_to_end,
    #         'path_end_before_leaf_nodes': path_count_not_to_end,
    #         'level_sample_number': level_num_dict,
    #         'prob_dict': prob_dict
    #     }
    #

    def get_data_statistic(self, file_name):
        all_label_num = 0
        label_num_dict = dict()
        f = open(file_name, 'r')
        data = f.readlines()
        f.close()
        count_data = len(data)
        sample_count_not_to_end = 0
        path_count_not_to_end = 0
        doc_length_all = 0
        label_doc_len_dict = dict()
        level_num_dict = defaultdict(int)
        for i in range(self.level + 1):
            level_num_dict[i] = 0
        prob_dict = copy.deepcopy(self.init_prior_prob_dict)

        for sample in data:
            sample_flag = False
            sample = json.loads(sample)
            sample_label = sample['label']
            all_label_num += len(sample_label)
            doc_length_all += len(sample['token'])
            # sample label : list of labels
            for label in sample_label:
                path_flag = False
                assert label in self.label_vocab
                level_num_dict[self.label_trees[label]._depth] += 1
                if label in self.init_prior_prob_dict.keys():
                    if label in self.hierarchical_label_dict[ROOT_LABEL]:
                        prob_dict[ROOT_LABEL][label] += 1
                        self.prior_prob_dict[ROOT_LABEL][label] += 1
                        if 'train' in file_name or 'val' in file_name:
                            self.total_train_prob_dict[ROOT_LABEL][label] += 1

                    for c in self.init_prior_prob_dict[label].keys():
                        if c in sample_label:
                            prob_dict[label][c] += 1
                            self.prior_prob_dict[label][c] += 1
                            if 'train' in file_name or 'val' in file_name:
                                self.total_train_prob_dict[label][c] += 1

                if label not in label_num_dict:
                    label_num_dict[label] = 1
                    label_doc_len_dict[label] = len(sample['token'])
                else:
                    label_num_dict[label] += 1
                    label_doc_len_dict[label] += len(sample['token'])

                if self.label_trees[label].num_children > 0 and not (sample_flag and path_flag):
                    # flag = False
                    for child in self.label_trees[label].children:
                        if child.idx in sample_label:
                            sample_flag = True
                            path_flag = True

                    if not path_flag:
                        path_count_not_to_end += 1
                    if not sample_flag:
                        sample_count_not_to_end += 1
                        # print(sample)
        avg_label_num = float(all_label_num) / count_data
        avg_doc_len = float(doc_length_all) / count_data

        for label in self.label_vocab:
            if label not in label_doc_len_dict.keys():
                label_doc_len_dict[label] = 0.0
            else:
                label_doc_len_dict[label] = float(label_doc_len_dict[label]) / label_num_dict[label]

        return {
            'num_of_samples': count_data,
            'average_label_num_per_sample': avg_label_num,
            'average_doc_length_per_sample': avg_doc_len,
            'label_num_dict': label_num_dict,
            'average_doc_length_per_label': label_doc_len_dict,
            'sample_end_before_leaf_nodes': sample_count_not_to_end,
            'path_end_before_leaf_nodes': path_count_not_to_end,
            'level_sample_number': level_num_dict,
            'prob_dict': prob_dict
        }


def prior_probs(prob_dict):
    for p in prob_dict.keys():
        total_sum = 0
        for c in prob_dict[p]:
            total_sum += prob_dict[p][c]
        if total_sum:
            for c in prob_dict[p]:
                prob_dict[p][c] = float(prob_dict[p][c]) / total_sum
    return prob_dict


def generate_tree_stat(configs):
    dataset_statistic = DatasetStatistic(configs)
    # rcv1_dataset_statistic.get_taxonomy_file()
    train_statistics = dataset_statistic.get_data_statistic(
        os.path.join(configs.data.data_dir, configs.data.train_file))
    val_statistics = dataset_statistic.get_data_statistic(
        os.path.join(configs.data.data_dir, configs.data.val_file))
    test_statistics = dataset_statistic.get_data_statistic(
        os.path.join(configs.data.data_dir, configs.data.test_file))

    print('*****TRAIN PRIOR PROBABILITIES *****')
    print(train_statistics)
    print(prior_probs(train_statistics['prob_dict']))
    print('*****val PRIOR PROBABILITIES *****')
    print(val_statistics)
    print(prior_probs(val_statistics['prob_dict']))
    print('*****TEST PRIOR PROBABILITIES *****')
    print(test_statistics)
    print(prior_probs(test_statistics['prob_dict']))

    # check_rcv1_level_data()
    print('*****TOTAL PRIOR PROBABILITIES *****')
    print(dataset_statistic.prior_prob_dict)
    print(prior_probs(dataset_statistic.prior_prob_dict))
    print('*****TOTAL TRAIN PRIOR PROBABILITIES *****')
    print(dataset_statistic.total_train_prob_dict)
    print(prior_probs(dataset_statistic.total_train_prob_dict))
    train_probs = prior_probs(dataset_statistic.total_train_prob_dict)
    with open(os.path.join(configs.data.data_dir, configs.data.prob_json), 'w') as json_file:
        json_str = json.dumps(train_probs)
        json_file.write(json_str)
