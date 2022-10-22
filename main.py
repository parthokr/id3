import csv
import json
from math import log2

import numpy as np
from pygments import highlight, lexers, formatters


class DecisionTree:
    def __init__(self, table):
        if isinstance(table, np.ndarray):
            """ this has been instantiated from recursion """
            self.table = table
        elif isinstance(table, str):
            """ this is the first instantiation by user"""
            data = []
            with open(table) as file:
                reader = csv.reader(file)
                for row in reader:
                    data.append(row)

            self.table = np.array(data)
        else:
            raise Exception("Please provide a valid CSV file path")

        self.entropies = np.array([])
        self.tree = {}
        self.run()  # init building tree by initiating entropies

    def run(self):
        for attr in self.table[0, 0:-1]:
            self.entropies = np.append(self.entropies, self.get_entropy(attr))
        self.build_tree()

    def get_rows(self, category, col_index):
        tmp_table = np.array([self.table[0]])
        for row in self.table:
            if row[col_index] == category:
                tmp_table = np.concatenate((tmp_table, [row]))
        return tmp_table

    def get_entropy(self, attribute):
        """ get all categories of an attribute"""
        col_index_of_attribute = np.where(self.table == attribute)[1][0]
        categories = np.unique(self.table[1: self.table.shape[0], col_index_of_attribute])

        entropy = 0

        for category in categories:
            subtable = self.get_rows(category, col_index_of_attribute)
            # print(subtable)
            x = subtable.shape[0] - 1  # no of rows in subtable
            y = self.table.shape[0] - 1  # no of rows in master table
            # print(x, y)
            prob_of_cat = x / y
            # print(prob_of_cat)
            # print((subtable.shape[0]-1)/(self.table.shape[0] - 1))
            yes_in_subtable = 0
            for row in subtable:
                if row[-1] == "Yes":
                    yes_in_subtable += 1
            no_in_subtable = x - yes_in_subtable
            prob1 = yes_in_subtable / x
            prob2 = no_in_subtable / x
            entropy += -(prob1 * (log2(prob1) if prob1 > 0 else 1) + prob2 * (
                log2(prob2) if prob2 > 0 else 1)) * prob_of_cat

        return entropy

    def build_tree(self):
        if self.entropies.shape[0] == 0:
            return {}
        root_index = np.argmin(self.entropies)
        categories = np.unique(self.table[1: self.table.shape[0], root_index])
        # print(self.entropies)
        for category in categories:
            subtable = self.get_rows(category, root_index)
            # print(subtable)
            yes_no = np.sort(np.unique(np.array([row[-1] for row in self.table[1:]])))  # useful to get over case issue

            # # count frequency of yes and no in last col
            yes_count = list(subtable[:, -1]).count(yes_no[1])
            no_count = list(subtable[:, -1]).count(yes_no[0])
            # print(yes_count)
            # print(no_count)
            if yes_count == subtable.shape[0] - 1:
                self.tree[self.table[0][root_index]] = {
                    **self.tree.get(self.table[0][root_index], {}), **{category: yes_no[1]}
                }
            elif no_count == subtable.shape[0] - 1:
                self.tree[self.table[0][root_index]] = {
                    **self.tree.get(self.table[0][root_index], {}), **{category: yes_no[0]}
                }
            else:
                mod_subtable = np.delete(subtable, root_index, 1)
                _dt = DecisionTree(mod_subtable)
                self.tree[self.table[0][root_index]] = {
                    **self.tree.get(self.table[0][root_index], {}), **{category: _dt.get_tree()}
                }

    def get_tree(self):
        return self.tree

    # def _query(self, obj, parent, tree):
    #     # if not isinstance(tree, dict):
    #     #     print(parent)
    #     #     return tree[parent]
    #
    #     for key in tree:
    #         if isinstance(tree[key], dict):
    #             print(key)
    #             if obj[parent] == key:
    #                 print('Match')
    #                 self._query(obj, key, tree[key])
    #         else:
    #             if obj[parent] == key:
    #                 return tree[parent]
    #
    # def query(self, obj):
    #     return self._query(obj, list(self.tree.keys())[0], self.tree)


dt = DecisionTree(input('Enter CSV path (default "data.csv"): ') or 'data.csv')
# print(json.dumps(dt.get_tree(), indent=4))
colorful_json = highlight(json.dumps(dt.get_tree(), indent=4), lexers.JsonLexer(), formatters.TerminalFormatter())
print(colorful_json)
