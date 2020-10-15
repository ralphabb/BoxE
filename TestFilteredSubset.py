import numpy as np
from KBUtils import load_kb_file
from KBUtils import MissingDict
from NELLProcessing import materialise_simple_hierarchy
import Cnst

test_facts = load_kb_file("Other/DatasetsMulti/NELLRuleInjSplit90Mat/test.kb")
ts_facts_reform = np.zeros_like(test_facts)
ts_facts_reform[:, 0] = test_facts[:, 1]
ts_facts_reform[:, 1] = test_facts[:, 0]
ts_facts_reform[:, 2] = test_facts[:, 2]
ts_facts_reform[:, 3] = test_facts[:, 3]
train_facts = load_kb_file("Other/DatasetsMulti/NELLRuleInjSplit90Mat/train.kb")

print(test_facts.shape)

tr_facts_reform = np.zeros_like(train_facts)
print(tr_facts_reform.shape)
tr_facts_reform[:, 0] = train_facts[:, 1]
tr_facts_reform[:, 1] = train_facts[:, 0]
tr_facts_reform[:, 2] = train_facts[:, 2]
tr_facts_reform[:, 3] = train_facts[:, 3]

print(tr_facts_reform.shape)
mat_tr_facts, new_mats = materialise_simple_hierarchy(tr_facts_reform, "RulesNELL.txt")  
in_training_mat = MissingDict(lambda: False)
for triple in mat_tr_facts:
    in_training_mat[tuple(triple)] = True

in_test_mat = MissingDict(lambda: False)
for triple in tr_facts_reform:
    in_test_mat[tuple(triple)] = True


new_mat_triples = np.array([triple for triple in ts_facts_reform if not in_training_mat[tuple(triple)]])
print(new_mat_triples.shape)
new_mat_triples_ref = np.zeros_like(new_mat_triples)
new_mat_triples_ref[:, 0] = new_mat_triples[:, 1]
new_mat_triples_ref[:, 1] = new_mat_triples[:, 0]
new_mat_triples_ref[:, 2] = new_mat_triples[:, 2]
new_mat_triples_ref[:, 3] = new_mat_triples[:, 3]


np.random.seed(Cnst.DEFAULT_RANDOM_SEED)
np.random.shuffle(new_mat_triples_ref)
number_of_splits = int(np.ceil(new_mat_triples_ref.shape[0] / 15000))
batches = np.array_split(new_mat_triples_ref, number_of_splits)
separated_batches = []
for batch in batches:
    separated_batches.append(batch)
print(len(separated_batches))

in_final_mat = MissingDict(lambda: False)
for triple in new_mat_triples_ref:
    in_final_mat[tuple(triple)] = True

remaining_test = np.array([triple for triple in test_facts if not in_final_mat[tuple(triple)]])
np.save("Other/DatasetsMulti/NELLRuleInjSplit90Mat/test_subset.kbb", separated_batches)

number_of_splits = int(np.ceil(remaining_test.shape[0] / 15000))
batches_2 = np.array_split(remaining_test, number_of_splits)
for batch in batches_2:
    separated_batches.append(batch)
np.save("Other/DatasetsMulti/NELLRuleInjSplit90Mat/test_subset_all.kbb", separated_batches)
print(len(separated_batches))
