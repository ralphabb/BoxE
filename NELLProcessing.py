from collections import Counter
from KBUtils import MissingDict
from RuleParser import RuleParser, get_terminal_nodes
import numpy as np
import Cnst
import os

SELECTED_RELATIONS = [
                      "https://w3id.org/nellrdf/ontology/agentbelongstoorganization",
                      "https://w3id.org/nellrdf/ontology/athletecoach",
                      "https://w3id.org/nellrdf/ontology/teamplaysinleague",
                      "https://w3id.org/nellrdf/ontology/personbelongstoorganization",
                      "https://w3id.org/nellrdf/ontology/athleteplayedforschool",
                      "https://w3id.org/nellrdf/ontology/athleteplaysforteam",
                      "https://w3id.org/nellrdf/ontology/athleteplaysinleague",
                      "https://w3id.org/nellrdf/ontology/coachesinleague",
                      "https://w3id.org/nellrdf/ontology/athleteledsportsteam",
                      "https://w3id.org/nellrdf/ontology/worksfor",
                      "https://w3id.org/nellrdf/ontology/coachesteam"]

SELECTED_SPORTS_RELATIONS = ["https://w3id.org/nellrdf/ontology/athletecoach",
                      "https://w3id.org/nellrdf/ontology/teamplaysinleague",
                      "https://w3id.org/nellrdf/ontology/athleteplayedforschool",
                      "https://w3id.org/nellrdf/ontology/athleteplaysforteam",
                      "https://w3id.org/nellrdf/ontology/athleteplaysinleague",
                      "https://w3id.org/nellrdf/ontology/coachesinleague",
                      "https://w3id.org/nellrdf/ontology/athleteledsportsteam",
                      "https://w3id.org/nellrdf/ontology/coachesteam"]

SELECTED_RELATIONS_WITH_TAGS = ["<"+x+">" for x in SELECTED_RELATIONS]
SELECTED_SPORTS_RELATIONS_WITH_TAGS = ["<"+x+">" for x in SELECTED_SPORTS_RELATIONS]
SELECTED_REL_DICT = {rel: idx for idx, rel in enumerate(SELECTED_RELATIONS_WITH_TAGS)}
NELL_FILE = "NELL2RDF_0.3_1100.nt"


def parse_ntriple(input_ntriple: str):
    split_triple = input_ntriple.strip().split(" ")
    if len(split_triple) > 4:  
        ent_2 = " ".join(split_triple[2:-1])
    else:
        ent_2 = split_triple[2]
    ent_1 = split_triple[0]
    relation = split_triple[1]
    return ent_1, relation, ent_2  


def materialise_simple_hierarchy_prep(rule_file):  
    
    rules = RuleParser(rule_file, enforce_same=False).get_parsed_rules()
    
    rule_lhs_terminals = [get_terminal_nodes(rule[0]) for rule in rules]
    rule_rhs_terminals = [get_terminal_nodes(rule[1]) for rule in rules]
    rule_implications = np.array([rule[2] == Cnst.IMPLIES for rule in rules])
    nb_lhs_terminals = np.array([len(x) for x in rule_lhs_terminals])
    nb_rhs_terminals = np.array([len(x) for x in rule_rhs_terminals])
    if np.amax(nb_lhs_terminals) > 1 or np.amax(nb_rhs_terminals) > 1:
        print("One or more rules consists of more than a simple atom on more than one side")
        return
    if not np.all(rule_implications):
        print("Equivalence Rules Provided. Please re-write these as implications and try again")
        return
    
    
    lhs_to_rhs_rel = [(rule[0].rel, rule[1].rel) for rule in rules]
    return lhs_to_rhs_rel


def materialise_simple_hierarchy_step(numeric_triples_np, lhs_to_rhs_rel):
    triples_to_inject = [numeric_triples_np]
    for old_rel, new_rel in lhs_to_rhs_rel:
        relevant_triples = numeric_triples_np[numeric_triples_np[:, 1] == old_rel]
        new_triples = np.copy(relevant_triples)
        new_triples[:, 1] = new_rel
        triples_to_inject.append(new_triples)
    all_triples = np.concatenate(triples_to_inject, axis=0)
    mat_triples = all_triples[numeric_triples_np.shape[0]:, :]
    return np.unique(all_triples, axis=0), mat_triples


def materialise_simple_hierarchy(numeric_triples_np, rule_file):  
    numeric_triples_np = np.array(numeric_triples_np)  
    lhs_to_rhs_rel = materialise_simple_hierarchy_prep(rule_file)
    step_count = 0  
    current_triples_np = numeric_triples_np
    new_triples = None
    while True:
        old_nb_facts = current_triples_np.shape[0]
        current_triples_np, new_trip = materialise_simple_hierarchy_step(current_triples_np, lhs_to_rhs_rel)
        step_count += 1
        if step_count == 1:
            new_triples = new_trip
        else:
            new_triples = np.concatenate([new_triples, new_trip], axis=0)
        new_nb_facts = current_triples_np.shape[0]
        if new_nb_facts == old_nb_facts:
            break
    return current_triples_np, np.unique(new_triples, axis=0)  


def count_entities_in_triples(triples):
    original_list = [x[0] for x in triples] + [x[2] for x in triples]
    return Counter(original_list)


def get_entities_in_triples(triples, relation_subset=None, unique=True, return_dict=True):
    if relation_subset is None:
        original_list = [x[0] for x in triples] + [x[2] for x in triples]  
    else:
        original_list = [x[0] for x in triples if x[1] in relation_subset] + [x[2] for x in triples if
                                                                              x[1] in relation_subset]
    if unique:
        original_list = list(set(original_list))
    if return_dict:
        m_dict = MissingDict(lambda: False)
        for ent in original_list:
            m_dict[ent] = True
        return original_list, m_dict
    else:
        return original_list


def map_triples(triples, ent_dict):  
    if ent_dict is None:  
        ent_dict = map_entities_to_idx(triples)
    return np.array([(ent_dict[triple[0]], SELECTED_REL_DICT[triple[1]], ent_dict[triple[2]]) for triple in triples])


def map_entities_to_idx(triples):
    original_list = sorted(get_entities_in_triples(triples, unique=True, return_dict=False))
    return {ent: idx for idx, ent in enumerate(original_list)}


def get_facts_with_relation(selected_relations, triple_store):
    retained_triples = []
    with open(triple_store, "r") as nell:
        line = nell.readline()
        line_count = 1
        while line:
            ent1, rel, ent2 = parse_ntriple(line)
            line = nell.readline()
            line_count += 1
            if rel in selected_relations:
                retained_triples.append((ent1, rel, ent2))
    nell.close()
    return retained_triples


def prepare_dataset(materialise=True, rule_dir="RulesNELLPreMap.txt", sel_rel=None, nell=NELL_FILE, pre_loaded_triples=None,
                    train_split=0.9, val_split=0.95, rotatefriendly=True, name="NELLHierarchies", mat_split=False):
    if sel_rel is None:  
        sel_rel = SELECTED_RELATIONS_WITH_TAGS
    if pre_loaded_triples is None:  
        retained_triples = get_facts_with_relation(selected_relations=sel_rel, triple_store=nell)
    else:
        retained_triples = pre_loaded_triples
    
    sports_entities, m_dict = get_entities_in_triples(retained_triples, SELECTED_SPORTS_RELATIONS_WITH_TAGS,
                                                      unique=True, return_dict=True)
    surviving_triples = [triple for triple in retained_triples if m_dict[triple[0]] and m_dict[triple[2]]]
    counter = count_entities_in_triples(surviving_triples)  
    filtered_triples = [triple for triple in surviving_triples if
                        (counter[triple[0]] >= 50 and counter[triple[2]] >= 50) or triple[1]
                        == SELECTED_RELATIONS_WITH_TAGS[4]]
    ent_dict = map_entities_to_idx(filtered_triples)
    mapped_triples = map_triples(filtered_triples, ent_dict=ent_dict)  
    write_dir = Cnst.DEFAULT_KB_DIR + name
    if materialise:  
        mapped_triples_2, mat_triples = materialise_simple_hierarchy(mapped_triples, rule_dir)
        write_dir = write_dir + "Mat"
    else:
        mapped_triples_2 = mapped_triples
        mat_triples = None
    write_dir = write_dir + "/"
    if mat_split:
        
        
        in_training = MissingDict(lambda: False)
        for triple in mapped_triples:
            in_training[tuple(triple)] = True
        new_mat_triples = np.array([triple for triple in mat_triples if not in_training[tuple(triple)]])
        np.random.shuffle(mapped_triples)  
        np.random.shuffle(new_mat_triples)  
        even_mat_split = new_mat_triples.shape[0] // 2  
        train_idx = int(train_split * mapped_triples.shape[0])
        val_idx = int(val_split * mapped_triples.shape[0])
        train_triples = mapped_triples[:train_idx, :]
        if train_split < 1:
            val_triples = np.concatenate([mapped_triples[train_idx:val_idx, :], new_mat_triples[:even_mat_split, :]],
                                         axis=0)
        else:
            val_triples = new_mat_triples[:even_mat_split, :]
        if train_split < 1 and val_split < 1:
            test_triples = np.concatenate([mapped_triples[val_idx:, :], new_mat_triples[even_mat_split:, :]],
                                          axis=0)
        else:
            test_triples = new_mat_triples[even_mat_split:, :]
    else:  
        np.random.shuffle(mapped_triples_2)  
        train_idx = int(train_split * mapped_triples_2.shape[0])
        val_idx = int(val_split * mapped_triples_2.shape[0])
        train_triples = mapped_triples_2[:train_idx, :]
        val_triples = mapped_triples_2[train_idx:val_idx, :]
        test_triples = mapped_triples_2[val_idx:, :]

    if not os.path.exists(Cnst.DEFAULT_KB_DIR):
        os.mkdir(Cnst.DEFAULT_KB_DIR)
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    np.savetxt(os.path.join(write_dir, "train.txt"), train_triples, delimiter='\t', fmt='%i')
    np.savetxt(os.path.join(write_dir, "valid.txt"), val_triples, delimiter='\t', fmt='%i')
    np.savetxt(os.path.join(write_dir, "test.txt"), test_triples, delimiter='\t', fmt='%i')
    if rotatefriendly:  
        with open(os.path.join(write_dir, "entities.dict"), "w") as f:
            lines = [str(x)+"\t"+str(x)+"\r\n" for x in range(len(ent_dict))]
            f.writelines(lines)
            f.close()
        with open(os.path.join(write_dir, "relations.dict"), "w") as f:
            lines = [str(x) + "\t" + str(x) + "\r\n" for x in range(len(SELECTED_RELATIONS))]
            f.writelines(lines)
            f.close()


if __name__ == "__main__":
    retained_triples = get_facts_with_relation(SELECTED_RELATIONS_WITH_TAGS, NELL_FILE)
    prepare_dataset(pre_loaded_triples=retained_triples, train_split=0.9, val_split=0.95,
                    name="NELLRuleInjSplit90", mat_split=True)


