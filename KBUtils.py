
import msgpack
import msgpack_numpy as m
import numpy as np
import os
import Cnst
import tensorflow as tf
from collections import defaultdict
m.patch()  

ENTITY_PADDING = "pad_entity_boxemulti183729"
MAX_ARITIES = {"JF17K": 6, "FB-AUTO": 5, "FB15k-237": 2, "WN18RR": 2,
               "YAGO3-10": 2, "NELLRuleInjSplit90Mat": 2}


class MissingDict(defaultdict):  
    def __missing__(self, key):
        return self.default_factory()


def prepare_eval_dataset(ds_np_arr, nb_ent, max_ar):
    nb_atoms = ds_np_arr.shape[0]
    ds_np_arr_extended = np.tile(ds_np_arr, (nb_ent * max_ar, 1))  
    for i in range(max_ar):
        pad_idx = ds_np_arr_extended[nb_ent * nb_atoms * i: nb_ent * nb_atoms * (i + 1), 1 + i] == nb_ent
        replacement = np.repeat(np.arange(nb_ent), nb_atoms)
        replacement[pad_idx] = nb_ent
        ds_np_arr_extended[nb_ent * nb_atoms * i: nb_ent * nb_atoms * (i + 1), 1 + i] = replacement
    final_ds = ds_np_arr_extended
    return final_ds


def parse_kb_fact(fact, max_arity=6):  
    components = fact.strip().split("\t")
    nb_entities = len(components) - 1
    components.extend([ENTITY_PADDING] * (max_arity - nb_entities) + [1])  
    return components  



def get_value(components):
    if len(components) >= 4:
        val = int(components[3])
    else:
        val = 1
    return val


def parse_kb_fact_htr(fact):
    
    components = fact.strip().split("\t")
    head, tail, relation = components[0], components[1], components[2]
    val = get_value(components)  
    return [relation, head, tail, val]


def parse_kb_fact_hrt(fact):
    
    components = fact.strip().split("\t")
    head, relation, tail = components[0], components[1], components[2]
    val = get_value(components)
    return [relation, head, tail, val]


def compute_kb_id_mapping_adapter(kb_directory=Cnst.DEFAULT_KB_DIR, kb_multi_directory=Cnst.DEFAULT_KB_MULTI_DIR,
                                  file_to_read="train.txt", use_eval_data=False):
    path_to_e2id = os.path.join(kb_multi_directory, Cnst.ENT_2_ID_DICT_NAME)
    path_to_r2id = os.path.join(kb_multi_directory, Cnst.REL_2_ID_DICT_NAME)
    kb_name = os.path.basename(os.path.normpath(kb_directory))
    if not os.path.exists(kb_multi_directory):
        os.mkdir(kb_multi_directory)
    if use_eval_data:
        files = ["train.txt", "test.txt", "valid.txt"]
    else:
        files = [file_to_read]
    entities = []
    relations = []
    for file in files:
        path_to_file = os.path.join(kb_directory, file)
        kb = open(path_to_file, "r")
        if kb_name in Cnst.HTR_KBs:
            parsing_function = parse_kb_fact_htr
        else:
            parsing_function = parse_kb_fact_hrt
        for index, fact in enumerate(kb.readlines()):
            components = parsing_function(fact)  
            entities.extend(components[1:-1])  
            relations.append(components[0])  
    
    entities_distinct = sorted(list(set(entities)))
    relations_distinct = sorted(list(set(relations)))  
    ent_id_dict = {entity: index for index, entity in enumerate(entities_distinct)}
    rel_id_dict = {relation: index for index, relation in enumerate(relations_distinct)}
    
    ent_id_dict[ENTITY_PADDING] = len(ent_id_dict) - 1  
    
    with open(path_to_e2id, 'wb') as f:
        msgpack.pack(ent_id_dict, f)
    with open(path_to_r2id, 'wb') as f:
        msgpack.pack(rel_id_dict, f)


def compute_kb_id_mapping(kb_directory=Cnst.DEFAULT_KB_MULTI_DIR, max_arity=6, file_to_read="train.txt",
                          use_eval_data=False):
    
    path_to_e2id = os.path.join(kb_directory, Cnst.ENT_2_ID_DICT_NAME)
    path_to_r2id = os.path.join(kb_directory, Cnst.REL_2_ID_DICT_NAME)
    if use_eval_data:
        files = ["train.txt", "test.txt", "valid.txt"]
    else:
        files = [file_to_read]
    entities = []
    relations = []
    for file in files:
        path_to_file = os.path.join(kb_directory, file)
        kb = open(path_to_file, "r")
        for index, fact in enumerate(kb.readlines()):
            components = parse_kb_fact(fact, max_arity=max_arity)  
            entities.extend(components[1:-1])  
            relations.append(components[0])  
    
    entities_distinct = list(set(entities))
    relations_distinct = list(set(relations))
    ent_id_dict = {entity: index for index, entity in enumerate(entities_distinct)}
    rel_id_dict = {relation: index for index, relation in enumerate(relations_distinct)}
    
    ent_id_dict[ENTITY_PADDING] = len(ent_id_dict) - 1  
    
    with open(path_to_e2id, 'wb') as f:
        msgpack.pack(ent_id_dict, f)
    with open(path_to_r2id, 'wb') as f:
        msgpack.pack(rel_id_dict, f)


def load_kb_file(kb_file_path):
    with open(kb_file_path, "rb") as f:  
        kb_file = msgpack.unpack(f, encoding="utf-8")
    np_kb_file = np.array(kb_file, dtype=np.int32)  
    return np_kb_file


def load_kb_metadata_multi(kb_name):
    with open(Cnst.KB_META_MULTI_FILE_NAME, 'rb') as f:
        metadata_dict = msgpack.unpack(f, encoding="utf-8")
        try:
            return metadata_dict[kb_name]
        except KeyError:
            print("No KB named "+str(kb_name)+" in the default KB folder during metadata extraction")
            return


def adapt_kbs_binary(kb_directory=Cnst.DEFAULT_KB_DIR, kb_multi_directory=Cnst.DEFAULT_KB_MULTI_DIR,
                     tr_batch_size=1024, tst_batch_size=1024, verbose=False, use_eval_data=False):
    knowledge_bases = [f for f in os.listdir(kb_directory) if not f.startswith('.')]
    for kb in knowledge_bases:
        if verbose:
            print("Processing KB: "+str(kb))
        individual_kb_path = os.path.join(kb_directory, kb)
        destination_kb_path = os.path.join(kb_multi_directory, kb)
        arity = 2  
        compute_kb_id_mapping_adapter(individual_kb_path, kb_multi_directory=destination_kb_path,
                                      use_eval_data=use_eval_data)  
        kb_files = ["train.txt", "test.txt", "valid.txt"]
        for index, kb_file in enumerate(kb_files):  
            try:
                convert_kb_to_id_rep_multi_adapter(individual_kb_path, file_to_read=kb_file, max_arity=arity)
                kb_file_no_ext = os.path.splitext(kb_file)[0]
                if index > 0:  
                    convert_id_representation_to_batches(destination_kb_path, file_to_convert=kb_file_no_ext +
                                                                                             Cnst.KB_FORMAT,
                                                         batch_size=tst_batch_size)
                else:
                    convert_id_representation_to_batches(destination_kb_path, file_to_convert=kb_file_no_ext +
                                                                                             Cnst.KB_FORMAT,
                                                         batch_size=tr_batch_size)

            except KeyError:
                print("Error Converting file " + str(kb_file) + ": Contains Entities / Relations Outside Training Set")
    
    compute_kb_metadata(kb_directory)


def prepare_kbs_multi(kb_directory=Cnst.DEFAULT_KB_MULTI_DIR, tr_batch_size=1024, tst_batch_size=1024,
                      verbose=False, use_eval_data=False):
    knowledge_bases = [f for f in os.listdir(kb_directory) if not f.startswith('.')
                       and MAX_ARITIES[f] > 2]  
    
    for kb in knowledge_bases:
        if verbose:
            print("Processing KB: "+str(kb))
        individual_kb_path = os.path.join(kb_directory, kb)
        arity = MAX_ARITIES[kb]
        compute_kb_id_mapping(individual_kb_path, use_eval_data=use_eval_data, max_arity=arity)  
        kb_files = ["train.txt", "test.txt", "valid.txt"]
        for index, kb_file in enumerate(kb_files):  
            try:
                convert_kb_to_id_rep_multi(individual_kb_path, file_to_read=kb_file, max_arity=arity)
                kb_file_no_ext = os.path.splitext(kb_file)[0]
                if index > 0:  
                    convert_id_representation_to_batches(individual_kb_path, file_to_convert=kb_file_no_ext +
                                                         Cnst.KB_FORMAT, batch_size=tst_batch_size)
                else:
                    convert_id_representation_to_batches(individual_kb_path, file_to_convert=kb_file_no_ext +
                                                         Cnst.KB_FORMAT, batch_size=tr_batch_size)

            except KeyError:
                print("Error Converting file "+str(kb_file)+": Contains Entities / Relations Outside Training Set")
    
    compute_kb_metadata(kb_directory)



def convert_id_representation_to_batches(kb_directory, batch_size=15000,
                                         file_to_convert="train"+Cnst.KB_FORMAT, random_seed=Cnst.DEFAULT_RANDOM_SEED):
    
    path_to_kb_file = os.path.join(kb_directory, file_to_convert)
    path_to_kb_batch_file = os.path.join(kb_directory, os.path.splitext(file_to_convert)[0] + Cnst.KBB_FORMAT)
    with open(path_to_kb_file, "rb") as f:  
        kb_file = msgpack.unpack(f, encoding="utf-8")
    np_kb_file = np.array(kb_file)  
    if random_seed is not None:  
        np.random.seed(random_seed)
    np.random.shuffle(np_kb_file)
    number_of_splits = int(np.ceil(np_kb_file.shape[0]/batch_size))
    batches = np.array_split(np_kb_file, number_of_splits)
    separated_batches = []
    for batch in batches:
        separated_batches.append(batch)
    np.save(path_to_kb_batch_file, separated_batches)  
    return separated_batches


def compute_kb_metadata(kb_directory=Cnst.DEFAULT_KB_MULTI_DIR):  
    knowledge_bases = [f for f in os.listdir(kb_directory) if not f.startswith('.')]
    metadata_dict = {}  
    for kb in knowledge_bases:
        e2id_dict, r2id_dict = load_kb_dicts(kb)
        metadata_dict[kb] = [len(e2id_dict) - 1, len(r2id_dict), MAX_ARITIES[kb]]  
    with open(Cnst.KB_META_MULTI_FILE_NAME, "wb") as f:
        msgpack.pack(metadata_dict, f)


def compute_all_kb_id_mappings(kb_directory=Cnst.DEFAULT_KB_DIR):
    knowledge_bases = [f for f in os.listdir(kb_directory) if not f.startswith('.')]
    for kb in knowledge_bases:
        compute_kb_id_mapping(os.path.join(kb_directory, kb))


def load_kb_dicts(kb_name):
    kb_directory = os.path.join(Cnst.DEFAULT_KB_MULTI_DIR, kb_name)
    path_to_e2id = os.path.join(kb_directory, Cnst.ENT_2_ID_DICT_NAME)
    path_to_r2id = os.path.join(kb_directory, Cnst.REL_2_ID_DICT_NAME)
    
    try:
        with open(path_to_e2id, 'rb') as f:
            e2id_dict = msgpack.unpack(f, encoding="utf-8")
        with open(path_to_r2id, 'rb') as f:
            r2id_dict = msgpack.unpack(f, encoding="utf-8")
        return e2id_dict, r2id_dict
    except FileNotFoundError:  
        max_arity = MAX_ARITIES[kb_name]
        compute_kb_id_mapping(kb_directory, file_to_read="train.txt", max_arity=max_arity)  
        return load_kb_dicts(kb_directory)  


def create_kb_filter_tf(kb_name):
    path_to_kb = os.path.join(Cnst.DEFAULT_KB_MULTI_DIR, kb_name)
    path_to_kb_train = os.path.join(path_to_kb, "train"+Cnst.KB_FORMAT)
    path_to_kb_valid = os.path.join(path_to_kb, "valid" + Cnst.KB_FORMAT)
    path_to_kb_test = os.path.join(path_to_kb, "test" + Cnst.KB_FORMAT)
    with open(path_to_kb_train, "rb") as f:
        train_facts = np.array(msgpack.unpack(f, encoding="utf-8"))[:, :-1]
    with open(path_to_kb_valid, "rb") as f:
        valid_facts = np.array(msgpack.unpack(f, encoding="utf-8"))[:, :-1]
    with open(path_to_kb_test, "rb") as f:
        test_facts = np.array(msgpack.unpack(f, encoding="utf-8"))[:, :-1]
    all_facts = np.concatenate([train_facts, valid_facts, test_facts], axis=0).tolist()
    all_facts_string_entries = [[str(x) for x in fact] for fact in all_facts]
    all_facts_str = np.array([Cnst.FACT_DELIMITER.join(fact) for fact in all_facts_string_entries])
    values_tensor = tf.ones(all_facts_str.shape)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(all_facts_str, values_tensor), 0)
    return table


def create_kb_filter_multi(kb_name):
    path_to_kb = os.path.join(Cnst.DEFAULT_KB_MULTI_DIR, kb_name)
    kb_triple_existence_dict = MissingDict(lambda: True)  
    path_to_kb_train = os.path.join(path_to_kb, "train"+Cnst.KB_FORMAT)
    path_to_kb_valid = os.path.join(path_to_kb, "valid" + Cnst.KB_FORMAT)
    path_to_kb_test = os.path.join(path_to_kb, "test" + Cnst.KB_FORMAT)

    with open(path_to_kb_train, "rb") as f:  
        train_triples = msgpack.unpack(f, encoding="utf-8")
    for triple in train_triples:
        kb_triple_existence_dict[(*triple[:-1],)] = False  
    
    with open(path_to_kb_valid, "rb") as f:  
        valid_triples = msgpack.unpack(f, encoding="utf-8")
    for triple in valid_triples:
        kb_triple_existence_dict[(*triple[:-1], )] = False
    
    with open(path_to_kb_test, "rb") as f:  
        test_triples = msgpack.unpack(f, encoding="utf-8")
    for triple in test_triples:
        kb_triple_existence_dict[(*triple[:-1], )] = False

    return kb_triple_existence_dict


def convert_kb_to_id_rep_multi_adapter(kb_directory, kb_multi_directory=Cnst.DEFAULT_KB_MULTI_DIR,
                                       file_to_read="train.txt", max_arity=2):
    
    path_to_file = os.path.join(kb_directory, file_to_read)
    kb_name = os.path.basename(os.path.normpath(kb_directory))
    if not os.path.exists(kb_multi_directory):
        os.mkdir(kb_multi_directory)
    path_to_kb_idrep = os.path.join(os.path.join(kb_multi_directory, kb_name)
                                    , os.path.splitext(file_to_read)[0] + Cnst.KB_FORMAT)
    e2id_dict, r2id_dict = load_kb_dicts(kb_name)
    
    kb = open(path_to_file, "r")
    facts = []
    for fact in kb.readlines():
        if kb_name in Cnst.HTR_KBs:
            parsing_function = parse_kb_fact_htr
        else:
            parsing_function = parse_kb_fact_hrt
        fact_cmpnts = parsing_function(fact)
        fact_comp_ids = [r2id_dict[cmpnt] if idx == 0 else e2id_dict[cmpnt] if idx < max_arity + 1 else cmpnt
                         for idx, cmpnt in enumerate(fact_cmpnts)]
        facts.append(fact_comp_ids)
    with open(path_to_kb_idrep, 'wb') as f:  
        msgpack.pack(facts, f)


def convert_kb_to_id_rep_multi(kb_directory, file_to_read="train.txt", max_arity=6):
    
    path_to_file = os.path.join(kb_directory, file_to_read)
    path_to_kb_idrep = os.path.join(kb_directory, os.path.splitext(file_to_read)[0] + Cnst.KB_FORMAT)  
    kb_name = os.path.basename(os.path.normpath(kb_directory))
    e2id_dict, r2id_dict = load_kb_dicts(kb_name)
    
    kb = open(path_to_file, "r")
    facts = []
    for fact in kb.readlines():
        fact_cmpnts = parse_kb_fact(fact, max_arity=max_arity)
        fact_comp_ids = [r2id_dict[cmpnt] if idx == 0 else e2id_dict[cmpnt] if idx < max_arity + 1 else cmpnt
                         for idx, cmpnt in enumerate(fact_cmpnts)]
        facts.append(fact_comp_ids)
    with open(path_to_kb_idrep, 'wb') as f:  
        msgpack.pack(facts, f)


def compute_statistics(kb_name, file_to_read="train"+Cnst.KB_FORMAT):
    nb_rel = load_kb_metadata_multi(kb_name)[1]
    kb_directory = Cnst.DEFAULT_KB_MULTI_DIR+kb_name+"/"
    path_to_file = os.path.join(kb_directory, file_to_read)
    with open(path_to_file, "rb") as f:  
        train_triples = msgpack.unpack(f, encoding="utf-8")
    stats = np.array([0] * nb_rel, dtype=np.float32)
    for fact in train_triples:
        stats[fact[0]] += 1   
    normalised_stats = np.expand_dims(stats / sum(stats) * nb_rel, axis=-1)
    return normalised_stats


if __name__ == "__main__":
    adapt_kbs_binary(verbose=True, use_eval_data=True, tr_batch_size=15000, tst_batch_size=15000)
    prepare_kbs_multi(verbose=True, use_eval_data=True, tr_batch_size=15000, tst_batch_size=15000)
    

