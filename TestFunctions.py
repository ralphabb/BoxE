import os
import Cnst
import numpy as np
from operator import itemgetter
from KBUtils import load_kb_metadata_multi, create_kb_filter_multi
import copy


def load_batches(kb_batch_file="train" + Cnst.KBB_FORMAT):  
    return np.load(kb_batch_file+".npy", allow_pickle=True)


def run_categorical_tests(model, kb, test_file="test"+Cnst.KBB_FORMAT, verbose=True):
    kb_directory = os.path.join(Cnst.DEFAULT_KB_MULTI_DIR, kb)
    test_file_path = os.path.join(kb_directory, test_file)  
    batches = load_batches(test_file_path)
    incorrect = 0
    total = 0
    for batch in batches:
        values = model.categorical_forward_pass(batch_components=batch, reload_params=False)
        values = 2*values - 1  
        comparison = np.array(batch[:, -1] - values)
        incorrect += np.count_nonzero(comparison)
        total += comparison.shape[0]
    accuracy = 1 - incorrect/total
    if verbose:
        print("Categorical Accuracy:" + str(accuracy*100)+" %")
    return accuracy


def position_replacement(nb_entities, verbose, setting, batch, filter_dict, model, hits_at_boundaries,
                         position):
    
    batch_size = batch.shape[0]  
    
    batch_ranks = np.full((batch_size, 1), 1, dtype=np.float)  
    batch_scores = model.score_forward_pass(batch_components=batch, reload_params=False)
    
    for entity in range(nb_entities):  
        if entity % 500 == 0 and verbose:  
            print("Entity :" + str(entity))
        replacement_batch = np.array([entity] * batch_size)  
        if setting == Cnst.FILTERED:
            filter_mask, new_batch = compute_filter(batch=batch, replacement_entity=replacement_batch,
                                                    position=position, fltr_dict=filter_dict, return_new_batch=True)
            final_batch = new_batch[filter_mask, :]
        else:  
            filter_mask = np.array([True] * batch_size)  
            final_batch = copy.deepcopy(batch)
            final_batch[:, 1+position] = replacement_batch  
        
        values = model.score_forward_pass(batch_components=final_batch, reload_params=False)
        replacement_batch_array = np.full((batch_size, 1), np.inf, dtype=np.float)
        replacement_batch_array[filter_mask, 0] = values[:, 0]  
        
        rank_update = (replacement_batch_array < batch_scores) * 1  
        batch_ranks += rank_update
    ranks = batch_ranks  
    reciprocal_ranks = 1.0 / ranks  
    rank_sum = np.sum(ranks)  
    reciprocal_rank_sum = np.sum(reciprocal_ranks)  
    
    
    hits_at_array = np.zeros(hits_at_boundaries.shape)
    for index, hits_at_boundary in enumerate(hits_at_boundaries):
        nb_hits = np.sum((ranks <= hits_at_boundary) * 1)
        hits_at_array[index] += nb_hits  
    return rank_sum, reciprocal_rank_sum, hits_at_array, batch_size


def run_ranking_tests(model, kb, test_file="test"+Cnst.KBB_FORMAT, setting=Cnst.FILTERED, verbose=True, max_batch=-1):
    hits_at_boundaries = np.array([1, 3, 5, 10])  
    kb_directory = os.path.join(Cnst.DEFAULT_KB_MULTI_DIR, kb)
    test_file_path = os.path.join(kb_directory, test_file)  
    metadata_dict = load_kb_metadata_multi(kb)
    nb_entities = metadata_dict[0]  
    
    batches = load_batches(test_file_path)  
    
    total_facts = 0
    mean_rank = 0
    mean_reciprocal_rank = 0
    hits_at_array = np.zeros(hits_at_boundaries.shape)
    if setting == Cnst.FILTERED:
        filter_dict = create_kb_filter_multi(kb)  
    else:
        filter_dict = None
    for index, batch in enumerate(batches):  
        
        batch_arities = np.sum(batch[:, 1:-1] != nb_entities, axis=1)
        if max_batch == index:
            break  
        if verbose:
            print("Batch "+str(index+1))  
        max_batch_arity = np.max(batch_arities)
        for i in range(max_batch_arity):  
            if verbose:
                print("Position "+str(i))
            arity_mask = (i + 1) <= batch_arities  
            pos_batch = batch[arity_mask, :]

            rank_pos_sum, reciprocal_rank_pos_sum, hits_at_pos_array, batch_size = \
                position_replacement(nb_entities=nb_entities, verbose=verbose, setting=setting, batch=pos_batch,
                                     filter_dict=filter_dict, model=model, hits_at_boundaries=hits_at_boundaries,
                                     position=i)
            mean_rank += rank_pos_sum
            mean_reciprocal_rank += reciprocal_rank_pos_sum
            total_facts += batch_size  
            hits_at_array += hits_at_pos_array
        
        if verbose:
            hits_at = hits_at_array / total_facts
            mean1_rank = mean_rank / total_facts
            mean1_reciprocal_rank = mean_reciprocal_rank / total_facts
            print(hits_at)
            print(mean1_rank)
            print(mean1_reciprocal_rank)
    hits_at_array = hits_at_array / total_facts
    mean_rank = mean_rank / total_facts
    mean_reciprocal_rank = mean_reciprocal_rank / total_facts
    if verbose:
        if setting == Cnst.FILTERED:
            print("Filtered Test Results")
        else:
            print("Raw Test Results")
        print("Mean Rank: "+str(mean_rank))
        print("Mean Reciprocal Rank: "+str(mean_reciprocal_rank))
        print("Hits@ Values: ")
        for index, hits_at_bndry in enumerate(hits_at_boundaries):  
            print("Hits@"+str(hits_at_bndry)+": "+str(hits_at_array[index]))
    return mean_rank, mean_reciprocal_rank, hits_at_array


def compute_filter(batch, replacement_entity, fltr_dict, position, return_new_batch=False):
    new_batch = copy.deepcopy(batch)
    position_entities = copy.deepcopy(batch[:, 1+position])
    new_batch[:, 1+position] = replacement_entity
    tuple_representation = tuple(map(tuple, new_batch[:, :-1]))  
    filter_output = np.array(itemgetter(*tuple_representation)(fltr_dict))  
    
    test_filter_exemptions = (replacement_entity == position_entities)
    if return_new_batch:
        return np.logical_or(test_filter_exemptions, filter_output), new_batch
    else:
        return np.logical_or(test_filter_exemptions, filter_output)
