import tensorflow as tf
import KBUtils
import Cnst
import time
import msgpack
import msgpack_numpy as m
import numpy as np
import os
import copy
from TestFunctions import run_categorical_tests
from ModelOptions import ModelOptions
from math import ceil
from RuleParser import RuleParser, enforce_rule, recursive_tensor
import json

m.patch()
tf.logging.set_verbosity(tf.logging.ERROR)

zero = tf.constant(0.0, name="Zero")
half = tf.constant(0.5, name="Half")
one = tf.constant(1.0, name="One")
neg_zero = tf.constant(0.0, shape=[1, 1], name="Neg_Zero")
SANITY_EPS = 10 ** -8
NORM_LOG_BOUND = 1

FACTORIAL_TRICK = {1: 1, 2: 2, 3: 6, 4: 12, 5: 60, 6: 60, 7: 420, 8: 840, 9: 2520, 10: 2520}
BOTTOM_VALUE = np.nan


def sanitize_scatter(input_tensor):
    return tf.where(tf.equal(input_tensor, 0), input_tensor + SANITY_EPS, input_tensor)


def delta_time_string(delta):
    seconds = int(delta) % 60
    minutes = int(delta / 60) % 60
    hours = int(delta / 3600)
    return str(hours) + ":" + str(minutes).zfill(2) + ":" + str(seconds).zfill(2)


def sg(x):
    return tf.stop_gradient(x)


def print_or_log(input_string, log, log_file_path="log.txt"):
    if not log:
        print(input_string)
    else:
        log_file = open(log_file_path, "a+")
        log_file.write(input_string + "\r\n")
        log_file.close()


def transform(input_list, transformation_function):
    output_tuple = tuple([transformation_function(x) for x in input_list])
    return output_tuple


def q2b_loss(name: str, points, lower_corner, upper_corner, scale_mults):  # Query2Box Loss Function
    with tf.name_scope(name):
        with tf.name_scope("Center"):
            centres = 1 / 2 * (lower_corner + upper_corner)
        dist_outside = tf.maximum(points - upper_corner, 0.0) + tf.maximum(lower_corner - points, 0.0)
        dist_inside = centres - tf.minimum(upper_corner, tf.maximum(lower_corner, points))
    return dist_outside, dist_inside


def polynomial_loss(name: str, points, lower_corner, upper_corner, scale_mults):  # Standard Loss Function
    with tf.name_scope(name):
        with tf.name_scope("Width"):
            widths = upper_corner - lower_corner
            widths_p1 = widths + one

        with tf.name_scope("Center"):
            centres = 1 / 2 * (lower_corner + upper_corner)
        with tf.name_scope("Width_Cond"):
            width_cond = tf.where(tf.logical_and(lower_corner <= points, points <= upper_corner),
                                  tf.abs(points - centres) / widths_p1,
                                  widths_p1 * tf.abs(points - centres) - (widths / 2) *
                                  (widths_p1 - 1 / widths_p1))
        return width_cond


def total_box_size_reg(rel_deltas, reg_lambda, log_box_size):   # Regularization based on total box size
    rel_mean_width = tf.reduce_mean(tf.log(tf.abs(rel_deltas) + SANITY_EPS), axis=2)
    min_width = sg(tf.reduce_min(rel_mean_width))
    rel_width_ratios = tf.exp(rel_mean_width - min_width)
    total_multiplier = tf.log(tf.reduce_sum(rel_width_ratios) + SANITY_EPS)
    total_width = total_multiplier + min_width
    size_constraint_loss = reg_lambda * (total_width - log_box_size) ** 2
    return size_constraint_loss


def drpt(tensor, rate):  # Dropout
    return tf.cond(tf.greater(rate, 0), lambda: tf.nn.dropout(tensor, rate=rate), lambda: tensor)


def loss_function_q2b(batch_points, batch_mask, rel_bx_low, rel_bx_high, batch_rel_mults,
                      dim_dropout_prob=zero, order=2, alpha=0.2):
    batch_box_inside, batch_box_outside = q2b_loss("Q2B_Box_Loss", batch_points, rel_bx_low, rel_bx_high,
                                                   batch_rel_mults)

    bbi = tf.norm(drpt(batch_box_inside, rate=dim_dropout_prob), axis=2, ord=order)
    bbi_masked = tf.reduce_sum(tf.multiply(bbi, batch_mask), axis=1)
    bbo = tf.norm(drpt(batch_box_outside, rate=dim_dropout_prob), axis=2, ord=order)
    bbo_masked = tf.reduce_sum(tf.multiply(bbo, batch_mask), axis=1)
    total_loss = alpha * bbi_masked + bbo_masked
    return total_loss


def loss_function_poly(batch_points, batch_mask, rel_bx_low, rel_bx_high, batch_rel_mults, dim_dropout_prob=zero,
                       order=1):
    poly_loss = polynomial_loss("Poly_Box_Loss", batch_points, rel_bx_low, rel_bx_high, batch_rel_mults)
    poly_loss = tf.norm(drpt(poly_loss, rate=dim_dropout_prob), axis=2, ord=order)
    total_loss = tf.reduce_sum(tf.multiply(poly_loss, batch_mask), axis=1)
    return total_loss


def compute_box(box_base, box_delta, name: str):
    box_second = box_base + half * box_delta
    box_first = box_base - half * box_delta
    box_low = tf.minimum(box_first, box_second, name=name + "_low")
    box_high = tf.maximum(box_first, box_second, name=name + "_high")
    return box_low, box_high


def compute_box_np(box_base, box_delta):
    box_second = box_base + 0.5 * box_delta
    box_first = box_base - 0.5 * box_delta
    box_low = np.minimum(box_first, box_second)
    box_high = np.maximum(box_first, box_second)
    return box_low, box_high


# Stop gradients enabled
def loss_computation_with_sg(batch_points, batch_mask, batch_rel_bases, batch_rel_deltas, batch_rel_mults,
                             bounded_pt_space: bool, bounded_box_space: bool, bound_scale: float, sgrad=Cnst.NO_STOPS,
                             original_batch_size=None, dim_drpt_prob=zero, loss_order=2, loss_fct=Cnst.POLY_LOSS):
    obs = original_batch_size
    loss_function = loss_function_poly if loss_fct == Cnst.POLY_LOSS else loss_function_q2b
    if sgrad == Cnst.NO_STOPS or obs is None:   # No Stop gradients, standard computation
        with tf.name_scope("Standard_Loss_Computation"):
            rel_bx_low, rel_bx_high = compute_box(batch_rel_bases, batch_rel_deltas, name="batch_rel_box")
            batch_points = bound_scale * tf.tanh(batch_points) if bounded_pt_space else batch_points
            if bounded_box_space:
                rel_bx_low, rel_bx_high = transform([rel_bx_low, rel_bx_high], lambda x: bound_scale * tf.tanh(x))
            loss_pos_neg = loss_function(batch_points=batch_points, batch_mask=batch_mask, rel_bx_low=rel_bx_low,
                                         rel_bx_high=rel_bx_high, batch_rel_mults=batch_rel_mults, order=loss_order,
                                         dim_dropout_prob=dim_drpt_prob)
        with tf.name_scope("Values_Split"):
            pos_loss = tf.cond(obs < tf.shape(loss_pos_neg)[0], lambda: loss_pos_neg[:obs], lambda: loss_pos_neg)
            neg_loss = tf.cond(obs < tf.shape(loss_pos_neg)[0], lambda: loss_pos_neg[obs:], lambda: neg_zero)
    else:   # Stop Gradients introduced, so loss split into two computation streams to enable their use.
        relation_all = [batch_rel_bases, batch_rel_deltas, batch_rel_mults]
        with tf.name_scope("Loss_With_Stop_Grads"):
            rel_bases_pos, rel_deltas_pos, rel_mults_pos = transform(relation_all, lambda x: x[:obs])
            rel_bases_neg = sg(batch_rel_bases[obs:]) if sgrad[0] == Cnst.STOP else batch_rel_bases[obs:]
            rel_deltas_neg = sg(batch_rel_deltas[obs:]) if sgrad[1] == Cnst.STOP else batch_rel_deltas[obs:]
            batch_points_pos = batch_points[:obs]
            batch_mask_pos = batch_mask[:obs]
            batch_points_neg = batch_points[obs:]
            batch_mask_neg = batch_mask[obs:]
            rel_mults_neg = batch_rel_mults[obs:]
            with tf.name_scope("Positive_Loss"):
                rel_bx_low_pos, rel_bx_high_pos = compute_box(rel_bases_pos, rel_deltas_pos, name="batch_rel_box_pos")
                batch_points_pos = bound_scale * tf.tanh(batch_points_pos) if bounded_pt_space else batch_points_pos
                if bounded_box_space:
                    rel_bx_low_pos, rel_bx_high_pos = transform([rel_bx_low_pos, rel_bx_high_pos],
                                                                lambda x: bound_scale * tf.tanh(x))

                pos_loss = loss_function(batch_points=batch_points_pos, dim_dropout_prob=dim_drpt_prob,
                                         rel_bx_low=rel_bx_low_pos, rel_bx_high=rel_bx_high_pos,
                                         batch_rel_mults=rel_mults_pos, order=loss_order, batch_mask=batch_mask_pos)

            with tf.name_scope("Negative_Loss"):
                rel_bx_low_neg, rel_bx_high_neg = compute_box(rel_bases_neg, rel_deltas_neg, name="batch_rel_box_neg")
                batch_points_neg = bound_scale * tf.tanh(batch_points_neg) if bounded_pt_space else batch_points_neg
                if bounded_box_space:
                    rel_bx_low_neg, rel_bx_high_neg = transform([rel_bx_low_neg, rel_bx_high_neg],
                                                                lambda x: bound_scale * tf.tanh(x))

                neg_loss = loss_function(batch_points=batch_points_neg, dim_dropout_prob=dim_drpt_prob,
                                         rel_bx_low=rel_bx_low_neg, rel_bx_high=rel_bx_high_neg,
                                         batch_rel_mults=rel_mults_neg, order=loss_order, batch_mask=batch_mask_neg)
    return pos_loss, neg_loss


# Negative Sampling function
def uniform_neg_sampling(nb_neg_examples_per_pos, batch_components, nb_entities, max_arity, return_replacements=False):
    if nb_neg_examples_per_pos == 0:
        return batch_components
    batch_size = tf.shape(batch_components)[0]
    batch_components_ent = batch_components[:, 1:-1]
    pre_arities = tf.where(tf.equal(batch_components_ent, nb_entities), tf.zeros_like(batch_components_ent),
                           tf.ones_like(batch_components_ent))
    arities = tf.reduce_sum(pre_arities, axis=1, keepdims=True)
    random_count = batch_size * nb_neg_examples_per_pos
    arities_tiled = tf.tile(arities, [nb_neg_examples_per_pos, 1])

    with tf.name_scope("Position_Replacement_Choices"):
        replacement_choice = tf.transpose(tf.random.categorical(tf.zeros([1, FACTORIAL_TRICK[max_arity]]), random_count,
                                                                dtype=tf.int32))
        replacement_choice = tf.floormod(replacement_choice, arities_tiled)
    with tf.name_scope("Replacement_Entity_Choice"):
        new_entities_pre = tf.transpose(tf.random.categorical(tf.zeros([1, nb_entities - 1]),
                                                              random_count, dtype=tf.int32))[:, 0]

    with tf.name_scope("Negative_Samples"):
        nd_indices = tf.concat([tf.expand_dims(tf.range(random_count), axis=-1), replacement_choice + 1], axis=1)
        negative_samples_pre = tf.tile(batch_components, [nb_neg_examples_per_pos, 1])[:, :-1]
        negative_samples_with_val = tf.concat([negative_samples_pre, tf.fill([random_count, 1], -1)], axis=1)
        replaced_ents = tf.gather_nd(negative_samples_with_val, nd_indices)
        increment_entities = tf.cast(tf.greater_equal(new_entities_pre, replaced_ents), tf.int32)
        new_entities = new_entities_pre + increment_entities
        negative_sample_update = tf.scatter_nd(nd_indices, new_entities + 1, tf.shape(negative_samples_with_val))
        negative_samples = tf.where(tf.greater(negative_sample_update, 0), negative_sample_update - 1,
                                    negative_samples_with_val)
        batch_components_neg = tf.concat([batch_components, negative_samples], axis=0)

        nd_indices_post = tf.concat([batch_size + tf.expand_dims(tf.range(random_count), axis=-1),
                                     replacement_choice + 1], axis=1)
    if return_replacements:
        return batch_components_neg, nd_indices_post
    else:
        return batch_components_neg


def create_uniform_var(name, shape, min_val, max_val):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            var = tf.get_variable(name="init_unif", initializer=tf.random_uniform(shape, min_val, max_val))
    return var


def instantiate_box_embeddings(name: str, scale_mult_shape, rel_tbl_shape, base_norm_shapes, sqrt_dim,
                               hard_size: bool, total_size: float, relation_stats, fixed_width: bool):
    with tf.name_scope(name):
        if relation_stats is not None:
            scale_multiples = relation_stats
        else:
            if fixed_width:
                scale_multiples = tf.zeros(scale_mult_shape)
            else:
                scale_multiples = create_uniform_var("scale_multiples_" + name, scale_mult_shape, -1.0, 1.0)
            if hard_size:
                scale_multiples = total_size * tf.nn.softmax(scale_multiples, axis=0)
            else:
                scale_multiples = tf.nn.elu(scale_multiples) + one

        embedding_base_points = create_uniform_var(name + "_base_point", rel_tbl_shape, -0.5 / sqrt_dim,
                                                   0.5 / sqrt_dim)
        embedding_deltas = tf.multiply(scale_multiples, base_norm_shapes, name=name + "delta")
        return embedding_base_points, embedding_deltas, scale_multiples


def apply_flag_sg(input_tensor, flag):
    return tf.multiply(flag, input_tensor) + tf.multiply(one - flag, sg(input_tensor))


def product_normalise(input_tensor, bounded_norm=True):
    step1_tensor = tf.abs(input_tensor)
    step2_tensor = step1_tensor + SANITY_EPS
    log_norm_tensor = tf.log(step2_tensor)
    step3_tensor = tf.reduce_mean(log_norm_tensor, axis=2, keepdims=True)
    norm_volume = tf.exp(step3_tensor)
    pre_norm_out = input_tensor / norm_volume
    if not bounded_norm:
        return pre_norm_out
    else:

        minsize_tensor = tf.minimum(tf.reduce_min(log_norm_tensor, axis=2, keepdims=True), -NORM_LOG_BOUND)
        maxsize_tensor = tf.maximum(tf.reduce_max(log_norm_tensor, axis=2, keepdims=True), NORM_LOG_BOUND)
        minsize_ratio = -NORM_LOG_BOUND / minsize_tensor
        maxsize_ratio = NORM_LOG_BOUND / maxsize_tensor
        size_norm_ratio = tf.minimum(minsize_ratio, maxsize_ratio)
        normed_tensor = log_norm_tensor * size_norm_ratio

        return tf.exp(normed_tensor)


def add_padding(input_tensor):
    return tf.concat([input_tensor, tf.zeros([1, tf.shape(input_tensor)[1]])], axis=0)


def apply_sg_neg(input_tensor, replacement_indices):
    input_tensor_2 = sanitize_scatter(input_tensor)
    values = sg(tf.gather_nd(input_tensor_2, replacement_indices))
    replacement_mask = tf.scatter_nd(replacement_indices, values, tf.shape(input_tensor))
    return tf.where(tf.greater(replacement_mask, 0), replacement_mask, input_tensor)


def corrupt_batch(batch, idx, nb_entities, hash_table, filtered=True):  # Generate Corrupted Data for validation
    nb_batch_facts = tf.shape(batch)[0]
    replacement_ents = tf.fill([nb_batch_facts], tf.cast(idx % nb_entities, tf.int32))  # the replacement values
    rep_ar_pos = tf.fill([nb_batch_facts, 1], tf.cast(idx // nb_entities, tf.int32))  # output_type not available
    replacement_idx = tf.concat([tf.expand_dims(tf.range(nb_batch_facts), axis=-1), rep_ar_pos + 1], axis=1)
    replacement_mask = tf.scatter_nd(replacement_idx, replacement_ents + 1, tf.shape(batch))
    new_batch = tf.where(tf.greater(replacement_mask, 0), replacement_mask - 1, batch)  # Replace everywhere
    if filtered:
        # Filtering Mechanism
        input_keys = tf.strings.reduce_join(tf.strings.as_string(new_batch[:, :-1]), axis=1,
                                            separator=Cnst.FACT_DELIMITER)
        fact_exists = hash_table.lookup(input_keys)  # Get which values are in there
        fact_exists_bool = tf.greater(fact_exists, 0)  # 0 implies not in KB, so keep replaced, else restore
        original_ents = batch[:, idx // nb_entities + 1]  # Get the original batch values
        original_ents_filt = tf.boolean_mask(original_ents, fact_exists_bool)
        replacement_idx_filt = tf.boolean_mask(replacement_idx, fact_exists_bool)  # Which of the replaced exists?
        # Replacement of entities with values where applicable: + 1 to avoid 0 index
        replacement_mask = tf.scatter_nd(replacement_idx_filt, original_ents_filt + 1, tf.shape(new_batch))
        new_batch = tf.where(tf.greater(replacement_mask, 0), replacement_mask - 1, new_batch)

    return new_batch

class BoxEMulti:
    def __init__(self, kb_name, options: ModelOptions, suffix: str = ""):
        self.options = options
        self.embedding_dim = options.embedding_dim
        self.neg_sampling_opt = options.neg_sampling_opt  # Negative Sampling mode
        self.adv_temp = options.adversarial_temp  # Adversarial Temperature (if applicable)
        self.nb_neg_examples_per_pos = options.nb_neg_examples_per_pos  # Number of negative samples per fact
        self.nb_neg = tf.Variable(self.nb_neg_examples_per_pos, dtype=tf.int32)
        self.learning_rate = options.learning_rate

        self.stop_gradient = options.stop_gradient   # Use stop gradient
        self.sg_neg = options.stop_gradient_negated   # Stop gradient for negative examples
        self.margin = options.loss_margin   # Margin for loss function
        self.reg_lambda = options.regularisation_lambda   # Regularization options
        self.reg_points = options.regularisation_points
        self.total_log_box_size = options.total_log_box_size
        self.batch_size = options.batch_size

        self.use_bumps = options.use_bumps  # Enable entity bumps
        self.hard_total_size = options.hard_total_size  # Fix a total both size

        self.shared_shape = options.shared_shape   # Fix a common box shape
        self.learnable_shape = options.learnable_shape   # Shape a fixed value or
        self.fixed_width = options.fixed_width  # Fix the overall box volume

        self.param_directory = "weights_" + kb_name + "/values.ckpt"
        self.saver = None
        self.sess = None
        self.use_tensorboard = options.use_tensorboard

        self.bounded_pt_space = options.bounded_pt_space  # Map points with tanh activation
        self.bounded_box_space = options.bounded_box_space  # Map boxes with tanh
        self.bound_scale = options.space_bound   # Multiplier to apply on ]-1,1[ range.

        self.obj_fct = options.obj_fct  # Objective Function
        self.loss_fct = options.loss_fct   # Loss Function
        self.loss_ord = options.loss_norm_ord  # Loss Norm Order (1,2,etc..)
        self.dim_dropout_prob = tf.Variable(initial_value=options.dim_dropout_prob, shape=(), name='dim_drpt_prob')
        self.dim_dropout_flt = options.dim_dropout_prob

        # KB setting
        self.kb_name = kb_name
        kb_metadata = KBUtils.load_kb_metadata_multi(kb_name)
        self.nb_entities = kb_metadata[0]
        self.nb_relations = kb_metadata[1]
        self.hard_code_size = options.hard_code_size  # Set all boxes to fixed sizes based on fact statistics
        self.sqrt_dim = tf.sqrt(self.embedding_dim + 0.0)

        self.gradient_clip = options.gradient_clip
        self.bounded_norm = options.bounded_norm

        self.max_arity = kb_metadata[2]
        self.augment_inv = options.augment_inv
        self.original_nb_rel = self.nb_relations

        if self.augment_inv:   # Data Augmentation (define inverse relation and train on inverse fact)
            if self.max_arity > 2:
                print("Unable to use data augmentation, dataset is not a knowledge graph. Setting Aug to False")
                self.augment_inv = False
            else:
                self.nb_relations = 2 * self.nb_relations

        if options.hard_code_size:
            relation_stats = KBUtils.compute_statistics(kb_name)
            relation_stats = relation_stats ** (1 / self.embedding_dim)
        else:
            relation_stats = None

        self.lr_decay = options.learning_rate_decay
        self.lr_decay_period = options.decay_period

        self.rule_dir = options.rule_dir
        self.normed_bumps = options.normed_bumps

        with tf.name_scope('Entity_Embeddings' + suffix):   # Instantiate Entity Embeddings
            entity_table_shape = [self.nb_entities, self.embedding_dim]
            self.entity_points = create_uniform_var("entity_embeddings" + suffix, entity_table_shape,
                                                    -0.5 / self.sqrt_dim, 0.5 / self.sqrt_dim)
            self.entities_with_pad = add_padding(self.entity_points)

            if self.use_bumps:  # Translational Bumps
                self.entity_bumps = create_uniform_var("entity_bump_embeddings" + suffix, entity_table_shape,
                                                       -0.5 / self.sqrt_dim, 0.5 / self.sqrt_dim)
                if self.normed_bumps:  # Normalization of bumps option
                    self.entity_bumps = tf.math.l2_normalize(self.entity_bumps, axis=1)
                self.bumps_with_pad = add_padding(self.entity_bumps)

        rel_tbl_shape = [self.nb_relations, self.max_arity, self.embedding_dim]
        scale_multiples_shape = [self.nb_relations, self.max_arity, 1]
        tile_shape = [self.nb_relations, 1, 1]
        with tf.name_scope('Relation_Embeddings' + suffix):   # Relation Embedding Instantiation
            if self.shared_shape:  # Shared box shape
                base_shape = [1, self.max_arity, self.embedding_dim]
                tile_var = True
            else:   # Variable box shape
                base_shape = rel_tbl_shape
                tile_var = False
            if self.learnable_shape:   # If shape is learnable, define variables accordingly
                self.rel_shapes = create_uniform_var("rel_shape" + suffix,
                                                     base_shape, -0.5 / self.sqrt_dim, 0.5 / self.sqrt_dim)
                self.norm_rel_shapes = product_normalise(self.rel_shapes, self.bounded_norm)
            else:   # Otherwise set all boxes as one-hypercubes
                self.norm_rel_shapes = tf.ones(base_shape, name="rel_shape" + suffix)

            if tile_var:
                self.norm_rel_shapes = tf.tile(self.norm_rel_shapes, tile_shape)
            self.total_size = np.exp(options.total_log_box_size) if self.hard_total_size else -1

            self.rel_bases, self.rel_deltas, self.rel_multiples = \
                instantiate_box_embeddings("rel" + suffix, scale_multiples_shape, rel_tbl_shape,
                                           self.norm_rel_shapes, self.sqrt_dim, self.hard_total_size,
                                           self.total_size, relation_stats, self.fixed_width)

            if self.rule_dir:   # Rule Injection logic
                self.rel_bx_lows, self.rel_bx_highs = compute_box(self.rel_bases, self.rel_deltas, name="rel_bx")
                if self.bounded_box_space:
                    self.rel_bx_lows = self.bound_scale * tf.tanh(self.rel_bx_lows)
                    self.rel_bx_highs = self.bound_scale * tf.tanh(self.rel_bx_highs)
                if self.sg_neg or self.stop_gradient[0] == Cnst.STOP or self.stop_gradient[1] == Cnst.STOP:
                    self.stop_gradient = Cnst.NO_STOPS
                    self.sg_neg = False
                    print("Stop Gradients Not Implemented Yet in Rule Injection Mode. No Stop Gradients Applied...")

                rule_parser = RuleParser(self.rule_dir)  # Parse the rules
                parsed_rules = rule_parser.get_parsed_rules()
                self.rule_boxes = [self.rel_bx_lows]

                for rule_i in parsed_rules:   # Iterate over them linearly (hence rules must be ordered appropriately)
                    recursive_tensor(rule_i[0], self.rel_bx_lows, self.rel_bx_highs)
                    recursive_tensor(rule_i[1], self.rel_bx_lows, self.rel_bx_highs)
                    self.rel_bx_lows, self.rel_bx_highs = enforce_rule(rule_i, self.rel_bx_lows, self.rel_bx_highs)
                    self.rule_boxes.append(self.rel_bx_lows)   # Keep a store of box configuration over rule injection

        with tf.name_scope("Training_Data_Pipeline"):   # Data setup
            tr_np_arr = KBUtils.load_kb_file(Cnst.DEFAULT_KB_MULTI_DIR + str(kb_name) + "/train" + Cnst.KB_FORMAT)
            if not options.restricted_training:
                self.nb_training_facts = tr_np_arr.shape[0]
            else:
                tr_np_arr = tr_np_arr[:options.restriction, :]
                self.nb_training_facts = options.restriction
            if self.augment_inv:
                tr_np_arr_augmentation = np.zeros_like(tr_np_arr)
                tr_np_arr_augmentation[:, 0] = tr_np_arr[:, 0] + self.original_nb_rel
                tr_np_arr_augmentation[:, 1] = tr_np_arr[:, 2]
                tr_np_arr_augmentation[:, 2] = tr_np_arr[:, 1]
                tr_np_arr_augmentation[:, 3] = tr_np_arr[:, 3]
                tr_np_arr = np.concatenate([tr_np_arr, tr_np_arr_augmentation], axis=0)
                self.nb_training_facts = 2 * self.nb_training_facts

            self.nb_tr_batches = ceil(self.nb_training_facts / self.batch_size)
            self.tr_dataset = tf.data.Dataset.from_tensor_slices(tr_np_arr)
            self.tr_dataset = self.tr_dataset.shuffle(self.nb_training_facts, reshuffle_each_iteration=True)
            self.tr_dataset = self.tr_dataset.batch(self.batch_size)

            # Negative Sampling
            if self.neg_sampling_opt == Cnst.UNIFORM or self.neg_sampling_opt == Cnst.SELFADV:
                self.tr_dataset = self.tr_dataset.map(lambda facts: uniform_neg_sampling(self.nb_neg_examples_per_pos,
                                                                                         batch_components=facts,
                                                                                         nb_entities=self.nb_entities,
                                                                                         max_arity=self.max_arity,
                                                                                         return_replacements=self.sg_neg
                                                                                         ), num_parallel_calls=8)

            self.tr_dataset.prefetch(1)

        hash_tbl = KBUtils.create_kb_filter_tf(self.kb_name)   # Creating filter for evaluation

        with tf.name_scope("Val_Data_Pipeline"):
            vl_np_arr = KBUtils.load_kb_file(Cnst.DEFAULT_KB_MULTI_DIR + str(kb_name) + "/valid" + Cnst.KB_FORMAT)
            self.nb_vl_facts = vl_np_arr.shape[0]

            self.vl_dataset = tf.data.Dataset.from_tensor_slices(vl_np_arr)
            self.vl_dataset = self.vl_dataset.batch(self.nb_vl_facts)
            self.vl_dataset.prefetch(16)
            self.vl_dataset_corr = tf.data.Dataset.from_tensor_slices(vl_np_arr).batch(self.nb_vl_facts) \
                .repeat(self.max_arity * self.nb_entities)
            rep_idx = tf.data.Dataset.range(self.max_arity * self.nb_entities)  # int64, output_type only in TF2, cast
            self.vl_dataset_corr = tf.data.Dataset.zip((self.vl_dataset_corr, rep_idx))
            self.vl_dataset_corr = self.vl_dataset_corr.map(lambda batch, idx: corrupt_batch(batch, idx,
                                                                                            self.nb_entities, hash_tbl),
                                                            num_parallel_calls=8)
            self.vl_dataset_corr.prefetch(16)

        with tf.name_scope("Tr_Tst_Data_Pipeline"):
            tr_ts_np_arr = tr_np_arr[:3 * self.batch_size, :]
            self.nb_tr_ts_facts = tr_ts_np_arr.shape[0]
            self.tr_ts_dataset = tf.data.Dataset.from_tensor_slices(tr_ts_np_arr)
            self.tr_ts_dataset = self.tr_ts_dataset.batch(self.nb_tr_ts_facts)
            self.tr_ts_dataset.prefetch(16)
            # Trying something different here...
            self.tr_ts_dataset_corr = tf.data.Dataset.from_tensor_slices(tr_ts_np_arr).batch(self.nb_tr_ts_facts)\
                           .repeat(self.max_arity * self.nb_entities)
            rep_idx = tf.data.Dataset.range(self.max_arity * self.nb_entities)  # int64, output_type only in TF2, cast
            self.tr_ts_dataset_corr = tf.data.Dataset.zip((self.tr_ts_dataset_corr, rep_idx))
            self.tr_ts_dataset_corr = self.tr_ts_dataset_corr.map(lambda batch,
                                                                   idx: corrupt_batch(batch, idx, self.nb_entities,
                                                                                      hash_tbl),
                                                                  num_parallel_calls=8)
            self.tr_ts_dataset_corr.prefetch(16)

        with tf.name_scope("Tst_Data_Pipeline"):
            ts_np_arr = KBUtils.load_kb_file(Cnst.DEFAULT_KB_MULTI_DIR + str(kb_name) + "/test" + Cnst.KB_FORMAT)
            self.nb_ts_facts = ts_np_arr.shape[0]

            self.ts_dataset = tf.data.Dataset.from_tensor_slices(ts_np_arr)
            self.ts_dataset = self.ts_dataset.batch(self.nb_ts_facts)
            self.ts_dataset.prefetch(16)
            self.ts_dataset_corr = tf.data.Dataset.from_tensor_slices(ts_np_arr).batch(self.nb_ts_facts) \
                .repeat(self.max_arity * self.nb_entities)
            rep_idx = tf.data.Dataset.range(self.max_arity * self.nb_entities)  # int64, output_type only in TF2, cast
            self.ts_dataset_corr = tf.data.Dataset.zip((self.ts_dataset_corr, rep_idx))
            self.ts_dataset_corr = self.ts_dataset_corr.map(lambda batch, idx: corrupt_batch(batch, idx,
                                                                                             self.nb_entities,
                                                                                             hash_tbl),
                                                            num_parallel_calls=8)
            self.ts_dataset_corr.prefetch(16)

        with tf.name_scope("Iterator"):  # Data Iterator
            self.iterator = tf.data.Iterator.from_structure(self.tr_dataset.output_types, self.tr_dataset.output_shapes)
            self.next_batch = self.iterator.get_next()
            if self.sg_neg and self.nb_neg_examples_per_pos > 0:
                self.batch_components, self.replaced_indices = self.next_batch
            else:
                self.batch_components = self.next_batch
            self.original_batch_size = tf.div(tf.shape(self.batch_components)[0], 1 + self.nb_neg)
            self.training_init_op = self.iterator.make_initializer(self.tr_dataset)
            self.valid_init_op = self.iterator.make_initializer(self.vl_dataset)
            self.valid_corr_init_op = self.iterator.make_initializer(self.vl_dataset_corr)
            self.test_init_op = self.iterator.make_initializer(self.ts_dataset)
            self.test_corr_init_op = self.iterator.make_initializer(self.ts_dataset_corr)
            self.tr_test_init_op = self.iterator.make_initializer(self.tr_ts_dataset)
            self.tr_test_corr_init_op = self.iterator.make_initializer(self.tr_ts_dataset_corr)

        with tf.name_scope("Batch_Points"):  # Batch Lookups
            self.batch_points = tf.nn.embedding_lookup(self.entities_with_pad,
                                                       self.batch_components[:, 1: self.max_arity + 1],
                                                       name="batch_pts")

            if self.use_bumps:
                self.batch_bumps = tf.nn.embedding_lookup(self.bumps_with_pad,
                                                          self.batch_components[:, 1: self.max_arity + 1],
                                                          name="bump_pts")

        with tf.name_scope("Bumps"):   # Application of bumps
            self.batch_bump_sum = tf.reduce_sum(self.batch_bumps, axis=1, keepdims=True)
            self.batch_point_representations = self.batch_points
            if self.use_bumps:
                self.batch_point_representations += self.batch_bump_sum - self.batch_bumps

        self.batch_components_ent = self.batch_components[:, 1:-1]
        self.batch_mask = tf.where(tf.equal(self.batch_components_ent, self.nb_entities),
                                   tf.zeros_like(self.batch_components_ent, dtype=tf.float32),
                                   tf.ones_like(self.batch_components_ent, dtype=tf.float32))

        with tf.name_scope("Batch_Rel_Params"):
            self.batch_rel_bases = tf.nn.embedding_lookup(self.rel_bases, self.batch_components[:, 0],
                                                          name='batch_rel_bases')
            self.batch_rel_deltas = tf.nn.embedding_lookup(self.rel_deltas, self.batch_components[:, 0],
                                                           name='batch_rel_deltas')
            self.batch_rel_mults = tf.nn.embedding_lookup(self.rel_multiples, self.batch_components[:, 0],
                                                          name='batch_rel_multiples')

        if self.rule_dir:

            self.batch_rel_bx_lows = tf.nn.embedding_lookup(self.rel_bx_lows, self.batch_components[:, 0],
                                                            name='batch_rel_bx_lows')
            self.batch_rel_bx_highs = tf.nn.embedding_lookup(self.rel_bx_highs, self.batch_components[:, 0],
                                                             name='batch_rel_bx_highs')

            loss_function = loss_function_poly if self.loss_fct == Cnst.POLY_LOSS else loss_function_q2b
            with tf.name_scope("Standard_Loss_Computation"):
                obs = self.original_batch_size
                if self.bounded_pt_space:
                    self.batch_point_representations = self.bound_scale * tf.tanh(self.batch_point_representations)
                loss_pos_neg = loss_function(batch_points=self.batch_point_representations, batch_mask=self.batch_mask,
                                             rel_bx_low=self.batch_rel_bx_lows, rel_bx_high=self.batch_rel_bx_highs,
                                             batch_rel_mults=self.batch_rel_mults, order=self.loss_ord,
                                             dim_dropout_prob=self.dim_dropout_prob)
                with tf.name_scope("Values_Split"):
                    self.positive_loss = tf.cond(obs < tf.shape(loss_pos_neg)[0], lambda: loss_pos_neg[:obs],
                                                 lambda: loss_pos_neg)
                    self.negative_loss = tf.cond(obs < tf.shape(loss_pos_neg)[0], lambda: loss_pos_neg[obs:],
                                                 lambda: neg_zero)
        else:
            if self.sg_neg and self.nb_neg_examples_per_pos > 0:
                with tf.name_scope("Negated_Replacement_Stop_Grad"):
                    self.batch_rel_bases = apply_sg_neg(self.batch_rel_bases, self.replaced_indices)
                    self.batch_rel_deltas = apply_sg_neg(self.batch_rel_deltas, self.replaced_indices)
                    self.batch_rel_mults = apply_sg_neg(self.batch_rel_mults, self.replaced_indices)

            self.positive_loss, self.negative_loss = \
                loss_computation_with_sg(
                    batch_points=self.batch_point_representations, batch_mask=self.batch_mask, loss_fct=self.loss_fct,
                    batch_rel_deltas=self.batch_rel_deltas, batch_rel_mults=self.batch_rel_mults,
                    bounded_box_space=self.bounded_box_space, bound_scale=self.bound_scale, sgrad=self.stop_gradient,
                    bounded_pt_space=self.bounded_pt_space, original_batch_size=self.original_batch_size,
                    loss_order=self.loss_ord, dim_drpt_prob=self.dim_dropout_prob, batch_rel_bases=self.batch_rel_bases)

        if self.obj_fct == Cnst.NEG_SAMP:
            self.loss_pos = tf.log(tf.nn.sigmoid(self.margin - self.positive_loss) + SANITY_EPS)
        elif self.obj_fct == Cnst.MARGIN_BASED:
            self.loss_pos = self.positive_loss

        if self.nb_neg_examples_per_pos > 0:
            if self.neg_sampling_opt == Cnst.UNIFORM:
                if self.obj_fct == Cnst.NEG_SAMP:  # Standard Objective
                    self.loss_neg = tf.log(tf.nn.sigmoid(self.negative_loss - self.margin) + SANITY_EPS)
                    self.loss_n_term = tf.reduce_sum(self.loss_neg) / self.nb_neg_examples_per_pos
                elif self.obj_fct == Cnst.MARGIN_BASED:   # Objective used in TransE
                    self.reshaped_neg_dists = tf.reshape(self.negative_loss, [self.nb_neg_examples_per_pos,
                                                                              self.original_batch_size])
                    self.reshaped_neg_dists = tf.transpose(self.reshaped_neg_dists, perm=[1, 0],
                                                           name='transposed_neg', conjugate=False)
                    self.loss_neg = tf.reduce_mean(self.reshaped_neg_dists, axis=1)
                    self.loss_n_term = tf.reduce_sum(
                        self.loss_neg)

            elif self.neg_sampling_opt == Cnst.SELFADV:

                self.reshaped_neg_dists = tf.reshape(self.negative_loss, [self.nb_neg_examples_per_pos,
                                                                          self.original_batch_size])
                self.reshaped_neg_dists = tf.transpose(self.reshaped_neg_dists, perm=[1, 0],
                                                       name='transposed_neg', conjugate=False)
                self.softmax_pre_scores = tf.negative(self.reshaped_neg_dists, name="Negated_Dists") * self.adv_temp

                self.neg_softmax = sg(tf.nn.softmax(self.softmax_pre_scores, axis=1, name="softmax_weights"))
                if self.obj_fct == Cnst.NEG_SAMP:
                    self.loss_neg_batch = tf.log(tf.nn.sigmoid(self.reshaped_neg_dists - self.margin) + SANITY_EPS)
                    self.loss_neg = tf.multiply(self.neg_softmax, self.loss_neg_batch, name="Self-Adversarial_Loss")
                elif self.obj_fct == Cnst.MARGIN_BASED:
                    self.loss_neg = tf.multiply(self.neg_softmax, self.reshaped_neg_dists, name="Self-Adversarial_Loss")
                self.loss_n_term = tf.reduce_sum(self.loss_neg)


        else:
            self.loss_n_term = tf.constant(0.0)

        self.loss_p_term = tf.reduce_sum(self.loss_pos)

        if self.reg_lambda > 0 and not self.hard_total_size:
            if self.fixed_width:
                print("Box size regularization with fixed widths is redundant, so regularization has been disabled")
                self.reg_lambda = -1
                self.reg_loss = 0.0
            else:
                self.reg_loss = total_box_size_reg(rel_deltas=self.rel_deltas, reg_lambda=self.reg_lambda,
                                                   log_box_size=self.total_log_box_size)
        else:
            self.reg_loss = 0.0
        if self.obj_fct == Cnst.NEG_SAMP:
            self.loss = - self.loss_n_term - self.loss_p_term + self.reg_loss
        elif self.obj_fct == Cnst.MARGIN_BASED:
            self.loss = tf.reduce_sum(tf.maximum(0.0, self.margin + self.loss_pos - self.loss_neg))

        if self.reg_points > 0:
            self.loss += self.reg_points * (tf.nn.l2_loss(self.batch_point_representations) +
                                            tf.nn.l2_loss(self.batch_rel_bases))

        if self.use_tensorboard:
            with tf.name_scope('Loss_Terms'):
                if self.obj_fct == Cnst.NEG_SAMP:
                    self.pos_loss_summary = tf.summary.scalar('pos_loss', - self.loss_p_term)
                    self.neg_loss_summary = tf.summary.scalar('neg_loss', - self.loss_n_term)
                elif self.obj_fct == Cnst.MARGIN_BASED:
                    self.pos_loss_summary = tf.summary.scalar('pos_loss', self.loss_p_term)
                    self.neg_loss_summary = tf.summary.scalar('neg_loss', self.loss_n_term)
                self.reg_loss_summary = tf.summary.scalar('reg_loss', self.reg_loss)
                self.total_loss_summary = tf.summary.scalar('loss', self.loss)
            self.loss_summaries = tf.summary.merge([self.pos_loss_summary, self.neg_loss_summary,
                                                    self.reg_loss_summary, self.total_loss_summary])

        self.global_step = tf.Variable(0, trainable=False)
        if self.lr_decay > 0:
            decay_step = self.lr_decay_period * self.nb_tr_batches
            self.lr_with_decay = tf.train.inverse_time_decay(self.learning_rate, global_step=self.global_step,
                                                             decay_rate=self.lr_decay, decay_steps=decay_step)
            self.optimiser = tf.train.AdamOptimizer(learning_rate=self.lr_with_decay)
        else:
            self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.gradient_clip > 0:
            gradients, variables = zip(*self.optimiser.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
            self.train_op = self.optimiser.apply_gradients(zip(gradients, variables), name='minimize',
                                                           global_step=self.global_step)
        else:
            self.train_op = self.optimiser.minimize(self.loss, name='minimize', global_step=self.global_step)

        self.scores = tf.expand_dims(self.positive_loss, axis=-1)
        if self.use_tensorboard:   # TensorBoard configuration
            self.sess = tf.Session()
            self.summary_writer = None
            self.average_epoch_loss = tf.placeholder(tf.float32, shape=None, name='per_epoch_loss')
            self.epoch_loss_summary = tf.summary.scalar('Average Epoch Loss', self.average_epoch_loss)

            with tf.name_scope('Training_Acc'):
                self.train_cat_acc = tf.placeholder(tf.float32, shape=None, name='training_cat_acc')
                self.train_cat_acc_summary = tf.summary.scalar('Training Cat Accuracy', self.train_cat_acc)
                self.train_mr = tf.placeholder(tf.float32, shape=None, name='train_mean_rank')
                self.train_mr_summary = tf.summary.scalar('Training Mean Rank', self.train_mr)
                self.train_mrr = tf.placeholder(tf.float32, shape=None, name='train_mean_reciprocal_rank')
                self.train_mrr_summary = tf.summary.scalar('Training Mean Reciprocal Rank', self.train_mrr)
                self.train_h_at_1 = tf.placeholder(tf.float32, shape=None, name='train_hits_at_1')
                self.train_h_at_1_summary = tf.summary.scalar('Train Hits@1', self.train_h_at_1)
                self.train_h_at_3 = tf.placeholder(tf.float32, shape=None, name='train_hits_at_3')
                self.train_h_at_3_summary = tf.summary.scalar('Train Hits@3', self.train_h_at_3)
                self.train_h_at_5 = tf.placeholder(tf.float32, shape=None, name='train_hits_at_5')
                self.train_h_at_5_summary = tf.summary.scalar('Train Hits@5', self.train_h_at_5)
                self.train_h_at_10 = tf.placeholder(tf.float32, shape=None, name='train_hits_at_10')
                self.train_h_at_10_summary = tf.summary.scalar('Train Hits@10', self.train_h_at_10)
                self.train_summaries = tf.summary.merge([self.train_cat_acc_summary, self.train_mr_summary,
                                                         self.train_mrr_summary, self.train_h_at_1_summary,
                                                         self.train_h_at_3_summary, self.train_h_at_5_summary,
                                                         self.train_h_at_10_summary])

            with tf.name_scope("Validation_Acc"):
                self.valid_cat_acc = tf.placeholder(tf.float32, shape=None, name='valid_cat_acc')
                self.valid_cat_acc_summary = tf.summary.scalar('Validation Cat Accuracy', self.valid_cat_acc)
                self.valid_mr = tf.placeholder(tf.float32, shape=None, name='valid_mean_rank')
                self.valid_mr_summary = tf.summary.scalar('Validation Mean Rank', self.valid_mr)
                self.valid_mrr = tf.placeholder(tf.float32, shape=None, name='valid_mean_reciprocal_rank')
                self.valid_mrr_summary = tf.summary.scalar('Valid Mean Reciprocal Rank', self.valid_mrr)
                self.valid_h_at_1 = tf.placeholder(tf.float32, shape=None, name='valid_hits_at_1')
                self.valid_h_at_1_summary = tf.summary.scalar('Valid Hits@1', self.valid_h_at_1)
                self.valid_h_at_3 = tf.placeholder(tf.float32, shape=None, name='valid_hits_at_3')
                self.valid_h_at_3_summary = tf.summary.scalar('Valid Hits@3', self.valid_h_at_3)
                self.valid_h_at_5 = tf.placeholder(tf.float32, shape=None, name='valid_hits_at_5')
                self.valid_h_at_5_summary = tf.summary.scalar('Valid Hits@5', self.valid_h_at_5)
                self.valid_h_at_10 = tf.placeholder(tf.float32, shape=None, name='valid_hits_at_10')
                self.valid_h_at_10_summary = tf.summary.scalar('Valid Hits@10', self.valid_h_at_10)
                self.valid_summaries = tf.summary.merge([self.valid_cat_acc_summary, self.valid_mr_summary,
                                                         self.valid_mrr_summary, self.valid_h_at_1_summary,
                                                         self.valid_h_at_3_summary, self.valid_h_at_5_summary,
                                                         self.valid_h_at_10_summary])

    def get_rule_boxes(self):
        if not self.rule_dir:
            print("Not in Rule Mode!")
            return None
        else:
            rel_bx_low, rel_bx_high = self.sess.run([self.rel_bx_lows, self.rel_bx_highs])
            return rel_bx_low, rel_bx_high

    def create_feed_dict(self, batch_components=None):
        feed_dict = {}
        if batch_components is not None:
            feed_dict[self.batch_components] = batch_components
        return feed_dict

    def check_scale_mults(self, reload_params=True, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        scores = self.sess.run(self.rel_multiples)
        return scores

    def check_shapes(self, reload_params=True, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        scores = self.sess.run(self.norm_rel_shapes)
        return scores

    def check_box_pos(self, reload_params=True, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        scores = self.sess.run(self.rel_bases)
        return scores

    def check_reg_loss(self, reload_params=True, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        scores = self.sess.run(self.reg_loss)
        return scores

    def score_forward_pass(self, batch_components, reload_params=True, param_loc=None):
        feed_dict = self.create_feed_dict(batch_components)
        if self.nb_neg_examples_per_pos > 0:
            feed_dict[self.original_batch_size] = batch_components.shape[0]
            if self.sg_neg:
                feed_dict[self.replaced_indices] = np.array([[0, 0]])
        feed_dict[self.dim_dropout_prob] = 0.0
        if reload_params:
            self.load_params(param_loc)
        scores = self.sess.run(self.scores, feed_dict=feed_dict)
        return scores

    def get_mask(self, batch_components, reload_params=True, param_loc=None):
        feed_dict = self.create_feed_dict(batch_components)
        if self.nb_neg_examples_per_pos > 0:
            feed_dict[self.original_batch_size] = batch_components.shape[0]
            if self.sg_neg:
                feed_dict[self.replaced_indices] = np.array([[0, 0]])
        feed_dict[self.dim_dropout_prob] = 0.0
        if reload_params:
            self.load_params(param_loc)
        mask = self.sess.run(self.batch_mask, feed_dict=feed_dict)
        return mask

    def compute_box_volume(self, reload_params=True, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        r_bases, r_deltas = self.sess.run([self.rel_bases, self.rel_deltas])
        r_low, r_high = compute_box_np(r_bases, r_deltas)
        if self.bounded_box_space:
            r_low = self.bound_scale * np.tanh(r_low)
            r_high = self.bound_scale * np.tanh(r_high)
        r_log_widths = np.mean(np.log(r_high - r_low + SANITY_EPS), axis=-1, keepdims=False)
        r_geom_width = np.exp(r_log_widths)

        return r_geom_width

    def categorical_forward_pass(self, batch_components, reload_params=True, param_loc=None):
        feed_dict = self.create_feed_dict(batch_components)
        if self.nb_neg_examples_per_pos > 0:
            feed_dict[self.original_batch_size] = batch_components.shape[0]
            if self.sg_neg:
                feed_dict[self.replaced_indices] = np.array([[0, 0]])
        if reload_params:
            self.load_params(param_loc)
        points, mask, r_bases, r_deltas = self.sess.run([self.batch_point_representations, self.batch_mask,
                                                         self.batch_rel_bases, self.batch_rel_deltas],
                                                        feed_dict=feed_dict)
        mask_bool = (mask <= 0.0)
        r_low, r_high = compute_box_np(r_bases, r_deltas)
        points_inside = np.logical_and(points >= r_low, points <= r_high)
        points_inside_masked = np.logical_or(points_inside, np.expand_dims(mask_bool, axis=-1))

        points_inside_boxes = np.all(points_inside_masked, axis=2)
        scores = np.all(points_inside_boxes, axis=1) * 1
        return scores

    def load_params(self, param_loc=None):
        if self.saver is None:
            self.saver = tf.train.Saver(name="Saver")
        self.sess = tf.Session()
        if param_loc is None:
            param_loc = self.param_directory
        try:
            self.saver.restore(self.sess, param_loc)
            self.sess.run(tf.tables_initializer())
        except Exception as e:
            print(e)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.tables_initializer())

    def get_pair_points(self, batch_components, reload_params=True, param_loc=None):
        feed_dict = self.create_feed_dict(batch_components)
        if reload_params:
            self.load_params(param_loc)
        batch_points = self.sess.run([self.batch_points], feed_dict=feed_dict)
        return batch_points

    def get_relation_average_norm_width(self):
        rel_batch = np.arange(self.nb_relations)
        batch_components = np.zeros([self.nb_relations, self.max_arity + 2])
        batch_components[:, 0] = rel_batch
        feed_dict = self.create_feed_dict(batch_components=batch_components)
        self.load_params(None)
        rel_deltas = self.sess.run([self.rel_deltas], feed_dict=feed_dict)
        width_arith_mean = np.mean(np.mean(rel_deltas, axis=2), axis=1)
        return width_arith_mean

    def get_entity_embeddings(self, reload_params=True, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        feed_dict = {}
        ent_emb = self.sess.run([self.entity_points], feed_dict=feed_dict)
        return ent_emb

    def forward_pass(self, batch_components, reload_params=True, param_loc=None):

        feed_dict = self.create_feed_dict(batch_components)
        if reload_params:
            self.load_params(param_loc)

        batch_points, batch_mask, rel_bases, rel_deltas = self.sess.run([self.batch_points, self.batch_mask,
                                                                         self.batch_rel_bases, self.batch_rel_deltas],
                                                                        feed_dict=feed_dict)
        return batch_points, rel_bases, rel_deltas

    def validate(self, hits_at=None, verbose=True, dataset=Cnst.VALID):  # Validation Function
        init_op = self.valid_init_op if dataset == Cnst.VALID else self.test_init_op if dataset == Cnst.TEST \
            else self.tr_test_init_op
        corr_init_op = self.valid_corr_init_op if dataset == Cnst.VALID else self.test_corr_init_op \
            if dataset == Cnst.TEST else self.tr_test_corr_init_op
        nb_facts = self.nb_vl_facts if dataset == Cnst.VALID else self.nb_ts_facts if dataset == Cnst.TEST \
            else self.nb_tr_ts_facts
        if hits_at is None:
            hits_at = [1, 3, 5, 10]
        self.sess.run(tf.assign(self.nb_neg, 0))
        self.sess.run(tf.assign(self.dim_dropout_prob, 0.0))   # Disable dropout during testing
        self.sess.run(init_op)
        reference_scores, batch_mask = self.sess.run([self.scores, self.batch_mask])
        ranks = np.full((nb_facts, self.max_arity), 1)
        self.sess.run(corr_init_op)
        entities_seen = 0
        while entities_seen < self.nb_entities * self.max_arity:
            try:
                current_ar = entities_seen // self.nb_entities
                scores = self.sess.run(self.scores)
                nb_ent = scores.shape[0] // nb_facts
                reshaped_scores = np.reshape(scores, (nb_ent, nb_facts, 1))
                if entities_seen // self.nb_entities < (entities_seen + nb_ent) // self.nb_entities:
                    ents_to_go = self.nb_entities - (entities_seen % self.nb_entities)
                    rank_ind_low = np.sum((reshaped_scores < reference_scores)[:ents_to_go, :, :] * 1,
                                          axis=0, keepdims=False)
                    ranks[:, current_ar] += rank_ind_low[:, 0]
                    rank_ind_high = np.sum((reshaped_scores < reference_scores)[ents_to_go:, :, :] * 1,
                                           axis=0, keepdims=False)
                    if current_ar + 1 < self.max_arity:
                        ranks[:, current_ar + 1] += rank_ind_high[:, 0]
                else:
                    rank_indicator = np.sum((reshaped_scores < reference_scores) * 1, axis=0, keepdims=False)
                    ranks[:, current_ar] += rank_indicator[:, 0]
                entities_seen += nb_ent
                if verbose:
                    print(entities_seen)
            except tf.errors.OutOfRangeError:
                break
        all_ranks = ranks[batch_mask > 0]
        mean_rank = np.mean(all_ranks)
        mean_reciprocal_rank = np.mean(1 / all_ranks)
        hits_at_values = []
        for x in hits_at:
            hits_at_values.append(np.mean((all_ranks <= x) * 1))
        if verbose:
            print("MR:" + str(mean_rank))
            print("MRR:" + str(mean_reciprocal_rank))
            for i in range(len(hits_at)):
                print("Hits@" + str(hits_at[i]) + ":" + str(hits_at_values[i]))
        self.sess.run(tf.assign(self.nb_neg, self.nb_neg_examples_per_pos))
        self.sess.run(tf.assign(self.dim_dropout_prob, self.dim_dropout_flt))  # Restore dropout after eval complete
        self.sess.run(self.training_init_op)
        return mean_rank, mean_reciprocal_rank, hits_at_values

    def set_up_valid_net(self):
        options_no_neg = copy.deepcopy(self.options)
        options_no_neg.nb_neg_examples_per_pos = 0
        with tf.name_scope("Valid_Net"):
            valid_net = BoxEMulti(self.kb_name, options_no_neg, "_val")
        return valid_net

    def train_with_valid(self, separate_valid_model=True, print_period=1, epoch_ckpt=50, save_period=1000,
                         num_epochs=1000, reset_weights=True, loss_file_name="losses", log_to_file=True,
                         log_file_name="training_log.txt", viz_mode=False):

        if separate_valid_model:
            valid_model = self.set_up_valid_net()
        else:
            valid_model = self
        if self.use_tensorboard:
            if not os.path.exists('summaries'):
                os.mkdir('summaries')
            summary_descriptor = str(self.kb_name) + "_" + str(self.stop_gradient) + "_" + str(self.learning_rate) + \
                                 "_" + "nb_neg-" + str(self.nb_neg_examples_per_pos) + "_" + "loss_margin-" \
                                 + str(self.margin) + "_" + "emb_dim-" + str(self.embedding_dim) + "_ " + \
                                 "neg_opt-" + str(self.neg_sampling_opt) + "_" \
                                 + time.asctime()
            if not os.path.exists(os.path.join('summaries', summary_descriptor)):
                os.mkdir(os.path.join('summaries', summary_descriptor))
            self.summary_writer = tf.summary.FileWriter(os.path.join('summaries', summary_descriptor), self.sess.graph)
        if log_to_file:
            open(log_file_name, "w")
        losses = []
        self.sess = tf.Session()
        if self.saver is None:
            self.saver = tf.train.Saver(name="Saver")
        self.sess.run(tf.tables_initializer())
        if reset_weights:
            self.sess.run(tf.global_variables_initializer())
        else:
            try:
                self.saver.restore(self.sess, self.param_directory)
            except ValueError:
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.tables_initializer())
        batch_total_count = 0
        try:
            if not os.path.exists('training_ckpts'):
                os.mkdir('training_ckpts')
            tim = time.time()

            print_or_log("BoxEMulti: ", log_to_file, log_file_name)
            print_or_log("Training for " + str(self.kb_name) + ":", log_to_file, log_file_name)
            print_or_log("Embedding Dimension: " + str(self.embedding_dim), log_to_file, log_file_name)
            print_or_log("Checkpoint Frequency: " + str(epoch_ckpt), log_to_file, log_file_name)
            print_or_log("Number of Epochs: " + str(num_epochs), log_to_file, log_file_name)
            print_or_log("Learning Rate: " + str(self.learning_rate), log_to_file, log_file_name)
            print_or_log("LR Decay: " + str(self.lr_decay) + "/" + "Period: " + str(self.lr_decay_period),
                         log_to_file, log_file_name)
            print_or_log("Loss Margin: " + str(self.margin), log_to_file, log_file_name)
            print_or_log("Shared Shape: " + str(self.shared_shape), log_to_file, log_file_name)
            print_or_log("Learnable Shape: " + str(self.learnable_shape), log_to_file, log_file_name)
            print_or_log("Fixed Width: " + str(self.fixed_width), log_to_file, log_file_name)
            print_or_log("Stop Negated: " + str(self.sg_neg), log_to_file, log_file_name)
            print_or_log("Stop Gradients: " + str(self.stop_gradient), log_to_file, log_file_name)
            print_or_log("Negative Sampling: " + str(self.neg_sampling_opt), log_to_file, log_file_name)
            print_or_log("Bumps: " + str(self.use_bumps), log_to_file, log_file_name)
            print_or_log("Reg Lambda: " + str(self.reg_lambda), log_to_file, log_file_name)
            print_or_log("Reg Points: " + str(self.reg_points), log_to_file, log_file_name)
            print_or_log("Bounded Pt: " + str(self.bounded_pt_space), log_to_file, log_file_name)
            print_or_log("Bounded Box: " + str(self.bounded_box_space), log_to_file, log_file_name)
            if self.bounded_pt_space or self.bounded_box_space:
                print_or_log("Bound Scale: " + str(self.bound_scale), log_to_file, log_file_name)
            if self.hard_total_size:
                print_or_log("Hard Size: " + str(self.total_size), log_to_file, log_file_name)
            else:
                print_or_log("Hard Size: NO", log_to_file, log_file_name)
            print_or_log("Hard Code Size: " + str(self.hard_code_size), log_to_file, log_file_name)
            print_or_log("Objective Function: " + str(self.obj_fct), log_to_file, log_file_name)
            print_or_log("Loss Function: " + str(self.loss_fct), log_to_file, log_file_name)
            print_or_log("Loss Order: " + str(self.loss_ord), log_to_file, log_file_name)
            print_or_log("Dim_Dropout Probability: " + str(self.dim_dropout_flt), log_to_file, log_file_name)
            print_or_log("Gradient Clip: " + str(self.gradient_clip), log_to_file, log_file_name)
            print_or_log("Bounded Norm: " + str(self.bounded_norm), log_to_file, log_file_name)
            print_or_log("Data Augmentation: " + str(self.augment_inv), log_to_file, log_file_name)
            if self.rule_dir:
                print_or_log("Rule Incorporation: " + str(self.rule_dir), log_to_file, log_file_name)
            if self.reg_lambda > 0 and self.hard_total_size:
                print_or_log("WARNING, Regularising with hard constraints. Disable one.", log_to_file, log_file_name)
            print_or_log("Normalized Bumps: " + str(self.normed_bumps), log_to_file, log_file_name)

            json_data = {"Dataset": self.kb_name, "Max Arity": self.max_arity, "Relations": [], "Entities": []}
            if viz_mode:
                with open(Cnst.VIZ_TOOL_NAME + '/settings.json') as f:
                    settings = json.load(f)
                selected_dims = np.array(settings["selectedDims"])
                selected_entities = np.array(settings["selectedEntities"])
                epoch_frequency = settings["epochFrequency"]
                json_data["EpochFrequency"] = epoch_frequency  # Append this information to the JSON. TODO: Date info?
                json_data["animTime"] = settings["animTime"]  # Animation Duration
                data_file_path = Cnst.VIZ_TOOL_NAME + "/" + settings["dataFile"]
                e_dict, r_dict = KBUtils.load_kb_dicts(self.kb_name)  # Get names of things
                e_id2e = {v: k for k, v in e_dict.items()}
                r_id2r = {v: k for k, v in r_dict.items()}
                for idx in range(self.original_nb_rel):  # Ignoring Higher-Arity redundant box removal for now
                    rel_obj = {"name": r_id2r[idx], "x": [], "y": [], "width": [], "height": []}
                    json_data["Relations"].append(rel_obj)
                for idx in selected_entities:
                    ent_obj = {"name": e_id2e[idx], "x": [], "y": [], "b_x": [], "b_y": []}
                    json_data["Entities"].append(ent_obj)
            for epoch_index in range(num_epochs):
                self.sess.run(self.training_init_op)
                if viz_mode:  # Update the files
                    if epoch_index % epoch_frequency == 0:
                        e_pos, e_bumps, r_bases, r_deltas = self.sess.run([self.entity_points, self.entity_bumps,
                                                                           self.rel_bases, self.rel_deltas])
                        r_low, r_high = compute_box_np(r_bases, r_deltas)
                        if self.augment_inv:  # Remove Inverse relations
                            r_low = r_low[self.original_nb_rel:, :, :]
                            r_high = r_high[self.original_nb_rel:, :, :]
                        # Ignoring mapping to bounded space for visualisation purposes
                        sel_e_pos = e_pos[selected_entities, :][:, selected_dims]
                        sel_e_bumps = e_bumps[selected_entities, :][:, selected_dims]
                        r_low = r_low[:, :, selected_dims]
                        r_high = r_high[:, :, selected_dims]
                        r_widths = r_high - r_low
                        for idx in range(self.original_nb_rel):
                            json_data["Relations"][idx]["x"].append(r_low[idx, :, 0].tolist())
                            json_data["Relations"][idx]["y"].append(r_high[idx, :, 1].tolist())
                            json_data["Relations"][idx]["width"].append(r_widths[idx, :, 0].tolist())
                            json_data["Relations"][idx]["height"].append(r_widths[idx, :, 1].tolist())
                        for idx in range(len(selected_entities)):
                            json_data["Entities"][idx]["x"].append(sel_e_pos[idx, 0].tolist())
                            json_data["Entities"][idx]["y"].append(sel_e_pos[idx, 1].tolist())
                            json_data["Entities"][idx]["b_x"].append(sel_e_bumps[idx, 0].tolist())
                            json_data["Entities"][idx]["b_y"].append(sel_e_bumps[idx, 1].tolist())
                        json_data["nbStages"] = epoch_index // epoch_frequency + 1
                        with open(data_file_path, "w") as f:
                            json.dump(json_data, f)

                if epoch_index % epoch_ckpt == 0 and epoch_index > 0:
                    self.saver.save(self.sess, self.param_directory)
                    if separate_valid_model:
                        valid_model.load_params()
                    print_or_log("Checkpoint Reached. Evaluating Metrics...", log_to_file, log_file_name)
                    print_or_log("3 Training Set Batches:", log_to_file, log_file_name)
                    tr_cat_acc = run_categorical_tests(valid_model, self.kb_name, "train.kbb", verbose=False)
                    print_or_log("Cat Acc: " + str(tr_cat_acc * 100) + " %", log_to_file, log_file_name)
                    tr_mr, tr_mrr, tr_hits = self.validate(dataset=Cnst.TRAIN, verbose=False)

                    print_or_log("MR: " + str(tr_mr), log_to_file, log_file_name)
                    print_or_log("MRR: " + str(tr_mrr), log_to_file, log_file_name)
                    print_or_log("Hits@: " + str(tr_hits), log_to_file, log_file_name)
                    tr_hits_1, tr_hits_3, tr_hits_5, tr_hits_10 = tr_hits[0], tr_hits[1], tr_hits[2], tr_hits[3]
                    if self.use_tensorboard:
                        tr_feed_dict = {self.train_cat_acc: tr_cat_acc, self.train_mr: tr_mr, self.train_mrr: tr_mrr,
                                        self.train_h_at_1: tr_hits_1, self.train_h_at_3: tr_hits_3,
                                        self.train_h_at_5: tr_hits_5, self.train_h_at_10: tr_hits_10}

                        tr_loss_summary = self.sess.run(self.train_summaries, feed_dict=tr_feed_dict)
                        self.summary_writer.add_summary(tr_loss_summary, epoch_index)

                    print_or_log("Validation Set Results:", log_to_file, log_file_name)
                    vl_cat_acc = run_categorical_tests(valid_model, self.kb_name, "valid.kbb", verbose=False)
                    print_or_log("Cat Acc: " + str(vl_cat_acc * 100) + " %", log_to_file, log_file_name)
                    vl_mr, vl_mrr, vl_hits = self.validate(dataset=Cnst.VALID, verbose=False)

                    print_or_log("MR: " + str(vl_mr), log_to_file, log_file_name)
                    print_or_log("MRR: " + str(vl_mrr), log_to_file, log_file_name)
                    print_or_log("Hits@: " + str(vl_hits), log_to_file, log_file_name)
                    vl_hits_1, vl_hits_3, vl_hits_5, vl_hits_10 = vl_hits[0], vl_hits[1], vl_hits[2], vl_hits[3]
                    if self.use_tensorboard:
                        vl_feed_dict = {self.valid_cat_acc: vl_cat_acc, self.valid_mr: vl_mr, self.valid_mrr: vl_mrr,
                                        self.valid_h_at_1: vl_hits_1, self.valid_h_at_3: vl_hits_3,
                                        self.valid_h_at_5: vl_hits_5, self.valid_h_at_10: vl_hits_10}
                        vl_loss_summary = self.sess.run(self.valid_summaries, feed_dict=vl_feed_dict)
                        self.summary_writer.add_summary(vl_loss_summary, epoch_index)
                    print_or_log("Saving Checkpoint Weights...", log_to_file, log_file_name)
                    self.saver.save(self.sess,
                                    "training_ckpts/" + self.kb_name + "_ep" + str(epoch_index) + "/values.ckpt")
                    print_or_log("Save Complete", log_to_file, log_file_name)

                average_epoch_loss = 0
                print_or_log("Epoch " + str(epoch_index + 1), log_to_file, log_file_name)
                for batch_index in range(self.nb_tr_batches):
                    try:
                        overall_batch = batch_index + batch_total_count
                        if not self.use_tensorboard:
                            _, loss, loss_pos, loss_neg, batch_mask = self.sess.run([self.train_op, self.loss,
                                                                                     self.loss_p_term, self.loss_n_term,
                                                                                     self.batch_mask])
                            losses.append([loss, loss_pos, loss_neg])
                        else:
                            _, loss, loss_pos, loss_neg, batch_mask, loss_summary = self.sess.run([self.train_op,
                                                                                                   self.loss,
                                                                                                   self.loss_p_term,
                                                                                                   self.loss_n_term,
                                                                                                   self.batch_mask,
                                                                                                   self.loss_summaries])
                            self.summary_writer.add_summary(loss_summary, overall_batch)
                        average_epoch_loss += loss
                        if overall_batch % print_period == 0:
                            tim2 = time.time()
                            delta = tim2 - tim
                            delta_string = delta_time_string(delta)
                            print_or_log(delta_string + ") Loss @Batch " + str(overall_batch) + ":" + str(loss) +
                                         ", +:" + str(-loss_pos) + ",-:" + str(-loss_neg), log_to_file, log_file_name)
                        if overall_batch % save_period == 0:
                            self.saver.save(self.sess, self.param_directory)
                    except tf.errors.InvalidArgumentError as e:
                        print("NaN or Inf Gradient at Batch:" + str(batch_index))
                        print(e)

                batch_total_count += self.nb_tr_batches
                average_epoch_loss /= self.nb_tr_batches
                print_or_log("Epoch " + str(epoch_index + 1) + " Complete. Average Epoch Loss: " +
                             str(average_epoch_loss), log_to_file, log_file_name)
                if self.use_tensorboard:
                    av_loss_summary = self.sess.run(self.epoch_loss_summary,
                                                    feed_dict={self.average_epoch_loss: average_epoch_loss})
                    self.summary_writer.add_summary(av_loss_summary, epoch_index + 1)
        except KeyboardInterrupt:
            print_or_log("Training Stopped", log_to_file, log_file_name)

        self.saver.save(self.sess, self.param_directory)
        print_or_log("Weights saved to " + str(self.param_directory), log_to_file, log_file_name)
        if not self.use_tensorboard:
            with open(loss_file_name + ".ls", "wb") as f:
                msgpack.dump(losses, f)
            self.sess.close()
            return losses
        self.sess.close()
