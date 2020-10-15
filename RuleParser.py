
from Cnst import IMPLIES, EQUIV, TERMINAL, AND, OR
import re
import numpy as np
from collections import Counter
import tensorflow as tf

atom = re.compile("(?P<rel>[0-9]+)\[(?P<ents>([A-Za-z0-9]+,)*[A-Za-z0-9]+)\]")
exclusive_atom = re.compile("^(?P<rel>[0-9]+)\[(?P<ents>([A-Za-z0-9]+,)*[A-Za-z0-9]+)\]$")
composition_atom = re.compile("\(.*\)")  
operators = re.compile("["+AND+OR+"]")

BOTTOM_VALUE = np.nan
SANITY_EPS = 10**-8


def sanitize_scatter(input_tensor):
    return tf.where(tf.equal(input_tensor, 0), input_tensor + SANITY_EPS, input_tensor)


class TreeNode:
    def __init__(self, ntype, rel=-1, ents=-1):
        self.type = ntype
        self.children = []  
        self.parent = None
        self.box_tensor_low = None
        self.box_tensor_high = None
        if self.type == TERMINAL:  
            self.rel = rel
            self.ents = ents
            self.indices = np.array([[self.rel, ent] for ent in self.ents])

    def add_child(self, child):
        self.children.append(child)  
        child.parent = self  

    def get_child(self, child_idx):
        return self.children[child_idx]

    def __repr__(self):
        if self.type == TERMINAL:
            return self.type+":"+str(self.rel)+str(self.ents)
        else:
            return self.type+": Children: " + str([str(child) for child in self.children])


def box_intersection(b1_low, b1_high, b2_low, b2_high):  
    max_l = tf.maximum(b1_low, b2_low)
    min_h = tf.minimum(b1_high, b2_high)
    intersection_existence = tf.reduce_min(min_h - max_l)
    with tf.name_scope("Intersection-Standard"):
        output = tf.cond(tf.less(intersection_existence, 0), lambda: (tf.fill(tf.shape(max_l), BOTTOM_VALUE),
                                                                      tf.fill(tf.shape(max_l), BOTTOM_VALUE)),
                         lambda: (max_l, min_h))
    return output


def box_intersection_with_bottom(b1_low, b1_high, b2_low, b2_high):
    pred = tf.logical_or(tf.reduce_any(tf.is_nan(b1_low)), tf.reduce_any(tf.is_nan(b2_low)))
    with tf.name_scope("Intersection-With-btm"):
        out = tf.cond(pred, lambda: (tf.fill(tf.shape(b1_low), BOTTOM_VALUE), tf.fill(tf.shape(b1_low), BOTTOM_VALUE)),
                            lambda: box_intersection(b1_low, b1_high, b2_low, b2_high))
    return out


def box_union(b1_low, b1_high, b2_low, b2_high):
    min_l = tf.minimum(b1_low, b2_low)
    max_h = tf.maximum(b1_high, b2_high)
    return min_l, max_h


def box_union_with_bottom(b1_low, b1_high, b2_low, b2_high):
    with tf.name_scope("Union-With-Btm"):
        return tf.cond(tf.reduce_any(tf.is_nan(b1_low)), lambda: tf.cond(tf.reduce_any(tf.is_nan(b2_low)),
                                                                         lambda: (tf.fill(tf.shape(b1_low),
                                                                                          BOTTOM_VALUE),
                                                                                  tf.fill(tf.shape(b1_low),
                                                                                          BOTTOM_VALUE)),
                                                                         lambda: (b2_low, b2_high)),
                                                         lambda: tf.cond(tf.reduce_any(tf.is_nan(b2_low)),
                                                                         lambda: (b1_low, b1_high),
                                                                         lambda: box_union(b1_low, b1_high, b2_low,
                                                                                           b2_high)))


def recursive_tensor(node: TreeNode, embedding_low=None, embedding_high=None):
    if node.type == TERMINAL:  
        node.box_tensor_low = tf.gather_nd(embedding_low, node.indices)
        node.box_tensor_high = tf.gather_nd(embedding_high, node.indices)
    else:
        child0_l, child0_h = recursive_tensor(node.get_child(0), embedding_low, embedding_high)
        child1_l, child1_h = recursive_tensor(node.get_child(1), embedding_low, embedding_high)
        if node.type == AND:
            node.box_tensor_low, node.box_tensor_high = box_intersection_with_bottom(child0_l, child0_h,
                                                                                     child1_l, child1_h)
        if node.type == OR:
            node.box_tensor_low, node.box_tensor_high = box_union_with_bottom(child0_l, child0_h,
                                                                              child1_l, child1_h)

    return node.box_tensor_low, node.box_tensor_high


def recursive_parse(input_expression):  
    
    m = re.match(exclusive_atom, input_expression)  
    if m:  
        rel = int(m.group("rel"))
        ents = np.array([int(ent) for ent in m.group("ents").split(",")])  
        
        return TreeNode(TERMINAL, rel, ents)
    else:
        
        m_sub = re.findall(composition_atom, input_expression)  
        bracketless = input_expression
        for index, match in enumerate(m_sub):
            if match[0] == "(" and match[-1] == ")": 
                bracketless = input_expression.replace(match, "")  
                m_sub[index] = match[1:-1]  
            else:
                raise SyntaxError("Unbalanced Brackets at:"+str(match))
        
        bracketless_operands = re.split(operators, bracketless)  
        bracketless_operators = re.findall(operators, bracketless)  
        bracketed_idx = np.array([x for x in range(len(bracketless_operands)) if bracketless_operands[x] == ""])
        for hl_indx, index in enumerate(bracketed_idx):  
            bracketless_operands[index] = m_sub[hl_indx]
        nb_op = len(bracketless_operators)
        last_node = recursive_parse(bracketless_operands[0])  
        for i in range(nb_op):
            new_op_node = TreeNode(bracketless_operators[i])  
            new_op_node.add_child(last_node)  
            new_op_node.add_child(recursive_parse(bracketless_operands[i+1]))
            last_node = new_op_node  
        return last_node


def get_terminal_nodes(tree_root):
    if tree_root.type == TERMINAL:
        return [tree_root]
    else:  
        my_terminal_nodes = []
        for child in tree_root.children:
            my_terminal_nodes.extend(get_terminal_nodes(child))
        return my_terminal_nodes


def parse_rule(rule_line, rule_idx, enforce_same=True):
    
    
    impl_count, equiv_count = rule_line.count(IMPLIES), rule_line.count(EQUIV)
    if impl_count + equiv_count != 1:  
        raise SyntaxError("Rule " + str(rule_idx) + " has none or more than 1 implication/equivalence operator")
    else:
        rule_line_no_sp = rule_line.replace(" ", "")  
        if enforce_same:  
            
            entity_sets = [Counter(m.group("ents").split(",")) for m in re.finditer(atom, rule_line_no_sp)]
            all_same = entity_sets[1:] == entity_sets[:-1]
            if not all_same:
                raise AssertionError('Invalid Entity Configuration. Relation entities'
                                     ' are not the same among all atoms.')
        
        if impl_count == 1:  
            lhs, rhs = rule_line_no_sp.split(IMPLIES)
            mid_param = IMPLIES
        else:
            lhs, rhs = rule_line_no_sp.split(EQUIV)
            mid_param = EQUIV
        root_lhs = recursive_parse(lhs)  
        root_rhs = recursive_parse(rhs)
        return root_lhs, root_rhs, mid_param


def enforce_rule(rule, emb_low, emb_high):  
    lhs = rule[0]  
    rhs = rule[1]  
    rtype = rule[2]  
    if rtype == IMPLIES:  
        entailment_union = box_union_with_bottom(rhs.box_tensor_low, rhs.box_tensor_high,
                                                                     lhs.box_tensor_low, lhs.box_tensor_high)
        new_bx_l = tf.scatter_nd(rhs.indices, sanitize_scatter(entailment_union[0]), tf.shape(emb_low,
                                                                                              out_type=tf.dtypes.int64))
        new_bx_h = tf.scatter_nd(rhs.indices, sanitize_scatter(entailment_union[1]), tf.shape(emb_high,
                                                                                              out_type=tf.dtypes.int64))

    elif rtype == EQUIV:  
        if lhs.type == TERMINAL:  
            if lhs.rel == rhs.rel and len(lhs.ents) == 2:  
                lhs.indices[:, -1] = 0
                recursive_tensor(lhs, emb_low, emb_high)  
        new_bx_l = tf.scatter_nd(rhs.indices, sanitize_scatter(lhs.box_tensor_low), tf.shape(emb_low,
                                                                                             out_type=tf.dtypes.int64))
        new_bx_h = tf.scatter_nd(rhs.indices, sanitize_scatter(lhs.box_tensor_high), tf.shape(emb_high,
                                                                                              out_type=tf.dtypes.int64))
    else:
        return ValueError("Invalid Rule Type for enforcement")
    new_emb_low = tf.where(tf.not_equal(new_bx_l, 0), new_bx_l, emb_low)  
    
    new_emb_high = tf.where(tf.not_equal(new_bx_h, 0), new_bx_h, emb_high)
    return new_emb_low, new_emb_high


class RuleParser:
    def __init__(self, source_path, enforce_same=True):  
        self.source_path = source_path
        self.rule_lines = open(source_path, 'r').readlines()
        self.rules_parsed = [parse_rule(rule, idx, enforce_same=enforce_same)
                             for idx, rule in enumerate(self.rule_lines)]

    def get_parsed_rules(self):
        return self.rules_parsed

