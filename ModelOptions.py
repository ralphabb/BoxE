import Cnst


class ModelOptions:
    def __init__(self):
        
        self.embedding_dim = 64  
        
        self.neg_sampling_opt = Cnst.SELFADV  
        
        self.adversarial_temp = 1.0
        self.nb_neg_examples_per_pos = 25  
        self.loss_margin = 3.0  
        self.loss_fct = Cnst.POLY_LOSS  
        self.obj_fct = Cnst.NEG_SAMP  
        self.batch_size = 1024  
        self.loss_norm_ord = 2  
        self.regularisation_lambda = 0  
        self.regularisation_points = 0  
        self.total_log_box_size = -5  
        self.hard_total_size = False  
        self.hard_code_size = False  
        
        
        self.learning_rate = 10 ** -2  
        self.ranking_function = Cnst.DIST_RANK  
        self.stop_gradient = Cnst.NO_STOPS  
        
        
        self.restricted_training = False
        self.restriction = 1024

        self.stop_gradient_negated = False
        self.use_bumps = True  
        
        
        self.shared_shape = False  
        self.learnable_shape = True
        self.fixed_width = False  
        
        
        self.bounded_pt_space = True
        self.bounded_box_space = True
        self.space_bound = 1.0

        self.dim_dropout_prob = 0.0  
        self.use_tensorboard = True  
        self.gradient_clip = -1.0  

        self.bounded_norm = False
        self.learning_rate_decay = 0  
        self.decay_period = 100
        
        self.rule_dir = False
        self.normed_bumps = False  

        self.augment_inv = False


