import Cnst
from ModelOptions import ModelOptions
from BoxEModel import BoxEMulti
import argparse


def stop_grad(v):
    if v in (Cnst.NO_STOPS, Cnst.STOP_SIZES, Cnst.STOP_POSITIONS, Cnst.STOP_BOTH):
        return v
    else:
        raise argparse.ArgumentTypeError('Invalid Stop Gradient Setting Entered')


def obj_fct(v):
    if v.lower() in ('negSamp', Cnst.NEG_SAMP, 'ns', 'n'):
        return Cnst.NEG_SAMP
    elif v.lower() in ('marginBased', Cnst.MARGIN_BASED, 'mb', 'm'):
        return Cnst.MARGIN_BASED
    else:
        raise argparse.ArgumentTypeError("Invalid Final Objective Function Used")


def loss_fct(v):
    if v.lower() in ('ply', Cnst.POLY_LOSS, 'p', 'polynomial'):
        return Cnst.POLY_LOSS
    elif v.lower() in ('q2box', Cnst.Q2B_LOSS, 'q', 'query2box'):
        return Cnst.Q2B_LOSS
    else:
        raise argparse.ArgumentTypeError("Invalid Final Loss Function Used")


def neg_samp(v):
    if v.lower() in ('u', 'unif', 'uniform'):
        return Cnst.UNIFORM
    elif v.lower() in ('gan', 'adversarial'):
        return Cnst.GAN  
    elif v.lower() in ('self-adv', 'self', 'sa', 'self-adversarial', 'selfadv'):
        return Cnst.SELFADV
    else:
        raise argparse.ArgumentTypeError('Invalid Negative Sampling Option selected')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_commandline():
    default_options = ModelOptions()  
    default_options.running_mode = Cnst.TRAIN  
    parser = argparse.ArgumentParser(description='Set up BoxE training over a given KB')
    parser.add_argument("targetKB", type=str, help="The Knowledge Base on which to train")
    parser.add_argument("-validation", metavar='', type=str2bool, default=False, help=" Use Validation-Based "
                                                                          "training (early stopping)")
    parser.add_argument("-printFreq", metavar='', type=int, default=50, help="Batch Interval between which to log")
    parser.add_argument("-validCkpt", metavar='', type=int, default=50, help="Epoch Gap between validation "
                        + "tests (if applicable)")
    parser.add_argument("-savePeriod", type=int, default=10000, metavar='',
                        help="If no early stopping, batch intervals at which weight saving is done")
    parser.add_argument("-epochs", type=int, default=100000, metavar='', help="Maximum Number of Epochs to run")
    parser.add_argument("-resetWeights", type=str2bool, default=True, metavar='',
                        help="Initialize weights (default) or start with existing weights")
    parser.add_argument("-lossFName", type=str, metavar='', default="losses", help="Loss Log File Name")
    parser.add_argument("-logToFile", type=str2bool, metavar='', default=True, help="Log to file (default)" +
                        "or print to console")
    parser.add_argument("-logFName", type=str, metavar='', default="training_log.txt", help="Loss Log File Name")

    
    
    parser.add_argument("-useTB", type=str2bool, default=default_options.use_tensorboard, metavar='',
                        help="Enable Use of TensorBoard during training")
    parser.add_argument('-embDim', type=int, default=default_options.embedding_dim, metavar='',
                        help="Embedding Dimensionality for points and boxes")
    parser.add_argument('-negSampling', type=neg_samp, default=default_options.neg_sampling_opt, metavar='',
                        help="Type of Negative Sampling to use (Default and Only Current Option: Uniform")
    parser.add_argument("-nbNegExp", type=int, default=default_options.nb_neg_examples_per_pos, metavar='',
                        help="Number of Negative Examples per positive example (default "
                        + str(default_options.nb_neg_examples_per_pos)+")")
    parser.add_argument("-batchSize", type=int, default=default_options.batch_size, metavar='',
                        help="Batch Size to use for Training (default "+str(default_options.batch_size)+")")
    parser.add_argument("-lossMargin", type=float, default=default_options.loss_margin, metavar='',
                        help="The maximum negative distance to consider")
    parser.add_argument("-advTemp", type=float, default=default_options.adversarial_temp, metavar='',
                        help="The temperature to use for self-adversarial negative sampling")
    parser.add_argument("-regLambda", type=float, default=default_options.regularisation_lambda, metavar='',
                        help="The weight of L2 regularization over bound width (BOX model) to apply")
    parser.add_argument("-totalLogBoxSize", type=float, default=default_options.total_log_box_size, metavar='',
                        help="The total log box size to target during training")
    parser.add_argument("-boundScale", type=float, default=default_options.space_bound, metavar='',
                        help="The finite bounds of the space (if bounded)")
    parser.add_argument("-learningRate", type=float, default=default_options.learning_rate, metavar='',
                        help="Learning Rate to use for training (default "+str(default_options.learning_rate)+")")
    parser.add_argument("-lrDecay", type=float, default=default_options.learning_rate_decay, metavar='',
                        help="Learning Rate Decay to use in training (default " +
                             str(default_options.learning_rate_decay) + ")")
    parser.add_argument("-lrDecayStep", type=float, default=default_options.decay_period, metavar='',
                        help="Decay Period for LR (default " +
                             str(default_options.decay_period) + ")")
    parser.add_argument("-stopGradients", type=stop_grad, default=default_options.stop_gradient, metavar='',
                        help="Stop Gradient Configuration for negative examples. NO STOP:" + Cnst.NO_STOPS
                             + "|| STOP REL BOUNDS:" + Cnst.STOP_SIZES + "  (default " + default_options.stop_gradient +
                             ")")
    parser.add_argument("-stopNegated", type=str2bool, default=default_options.stop_gradient_negated, metavar='',
                        help="Disable Gradients from non-replaced negative example components")
    parser.add_argument("-sharedShape", type=str2bool, default=default_options.shared_shape, metavar='',
                        help="Specifies whether shape is shared by all boxes during training")
    parser.add_argument("-fixedWidth", type=str2bool, default=default_options.fixed_width, metavar='',
                        help="Specifies whether box width (size) is learned during training")
    parser.add_argument("-learnableShape", type=str2bool, default=default_options.learnable_shape, metavar='',
                        help="Specifies whether shape is learned during training")
    parser.add_argument("-useBumps", type=str2bool, default=default_options.use_bumps, metavar='',
                        help="Allow pairwise bumping to occur, to prevent all-pair correctness (default " +
                             str(default_options.use_bumps)+")")
    parser.add_argument("-hardSize", type=str2bool, default=default_options.hard_total_size, metavar='',
                        help="Use Softmax to enforce that all boxes share a hard total size")
    parser.add_argument("-hardCodeSize", type=str2bool, default=default_options.hard_total_size, metavar='',
                        help="Hard Code Size based on statistical appearances of relations in set (works only "
                             "with shared shape)")
    parser.add_argument("-boundedPt", type=str2bool, default=default_options.bounded_pt_space, metavar='',
                        help="Limit points (following bumps and all processing in the unbounded space) to be mapped to "
                             "the bounded tanh ]-1,1[ space")
    parser.add_argument("-regPoints", type=float, default=default_options.regularisation_points, metavar='',
                        help="Regularisation factor to apply to batch to prevent excessive divergence from center")
    parser.add_argument("-lossOrd", type=int, default=default_options.loss_norm_ord, metavar='',
                        help="Order of loss normalisation to use (Default "+str(default_options.loss_norm_ord)+" )")
    parser.add_argument("-boundedBox", type=str2bool, default=default_options.bounded_box_space, metavar='',
                        help="Limit boxes (following bumps and all processing in the unbounded space) to be mapped to "
                             "the bounded tanh ]-1,1[ space")
    parser.add_argument("-objFct", type=obj_fct, default=default_options.obj_fct, metavar='',
                        help="Choice of Objective Function in Training (Default " + str(default_options.obj_fct) + ")")
    parser.add_argument("-lossFct", type=loss_fct, default=default_options.loss_fct, metavar='',
                        help="Choice of Loss Function in Training (Default " + str(default_options.obj_fct) + ")")
    parser.add_argument("-dimDropout", type=float, default=default_options.dim_dropout_prob, metavar='',
                        help="Dropout probability to use when training the model (Default "
                             + str(default_options.dim_dropout_prob)+")")
    parser.add_argument("-gradClip", type=float, default=default_options.gradient_clip, metavar='',
                        help="Value to apply for gradient clipping (Default "
                             + str(default_options.gradient_clip)+")")
    parser.add_argument("-boundedNorm", type=str2bool, default=default_options.bounded_box_space, metavar='',
                        help="Limit boxes (following bumps and all processing in the unbounded space) to a minimum "
                             "and maximum size per dimension")
    parser.add_argument("-normedBumps", type=str2bool, default=default_options.normed_bumps, metavar='',
                        help="Restrict all bumps to be of L2 norm 1 (default +"+str(default_options.normed_bumps)+")")

    parser.add_argument("-separateValid", type=str2bool, default=False, metavar='',
                        help="Use a duplicate model without negative sampling to perform quicker testing")

    parser.add_argument("-ruleDir", type=str, default=False, metavar='', help="Specify the txt "
                                                                              "file to read rules from (default no)")

    parser.add_argument("-augmentInv", type=str2bool, default=default_options.augment_inv, metavar='',
                        help="For binary KBs, augment training set with inverse relations (default "
                             + str(default_options.augment_inv)+")")
    parser.add_argument("-viz", type=str2bool, default=False, metavar='',
                        help="Enable Data Logging for subsequent BoxEViz visualization")

    
    args = parser.parse_args()
    target_kb = args.targetKB
    feedback_period = args.printFreq
    save_period = args.savePeriod
    epoch_checkpoint = args.validCkpt
    num_epochs = args.epochs
    reset_weights = args.resetWeights
    loss_file_name = args.lossFName
    log_to_file = args.logToFile
    log_file_name = args.logFName

    
    default_options.batch_size = args.batchSize
    default_options.use_tensorboard = args.useTB
    default_options.embedding_dim = args.embDim
    default_options.neg_sampling_opt = args.negSampling
    default_options.nb_neg_examples_per_pos = args.nbNegExp
    default_options.learning_rate = args.learningRate
    default_options.learning_rate_decay = args.lrDecay
    default_options.decay_period = args.lrDecayStep
    default_options.stop_gradient = args.stopGradients
    default_options.adversarial_temp = args.advTemp
    default_options.total_log_box_size = args.totalLogBoxSize
    default_options.loss_margin = args.lossMargin
    default_options.regularisation_lambda = args.regLambda
    default_options.stop_gradient_negated = args.stopNegated

    default_options.use_bumps = args.useBumps
    default_options.shared_shape = args.sharedShape
    default_options.learnable_shape = args.learnableShape
    default_options.fixed_width = args.fixedWidth
    default_options.hard_total_size = args.hardSize
    default_options.hard_code_size = args.hardCodeSize
    default_options.bounded_pt_space = args.boundedPt
    default_options.bounded_box_space = args.boundedBox
    default_options.space_bound = args.boundScale
    default_options.obj_fct = args.objFct
    default_options.loss_fct = args.lossFct
    default_options.dim_dropout_prob = args.dimDropout
    default_options.regularisation_points = args.regPoints
    default_options.loss_norm_ord = args.lossOrd
    default_options.gradient_clip = args.gradClip
    default_options.bounded_norm = args.boundedNorm
    default_options.rule_dir = args.ruleDir
    default_options.normed_bumps = args.normedBumps
    default_options.augment_inv = args.augmentInv

    sepValid = args.separateValid

    model = BoxEMulti(target_kb, default_options)
    model.train_with_valid(print_period=feedback_period, epoch_ckpt=epoch_checkpoint, num_epochs=num_epochs,
                           reset_weights=reset_weights, loss_file_name=loss_file_name, log_to_file=log_to_file,
                           log_file_name=log_file_name, save_period=save_period, separate_valid_model=sepValid,
                           viz_mode=args.viz)


if __name__ == "__main__":
    train_commandline()
