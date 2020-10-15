import Cnst
from ModelOptions import ModelOptions
from BoxEModel import BoxEMulti
import TestFunctions
import argparse


def loss_fct(v):
    if v.lower() in ('ply', Cnst.POLY_LOSS, 'p', 'polynomial'):
        return Cnst.POLY_LOSS
    elif v.lower() in ('q2box', Cnst.Q2B_LOSS, 'q', 'query2box'):
        return Cnst.Q2B_LOSS
    else:
        raise argparse.ArgumentTypeError("Invalid Final Loss Function Used")


def test_type(v):
    if v.lower() in ('c', 'cat', 'categorical'):
        return Cnst.CATEGORICAL
    elif v.lower() in ('r', 'rank', 'ranking'):
        return Cnst.RANKING
    else:
        raise argparse.ArgumentTypeError("Invalid Testing Setting entered")


def dataset(v):
    if v.lower() in ('tr','training','train'):
        return Cnst.TRAIN
    elif v.lower() in ('vl', 'val', 'valid', 'validation'):
        return Cnst.VALID
    elif v.lower() in ('tst', 'test', 'testing'):
        return Cnst.TEST
    else:
        raise argparse.ArgumentTypeError('Invalid dataset choice')


def test_setting(v):
    if v.lower() in ('r', 'raw'):
        return Cnst.RAW
    elif v.lower() in ('f', 'filt', 'filtered'):
        return Cnst.FILTERED
    else:
        raise argparse.ArgumentTypeError('Invalid Test Setting.')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test_commandline():
    default_options = ModelOptions()  
    default_options.running_mode = Cnst.TRAIN  
    parser = argparse.ArgumentParser(description='Set up BoxE training over a given KB')
    parser.add_argument("targetKB", type=str, help="The Knowledge Base on which tests must be run")
    parser.add_argument("testType", type=test_type, default=Cnst.RANKING, help=" The test (categorical"
                                                                               " accuracy or ranking metric-based) "
                                                                               "which the model must run")
    parser.add_argument("-testFile", metavar='', type=str, default=Cnst.TEST, help="Test triples file path")
    parser.add_argument("-testSetting", metavar='', type=test_setting, default=Cnst.FILTERED, help="Run filtered/raw tests")
    parser.add_argument("-verbosity", metavar='', type=str2bool, default=True, help="Print periodic Progress updates")
    
    
    parser.add_argument('-embDim', type=int, default=default_options.embedding_dim, metavar='',
                        help="Embedding Dimensionality for points and boxes")
    parser.add_argument("-totalLogBoxSize", type=float, default=default_options.total_log_box_size, metavar='',
                        help="The total log box size to target during training")
    parser.add_argument("-boundScale", type=float, default=default_options.space_bound, metavar='',
                        help="The finite bounds of the space (if bounded)")
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
    parser.add_argument("-boundedBox", type=str2bool, default=default_options.bounded_box_space, metavar='',
                        help="Limit boxes (following bumps and all processing in the unbounded space) to be mapped to "
                             "the bounded tanh ]-1,1[ space")
    parser.add_argument("-boundedNorm", type=str2bool, default=default_options.bounded_box_space, metavar='',
                        help="Limit boxes (following bumps and all processing in the unbounded space) to a minimum "
                             "and maximum size per dimension")
    parser.add_argument("-lossOrd", type=int, default=default_options.loss_norm_ord, metavar='',
                        help="Order of loss normalisation to use (Default "+str(default_options.loss_norm_ord)+" )")
    parser.add_argument("-lossFct", type=loss_fct, default=default_options.loss_fct, metavar='',
                        help="Choice of Loss Function in Training (Default " + str(default_options.obj_fct) + ")")
    parser.add_argument("-normedBumps", type=str2bool, default=default_options.normed_bumps, metavar='',
                        help="Restrict all bumps to be of L2 norm 1 (default "+str(default_options.normed_bumps)+")")

    parser.add_argument("-ruleDir", type=str, default=False, metavar='', help="Specify the txt "
                                                                              "file to read rules from (default no)")
    parser.add_argument("-augmentInv", type=str2bool, default=default_options.augment_inv, metavar='',
                        help="For binary KBs, augment training set with inverse relations (default "
                             + str(default_options.augment_inv) + ")")

    
    args = parser.parse_args()
    target_kb = args.targetKB

    
    default_options.embedding_dim = args.embDim
    default_options.total_log_box_size = args.totalLogBoxSize

    default_options.use_bumps = args.useBumps
    default_options.shared_shape = args.sharedShape
    default_options.learnable_shape = args.learnableShape
    default_options.fixed_width = args.fixedWidth
    default_options.hard_total_size = args.hardSize
    default_options.hard_code_size = args.hardCodeSize
    default_options.bounded_pt_space = args.boundedPt
    default_options.bounded_box_space = args.boundedBox
    default_options.space_bound = args.boundScale
    default_options.bounded_norm = args.boundedNorm
    default_options.loss_norm_ord = args.lossOrd
    default_options.loss_fct = args.lossFct
    default_options.normed_bumps = args.normedBumps
    default_options.rule_dir = args.ruleDir  
    default_options.augment_inv = args.augmentInv 

    model = BoxEMulti(target_kb, default_options)
    model.load_params()
    if args.testType == Cnst.CATEGORICAL:
        file = ""
        if args.testFile == Cnst.TEST:
            file = "test.kbb"
        elif args.testFile == Cnst.VALID:
            file = "valid.kbb"
        elif args.testFile == Cnst.TRAIN:
            file = "train.kbb"
        TestFunctions.run_categorical_tests(model=model, kb=args.targetKB, test_file=file, verbose=args.verbosity)
    else:
        if args.targetKB == "NELLRuleInjSplit90Mat" and args.testFile == "test_subset.kbb":  # Not yet supported
            TestFunctions.run_ranking_tests(model=model, kb=args.targetKB, test_file=args.testFile,
                                            setting=args.testSetting, verbose=args.verbosity)  # Use old code for filt
        else:
            mr, mrr, hits_at = model.validate(verbose=args.verbosity, dataset=args.testFile)  # Filtered eval only
            # For Raw evaluation, please use run_ranking_tests, or disable the filter hash table in validate
            if not args.verbosity:  # Just to print the output
                print("MR:" + str(mr))
                print("MRR:" + str(mrr))
                hai = [1, 3, 5, 10]
                for i in range(len(hits_at)):
                    print("Hits@" + str(hai[i]) + ":" + str(hits_at[i]))


if __name__ == "__main__":
    test_commandline()
