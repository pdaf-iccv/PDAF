"""
Miscellanous Functions
"""

import os
import torch
from datetime import datetime
from tensorboardX import SummaryWriter
import logging


# Create unique output dir name based on non-default command line args
def make_exp_name(args, parser):
    exp_name = '{}-{}'.format(args.dataset[:4], args.arch[:])
    dict_args = vars(args)

    # sort so that we get a consistent directory name
    argnames = sorted(dict_args)
    ignorelist = ['date', 'exp', 'arch','prev_best_filepath', 'lr_schedule', 'max_cu_epoch', 'max_epoch',
                  'strict_bdr_cls', 'world_size', 'tb_path','best_record', 'test_mode', 'ckpt', 'coarse_boost_classes',
                  'crop_size', 'dist_url', 'syncbn', 'max_iter', 'color_aug', 'scale_max', 'scale_min', 'bs_mult',
                  'class_uniform_pct', 'class_uniform_tile']
    # build experiment name with non-default args
    for argname in argnames:
        if dict_args[argname] != parser.get_default(argname):
            if argname in ignorelist:
                continue
            if argname == 'snapshot':
                arg_str = 'PT'
                argname = ''
            elif argname == 'nosave':
                arg_str = ''
                argname=''
            elif argname == 'freeze_trunk':
                argname = ''
                arg_str = 'ft'
            elif argname == 'syncbn':
                argname = ''
                arg_str = 'sbn'
            elif argname == 'jointwtborder':
                argname = ''
                arg_str = 'rlx_loss'
            elif isinstance(dict_args[argname], bool):
                arg_str = 'T' if dict_args[argname] else 'F'
            else:
                arg_str = str(dict_args[argname])[:7]
            if argname is not '':
                exp_name += '_{}_{}'.format(str(argname), arg_str)
            else:
                exp_name += '_{}'.format(arg_str)
    # clean special chars out    exp_name = re.sub(r'[^A-Za-z0-9_\-]+', '', exp_name)
    return exp_name

def save_log(prefix, output_dir, date_str, rank=0):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' + date_str +'_rank_' + str(rank) +'.log')
    print("Logging :", filename)
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    if rank == 0:
        logging.getLogger('').addHandler(console)
    else:
        fh = logging.FileHandler(filename)
        logging.getLogger('').addHandler(fh)

def prep_experiment(args, parser):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    ckpt_path = args.ckpt
    tb_path = args.tb_path
    exp_name = make_exp_name(args, parser)
    args.exp_path = os.path.join(ckpt_path, args.date, args.exp, str(datetime.now().strftime('%m_%d_%H')))
    args.tb_exp_path = os.path.join(tb_path, args.date, args.exp, str(datetime.now().strftime('%m_%d_%H')))
    args.ngpu = torch.cuda.device_count()
    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    args.best_record = {}

    args.last_record = {}
    if args.local_rank == 0:
        os.makedirs(args.exp_path, exist_ok=True)
        os.makedirs(args.tb_exp_path, exist_ok=True)
        save_log('log', args.exp_path, args.date_str, rank=args.local_rank)
        open(os.path.join(args.exp_path, args.date_str + '.txt'), 'w').write(
            str(args) + '\n\n')
        writer = SummaryWriter(log_dir=args.tb_exp_path, comment=args.tb_tag)
        return writer
    return None
