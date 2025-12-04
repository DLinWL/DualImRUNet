import argparse

parser = argparse.ArgumentParser(description='DualImRUNet PyTorch Training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--data-dir', type=str, required=False,
                    help='the path of dataset.')
parser.add_argument('--scenario', type=str, required=False, choices=["in", "out"],default="out",
                    help="the channel scenario")
parser.add_argument('-b', '--batch-size', type=int, required=False, metavar='N',default=200,
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, metavar='N', required=False, default=3,
                    help='number of data loading workers')


# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N', default=60,
                    help='number of total epochs to run')
parser.add_argument('--cr', metavar='N', type=int, default=32,
                    help='compression ratio')
parser.add_argument('-d', '--d_model', type=int, default=32, metavar= 'N', help= 'number of Transformer feature dimension.' )
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')
parser.add_argument('--eig_flag', metavar='N', type=int, choices=[0, 1],  default=1,
                    help='eigenvector flag')
parser.add_argument('--ad_flag', metavar='N', type=int, choices=[0, 1],  default=1,
                    help='angular delay domain flag')
parser.add_argument('--spalign_flag', metavar='N', type=int, choices=[0, 1],  default=0,
                    help='sparsity aligning flag')
parser.add_argument('--enhanced_eigenvector_flag', metavar='N', type=int, choices=[0, 1],  default=0,
                    help='enhanced eigenvector design flag')
parser.add_argument('--env_num', metavar='N', type=int, choices=[1, 2, 4, 8, 16, 32, 70, 100],  default=70,
                    help='number of environment for training set')
parser.add_argument('--quan_bits', type=int, default=4, choices=[1, 2, 3, 4, 5, 6, 7, 8], help='from {1, 2, 3, 4, 5, 6, 7, 8}')
parser.add_argument('--quan_compander', type=str, default='no', help='from {mean, mu, no}')

args = parser.parse_args()
