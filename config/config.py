import argparse


def train_config():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, required=True, default='pep', choices=['pep', 'mpep'])
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')

    # training related
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--patience', type=int, default=8, help='patience before early stopping')
    parser.add_argument('--warmup', type=int, default=1000, help='warm-up steps during training')
    parser.add_argument('--save_topk', type=int, default=-1,
                        help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=8)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # model
    parser.add_argument('--model_type', type=str, default='bbm', required=True, choices=['bbm', 'fbm'])
    parser.add_argument('--baseline', type=str, help='baseline model ckpt if train FBM')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden state')
    parser.add_argument('--rbf_dim', type=int, default=8, help='dimension of RBF')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--layers', type=int, default=2, help='number of layers')
    parser.add_argument('--cutoff', type=float, default=5.0, help='radial threshold, unit: Angstrom')
    parser.add_argument('--s_eu', type=float, default=0.1, help='noise scale for bridge matching')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    return parser.parse_args()


def inference_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, default='none', help='specify any name for the test protein')
    parser.add_argument('--test_set', type=str, required=True, help='path to test pdb')
    parser.add_argument('--ckpt', type=str, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to save inference structures')
    parser.add_argument('--inf_step', type=int, required=True, default=1000, help='inference steps')
    parser.add_argument('--sde_step', type=int, default=30, help='SDE steps per sample')
    parser.add_argument('--seed', type=int, default=42, help='random seed for inference')
    parser.add_argument('--guidance', type=float, default=0.1, help='guidance strength for FBM inference')
    parser.add_argument('--bs', type=int, default=1, help='batch size for inference')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')

    return parser.parse_args()
