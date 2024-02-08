import argparse
import json
import os
import torch
import warnings

from pushover import Client
from tqdm import tqdm

from attack_methods import fgsm_attack, igs_attack, mim_attack, style_trans_attack, no_style_trans_attack, pgd_attack, \
    pgd_attack_st, cw_attack, vmi_fgsm_attack, pgd_attack_fg
from loader.attack_loader import get_attack_loader
from loss.contrastive_loss import ContrastiveLoss
from net.signet import Encoder

torch.manual_seed(3407)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--attack_method', type=str, default='pgd_fg',
                    choices=['fgsm', 'igs', 'mim', 'ours', 'pgd', 'pgd_st', 'pgd_fg', 'deepfool', 'no_stytrans', 'cw',
                             'vmi-fgsm'])
parser.add_argument('--dataset', type=str, choices=['CEDAR', 'Bengali'],default='CEDAR')
parser.add_argument('--attack_label', type=int, choices=[0, 1])
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--beta', type=float, default=1.5)
parser.add_argument('--mu', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--c', type=float, default=1e-4)
parser.add_argument('--overshoot', type=float, default=0.02)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--num_VT', type=int, default=5)
parser.add_argument('--layers', type=list, default=[0, 4, 8, 9])
parser.add_argument('--attack_weight', type=float, default=1e1)
parser.add_argument('--tv_weight', type=float, default=1e2)
parser.add_argument('--style_weight', type=float, default=1e11)
parser.add_argument('--diff_weight', type=float, default=1e-3)
parser.add_argument('--device', default='cuda')
parser.add_argument('--threshold', default=0.03)
parser.add_argument('--contrastive_loss_fn', default=None)
args = parser.parse_args()
if args.attack_method in ['ours', 'no_stytrans'] and args.attack_label == 1:
    raise ValueError('The selected attack method can only attack data pairs with label==1.')

# load the dataset info
current_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_script_dir, r'configs/dataset_config.json'), 'r') as f:
    dataset_config = json.load(f)
contrastive_loss_fn = ContrastiveLoss(1, 1, 1).to(args.device)

# give the correct values
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.threshold = dataset_config[args.dataset]['threshold']
args.contrastive_loss_fn = ContrastiveLoss(1, 1, 1).to(args.device)

# map the attack methods
attack_methods = {
    'fgsm': fgsm_attack,
    'mim': mim_attack,
    'igs': igs_attack,
    'pgd': pgd_attack,
    'ours': style_trans_attack,
    'no_stytrans': no_style_trans_attack,
    'pgd_st': pgd_attack_st,
    'pgd_fg': pgd_attack_fg,
    'cw': cw_attack,
    'vmi-fgsm': vmi_fgsm_attack
}
attack_method = attack_methods[args.attack_method]
print_info = {
    'vmi-fgsm': {
        'epsilon': args.epsilon,
        'epochs': args.epochs,
        'mu': args.mu,
        'threshold': args.threshold,
        'num_VT': args.num_VT,
        'beta': args.beta
    },
    'fgsm': {
        'epsilon': args.epsilon,
        'threshold': args.threshold,
    },
    'igs': {
        'epsilon': args.epsilon,
        'epochs': args.epochs,
        'threshold': args.threshold,
    },
    'mim': {
        'epsilon': args.epsilon,
        'epochs': args.epochs,
        'mu': args.mu,
        'threshold': args.threshold,
    },
    'pgd': {
        'epsilon': args.epsilon,
        'epochs': args.epochs,
        'alpha': args.alpha,
        'threshold': args.threshold,
    },
    'pgd_st': {
        'epsilon': args.epsilon,
        'epochs': args.epochs,
        'alpha': args.alpha,
        'threshold': args.threshold,
        'tv_weight': args.tv_weight,
        'style_weight': args.style_weight,
    },
    'pgd_fg': {
        'epsilon': args.epsilon,
        'epochs': args.epochs,
        'alpha': args.alpha,
        'threshold': args.threshold,
    },
    'cw': {
        'epochs': args.epochs,
        'c': args.c,
        'threshold': args.threshold,
        'lr': args.lr,
    },

    'ours': {
        'lr': args.lr,
        'layers': args.layers,
        'attack_weight': args.attack_weight,
        'tv_weight': args.tv_weight,
        'style_weight': args.style_weight,
        'diff_weight': args.diff_weight,
        'threshold': args.threshold,
    },
    'no_stytrans': {
        'lr': args.lr,
        'layers': args.layers,
        'attack_weight': args.attack_weight,
        'tv_weight': args.tv_weight,
        'diff_weight': args.diff_weight,
        'threshold': args.threshold,
    }
}

print('-' * 60)
print('Starting "{} attack" on dataset "{}", targeting on the data pairs have label = {}'.format(args.attack_method,
                                                                                                 args.dataset,
                                                                                                 args.attack_label))
print('Hyperparameters:')
for k in print_info[args.attack_method].keys():
    print('\t{}: {}'.format(k, print_info[args.attack_method][k]))

# prepare the model we will attack
net = Encoder()
net.load_state_dict(torch.load(dataset_config[args.dataset]['model_path']))
net.requires_grad_(False)
net = net.eval()
net = net.to(args.device)

# prepare data loader
loader = get_attack_loader(
    label_path=dataset_config[args.dataset]['label'],
    batch_size=1,
    data_dir=dataset_config[args.dataset]['data_path'],
)

# starting attack
acc_item = 0
diffs = []
pbar = tqdm(total=len(loader))
for batch_idx, (x1, x2) in enumerate(loader):
    x1, x2 = x1.to(args.device), x2.to(args.device)
    gen_img, acc, loss_his, diff = attack_method(model=net, data=(x1, x2), args=args)
    acc_item += acc
    diffs.append(diff)
    pbar.update(1)
    pbar.set_postfix(num_acc=acc_item, acc=acc_item / (batch_idx + 1))
pbar.close()

print(r'Attack finished, total success rate: {} \ {} = {:.2%}'.format(acc_item, len(loader), acc_item / len(loader)))

client = Client(config_path=os.path.join(current_script_dir, r'configs/pushover.conf'), profile='S23U')
msg = 'Starting "{} attack" on dataset "{}", targeting on the data pairs have label = {}\n'.format(args.attack_method,
                                                                                                   args.dataset,
                                                                                                   args.attack_label)
msg += 'Hyperparameters:\n' + '\n'.join(
    '\t{}: {}'.format(k, print_info[args.attack_method][k]) for k in print_info[args.attack_method].keys())
msg += '\nAttack finished, total success rate: {} \ {} = {:.2%}'.format(acc_item, len(loader), acc_item / len(loader))
if args.attack_method in ['ours', 'no_stytrans']:
    msg += '\naverage diff: {:.4f}'.format(sum(diffs) / len(diffs))
client.send_message(msg)
print('Message is pushed to mobile phone')
