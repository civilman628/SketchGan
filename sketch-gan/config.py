import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Model hyper-parameters
parser.add_argument('--model', type=str, default='sagan', choices=['sagan', 'qgan'])
parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['wgan-gp', 'hinge'])
parser.add_argument('--imsize', type=int, default=256)
#parser.add_argument('--g_num', type=int, default=5)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--g_conv_dim', type=int, default=64)
parser.add_argument('--d_conv_dim', type=int, default=64)
parser.add_argument('--lambda_gp', type=float, default=10)
parser.add_argument('--version', type=str, default='sagan_1')
parser.add_argument('--ch', type=int, default=64)

# Training setting
parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
parser.add_argument('--d_iters', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--g_lr', type=float, default=0.00005)
parser.add_argument('--d_lr', type=float, default=0.0002)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.999)

# using pretrained
parser.add_argument('--pretrained_model', type=int, default=None)

# Misc
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--parallel', type=str2bool, default=True)
parser.add_argument('--dataset', type=str, default='dresses', choices=['dresses', 'tops'])
parser.add_argument('--use_tensorboard', type=str2bool, default=False)

# Path
parser.add_argument('--image_path', type=str, default='../../data/')
parser.add_argument('--log_path', type=str, default='./logs')
parser.add_argument('--model_save_path', type=str, default='./models')
parser.add_argument('--sample_path', type=str, default='./samples')
parser.add_argument('--attn_path', type=str, default='./attn')

# Step size
parser.add_argument('--log_step', type=int, default=10) #10
parser.add_argument('--sample_step', type=int, default=500) #100
parser.add_argument('--model_save_step', type=float, default=500) #1.0

def get_config():
    args = parser.parse_args()
    return args
