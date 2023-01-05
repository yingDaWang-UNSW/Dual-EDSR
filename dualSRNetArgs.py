import argparse

def args():
    # TODO modularise this into concentricGAN noise->clean->SR->seg->vels
    parser = argparse.ArgumentParser(description='')
    #training arguments
    parser.add_argument('--mixedPrecision', dest='mixedPrecision', type=str2bool, default=False, help='16bit computes')
    parser.add_argument('--gpuIDs', dest='gpuIDs', type=str, default='2', help='IDs for the GPUs. Empty for CPU. Nospaces')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='/media/user/SSD2/fuelCellDataset2Dxy/', help='dataset path - include last slash')
    parser.add_argument('--augFlag', dest='augFlag', type=str2bool, default=False, help='augmentation')

    parser.add_argument('--scale', dest='scale', type=str2int, default=4, help='sr scale factor')
    parser.add_argument('--batch_size', dest='batch_size', type=str2int, default=64, help='# 2D images in subbatch')
    parser.add_argument('--subBlocks', dest='subBlocks', type=str2int, default=4, help='# 3D images in batch')
    parser.add_argument('--fine_size', dest='fine_size', type=str2int, default=64, help='then crop LR to this size')

    
    parser.add_argument('--epoch', dest='epoch', type=str2int, default=500, help='# of epoch')        
    parser.add_argument('--itersPerEpoch', dest='itersPerEpoch', type=str2int, default=300, help='# iterations per epoch') 
    parser.add_argument('--iterCyclesPerEpoch', dest='iterCyclesPerEpoch', type=str2int, default=3, help='# iteration cycles per epoch') 

    parser.add_argument('--valNum', dest='valNum', type=str2int, default=10, help='# max val images') 
    parser.add_argument('--valTest', dest='valTest', type=str2bool, default=False, help='# max val images') 

    # base model uses dualEDSR
    parser.add_argument('--ngsrf', dest='ngsrf', type=str2int, default=64, help='# of gen SR filters in first conv layer')
    parser.add_argument('--numResBlocks', dest='numResBlocks', type=str2int, default=16, help='# of resBlocks in SR')

    # extra loss functions

    # base model uses SCGAN
    parser.add_argument('--ganFlag', dest='ganFlag', type=str2bool, default=False, help='if gan is active') 
    parser.add_argument('--ndsrf', dest='ndsrf', type=str2int, default=64, help='# of disc SR filters in first conv layer')
    parser.add_argument('--srAdv_lambda', dest='srAdv_lambda', type=str2float, default=1e-2, help='weight on Adv term for normal sr')
    parser.add_argument('--disc_size_max', dest='disc_size', type=str2int, default=64, help='then crop HR to this size during disc')
    
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--epoch_step', dest='epoch_step', type=str2int, default=50, help='# of epoch to decay lr')

    parser.add_argument('--phase', dest='phase', type=str, default='train', help='train, test')
    
    # Model IO
    parser.add_argument('--save_freq', dest='save_freq', type=str2int, default=10, help='save a model every save_freq epochs')
    parser.add_argument('--print_freq', dest='print_freq', type=str2int, default=10, help='print the validation images every X epochs')
    parser.add_argument('--continue_train', dest='continue_train', type=str2bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
    parser.add_argument('--continueEpoch', dest='continueEpoch', type=str2int, default=0, help='')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoints', help='models are saved here')
    parser.add_argument('--modelName', dest='modelName', default='dual2DSRTest', help='models are loaded here')
    
    # testing arguments
    parser.add_argument('--test_dir', dest='test_dir', default='/media/user/SSD2/testLR/', help='test sample slices are saved here as png slices')
    parser.add_argument('--test_temp_save_dir', dest='test_temp_save_dir', default='/media/user/SSD2/', help='test sample are saved here')
    parser.add_argument('--test_save_dir', dest='test_save_dir', default='/media/user/SSD2/', help='test sample are saved here')
    args = parser.parse_args()

    return args
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2int(v):
    if v=='M':
        return v
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError('int value expected.')
    return v
    
def str2float(v):
    if v=='M':
        return v
    try:
        v = float(v)
    except:
        raise argparse.ArgumentTypeError('float value expected.')
    return v
        
