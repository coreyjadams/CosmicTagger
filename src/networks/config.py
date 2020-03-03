

# This function is to parse strings from argparse into bool
def str2bool(v):
    '''Convert string to boolean value

    This function is from stackoverflow:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Arguments:
        v {str} -- [description]

    Returns:
        bool -- [description]

    Raises:
        argparse -- [description]
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class UResNetConfig(object):

    def __init__(self):
        self._name = "UResNet"
        self._help = "Tensorflow Implementation of multi-plane UResNet"

    def build_parser(self, parser):

        ##################################################################
        # Parameters to control the network implementation
        ##################################################################

        # Layerwise parameters:
        parser.add_argument('-ub','--use-bias', 
            type    = str2bool, 
            default = True,
            help    = "Whether or not to include bias terms in all learned layers.")

        parser.add_argument('-bn','--batch-norm', 
            type    = str2bool, 
            default = True,
            help    = "Whether or not to use batch normalization in all mlp layers.")

        # Network Architecture parameters:
        parser.add_argument('--n-initial-filters', 
            type    = int, 
            default = 16,
            help    = "Number of filters applied, per plane, for the initial convolution.")

        parser.add_argument('--blocks-per-layer', 
            type    = int, 
            default = 2,
            help    = "Number of blocks per layer.")

        parser.add_argument('--blocks-deepest-layer', 
            type    = int, 
            default = 5,
            help    = "Number of blocks applied at the deepest, merged layer.")

        parser.add_argument('--blocks-final', 
            type    = int, 
            default = 0,
            help    = "Number of blocks applied at full, final resolution.")

        parser.add_argument('--network-depth', 
            type    = int, 
            default = 6,
            help    = "Total number of downsamples to apply. Note: ensure depth + downsample =< 8")

        parser.add_argument("--filter-size-deepest", 
            type    = int, 
            default = 5,
            help    = "Convolutional window size for the deepest layer convolution.")

        parser.add_argument("--bottleneck-deepest", 
            type    = int, 
            default = 256,
            help    = "Bottleneck size for deepest layer convolution.")

        parser.add_argument('--connections', 
            type    = str, 
            choices = ['sum', 'concat', 'none'], 
            default = 'sum',
            help    = "Connect shortcuts with sums, concat+bottleneck, or no connections.")

        parser.add_argument('--upsampling', 
            type    = str,
            choices = ["convolutional", "interpolation"], 
            default = "interpolation",
            help    = "Which operation to use for upsamplign.")

        parser.add_argument('--downsampling', 
            type    = str,
            choices = ["convolutional", "max_pooling"], 
            default = "max_pooling",
            help    = "Which operation to use for downsamplign.")

        parser.add_argument('--residual', 
            type    = str2bool, 
            default = True,
            help    = "Use residual units instead of convolutions.")


        parser.add_argument('-gr', '--growth-rate', 
            type    = str, 
            choices = ['multiplicative','additive'], 
            default = "additive",
            help    = "Either double at each layer, or add a constant factor, to the number of filters.")

        parser.add_argument('--block-concat', 
            type    = str2bool, 
            default = False,
            help    = "Block the concatenations at the deepest layer (2D only).")



