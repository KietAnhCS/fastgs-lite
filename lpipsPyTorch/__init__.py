import torch

from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1',
          normalize: bool = True):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
        normalize (bool): inputs are in [0, 1] and must be mapped to [-1, 1]
                        first. The ScalingLayer constants in BaseNet.z_score
                        (mean -.030/-.088/-.188, std .458/.448/.450) are the
                        official LPIPS ones and assume [-1, 1] input; feeding
                        [0, 1] directly gives a value that is not LPIPS as
                        defined in Zhang et al. Same as `normalize=True` in the
                        official `lpips` package.
    """
    if normalize:
        x, y = 2.0 * x - 1.0, 2.0 * y - 1.0
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
