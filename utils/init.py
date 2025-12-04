import os
import random
from pathlib import Path
import thop
import torch

from models import dualimrunet_up
from utils import logger, line_seg

__all__ = ["init_device", "init_model_up", "build_run_tag"]


def build_run_tag(args, include_quant=False):
    """Create a consistent tag used for checkpoints and logging."""
    tag = (
        f"dual_env{args.scenario}{args.env_num}"
        f"_eig{args.eig_flag}"
        f"_enh{args.enhanced_eigenvector_flag}"
        f"_ad{args.ad_flag}"
        f"_sp{args.spalign_flag}"
        f"_cr{args.cr}"
        f"_d{args.d_model}"
        f"_lr{args.scheduler}"
    )
    # quantization args removed; hook kept for compatibility
    return tag


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model_up(args):
    # Model loading
    model = dualimrunet_up(reduction=args.cr, d_model=args.d_model)

    if args.pretrained is not None:
        pretrained_dir = Path("./checkpoints") / build_run_tag(args) / "best_rho.pth"
        assert os.path.isfile(pretrained_dir)
        state_dict = torch.load(pretrained_dir,
                                map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict,strict=False)
        logger.info("pretrained model loaded from {}".format(pretrained_dir))

    # Model flops and params counting
    H_a = torch.randn([1,2,32,13])
    H_mag_up = torch.randn([1,1,32,13])
    flops, params = thop.profile(model, inputs=(H_a,H_mag_up,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    logger.info(f'=> Model Name: DualImRUNet [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
