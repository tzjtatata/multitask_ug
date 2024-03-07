import os

from torchlet.config import CfgNode as CN
from fvcore.common.file_io import PathManager


def add_multitask_config(cfg):
    """
    Add Config for multitask Project.
    """
    cfg.TRAIN.OPTIMIZER = 'Adam'  # I hear that Adam is better than SGD in SemSeg
    cfg.TRAIN.BALANCER_SWITCH_SCHEDULE = []
    cfg.TRAIN.BALANCER_INHERENT = True
    cfg.TRAIN.SPEC_CHECKPOINTS = []


    cfg.MODEL.BALANCER = ['DefaultBalancer']
    cfg.MODEL.HEADS = ['SemSeg1Conv', 'Depth1Conv']  ## Compact for older version.
    cfg.MODEL.TASK = ['SEMSEG', 'DEPTH']


    cfg.BALANCER = CN()
    cfg.BALANCER.LR = 0.025  # From GradNorm papers, used for both GradNorm and Uncertainty Weights.

    cfg.BALANCER.GRADNORM = CN()
    cfg.BALANCER.GRADNORM.ALPHA = 1.5  # comes from original GradNorm paper 5.4 for NYUv2

    cfg.BALANCER.DTP = CN()
    cfg.BALANCER.DTP.ALPHA = 0.9
    cfg.BALANCER.DTP.GAMMA = 1.0

    cfg.BALANCER.DWA = CN()
    cfg.BALANCER.DWA.T = 2.0

    cfg.BALANCER.UW = CN()
    cfg.BALANCER.UW.INIT_WEIGHTS = [1.0, 1.0]

    cfg.BALANCER.PCGRAD = CN()
    cfg.BALANCER.PCGRAD.MODE = 0  # mode 0: use whole grad; 1: use input related grad; 2: use output related grad


    cfg.TASK = CN()
    cfg.TASK.SEMSEG = CN()
    cfg.TASK.SEMSEG.NUMCLASSES = 40  # For NYU.
    cfg.TASK.SEMSEG.HEAD = 'SemSeg1Conv'

    cfg.TASK.DEPTH = CN()
    cfg.TASK.DEPTH.HEAD = 'Depth1Conv'

    cfg.TASK.CLASSPREDICT = CN()
    cfg.TASK.CLASSPREDICT.NUMCLASSES = 10
    cfg.TASK.CLASSPREDICT.HEAD = "CPFC"


    cfg.DATA = CN()
    cfg.DATA.OUTPUTRES = (80, 80)
    cfg.DATA.INPUTRES = (320, 320)
    cfg.DATA.MAPPER = "NYU"


    cfg.MONITOR = CN()
    cfg.MONITOR.GRAD = CN()
    cfg.MONITOR.GRAD.MODE = 0  # 0 for in_c * out_c, 1 for in_c, 2 for out_c
    cfg.MONITOR.GRAD.OPEN = False
    cfg.MONITOR.GRAD.IS_WEIGHTED = False


def postproc_cfg(cfg):
    # Build Heads according to multi task setting.
    tasks = cfg.MODEL.TASK
    heads = []
    for task in tasks:
        heads.append(cfg.TASK[task].HEAD)
    cfg.MODEL.HEADS = heads

    # compact  Old Version Config before Supporting Balancer Switch Feature.
    if isinstance(cfg.MODEL.BALANCER, str):
        cfg.MODEL.BALANCER = [cfg.MODEL.BALANCER]
        cfg.TRAIN.BALANCER_SWITCH_SCHEDULE = []
    assert cfg.BALANCER.PCGRAD.MODE in [0, 1, 2], "PCGrad's mode should in [0, 1, 2]"

    # detect and write down real output_dir
    root = cfg.LOG.OUTPUT_DIR
    PathManager.mkdirs(root)
    max_num = -1

    for f in os.listdir(root):
        if os.path.isdir(os.path.join(root, f)) and f.startswith("try_"):
            try:
                cur_num = int(f[4:])
            except:
                continue
            if cur_num > max_num:
                max_num = cur_num
    max_num += 1
    cfg.LOG.OUTPUT_DIR = os.path.join(root, "try_{}".format(max_num))

