import torch
import torch.nn as nn
import torch.nn.functional as F
from project_code.models.deepLab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from project_code.models.deepLab.aspp import build_aspp
from project_code.models.deepLab.decoder import build_decoder
from project_code.models.deepLab.backbone import build_backbone

class RegressionModel(nn.Module):
    def __init__(self, options, backbone='resnet', output_stride=16,
                 sync_bn=True, freeze_bn=False):
        super(RegressionModel, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, options)
        