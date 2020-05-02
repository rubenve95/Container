from project_code.models.deepLab.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, options):
    if backbone == 'resnet':
        return resnet.ResNet(resnet.Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, options, pretrained=True)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
