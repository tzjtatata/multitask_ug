from torchlet.backbone import BACKBONE_REGISTRY


def build_backbone(cfg):
    return BACKBONE_REGISTRY.get(cfg.dataset.model.backbone.type)(cfg)