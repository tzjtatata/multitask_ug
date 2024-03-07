def build_balancer(balancer_type, cfg):
    from lightning.balancer.regist_balancer import BALANCER_REGISTRY
    return BALANCER_REGISTRY.get(balancer_type)(cfg)