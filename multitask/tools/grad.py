import torch


def normalize(v, eps=1e-8):
    return v / (torch.norm(v) + 1e-8)


def cal_gradnorm(grad):
    return torch.norm(grad.flatten(), dim=0, p=2)


def process_grad(v, mode=0):
    out_c, in_c, _, _ = v.shape
    assert mode in [0, 1, 2, 3]
    if mode == 0:
        new_v = v.reshape(in_c * out_c, -1)
    elif mode == 1:
        new_v = torch.transpose(v, 0, 1).contiguous().reshape(in_c, -1)
    elif mode == 3:
        new_v = v.flatten().unsqueeze(0)  # This shape will be (1, n)
    else:
        new_v = v.reshape(out_c, -1)

    return new_v


def recover_grad(v, mode=0, grad_shape=None):
    if mode == 0:
        new_v = v.reshape(grad_shape)
    elif mode == 1:
        new_v = v.reshape(grad_shape[1], grad_shape[0], grad_shape[2], grad_shape[3])
        new_v = torch.transpose(new_v, 0, 1).contiguous()
    elif mode == 3:
        new_v = v.squeeze(0).reshape(grad_shape)
    else:
        new_v = v.reshape(grad_shape)
    return new_v


def get_grad_from_storage(task_names):
    from torchlet.events import get_current_storage
    storage = get_current_storage()
    grads = []
    for k in task_names:
        grad_k = storage.get_buffer(
            "grad/{}".format(k.lower())
        ).clone()
        grads.append(grad_k)
    return grads