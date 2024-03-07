from textwrap import wrap
from torchlet.registry import Registry


HEAD_REGISTRY = Registry("HEAD")


def build_heads(cfg):
    if cfg.dataset.name =='celeba' and cfg.dataset.task.multi_label:
        assert len(cfg.dataset.task.tasks) == 1, "dataset.task.multi_label is True while len(task.tasks) == 1."
        new_task_names = [str(i) for i in range(40)]
        heads = {
            task_name: HEAD_REGISTRY.get(
                cfg.dataset.model['attr'].head)(cfg, 'attr')
            for task_name in new_task_names
        }
        wrapper = {
            cfg.dataset.task.tasks[0]: HEAD_REGISTRY.get("CelebAWrapper")(cfg, heads, new_task_names)
        }
        return wrapper
    
    task_names = [t.lower() for t in cfg.dataset.task.tasks]
    if 'aux_tasks' in cfg.dataset.task: 
        task_names += [t.lower() for t in cfg.dataset.task.aux_tasks]
    heads = {
        task_name: HEAD_REGISTRY.get(
            cfg.dataset.model[task_name].head)(cfg, task_name)
        for task_name in task_names
    }
    return heads