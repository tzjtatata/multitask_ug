#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from lightning.utils.utils import get_output, mkdir_if_missing


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, db_name, tasks):
        self.database = db_name
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        tasks = self.tasks
        if len(pred.keys()) != len(self.tasks):
            tasks = list(pred.keys())
        for t in tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict


def calculate_multi_task_performance(eval_dict, single_task_dict):
    # assert(set(eval_dict.keys()) == set(single_task_dict.keys())), "Eval: {}, STL: {}".format(list(eval_dict.keys()), list(single_task_dict.keys()))
    # for k in eval_dict.keys():
    #     assert k in single_task_dict, "Key {} Not in Single Task Results.".format(k)
    # tasks = eval_dict.keys()  # origin version
    tasks = [k for k in single_task_dict.keys() if k in eval_dict]  # 我们只计算给出了Single Task数据的任务，并且用于计算的任务。
    num_tasks = len(tasks)    
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]
        
        if task == 'depth' or task == 'city_depth': # rmse lower is better
            if 'rmse' in stl:
                mtl_performance -= (mtl['rmse'] - stl['rmse'])/stl['rmse']
            else:
                mtl_performance -= (mtl['abs_err'] - stl['abs_err'])/stl['abs_err']

        elif task in ['semseg', 'sal', 'human_parts', 'city_semseg', 'part_seg']: # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU'])/stl['mIoU']

        elif task in ['detection']: # mIoU higher is better
            mtl_performance += (mtl['AP'] - stl['AP'])/stl['AP']

        elif task == 'normals': # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean'])/stl['mean']

        elif task == 'edge': # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF'])/stl['odsF']

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks

SUPPORT_EVAL_TASKS = ['semseg', 'depth', 'coco_semseg', 'city_semseg', 'normals', 'part_seg']
def get_single_task_meter(database, task):
    """ Retrieve a meter to measure the single-task performance """
    if task == 'semseg':
        from lightning.eval.eval_semseg import SemsegMeter
        if database.startswith('NYU'):
            return SemsegMeter(database)
        elif database.startswith('loren'):
            return SemsegMeter("Cityscapes")

    elif task == 'part_seg':
        from lightning.eval.eval_semseg import PartSemsegMeter
        return PartSemsegMeter(database)
    
    elif task == 'city_semseg':
        from lightning.eval.eval_semseg import SemsegMeter
        return SemsegMeter("Cityscapes")
    
    elif task == 'coco_semseg':
        from lightning.eval.eval_semseg import SemsegMeter
        return SemsegMeter("PASCALContext")

    elif task == 'depth' or task.endswith('depth'):
        from lightning.eval.eval_depth import DepthMeter, LorenDepthMeter
        if database.startswith('loren'):
            return LorenDepthMeter()
        
        return DepthMeter()
    
    elif task == 'normals':
        from lightning.eval.eval_normals import NormalsMeter
        return NormalsMeter()
    
    elif task == 'imagenet':
        return 

    else:
        print(task)
        raise NotImplementedError


def validate_results(p, current, reference):
    """
        Compare the results between the current eval dict and a reference eval dict.
        Returns a tuple (boolean, eval_dict).
        The boolean is true if the current eval dict has higher performance compared
        to the reference eval dict.
        The returned eval dict is the one with the highest performance.
    """
    tasks = p.TASKS.NAMES
    
    if len(tasks) == 1: # Single-task performance
        task = tasks[0]
        if task == 'semseg': # Semantic segmentation (mIoU)
            if current['semseg']['mIoU'] > reference['semseg']['mIoU']:
                print('New best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = True
            else:
                print('No new best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = False
        
        elif task == 'human_parts': # Human parts segmentation (mIoU)
            if current['human_parts']['mIoU'] > reference['human_parts']['mIoU']:
                print('New best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = True
            else:
                print('No new best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = False

        elif task == 'sal': # Saliency estimation (mIoU)
            if current['sal']['mIoU'] > reference['sal']['mIoU']:
                print('New best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = True
            else:
                print('No new best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = False

        elif task == 'depth': # Depth estimation (rmse)
            if current['depth']['rmse'] < reference['depth']['rmse']:
                print('New best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = True
            else:
                print('No new best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = False
        
        elif task == 'normals': # Surface normals (mean error)
            if current['normals']['mean'] < reference['normals']['mean']:
                print('New best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = True
            else:
                print('No new best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = False

        elif task == 'edge': # Validation happens based on odsF
            if current['edge']['odsF'] > reference['edge']['odsF']:
                print('New best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = True
            
            else:
                print('No new best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = False


    else: # Multi-task performance
        if current['multi_task_performance'] > reference['multi_task_performance']:
            print('New best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = True

        else:
            print('No new best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = False

    if improvement: # Return result
        return True, current

    else:
        return False, reference


@torch.no_grad()
def eval_model(p, val_loader, model):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p)

    model.eval()

    for i, batch in enumerate(val_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}
        output = model(images)

        # Measure performance
        performance_meter.update({t: get_output(output[t], t) for t in tasks}, targets)

    eval_results = performance_meter.get_score(verbose = True)
    return eval_results
