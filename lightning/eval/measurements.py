import torch
import math
import numpy as np
from typing import Dict


def build_measurement(cfg, task_names) -> Dict:
        d_name = cfg.dataset.name
        predefined = {
            # from 0 to 1, 'semseg' and 'depth' mode
            "NYUMT": (1, 0),
            "cityscapes": (-1, 1),
            "cityscapes_npy": (0, 1)
        }

        def _consturct(t):
            if t == "semseg":
                return SemanticMeasurement(cfg.dataset.meta_data.semseg.num_classes, mode=predefined[d_name][0])
            elif t == "depth":
                return DepthMeasurement(mode=predefined[d_name][1])
            elif t == 'attr':
                return MultiLabelMeasurement(cfg.dataset.meta_data[t].num_classes, t)
            elif t in ["upper", "lower", "middle", "whole"]:
                return MultiLabelMeasurement(cfg.task[t].num_classes, t)
            elif t == 'landmarks':
                return LandmarkMeasurement()
            elif t == 'instance':
                return InstanceMeasurement()
            elif t == 'classpredict':
                return ClassifyMeasurement(cfg.dataset.meta_data[t].num_classes, t)
            else:
                raise NotImplementedError("Not Support task {}".format(t))
        return {
            t: _consturct(t)
            for t in task_names
        }


class DepthMeasurement:

    def __init__(self, mode=0):
        """
        params:
            mode: Mask Invalid depth or not. In nyuv2_13, no invalid. In cityscapes, 0.0 is invalid
        """
        self.mode = mode
        assert self.mode in [0, 1], "Not SUPPORT mode {}".format(self.mode)

    def step(self, pred, target, eps=1e-8):
        if self.mode == 0:
            valid_num = target.flatten().shape[0]
            diff = target.flatten() - pred.flatten()
            real_pred = pred.flatten()
        else:
            binary_mask = (target != 0)
            pred_masked = pred.squeeze().masked_select(binary_mask)
            target_masked = target.masked_select(binary_mask)
            diff = pred_masked - target_masked
            real_pred = pred_masked
            valid_num = torch.nonzero(binary_mask, as_tuple=False).size(0)
            # print(valid_num, binary_mask.shape)
        mse = torch.sum(diff ** 2)
        abs_err = torch.sum(torch.abs(diff))
        rel_err = torch.sum(torch.abs(diff) / real_pred)
        return np.array([mse.item(), abs_err.item(), rel_err.item(), valid_num], dtype=np.float)

    def summary(self, outs):
        rmse = math.sqrt(outs[0] / outs[-1])
        abs_err = outs[1] / (1.0 * outs[-1])
        rel_err = outs[2] / (1.0 * outs[-1])
        ret = {
            'depth/rmse': rmse,
            'depth/abs_err': abs_err,
            'depth/rel_err': rel_err
        }
        return ret


class InstanceMeasurement:

    def step(self, pred, target, eps=1e-8):
        binary_mask = (target != 250.0)
        pred_masked = pred.masked_select(binary_mask)
        target_masked = target.masked_select(binary_mask)
        diff = pred_masked - target_masked
        valid_num = torch.nonzero(binary_mask, as_tuple=False).size(0)
        mse = torch.sum(diff ** 2)
        abs_err = torch.sum(torch.abs(diff))
        return np.array([mse.item(), abs_err.item(), valid_num], dtype=np.float)

    def summary(self, outs):
        rmse = math.sqrt(outs[0] / outs[-1])
        abs_err = outs[1] / (1.0 * outs[-1])
        ret = {
            'instance/rmse': rmse,
            'instance/abs_err': abs_err,
        }
        return ret


class SemanticMeasurement:

    def __init__(self, num_classes, mode=0):
        """
        params:
            num_classes: the number of classes, which is valid in evaluation.
            mode: Evaluation mode, 0 for No Ignored class, 1 for ignore last class, -1 for ignore class 0
        """
        self.num_classes = num_classes
        self.mode = mode
        assert self.mode in [0, 1, -1], "Not SUPPORT MODE {}".format(self.mode)
        self.real_num_classes = self.num_classes if self.mode == 0 else self.num_classes+1

    def step(self, pred, target):
        num_classes = self.real_num_classes
        valid_mask = (target >= 0) & (target < self.num_classes)
        pred = pred.argmax(dim=1)[valid_mask].flatten()
        gt = target[valid_mask].flatten()
        return torch.bincount(num_classes * pred + gt, minlength=num_classes ** 2).reshape(
            num_classes, num_classes).detach().cpu().numpy()

    def summary(self, outs):
        num_classes = self.num_classes
        acc = np.full(num_classes, np.nan, dtype=np.float)
        iou = np.full(num_classes, np.nan, dtype=np.float)
        # below, we do not count for the last class -- class 40, cause it is None Class.
        if self.mode == 1:
            outs = outs[:-1, :-1]
        elif self.mode == -1:
            outs = outs[1:, 1:]
        tp = outs.diagonal().astype(np.float)
        pos_gt = np.sum(outs, axis=0).astype(np.float)
        pos_pred = np.sum(outs, axis=1).astype(np.float)

        acc_valid = pos_gt > 0
        iou_valid = (pos_gt + pos_pred) > 0

        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]

        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        pacc = np.sum(tp) / np.sum(pos_gt)

        ret = {
            "semseg/mIoU": 100 * miou.item(),
            "semseg/mAcc": 100 * macc.item(),
            "semseg/pAcc": 100 * pacc.item(),
        }
        return ret


class MultiLabelMeasurement:

    def __init__(self, num_classes, task_name):
        self.num_classes = num_classes
        self.task_name = task_name

    def step(self, pred, target):
        # pred, target: (BS, num_classes)
        pred = torch.round(pred)
        count = pred.size(0)
        # print(pred.shape)
        # print(target.shape)
        # print(count)
        pos = torch.sum((pred == target), dim=0).to(torch.int64)  # (1, 40)
        # print(pos)
        return pos, count

    def summary(self, outs):
        return {
            "{}/mAcc".format(self.task_name): torch.mean(outs[0] * 1.0 / outs[1]).item() * 100.0
        }


class ClassifyMeasurement(MultiLabelMeasurement):

    def step(self, pred, target):
        # pred, target: (BS, num_classes)
        pred = torch.argmax(pred, dim=1)
        # print(pos)
        return torch.bincount(self.num_classes * pred + target, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes).detach().cpu().numpy()

    def summary(self, outs):
        num_classes = self.num_classes
        acc = np.full(num_classes, np.nan, dtype=np.float)
        tp = outs.diagonal().astype(np.float)
        pos_gt = np.sum(outs, axis=0).astype(np.float)

        acc_valid = pos_gt > 0

        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]

        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)

        ret = {
            "Classify/mAcc": 100 * macc.item(),
        }
        return ret


class LandmarkMeasurement:

    def step(self, pred, target):
        # pred, target: (BS, 10)
        with torch.no_grad():
            # sum on batch, not on a image.
            error = torch.mean(torch.abs(pred - target))
        return error, 1.0

    def summary(self, outs):
        return {
            "LandmarkReg/abs_err": outs[0] * 1.0 / outs[1]
        }
