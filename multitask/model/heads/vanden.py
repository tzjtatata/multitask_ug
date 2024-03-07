from asyncio import Task
from tkinter import Image
import torch
from torch import nn
from torch.nn import functional as F

from lightning.utils.configs import configurable
from multitask.model.heads.build import HEAD_REGISTRY
from .base import BasicHead
from .aspp_head import DeepLabHead
from .hrnet_head import HighResolutionHead


class SoftMaxwithLoss(nn.Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self, ignore_index=255):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=ignore_index)
    
    def _forward_without_reduction(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = F.nll_loss(self.softmax(out), label, reduction='none', ignore_index=255)
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=1)
        return loss

    def forward(self, out, label, ret_example_level=False):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = self.criterion(self.softmax(out), label)
        return loss


class SoftMaxwithExampleLoss(nn.Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self):
        super(SoftMaxwithExampleLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)
    
    def _forward_without_reduction(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = F.nll_loss(self.softmax(out), label, reduction='none', ignore_index=255)
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=1)
        return loss

    # example-level loss
    def forward(self, out, label, ret_example_level=False):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = F.nll_loss(self.softmax(out), label, reduction='none', ignore_index=255)
        loss = torch.mean(loss.reshape(loss.shape[0], -1), dim=1)
        if ret_example_level:
            return loss
        # 不做平衡
        return torch.mean(loss)

        # 进行平衡
        # return torch.mean(loss / loss.detach())


class SoftMaxwithSoftLabel(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x 1000
        # label shape batch_size x 1000 是软标签
        loss = -torch.sum(self.softmax(out) * label)  # - sum ( q_k * log p_k ) q_k和p_k都是logits，是softmax之后的结果。
        return loss


class SoftMaxwithHardLabel(nn.Module):

    def __init__(self):
        super().__init__()
        # self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x 1000
        # label shape batch_size x 1
        label = label.long()
        # loss = -torch.sum(self.softmax(out) * label)  # - sum ( q_k * log p_k ) q_k和p_k都是logits，是softmax之后的结果。
        loss = self.criterion(out, label)
        return loss


class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.
    """
    def __init__(self, loss='l1'):
        super(DepthLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError('Loss {} currently not supported in DepthLoss'.format(loss))
        
    def _forward_without_reduction(self, out, label):
        mask = (label != 255)
        loss = F.l1_loss(out, label, reduction='none')  # (bs, 1, 640, 480)
        loss = loss * mask.int()
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=1) / torch.sum(mask.reshape(mask.shape[0], -1), dim=1)
        return loss

    def forward(self, out, label, ret_example_level=False):
        mask = (label != 255)
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class LorenDepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.
    """

    def _forward_without_reduction(self, out, label):
        pass

    def forward(self, out, label, ret_example_level=False):
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1
        valid_mask = (torch.sum(label, dim=1, keepdim=True) != invalid_idx)
        loss = torch.sum(F.l1_loss(out, label, reduction='none').masked_select(valid_mask)) \
               / torch.nonzero(valid_mask, as_tuple=False).size(0)
        return loss


class DepthExampleLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.
    """
    def __init__(self, loss='l1'):
        super(DepthExampleLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError('Loss {} currently not supported in DepthLoss'.format(loss))
        
    def _forward_without_reduction(self, out, label):
        mask = (label != 255)
        loss = F.l1_loss(out, label, reduction='none')  # (bs, 1, 640, 480)
        loss = loss * mask.int()
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=1) / torch.sum(mask.reshape(mask.shape[0], -1), dim=1)
        return loss

    # example-level loss
    def forward(self, out, label, ret_example_level=False):
        mask = (label != 255)
        loss = F.l1_loss(out, label, reduction='none')  # (bs, 1, 640, 480)
        loss = loss * mask.int()
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=1) / torch.sum(mask.reshape(mask.shape[0], -1), dim=1)
        # 不做平衡
        return torch.mean(loss)

        # 进行平衡
        # return torch.mean(loss / loss.detach())


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(nn.Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """
    def __init__(self, size_average=True, normalize=True, norm=1):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError
    
    def _forward_without_reduction(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)

        if self.normalize is not None:
            out_norm = self.normalize(out)
        else:
            out_norm = out
        
        loss = self.loss_func(out_norm, label, reduction='none')  # (bs, 1, 640, 480)
        loss = loss * (mask.int() * 1.0)
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=1)
        return loss

    def forward(self, out, label, ignore_label=255, ret_example_level=False):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
        else:
            out_norm = out
        
        loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss


class NormalsExampleLoss(nn.Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """
    def __init__(self, size_average=True, normalize=True, norm=1):
        super(NormalsExampleLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError
    
    def _forward_without_reduction(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)

        if self.normalize is not None:
            out_norm = self.normalize(out)
        else:
            out_norm = out
        
        loss = self.loss_func(out_norm, label, reduction='none')  # (bs, 1, 640, 480)
        loss = loss * (mask.int() * 1.0)
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=1)
        return loss
    
    # provides example-level loss.
    def forward(self, out, label, ignore_label=255, ret_example_level=False):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
        else:
            out_norm = out
        
        loss = self.loss_func(out_norm, label, reduction='none')  # (bs, 1, 640, 480)
        loss = loss * (mask.int() * 1.0)
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=1)

        if ret_example_level:
            return torch.div(loss, max(n_valid, 1e-6))

        if self.size_average:
            if ignore_label:
                # 不做平衡
                ret_loss = torch.div(torch.sum(loss), max(n_valid, 1e-6))
                # 做example-level的平衡
                # ret_loss = torch.div(loss, max(n_valid, 1e-6))
                # ret_loss = torch.sum(ret_loss / ret_loss.detach())
                # return ret_loss
            else:
                ret_loss = torch.div(torch.sum(loss), float(np.prod(label.size())))
                return ret_loss


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class TokenLabelCrossEntropy(nn.Module):
    """
    Token labeling loss.
    """
    def __init__(self, dense_weight=1.0, cls_weight = 1.0, mixup_active=True, classes = 1000, ground_truth = False):
        """
        Constructor Token labeling loss.
        """
        super(TokenLabelCrossEntropy, self).__init__()


        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        self.ground_truth = ground_truth
        assert dense_weight+cls_weight>0


    def forward(self, x, target):
        # 这里原来输出会给出一个bbx，我看了一下是mix augmentation方面的，我们暂时不做那个，所以我先删去了。
        output, aux_output = x

        B,N,C = aux_output.shape
        if len(target.shape)==2:
            target_cls=target
            target_aux = target.repeat(1,N).reshape(B*N,C)
        else: 
            target_cls = target[:,:,1]
            # print("In Loss fn, target_cls is:", target_cls)
            # print("In Loss fn, gt is:", target[:, :, 0])
            if self.ground_truth:
                # use ground truth to help correct label.
                # rely more on ground truth if target_cls is incorrect.
                ground_truth = target[:,:,0]
                ratio = (0.9 - 0.4 * (ground_truth.max(-1)[1] == target_cls.max(-1)[1])).unsqueeze(-1)
                target_cls = target_cls * ratio + ground_truth * (1 - ratio)
            target_aux = target[:,:,2:]
            target_aux = target_aux.transpose(1,2).reshape(-1,C)

        aux_output = aux_output.reshape(-1,C)
        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)
        return self.cls_weight*loss_cls+self.dense_weight* loss_aux
        # return loss_aux


TASK_INFO = {
    "semseg": {
        "num_classes": 40,
        "criterion": SoftMaxwithLoss,
        "kwargs": {}
    },
    "example_semseg": {
        "num_classes": 40,
        "criterion": SoftMaxwithExampleLoss,
        "kwargs": {}
    },
    "city_semseg": {
        "num_classes": 19,
        "criterion": SoftMaxwithLoss,
        "kwargs": {}
    },
    "coco_semseg": {
        "num_classes": 21,
        "criterion": SoftMaxwithLoss,
        "kwargs": {}
    },
    "imagenet": {
        "num_classes": 1000,
        "criterion": SoftMaxwithHardLabel,
        "kwargs": {}
    },
    "label_map": {
        "num_classes": 1000,
        "criterion": TokenLabelCrossEntropy,
        "kwargs": {}
    },
    "depth": {
        "num_classes": 1,
        "criterion": DepthLoss,
        "kwargs": {}
    },
    "example_depth": {
        "num_classes": 1,
        "criterion": DepthExampleLoss,
        "kwargs": {}
    },
    "city_depth": {
        "num_classes": 1,
        "criterion": DepthLoss,
        "kwargs": {}
    },
    "normals": {
        "num_classes": 3,
        "criterion": NormalsLoss,
        "kwargs": {}
    },
    "example_normals": {
        "num_classes": 3,
        "criterion": NormalsExampleLoss,
        "kwargs": {}
    },
    "loren_cityscapes_semseg": {
        "num_classes": 19,
        "criterion": SoftMaxwithLoss,
        "kwargs": {'ignore_index': -1}
    },
    "loren_cityscapes_part_seg": {
        "num_classes": 10,
        "criterion": SoftMaxwithLoss,
        "kwargs": {'ignore_index': -1}
    },
    "loren_cityscapes_depth": {
        "num_classes": 1,
        "criterion": LorenDepthLoss,
        "kwargs": {}
    }
}


@HEAD_REGISTRY.register()
class VandenHead(BasicHead):

    @configurable
    def __init__(
        self,
        num_classes,
        criterion,
        **kwargs
    ):
        self.num_classes = num_classes
        super().__init__(**kwargs)
        self.criterion = criterion
    
    @classmethod
    def from_config(cls, cfg, task_name):
        ret = super().from_config(cfg, task_name)
        if cfg.dataset.name.startswith('loren'):
            _task_name = 'loren_cityscapes_'+ task_name
        else:
            _task_name = task_name
        assert _task_name in TASK_INFO.keys()
        if 'example_test' in cfg.dataset:
            _task_name = 'example_'+ _task_name
            print("Using Example Level Loss.")
        num_classes = TASK_INFO[_task_name]['num_classes']
        kwargs = TASK_INFO[_task_name]['kwargs']
        ret['criterion'] = TASK_INFO[_task_name]['criterion'](**kwargs)
        ret["num_classes"] = num_classes
        return ret

    def build_head(self):
        return DeepLabHead(2048, self.num_classes)
    
    def get_features(self, x, key=None):
        ret = self.map_func.get_features(x)
        if key is None:
            return ret
        return {k: v for k, v in ret.items() if k in key}

    def forward(self, data):
        return self.map_func(data)
    
    def _forward_without_reduction(self, out, label, **kwargs):
        return self.criterion._forward_without_reduction(out, label, **kwargs)

    def losses(self, pred, gt, ret_example_level=False):
        return self.criterion(pred, gt, ret_example_level=ret_example_level)


class ImagenetHead(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@HEAD_REGISTRY.register()
class ImagenetKDHead(BasicHead):

    @configurable
    def __init__(
        self,
        num_classes,
        criterion,
        **kwargs
    ):
        self.num_classes = num_classes
        super().__init__(**kwargs)
        self.criterion = criterion
    
    @classmethod
    def from_config(cls, cfg, task_name):
        ret = super().from_config(cfg, task_name)
        assert task_name in TASK_INFO.keys()
        num_classes = TASK_INFO[task_name]['num_classes']
        kwargs = TASK_INFO[task_name]['kwargs']
        ret['criterion'] = TASK_INFO[task_name]['criterion'](**kwargs)
        ret["num_classes"] = num_classes
        return ret

    def build_head(self):
        return ImagenetHead(self.num_classes)

    def forward(self, data):
        return self.map_func(data)

    def losses(self, pred, gt, ret_example_level=False):
        return self.criterion(pred, gt)


class LMHead(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.dense_avgpool = nn.AvgPool2d(kernel_size=(6, 8))  # 把60x80的分辨率变成(10,10)的块,当然，这里用conv可能会更好。
        self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
    
    def forward(self, x):
        out_aux = self.dense_avgpool(x)
        out_aux = self.conv1(out_aux)
        # print("Aux output shape: ", out_aux.shape)
        # 这里要把h x w摊平, to (B, 1000, M*M)
        out_aux = out_aux.reshape(out_aux.shape[0], out_aux.shape[1], -1)
        return None, out_aux


# class LM2LHead(nn.Module):

#     def __init__(self, num_classes):
#         super().__init__()
#         self.dense_avgpool = nn.AvgPool2d(kernel_size=(6, 8))  # 把60x80的分辨率变成(10,10)的块,当然，这里用conv可能会更好。
#         self.fc = nn.Linear(num_classes, num_classes)
#         self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
    
#     def forward(self, x):
#         out_aux = self.dense_avgpool(x)
#         out_aux = self.conv1(out_aux)
#         # print("Aux output shape: ", out_aux.shape)
#         # 这里要把h x w摊平, to (B, 1000, M*M)
#         out_aux = out_aux.reshape(out_aux.shape[0], out_aux.shape[1], -1)
#         out = self.fc(torch.mean(out_aux, dim=2))
#         return out, out_aux


class LM2LHead(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.dense_avgpool = nn.AvgPool2d(kernel_size=(6, 8))  # 把60x80的分辨率变成(10,10)的块,当然，这里用conv可能会更好。
        self.cls_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
    
    def forward(self, x):
        out_aux = self.dense_avgpool(x)
        out = self.fc(torch.flatten(self.cls_pool(out_aux), 1))
        out_aux = self.conv1(out_aux)
        # print("Aux output shape: ", out_aux.shape)
        # 这里要把h x w摊平, to (B, 1000, M*M)
        out_aux = out_aux.reshape(out_aux.shape[0], out_aux.shape[1], -1)
        return out, out_aux


class LMConvHead(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.dense_pool = nn.Conv2d(2048, 1024, kernel_size=(6, 8), stride=(6, 8))  # 这里的训练结果和LMHead差不多，可以进行下一步了。
        self.cls_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        self.conv1 = nn.Conv2d(1024, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 2048, H, W) -> out_aux: (B, 2048, 10, 10)
        out_aux = self.dense_pool(x)
        out = self.cls_pool(out_aux)
        out = self.fc(torch.flatten(out, 1))
        out_aux = self.conv1(out_aux)
        # print("Aux output shape: ", out_aux.shape)
        # 这里要把h x w摊平, to (B, 1000, M*M)
        out_aux = out_aux.reshape(out_aux.shape[0], out_aux.shape[1], -1)
        return out, out_aux


@HEAD_REGISTRY.register()
class LabelMapHead(BasicHead):

    @configurable
    def __init__(
        self,
        num_classes,
        criterion,
        use_conv, 
        **kwargs
    ):
        self.num_classes = num_classes
        self.use_conv = use_conv
        super().__init__(**kwargs)
        self.criterion = criterion
    
    @classmethod
    def from_config(cls, cfg, task_name):
        ret = super().from_config(cfg, task_name)
        assert task_name in TASK_INFO.keys()
        num_classes = TASK_INFO[task_name]['num_classes']
        kwargs = TASK_INFO[task_name]['kwargs']
        ret['criterion'] = TASK_INFO[task_name]['criterion'](**kwargs)
        ret["num_classes"] = num_classes
        ret['use_conv'] = cfg.dataset.model.label_map.use_conv
        return ret

    def build_head(self):
        if self.use_conv:
            return LMConvHead(self.num_classes)
        else:
            return LM2LHead(self.num_classes)

    def forward(self, data):
        # print("LabelMapHead input shape:", data.shape)
        return self.map_func(data)

    def losses(self, pred, gt, ret_example_level=False):
        return self.criterion(pred, gt)


@HEAD_REGISTRY.register()
class HRNetHead(VandenHead):

    def build_head(self):
        return HighResolutionHead([18, 36, 72, 144], self.num_classes)


from detectron2.modeling.roi_heads import Res5ROIHeads
class MyRes5ROIHeads(Res5ROIHeads):

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on
        return None, out_channels



@HEAD_REGISTRY.register()
class DetectionHead(BasicHead):

    @configurable
    def __init__(
            self,
            roi_heads,
            proposal_generator,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.roi_heads = roi_heads
        self.proposal_generator = proposal_generator

    @classmethod
    def from_config(cls, cfg, task_name):
        from detectron2.config import get_cfg
        from detectron2.modeling.proposal_generator import build_proposal_generator
        # from detectron2.modeling.roi_heads import build_roi_heads
        from detectron2.layers import ShapeSpec

        ret = super().from_config(cfg, task_name)
        head_cfg = get_cfg()
        head_cfg.merge_from_file(cfg.dataset.model.detection.head_cfg)
        output_shape = {
            'res4': ShapeSpec(
                channels=1024, stride=2
            )
        }
        ret['roi_heads'] = MyRes5ROIHeads(head_cfg, output_shape)
        ret['proposal_generator'] = build_proposal_generator(head_cfg, output_shape)
        return ret

    def build_head(self):
        return
    
    def losses(self, pred, gt):
        return 

    def forward(self, images, features, gt_instances, proposals=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        # images = self.preprocess_image(batched_inputs)
        # if "instances" in batched_inputs[0]:
        #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        # else:
        #     gt_instances = None
        from detectron2.structures import ImageList

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(ImageList.from_tensors([images[i] for i in range(images.shape[0])]), features, gt_instances)
        else:
            assert proposals is not None
            proposal_losses = {}

        results, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if not self.training:
            return results
        return losses