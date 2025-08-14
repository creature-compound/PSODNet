from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class IntermediateLayerGetter(nn.ModuleDict):

    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:

        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:

        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:

        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))


        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  
        return x, names

class SceneRelation(nn.Module):
    def __init__(self,
                 in_channels, 
                 channel_list,  
                 out_channels, 
                 scale_aware_proj=True): 
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1), 
                    nn.ReLU(True), 
                    nn.Conv2d(out_channels, out_channels, 1),  
                ) for _ in range(len(channel_list))]
            )
        else:
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), 
                nn.ReLU(True), 
                nn.Conv2d(out_channels, out_channels, 1), 
            )
        
        self.content_encoders = nn.ModuleList() 
        self.feature_reencoders = nn.ModuleList() 
        
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1), 
                    nn.BatchNorm2d(out_channels),  
                    nn.ReLU(True)  
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),  
                    nn.BatchNorm2d(out_channels), 
                    nn.ReLU(True) 
                )
            )

        self.normalizer = nn.Sigmoid()
        self.relation_maps = []


    def forward(self, scene_feature, features: list):

        self.relation_maps = []
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in zip(scene_feats, content_feats)]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]


        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]
        
        assert len(relations) == len(p_feats), f"relations长度 ({len(relations)}) 与 p_feats长度 ({len(p_feats)}) 不匹配"

        refined_feats = OrderedDict([
            (f"{i}", r * p)
            for i, (r, p) in enumerate(zip(relations, p_feats))
        ])

        self.relation_maps = list(refined_feats.values())


        return refined_feats  
    

class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers=None, in_channels_list=None, out_channels=256, extra_blocks=None, use_scene_relation=False):
        super().__init__()
        self.use_scene_relation = use_scene_relation
        
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=None)
        
        self.out_channels = out_channels
        

        if self.use_scene_relation:
            self.scene_relation = SceneRelation(in_channels=256, channel_list=[256, 256, 256, 256], out_channels=256)

    def forward(self, x):

        x = self.body(x)
        x = self.fpn(x)


        if self.use_scene_relation:
           
            fpn_output_names = list(x.keys())
            last_fpn_feature = x[fpn_output_names[-1]]  
            scene_feature = torch.mean(last_fpn_feature, dim=(2, 3), keepdim=True)
        
            features = [x[name] for name in fpn_output_names]
            
            x = self.scene_relation(scene_feature, features)
            
        return x



