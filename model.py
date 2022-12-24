import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34, mobilenet_v3_large
import config as c
from freia_funcs import *

WEIGHT_DIR = './weights'
MODEL_DIR = './models/tmp'


def get_cs_flow_model(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, c.map_size[0], c.map_size[1], name='input'))
    nodes.append(InputNode(input_dim, c.map_size[0] // 2, c.map_size[1] // 2, name='input2'))
    nodes.append(InputNode(input_dim, c.map_size[0] // 4, c.map_size[1] // 4, name='input3'))

    for k in range(c.n_coupling_blocks):
        if k == 0:
            node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
        else:
            node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

        nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0, nodes[-1].out1, nodes[-1].out2], parallel_glow_coupling_layer,
                          {'clamp': c.clamp, 'F_class': CrossConvolutions,
                           'F_args': {'channels_hidden': c.fc_internal,
                                      'kernel_size': c.kernel_sizes[k], 'block_no': k}},
                          name=F'fc1_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
    nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
    nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
    nf = ReversibleGraphNet(nodes, n_jac=3)
    return nf

def nf_forward(model, inputs):
    return model(inputs), model.jacobian(run_forward=False)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')

    def eff_ext(self, x, use_layer=35):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == use_layer:
                return x

    def forward(self, x):
        y = list()
        for s in range(c.n_scales):
            feat_s = F.interpolate(x, size=(c.img_size[0] // (2 ** s), c.img_size[1] // (2 ** s))) if s > 0 else x
            feat_s = self.eff_ext(feat_s)

            y.append(feat_s)
        return y

class FeatureExtractorResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = resnet34(pretrained=True)
    
    def res_ext(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)

        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        return x
        
    def forward(self, x):
        y = list()
        for s in range(c.n_scales):
            feat_s = F.interpolate(x, size=(c.img_size[0] // (2 ** s), c.img_size[1] // (2 ** s))) if s > 0 else x
            feat_s = self.res_ext(feat_s)

            y.append(feat_s)
        return y

class FeatureExtractorMobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = mobilenet_v3_large(pretrained=True)
    
    def mob_ext(self, x, use_layer=14):
        for idx, m in enumerate(self.feature_extractor.features.children()):
            x = m(x)
            if idx >= use_layer:
                return x
    def forward(self, x):
        y = list()
        for s in range(c.n_scales):
            feat_s = F.interpolate(x, size=(c.img_size[0] // (2 ** s), c.img_size[1] // (2 ** s))) if s > 0 else x
            # feat_s = self.feature_extractor.features(feat_s)
            feat_s = self.mob_ext(feat_s)

            y.append(feat_s)
        return y

def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model
