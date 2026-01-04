import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
    
        self.backbone = CBraMod(
            in_dim=200,  
            out_dim=200,  
            d_model=200,  
            dim_feedforward=4 * 200,  
            seq_len=30,
            n_layer=12,
            nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(
                torch.load(param.foundation_dir, map_location=map_location)
            )
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(param.points_per_patch, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            feature_dim = param.n_channels * param.n_patches * param.points_per_patch
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(feature_dim, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(67*1*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(19*2*200, 2*200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(2*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
    
    def forward(self, x):
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out