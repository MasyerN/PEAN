import torch.nn as nn
import mae2
import torch
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from torchsummary import summary
import numpy as np
import random
import data
#model = mae.gcmae_vit_base_patch16_dec512d8b()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class mae_model(nn.Module):
    def __init__(self):
        super(mae_model, self).__init__()
        model = mae2.__dict__['vit_base_patch16'](
            #num_classes=9,
            #drop_path_rate=0.1,
            global_pool=False,
            )
        checkpoint = torch.load('/home/omnisky/hdd_15T_sdc/NanTH/GuidedCL/checkpoint-79.pth', map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

                # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
                # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
                # manually initialize fc layer
        trunc_normal_(model.head.weight, std=0.01)
        self.model = model#.to(torch.device('cuda:3'))
        

    def forward(self, x1, x2, len_list):
        x1 = self.model(x1)
        x2 = self.model(x2)
        x = torch.cat((x1, x2), 1)
        x = torch.split(x, len_list, 0)
        list1 = list(x)
        list1.sort(key=lambda x:-x.size()[0])
        return list1

    def forward2(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        x = torch.cat((x1, x2), -1)
        return x
    
    def forward3(self, x):
        return self.model(x)


class mae_model_nolevel(nn.Module):
    def __init__(self):
        super(mae_model_nolevel, self).__init__()
        model = mae2.__dict__['vit_base_patch16'](
            #num_classes=9,
            #drop_path_rate=0.1,
            global_pool=False,
            )
        checkpoint = torch.load('/home/omnisky/hdd_15T_sdc/NanTH/checkpoint-79.pth', map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

                # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
                # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
                # manually initialize fc layer
        trunc_normal_(model.head.weight, std=0.01)
        self.model = model#.to(torch.device('cuda:3'))
        

    def forward(self, x1, x2, len_list):
        x1 = self.model(x1)
        x2 = self.model(x2)
        x = torch.cat((x1, x2), 1)
        x = torch.split(x, len_list, 0)
        list1 = list(x)
        list1.sort(key=lambda x:-x.size()[0])
        return list1

    def forward2(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        x = torch.cat((x1, x2), -1)
        return x
#params = list(model.named_parameters())
#for p in params:
#    print(p)
#model = mae_model().to(torch.device('cuda:0'))
#summary(model, (3,224,224))
#checkpoint = torch.load('/home/omnisky/hdd_15T_sdc/NanTH/GuidedCL/checkpoint-79.pth', map_location='cpu')
#checkpoint_model = checkpoint['model']
#state_dict = model.state_dict()




