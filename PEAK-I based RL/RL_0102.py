import numpy as np
import torch
import data_tcia as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import openslide
import PIL.Image as Image
import random
import mae_model
import torch.nn as nn
import mae2
import cost_mae
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from torch.utils.data import Dataset
import os
import h5py
import  csv



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
seed = 180950483
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def find_matching_files(csv_path, folder_path):
    # 从 CSV 文件中读取文件名
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        filenames = [row[0] for row in reader]

    # 在指定文件夹中查找匹配的 .h5 文件
    matching_files = []
    for filename in filenames:
        for file in os.listdir(folder_path):
            if file.startswith(filename) and file.endswith('.h5'):
                matching_files.append(os.path.join(folder_path, file))

    return matching_files

# 使用示例


class ExampleDataset(Dataset):

    def __init__(self, data_list):
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data
        self.datalist = data_list
    
    def __getitem__(self, index):
        with h5py.File(self.datalist[index],'r') as hdf5_file:
            coord = hdf5_file['coords'][:]
            coord = coord * 0.25
            coord = coord.astype(int)
        return {'grid': coord, 'slide': '/home/omnisky/sde/NanTH/IRL_Data/all_slide/slides/' + self.datalist[index].split('/')[-1].split('.')[0] + '.ndpi'}
    
    def __len__(self):
        # 返回数据的长度
        return len(self.datalist)



normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5, 0.5, 0.5])
trans = transforms.Compose([transforms.ToTensor(), normalize]) 
#np.random.seed(seed)
#torch.manual_seed(seed)
# LOADING EXPERT/DEMO SAMPLES


csv_path = '/home/omnisky/sde/NanTH/IRL_Data/all_slide/cp/train.csv' # 替换为您的 CSV 文件路径
folder_path = '/home/omnisky/sde/NanTH/IRL_Data/all_slide/cp/patches' # 替换为您的文件夹路径
data_load = ExampleDataset(find_matching_files(csv_path, folder_path))
'''
data_load2 = torch.load('/home/omnisky/hdd_15T_sdd/NanTH/IRL_DATA/4th/path/test1_amplification_5.0.pth')
slides2 = []
grid2 = []
for i in range(len(data_load2['grid'])):
    if len(data_load2['grid'][i]) > 32:
        slides2.append(data_load2['slides'][i])
        grid2.append(data_load2['grid'][i])
data_load_test = {
        "slides": slides2,
        "grid": grid2,
        "level": 1
    }
'''
cuda_1 = torch.device('cuda:1')
cuda_2 = torch.device('cpu')

cost_f = cost_mae.Transformer_Cost_n3().to(cuda_1)
mae = mae_model.mae_model().to(cuda_1)
cost_f.load_state_dict(torch.load('/home/omnisky/sde/NanTH/result/irl/314cost_net.pth', map_location='cuda:1'))
mae.eval()
cost_f.eval()

'''
data_load1 = torch.load('/home/omnisky/hdd_15T_sdc/NanTH/slide/train_Slide.pth')
data_load2 = torch.load('/home/omnisky/hdd_15T_sdc/NanTH/slide/train2_Slide.pth')
print('data_load1:', len(data_load1['slides']), len(data_load1['grid']), len(data_load1['boundary']), len(data_load1['targets']))
print('data_load2:', len(data_load2['slides']), len(data_load2['grid']), len(data_load2['boundary']), len(data_load2['targets']))
data_load = {
        "slides": data_load1['slides'] + data_load2['slides'],
        "grid": data_load1['grid'] + data_load2['grid'],
        "boundary": data_load1['boundary'] + data_load2['boundary'],
        "targets": data_load1['targets'] + data_load2['targets'],
        "mult": data_load1['mult'] + data_load2['mult'],
        "level": data_load1['level']
    }

print('data_load:', len(data_load['slides']), len(data_load['grid']), len(data_load['boundary']), len(data_load['targets']))
print(len(data_load2['grid'][311]), len(data_load['grid'][311 + len(data_load1['slides'])]))
'''
BATCH_SIZE = 16
class mae_model_(nn.Module):
    def __init__(self):
        super(mae_model_, self).__init__()
        model = mae2.__dict__['vit_base_patch16'](
            #num_classes=9,
            #drop_path_rate=0.1,
            global_pool=False,
            )
        checkpoint = torch.load('/home/omnisky/hdd_15T_sdc/RL_project/save_checkpoints/checkpoint-79.pth', map_location='cpu')
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
    def forward(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        x = torch.cat((x1, x2), 1)
        return x


class ResidualBlock(nn.Module):
    #实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,stride,padding=1,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),#inplace = True原地操作
            nn.Conv2d(out_ch,out_ch,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_ch)
            )
        self.right = shortcut
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        
    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        out = self.relu(out)
        return out
        
class ResNet34(nn.Module):#224x224x3
    #实现主module:ResNet34
    def __init__(self, num_classes=1000):
        super(ResNet34,self).__init__()
        self.pre = nn.Sequential(
                nn.Conv2d(3,64,7,stride=2,padding=3,bias=True),# (224+2*p-)/2(向下取整)+1，size减半->112
                nn.BatchNorm2d(64),#112x112x64
                nn.LeakyReLU(negative_slope=0.01, inplace=False),
                nn.MaxPool2d(3,2,1)#kernel_size=3, stride=2, padding=1
                )#56x56x64
        
        #重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self.make_layer(64,128,3)#56x56x64,layer1层输入输出一样，make_layer里，应该不用对shortcut进行处理，但是为了统一操作。。。
        self.layer2 = self.make_layer(128,256,4,stride=2)#第一个stride=2,剩下3个stride=1;28x28x128
        self.layer3 = self.make_layer(256,512,6,stride=2)#14x14x256
        self.layer4 = self.make_layer(512,1024,3,stride=2)#7x7x512
        #分类用的全连接
        
    def make_layer(self,in_ch,out_ch,block_num,stride=1):
        #当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(#首个ResidualBlock需要进行option B处理
                nn.Conv2d(in_ch,out_ch,1,stride,bias=False),#1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch)
                )
        layers = []
        layers.append(ResidualBlock(in_ch,out_ch,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock(out_ch,out_ch))#后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)
        
    def forward(self,x):    #224x224x3
        x = self.pre(x)     #56x56x64
        x = self.layer1(x)  #56x56x64
        x = self.layer2(x)  #28x28x128
        x = self.layer3(x)  #14x14x256
        x = self.layer4(x)  #7x7x512
        x = F.avg_pool2d(x,7)#1x1x512
        x = x.view(x.size(0),-1)#将输出拉伸为一行  #1x1
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        return x



class DQN(nn.Module):
    def __init__(
        self
    ):
        super(DQN, self).__init__()
        self.mae = mae_model_()
        self.cnn = ResNet34()
        self.fc = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Linear(2560, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Linear(1024, 8),
            )
        self.act_net = nn.Linear(2, 256)
    
    def forward(self, big, small ,screen):
        out = self.mae(big, small)
        screen = self.cnn(screen)
        out = torch.cat((out, screen), dim = 1)
        out = self.fc(out)
        return out


class Agent(object):
    def __init__(self, training): 
        self.eval_net, self.target_net = DQN().to(cuda_1), DQN().to(cuda_1)
        if training == 1:
            self.eval_net.train()
            self.target_net.train()
        else:
            self.eval_net.eval()
            self.target_net.eval()
        self.loss_func = nn.MSELoss()
        #self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=2e-5, alpha=0.99, weight_decay = 1e-4)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=2e-5,weight_decay = 1e-4)
        self.learn_step_counter = 0
        self.BATCH_SIZE = BATCH_SIZE
        self.slide_num = len(data_load)
        self.gamma = 0.9
        self.stor = []
        self.result = []
        self.amplification = '-1'
        self.level = 2

    def change_amplification(self, a):
        self.amplification = a

    def replay_buffer(self, sample_index):
        point = self.stor[sample_index]['point']
        slide_index = self.stor[sample_index]['slide_num']
        act_dim = [[-0.415, -0.415], [0, -0.415], [0.415, -0.415], [0.415, 0], [0.415, 0.415], [0, 0.415], [-0.415, 0.415], [-0.415, 0]]
        slide = openslide.OpenSlide(data_load[slide_index]['slide'])
        win_img_big = torch.tensor([], dtype=torch.float, device = cuda_1)
        win_img_small = torch.tensor([], dtype=torch.float, device = cuda_1)
        img_screen_list = torch.tensor([], dtype=torch.float, device = cuda_1)
        act = []
        reward = []      
        for i in range(len(point)):
            p = point[i]
            img_ = data.get_img(p, p, data_load[slide_index]["slide"], self.level, cuda_1, self.amplification)
            img_big = img_.train_data_old
            img_small = img_.train_data_new
            img_big = torch.unsqueeze(img_big, dim=0)
            img_small = torch.unsqueeze(img_small, dim=0)
            win_img_big = torch.cat((win_img_big, img_big), 0)
            win_img_small = torch.cat((win_img_small, img_small), 0)
            img_screen = slide.read_region(((p[0] - 960) * (self.amplification ** self.level), (p[1] - 960) * (self.amplification ** self.level)), self.level, (1920, 1920)).convert('RGB')
            img_screen = img_screen.resize((224,224),Image.ANTIALIAS)
            img_screen = trans(img_screen)
            img_screen = img_screen.to(cuda_1)
            img_screen = torch.unsqueeze(img_screen, dim=0)
            img_screen_list = torch.cat((img_screen_list, img_screen), 0)
            act.append([self.stor[sample_index]['act'][i]])
            reward.append([self.stor[sample_index]['reward'][i]])
        a = [act_dim[x[0]] for x in act]
        return win_img_big, win_img_small, img_screen_list, a, reward, point, act #xuhao

    def next(self, sample_index, next_state):
        slide_index = self.stor[sample_index]['slide_num']
        #act_dim = [[-0.415, -0.415], [0, -0.415], [0.415, -0.415], [0.415, 0], [0.415, 0.415], [0, 0.415], [-0.415, 0.415], [-0.415, 0]]
        win_img_big = torch.tensor([], dtype=torch.float, device = cuda_1)
        win_img_small = torch.tensor([], dtype=torch.float, device = cuda_1)
        img_screen_list = torch.tensor([], dtype=torch.float, device = cuda_1)
        for i in range(len(next_state)):
            slide = openslide.OpenSlide(data_load[slide_index]['slide'])
            p = next_state[i]
            img_ = data.get_img(p, p, data_load[slide_index]["slide"], self.level, cuda_1, self.amplification)
            img_big = img_.train_data_old
            img_small = img_.train_data_new
            img_big = torch.unsqueeze(img_big, dim=0)
            img_small = torch.unsqueeze(img_small, dim=0)
            win_img_big = torch.cat((win_img_big, img_big), 0)
            win_img_small = torch.cat((win_img_small, img_small), 0)
            img_screen = slide.read_region(((p[0] - 960) * (self.amplification ** self.level), (p[1] - 960) * (self.amplification ** self.level)), self.level, (1920, 1920)).convert('RGB')
            img_screen = img_screen.resize((224,224),Image.ANTIALIAS)
            img_screen = trans(img_screen)
            img_screen = img_screen.to(cuda_1)
            img_screen = torch.unsqueeze(img_screen, dim=0)
            img_screen_list = torch.cat((img_screen_list, img_screen), 0)
        q_next = self.target_net(win_img_big, win_img_small, img_screen_list)
        q_next = torch.max(q_next, 1)[0]
        q_next = q_next.view(self.BATCH_SIZE, 1)
        return q_next

    def choose_action(self, sample_index0, state):        
        slide = openslide.OpenSlide(data_load[sample_index0]['slide'])     
        if np.random.uniform() < 0.9:
            win_img_big = torch.tensor([], dtype=torch.float, device = cuda_1)
            win_img_small = torch.tensor([], dtype=torch.float, device = cuda_1)
            img_screen_list = torch.tensor([], dtype=torch.float, device = cuda_1)
            for s in state:
                img_ = data.get_img(s, s, data_load[sample_index0]["slide"], self.level, cuda_1, self.amplification)
                img_big = img_.train_data_old
                img_small = img_.train_data_new
                img_big = torch.unsqueeze(img_big, dim=0)
                img_small = torch.unsqueeze(img_small, dim=0)
                win_img_big = torch.cat((win_img_big, img_big), 0)
                win_img_small = torch.cat((win_img_small, img_small), 0)
                img_screen = slide.read_region(((s[0] - 960) * (self.amplification ** self.level), (s[1] - 960) * (self.amplification ** self.level)), self.level, (1920, 1920)).convert('RGB')
                img_screen = img_screen.resize((224,224),Image.ANTIALIAS)
                img_screen = trans(img_screen)
                img_screen = img_screen.to(cuda_1)
                img_screen = torch.unsqueeze(img_screen, dim=0)
                img_screen_list = torch.cat((img_screen_list, img_screen), 0)
            actions_value = self.eval_net.forward(win_img_big, win_img_small, img_screen_list)  
            act = torch.argmax(actions_value, 1).cpu().tolist()                    
        else:                                                        
            act = np.random.choice(8, self.BATCH_SIZE)      


        return act  

    def storage(self, act, sample_index0, state, reward):
        if len(self.stor) <=5000:
            self.stor.append({'act': act, 'slide_num': sample_index0, 'point':state, 'reward': reward})
        else:
            del self.stor[0]
            self.stor.append({'act': act, 'slide_num': sample_index0, 'point':state, 'reward': reward})
        if len(self.stor) > 5017:
            print('storage error')

    def step(self, act, sample_index0, state):
        new_state = []
        act_dim = [[-0.415, -0.415], [0, -0.415], [0.415, -0.415], [0.415, 0], [0.415, 0.415], [0, 0.415], [-0.415, 0.415], [-0.415, 0]]
        for i in range(len(state)):
            new_state.append([int(state[i][0] + 540 * act_dim[act[i]][0]), int(state[i][1] + 540 * act_dim[act[i]][1])])
        if len(new_state) != BATCH_SIZE:
            print('new state len error')
        slide = openslide.OpenSlide(data_load[sample_index0]['slide'])  
        win_img_big = torch.tensor([], dtype=torch.float, device = cuda_1)
        win_img_small = torch.tensor([], dtype=torch.float, device = cuda_1)
        img_screen_list = torch.tensor([], dtype=torch.float, device = cuda_1)
        for ns in new_state:
            img_ = data.get_img(ns, ns, data_load[sample_index0]["slide"], self.level, cuda_1, self.amplification)
            img_big = img_.train_data_old
            img_small = img_.train_data_new
            img_big = torch.unsqueeze(img_big, dim=0)
            img_small = torch.unsqueeze(img_small, dim=0)
            win_img_big = torch.cat((win_img_big, img_big), 0)
            win_img_small = torch.cat((win_img_small, img_small), 0)
            img_screen = slide.read_region(((ns[0] - 960) * (self.amplification ** self.level), (ns[1] - 960) * (self.amplification ** self.level)), self.level, (1920, 1920)).convert('RGB')
            img_screen = img_screen.resize((224,224),Image.ANTIALIAS)
            img_screen = trans(img_screen)
            img_screen = img_screen.to(cuda_1)
            img_screen = torch.unsqueeze(img_screen, dim=0)
            img_screen_list = torch.cat((img_screen_list, img_screen), 0)
        encoder = mae.forward2(win_img_big, win_img_small)
        out= cost_f.forward4(encoder, img_screen_list)
        reward = out.cpu().tolist()
        re = [x[0] for x in reward]
        return new_state, re

    def learn(self):
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(len(self.stor)) 
        big, small, screen, act, reward, state, a = self.replay_buffer(sample_index) #a xuhao
        a = torch.LongTensor(a).to(cuda_1)
        q_eval = self.eval_net(big, small, screen).gather(1, a)
        next_state = []
        for m in range(self.BATCH_SIZE):
            next_state.append([int(state[m][0] + 540 * act[m][0]), int(state[m][1] + 540 * act[m][1])])
        with torch.no_grad(): 
            q_next = self.next(sample_index, next_state)
        q_target = torch.tensor(reward).to(cuda_1)
        q_target = q_target + 0.9 * q_next
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                         
        loss.backward()                                            
        self.optimizer.step()
        self.result.append(loss.item())

    def save(self, i) :
        RL_name = '/home/omnisky/sde/NanTH/result/rl/' + str(i) + 'DQN_net' + '.pth'
        torch.save(self.eval_net.state_dict(), RL_name) 

    def prin(self):
        print('loss:', np.mean(self.result))
        self.result.clear()

    def load(self):
        self.eval_net.load_state_dict(torch.load('/home/omnisky/verybigdisk/NanTH/Result/rl/zheng/5DQN_net.pth', map_location='cuda:1'))
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def testloss(self, sample_index0, new_state, reward, state, act):
        slide = openslide.OpenSlide(data_load[sample_index0]['slide'])
        win_img_big = torch.tensor([], dtype=torch.float, device = cuda_1)
        win_img_small = torch.tensor([], dtype=torch.float, device = cuda_1)
        img_screen_list = torch.tensor([], dtype=torch.float, device = cuda_1)
        for ns in new_state:
            img_ = data.get_img(ns, ns, data_load[sample_index0]["slide"], self.level, cuda_1, self.amplification)
            img_big = img_.train_data_old
            img_small = img_.train_data_new
            img_big = torch.unsqueeze(img_big, dim=0)
            img_small = torch.unsqueeze(img_small, dim=0)
            win_img_big = torch.cat((win_img_big, img_big), 0)
            win_img_small = torch.cat((win_img_small, img_small), 0)
            img_screen = slide.read_region(((ns[0] - 960) * (self.amplification ** self.level), (ns[1] - 960) * (self.amplification ** self.level)), self.level, (1920, 1920)).convert('RGB')
            img_screen = img_screen.resize((224,224),Image.ANTIALIAS)
            img_screen = trans(img_screen)
            img_screen = img_screen.to(cuda_1)
            img_screen = torch.unsqueeze(img_screen, dim=0)
            img_screen_list = torch.cat((img_screen_list, img_screen), 0)
        #q_eval = self.eval_net(win_img_big, win_img_small, img_screen_list).gather(1, a)
        q_next = self.target_net(win_img_big, win_img_small, img_screen_list)
        q_next = torch.max(q_next, 1)[0]
        q_next = q_next.view(self.BATCH_SIZE, 1)
        re = [[x] for x in reward]
        q_target = torch.tensor(re).to(cuda_1)
        q_target = q_target + 0.9 * q_next
        big = torch.tensor([], dtype=torch.float, device = cuda_1)
        small = torch.tensor([], dtype=torch.float, device = cuda_1)
        screen = torch.tensor([], dtype=torch.float, device = cuda_1)        
        for s in state:
            img_ = data.get_img(s, s, data_load[sample_index0]["slide"], self.level, cuda_1, self.amplification)
            img_big = img_.train_data_old
            img_small = img_.train_data_new
            img_big = torch.unsqueeze(img_big, dim=0)
            img_small = torch.unsqueeze(img_small, dim=0)
            big = torch.cat((big, img_big), 0)
            small = torch.cat((small, img_small), 0)
            img_screen = slide.read_region(((s[0] - 960) * (self.amplification ** self.level), (s[1] - 960) * (self.amplification ** self.level)), self.level, (1920, 1920)).convert('RGB')
            img_screen = img_screen.resize((224,224),Image.ANTIALIAS)
            img_screen = trans(img_screen)
            img_screen = img_screen.to(cuda_1)
            img_screen = torch.unsqueeze(img_screen, dim=0)
            screen = torch.cat((screen, img_screen), 0)
        a = torch.LongTensor([[x] for x in act]).to(cuda_1)
        q_eval = self.eval_net(big, small, screen).gather(1, a)
        loss = self.loss_func(q_eval, q_target)
        return loss.item()

        


setup_seed(seed + 10) 
dqn = Agent(1)  
#dqn.load()
dqn.change_amplification(2)                            
for i in range(0, 300):                                   
    print('<<<<<<<<<Episode: %s' % i)
    print(' ')
    re_list = []
    for step in range(0, 300):    
        if step % 3 ==0:
            sample_index0 = np.random.choice(len(data_load) - int(0.1 * len(data_load)))    
            points = np.array(data_load[sample_index0]['grid'])
            if len(points) < BATCH_SIZE * 0.5:
                continue
            max_x = max(points[:, 0])
            min_x = min(points[:, 0])
            max_y = max(points[:, 1])
            min_y = min(points[:, 1])
            samp = []
            for p in points:
                if p[0] > min_x + 960 and p[0] < max_x - 960 and p[1] > min_y + 960 and p[1] < max_y - 960:
                    samp.append(p.tolist())
            if len(samp) < BATCH_SIZE * 0.5:
                continue
            sample_index1 = np.random.choice(len(samp), BATCH_SIZE) 
            #state_dex = sample_index1
            state = [samp[x] for x in sample_index1]
            #state = [data_load['grid'][sample_index0][x] for x in state_dex]
        #seed = seed + 1
        with torch.no_grad():
            act = dqn.choose_action(sample_index0, state)
            state_new, reward = dqn.step(act, sample_index0, state)
            re_list.append(np.mean(reward))
            dqn.storage(act, sample_index0, state, reward)      
        dqn.learn()
        state = state_new
    dqn.save(i)
    print('reward:', np.mean(re_list))
    dqn.prin()
    torch.cuda.empty_cache()
    with torch.no_grad():
        test_reward_list = []
        test_loss_list = []
        for test_sam in range(len(data_load) - int(0.1 * len(data_load)), len(data_load)):
            if len(data_load[test_sam]['grid']) < 1:
                continue
            sample_index1_test = np.random.choice(len(data_load[test_sam]['grid']), BATCH_SIZE) 
            state_test = [data_load[test_sam]['grid'][x] for x in sample_index1_test]
            act_test = dqn.choose_action(test_sam, state_test)
            state_new, reward = dqn.step(act_test, test_sam, state_test)
            test_reward_list.append(np.mean(reward))
            loss = dqn.testloss(test_sam, state_new, reward, state_test, act_test)
            test_loss_list.append(loss)
        print('test:', np.mean(test_reward_list), '  ', np.mean(test_loss_list))
         

    ###################


