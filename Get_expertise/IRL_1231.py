import makedemo as make_Demo
import numpy as np
import torch
import data
import cost_mae
import copy
import torch.nn.functional as F
import torchvision.transforms as transforms
import openslide
import PIL.Image as Image
import random
import datetime
import math
import mae_model
import os
# SEEDS
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
seed = 180950483
#np.random.seed(seed)
#torch.manual_seed(seed)
# LOADING EXPERT/DEMO SAMPLES
cuda_1 = torch.device('cuda:1')
cuda_2 = torch.device('cpu')

#make_test_demo = make_Demo.demo_list('/home/omnisky/hdd_15T_sdc/NanTH/IRL_DATA/Mr,Zheng/epr', '/home/omnisky/hdd_15T_sdc/NanTH/IRL_DATA/Mr,Zheng/2021.10.19')
#demo_trajs2 = make_test_demo.demo__
#make_demo3 = make_Demo.demo_list('/home/omnisky/hdd_15T_sdc/RL_project/dataset/Mr_Zheng1/epr_repair', '/home/omnisky/hdd_15T_sdc/RL_project/dataset/Mr_Zheng1/ndpi_repair')
#demo_trajs3 = make_demo3.demo__
split = torch.load('')
make_demo = make_Demo.demo_list(split['train'])
demo_trajs = make_demo.demo__
make_demo_test = make_Demo.demo_list(split['test'])
demo_trajs_test = make_demo_test.demo__
#make_demo4 = make_Demo.demo_list('/home/omnisky/hdd_15T_sdd/NanTH/IRL_DATA/Mr,Zheng/epr_t', '/home/omnisky/hdd_15T_sdd/NanTH/IRL_DATA/Mr,Zheng/ndpi_t')
#demo_trajs4 = make_demo4.demo__
demo_trajs
#demo_trajs = make_demo1 + demo_trajs2 + demo_trajs3
cost_f = cost_mae.Transformer_Cost_n3().to(cuda_1)
mae = mae_model.mae_model().to(cuda_1)
#cost_f.load_state_dict(torch.load('/home/omnisky/hdd_15T_sdd/NanTH/Result/irl/bcc/8cost_net.pth', map_location='cuda:0'))
cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-5, weight_decay=1e-4)
flip = transforms.RandomVerticalFlip(p = 1)
normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5, 0.5, 0.5])
trans = transforms.Compose([transforms.ToTensor(), normalize]) 
policy_result = []
mean_rewards = []
mean_costs = []
mean_loss_rew = []
batch_size = 4
sample_trajs = []
max_sqlen = 9
our_demo = []
test_demo = []
epoch = 400
#for i in range(100):

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def loss_box(prob, label):
    prob = F.softmax(prob, dim = 1)
    prob = torch.log(prob)
    out = torch.tensor([], dtype=torch.float, device = cuda_1)
    for i in range(label.size()[0]):
        if label[i] == 0:
            loss = prob[i][0]
            out = torch.cat((out, loss.unsqueeze(dim = 0)), 0)
        else:
            loss = label[i] * prob[i][1]
            out = torch.cat((out, loss.unsqueeze(dim = 0)), 0)
    return  -torch.mean(out)

def loss_box2(prob, wsi_label, move_label):
    wsi = prob.index_select(1, torch.tensor([0, 1]).to(cuda_1))
    move = prob.index_select(1, torch.tensor([2, 3, 4]).to(cuda_1))
    wsi = F.softmax(wsi, dim = 1)
    wsi = torch.log(wsi)
    move_function = torch.nn.CrossEntropyLoss()
    move_label = torch.tensor(move_label, dtype=torch.long).to(cuda_1)
    move_loss = move_function(move, move_label)
    out = torch.tensor([], dtype=torch.float, device = cuda_1)
    for i in range(wsi_label.size()[0]):
        if wsi_label[i] < 0:
            continue
        if wsi_label[i] == 0:
            loss = wsi[i][0]
            out = torch.cat((out, loss.unsqueeze(dim = 0)), 0)
        else:
            loss = wsi_label[i] * wsi[i][1]
            out = torch.cat((out, loss.unsqueeze(dim = 0)), 0)
    if out.size()[0] == 0:
        out = torch.tensor(0, dtype=torch.float).to(cuda_1)
    else:
        out = - torch.mean(out)
    q = out.item()
    out = out + move_loss
    return  out, q
setup_seed(seed)
print("photo number:", len(demo_trajs))
for m in range(0, len(demo_trajs)):
    print(os.path.split(demo_trajs[m][0][0][0].ndpi)[-1].split('.')[0].split('_')[-2])
    for n in range(len(demo_trajs[m])):
            #print(demo_trajs[m][n][0][0].ndpi, demo_trajs[m][n][0][0].label)
        our_demo.append(demo_trajs[m][n])
print('test_data:------------------------------------------------------------')
for m in range(0, len(demo_trajs_test)):
    for n in range(len(demo_trajs_test[m])): 
            #print(demo_trajs_test[m][n][0][0].ndpi, demo_trajs_test[m][n][0][0].label)
        test_demo.append(demo_trajs_test[m][n])
print('len_train:', len(our_demo))
print('len_test:', len(test_demo))
#StepLR_1 = torch.optim.lr_scheduler.StepLR(cost_optimizer, int(120 * len(our_demo) / batch_size), gamma=0.1, last_epoch=-1)
for step in range(0, epoch):
    ture = 1
    our_demo_ = copy.deepcopy(our_demo)
    mae.eval()
    cost_f.eval()
    loss_rebuild_list = []
    loss_reward_list = []
    juli = []
    time_ = 0
    while(ture):
        if (time_ % 500 == 0):
            print('---------------------', 'step:', step, '     time:', time_, '----------------------')
            now = datetime.datetime.now()
            print (now.strftime("%Y-%m-%d %H:%M:%S"))
        time_ = time_ + 1
        if len(our_demo_) <= batch_size: #!!!!!!!!!
            break        
        selected_num = np.random.choice(len(our_demo_) - batch_size, batch_size)
        selected_demo = []
        for unm_ in selected_num:
            selected_demo.append(our_demo_[unm_])
            del our_demo_[unm_]
        
        if len(selected_demo) > (batch_size + 1) : #or len(our_demo) < 2000:
            print('error')
            break
        one_hot = []
        point_list =[]
        demo_old = []        
        ndpi_list = []
        level_list = []
        epr_list = []
        len_list = []
        center = []
        win_img_big = torch.tensor([], dtype=torch.float, device = cuda_1)
        win_img_small = torch.tensor([], dtype=torch.float, device = cuda_1)
        img_screen_list = torch.tensor([], dtype=torch.float, device = cuda_1)
        selected_demo.sort(key=lambda x:-len(x))
        wsi_lable = []
        move_lable = []
        for i in range(len(selected_demo)):
            points = []
            level = selected_demo[i][0][0].level
            ndpi = selected_demo[i][0][0].ndpi
            epr = selected_demo[i][0][0].epr
            wsi_l = selected_demo[i][0][0].label
            if selected_demo[i][-1][-1].next_level - level > 0:
                move_l = 2
            elif selected_demo[i][-1][-1].next_level - level == 0:
                move_l = 0
            elif selected_demo[i][-1][-1].next_level - level < 0:
                move_l = 1
            level_list.append(level)
            ndpi_list.append(ndpi)
            epr_list.append(epr)
            len_list.append(len(selected_demo[i]))
            wsi_lable.append(wsi_l)
            if wsi_l >= 0:
                wsi_lable.append(0)
            else:
                wsi_lable.append(-1)
            move_lable.append(move_l)
            move_lable.append(move_l)
            move_lable.append(0)
            screen_xy = [selected_demo[i][0][0].screen_x, selected_demo[i][0][0].screen_y]
            label1_list = []
            label2_list = []
            #img_s = img_screen.read_region(((screen_xy[0] * 2 ** level) - (p * 2 ** level), (screen_xy[1] * 2 ** level) - (p * 2 ** level)), level, (1080, 1080)).convert('RGB')
            for classes in range(len(selected_demo[i])):
                x_y = [int(np.mean([x.X for x in selected_demo[i][classes]])), int(np.mean([x.Y for x in selected_demo[i][classes]]))]
                points.append(x_y)
                img_ = data.get_img(x_y, x_y, ndpi, level, cuda_1)
                img_big = img_.train_data_old
                img_small = img_.train_data_new
                img_big = torch.unsqueeze(img_big, dim=0)
                img_small = torch.unsqueeze(img_small, dim=0)
                win_img_big = torch.cat((win_img_big, img_big), 0)
                win_img_small = torch.cat((win_img_small, img_small), 0)
                if len(selected_demo[i][classes]) <= 20:
                    label1_list.append(len(selected_demo[i][classes]))
                else:
                    label1_list.append(20)
            for j in range(len(points)):
                label2_list.append(min([math.sqrt((points[j][0] - points[j - 1][0]) ** 2 + (points[j][1] - points[j - 1][1]) ** 2), math.sqrt((points[j][0] - points[(j + 1) % len(points)][0]) ** 2 + (points[j][1] - points[(j + 1) % len(points)][1]) ** 2)]))
            for j2 in range(len(label1_list)):
                one_hot.append(0.5 * (0.9 ** (np.mean(label1_list) / label1_list[j2])) + 0.5 * (0.9 ** (label2_list[j2] / np.mean(label2_list))))
            for j3 in range(len(points)):
                one_hot[- j3 - 1] = one_hot[- j3 - 1] * 0.95 ** j3
            max_x = max([x[0] for x in points])
            min_x = min([x[0] for x in points])
            max_y = max([x[1] for x in points])
            min_y = min([x[1] for x in points])
            img_screen = openslide.OpenSlide(ndpi)
            #if (max_x - min_x) < 800 and (max_y - min_y) < 800:
                #img_s = img_screen.read_region((((min_x - 128) * (2 ** level)), ((min_y - 128) * (2 ** level))), level, (1080, 1080)).convert('RGB')
                #img_s = img_s.resize((224,224),Image.ANTIALIAS)
                #img_s = img_screen.read_region((((min_x - 128) * (2 ** level)), ((min_y - 128) * (2 ** level))), level + 2, (270, 270)).convert('RGB')
                #img_s = img_s.resize((224,224),Image.ANTIALIAS)
                #center = center + [[(max_x + min_x) / 2, (max_y + min_y) / 2] for row in range(len(selected_demo[i]))]
            #else:
            img_s = img_screen.read_region(((screen_xy[0] * (2 ** level)), (screen_xy[1] * (2 ** level)) - (420 * 2 ** level)), level + 3, (240, 240)).convert('RGB')
            img_s = img_s.resize((224,224),Image.ANTIALIAS)
            #######
            center = center + [[screen_xy[0] + 960, screen_xy[1] + 540] for row in range(len(selected_demo[i]))]
            img_s = trans(img_s)
            img_s = img_s.to(cuda_1)
            img_s = torch.unsqueeze(img_s, dim=0)
            img_screen_list = torch.cat((img_screen_list, img_s), 0)
            point_list.append(points)
        q = win_img_big.size()[0]
        for m in range(q):
            win_img_big = torch.cat((win_img_big, torch.unsqueeze(flip(win_img_big[m].clone().detach()), dim = 0)), 0)
            win_img_small = torch.cat((win_img_small, torch.unsqueeze(flip(win_img_small[m].clone().detach()), dim = 0)), 0)
        one_hot = one_hot + one_hot
        tra = [] 
        check_1 = 0       
        for l in range(len(point_list)):
            '''
            tr = []
            tr.append(point_list[l][0])
            w_img = data.get_img(point_list[l][0], point_list[l][0], ndpi_list[l], level_list[l])
            wib = w_img.train_data_old
            wis = w_img.train_data_new
            wib = torch.unsqueeze(wib, dim=0)
            wis = torch.unsqueeze(wis, dim=0)
            win_img_big = torch.cat((win_img_big, wib), 0)
            win_img_small = torch.cat((win_img_small, wis), 0)
            '''
            tr = []
            for p_ in range(len(point_list[l])):                
                trajs_x = random.uniform(-1, 1)
                trajs_x = trajs_x * 540 + center[check_1][0]
                trajs_x = int(trajs_x)
                trajs_y = random.uniform(-1, 1)
                trajs_y = trajs_y * 540 + center[check_1][1]
                trajs_y = int(trajs_y)
                w_img = data.get_img([trajs_x, trajs_y], [trajs_x, trajs_y], ndpi_list[l], level_list[l], cuda_1)
                wib = w_img.train_data_old
                wis = w_img.train_data_new
                wib = torch.unsqueeze(wib, dim=0)
                wis = torch.unsqueeze(wis, dim=0)
                win_img_big = torch.cat((win_img_big, wib), 0)
                win_img_small = torch.cat((win_img_small, wis), 0)            
                if min([(trajs_x - point_list[l][x][0]) ** 2 + (trajs_y - point_list[l][x][1]) ** 2 for x in range(len(point_list[l]))]) < 64:
                    one_hot.append(one_hot[check_1])
                else:
                    one_hot.append(0.0)
                tr.append([trajs_x, trajs_y])
                check_1 = check_1 + 1
            tra.append(tr)
        point_list = point_list + point_list + tra     

        point_new = []
        center = center + center + center
        for m in point_list:
            point_new = point_new + m 
        input_ = []
        for o in range(len(point_new)):
            if o < sum(len_list) or o >= 2 * sum(len_list):
                input_.append([(point_new[o][0] - center[o][0]) / 540, (point_new[o][1] - center[o][1]) / 540])
            else:
                input_.append([(point_new[o][0] - center[o][0]) / 540, (center[o][1] - point_new[o][1]) / 540])
        input__ = []
        one_hot_ = []
        for i in range(batch_size):
            input__ = input__ + input_[sum(len_list[: i + 1]) - len_list[i] : sum(len_list[: i + 1])]
            input__ = input__ + input_[sum(len_list) + sum(len_list[: i + 1]) - len_list[i] : sum(len_list) + sum(len_list[: i + 1])]
            input__ = input__ + input_[sum(len_list) + sum(len_list) + sum(len_list[: i + 1]) - len_list[i] : sum(len_list) + sum(len_list) + sum(len_list[: i + 1])]
            one_hot_ = one_hot_ + one_hot[sum(len_list[: i + 1]) - len_list[i] : sum(len_list[: i + 1])]
            one_hot_ = one_hot_ + one_hot[sum(len_list) + sum(len_list[: i + 1]) - len_list[i] : sum(len_list) + sum(len_list[: i + 1])]
            one_hot_ = one_hot_ + one_hot[sum(len_list) + sum(len_list) + sum(len_list[: i + 1]) - len_list[i] : sum(len_list) + sum(len_list) + sum(len_list[: i + 1])]
        one_hots = torch.tensor(one_hot_, dtype = torch.float).to(cuda_1)
        one_hots = torch.unsqueeze(one_hots, 1)
        input__ = torch.tensor(input__, dtype = torch.float).to(cuda_1)
        len_list_ = len_list + len_list + len_list
        '''
        l_list = copy.deepcopy(len_list_)
        l_list.sort(key=lambda x:-x)
        
        l_list2 = []
        checki = 0
        for i in range(len(l_list)):
            for j in range(l_list[i]):
                if j != 0:
                    l_list2.append(checki)
                checki = checki + 1
        #torch.cuda.empty_cache()
        one_hots = one_hots.index_select(0, torch.tensor(l_list2).to(cuda_1))
        '''
        with torch.no_grad():
            encoder = mae(win_img_big, win_img_small, len_list_)
            mse = torch.tensor([], dtype = torch.float).to(cuda_1)
            for en in range(len(encoder)):
                mse = torch.cat((mse, encoder[en].clone().detach()), dim = 0)
        rebuild, reward= cost_f(img_screen_list, encoder, input__, len_list_, batch_size, cuda_1, padding = True)
        
        loss1 = F.mse_loss(reward, one_hots)
        loss2 = F.mse_loss(rebuild, mse)
        cost_optimizer.zero_grad()
        loss_f = loss1 + 0.1 * loss2
        loss_f.backward()
        cost_optimizer.step()
        #StepLR_1.step()
        loss_rebuild_list.append(loss2.item())
        loss_reward_list.append(loss1.item())
        del loss1
        del loss2
        del loss_f

    cost_name = '/home/omnisky/sde/NanTH/result/irl/' + str(step) + 'cost_net' + '.pth'
    torch.save(cost_f.state_dict(), cost_name)
    torch.cuda.empty_cache()
    with torch.no_grad():
        ture = 1
        our_demo_ = copy.deepcopy(test_demo)
        mae.eval()
        cost_f.eval()
        test_loss_rebuild_list = []
        test_loss_reward_list = []
        while(ture):
            if (time_ % 500 == 0):
                print('---------------------', 'step:', step, '     time:', time_, '----------------------')
                now = datetime.datetime.now()
                print (now.strftime("%Y-%m-%d %H:%M:%S"))
            time_ = time_ + 1
            if len(our_demo_) <= batch_size: #!!!!!!!!!
                break        
            selected_num = np.random.choice(len(our_demo_) - batch_size, batch_size)
            selected_demo = []
            for unm_ in selected_num:
                selected_demo.append(our_demo_[unm_])
                del our_demo_[unm_]
            
            if len(selected_demo) > (batch_size + 1) : #or len(our_demo) < 2000:
                print('error')
                break
            one_hot = []
            point_list =[]
            demo_old = []        
            ndpi_list = []
            level_list = []
            epr_list = []
            len_list = []
            center = []
            win_img_big = torch.tensor([], dtype=torch.float, device = cuda_1)
            win_img_small = torch.tensor([], dtype=torch.float, device = cuda_1)
            img_screen_list = torch.tensor([], dtype=torch.float, device = cuda_1)
            selected_demo.sort(key=lambda x:-len(x))
            wsi_lable = []
            move_lable = []
            for i in range(len(selected_demo)):
                points = []
                level = selected_demo[i][0][0].level
                ndpi = selected_demo[i][0][0].ndpi
                epr = selected_demo[i][0][0].epr
                wsi_l = selected_demo[i][0][0].label
                if selected_demo[i][-1][-1].next_level - level > 0:
                    move_l = 2
                elif selected_demo[i][-1][-1].next_level - level == 0:
                    move_l = 0
                elif selected_demo[i][-1][-1].next_level - level < 0:
                    move_l = 1
                level_list.append(level)
                ndpi_list.append(ndpi)
                epr_list.append(epr)
                len_list.append(len(selected_demo[i]))
                wsi_lable.append(wsi_l)
                if wsi_l >= 0:
                    wsi_lable.append(0)
                else:
                    wsi_lable.append(-1)
                move_lable.append(move_l)
                move_lable.append(move_l)
                move_lable.append(0)
                screen_xy = [selected_demo[i][0][0].screen_x, selected_demo[i][0][0].screen_y]
                label1_list = []
                label2_list = []
                #img_s = img_screen.read_region(((screen_xy[0] * 2 ** level) - (p * 2 ** level), (screen_xy[1] * 2 ** level) - (p * 2 ** level)), level, (1080, 1080)).convert('RGB')
                for classes in range(len(selected_demo[i])):
                    x_y = [int(np.mean([x.X for x in selected_demo[i][classes]])), int(np.mean([x.Y for x in selected_demo[i][classes]]))]
                    points.append(x_y)
                    img_ = data.get_img(x_y, x_y, ndpi, level, cuda_1)
                    img_big = img_.train_data_old
                    img_small = img_.train_data_new
                    img_big = torch.unsqueeze(img_big, dim=0)
                    img_small = torch.unsqueeze(img_small, dim=0)
                    win_img_big = torch.cat((win_img_big, img_big), 0)
                    win_img_small = torch.cat((win_img_small, img_small), 0)
                    if len(selected_demo[i][classes]) <= 20:
                        label1_list.append(len(selected_demo[i][classes]))
                    else:
                        label1_list.append(20)
                for j in range(len(points)):
                    label2_list.append(min([math.sqrt((points[j][0] - points[j - 1][0]) ** 2 + (points[j][1] - points[j - 1][1]) ** 2), math.sqrt((points[j][0] - points[(j + 1) % len(points)][0]) ** 2 + (points[j][1] - points[(j + 1) % len(points)][1]) ** 2)]))
                for j2 in range(len(label1_list)):
                    one_hot.append(0.5 * (0.9 ** (np.mean(label1_list) / label1_list[j2])) + 0.5 * (0.9 ** (label2_list[j2] / np.mean(label2_list))))
                for j3 in range(len(points)):
                    one_hot[- j3 - 1] = one_hot[- j3 - 1] * 0.95 ** j3
                max_x = max([x[0] for x in points])
                min_x = min([x[0] for x in points])
                max_y = max([x[1] for x in points])
                min_y = min([x[1] for x in points])
                img_screen = openslide.OpenSlide(ndpi)
                #if (max_x - min_x) < 800 and (max_y - min_y) < 800:
                    #img_s = img_screen.read_region((((min_x - 128) * (2 ** level)), ((min_y - 128) * (2 ** level))), level, (1080, 1080)).convert('RGB')
                    #img_s = img_s.resize((224,224),Image.ANTIALIAS)
                    #img_s = img_screen.read_region((((min_x - 128) * (2 ** level)), ((min_y - 128) * (2 ** level))), level + 2, (270, 270)).convert('RGB')
                    #img_s = img_s.resize((224,224),Image.ANTIALIAS)
                    #center = center + [[(max_x + min_x) / 2, (max_y + min_y) / 2] for row in range(len(selected_demo[i]))]
                #else:
                img_s = img_screen.read_region(((screen_xy[0] * (2 ** level)), (screen_xy[1] * (2 ** level)) - (420 * 2 ** level)), level + 3, (240, 240)).convert('RGB')
                img_s = img_s.resize((224,224),Image.ANTIALIAS)
                center = center + [[screen_xy[0] + 960, screen_xy[1] + 540] for row in range(len(selected_demo[i]))]
                img_s = trans(img_s)
                img_s = img_s.to(cuda_1)
                img_s = torch.unsqueeze(img_s, dim=0)
                img_screen_list = torch.cat((img_screen_list, img_s), 0)
                point_list.append(points)
            q = win_img_big.size()[0]
            for m in range(q):
                win_img_big = torch.cat((win_img_big, torch.unsqueeze(flip(win_img_big[m].clone().detach()), dim = 0)), 0)
                win_img_small = torch.cat((win_img_small, torch.unsqueeze(flip(win_img_small[m].clone().detach()), dim = 0)), 0)
            one_hot = one_hot + one_hot
            tra = [] 
            check_1 = 0       
            for l in range(len(point_list)):
                '''
                tr = []
                tr.append(point_list[l][0])
                w_img = data.get_img(point_list[l][0], point_list[l][0], ndpi_list[l], level_list[l])
                wib = w_img.train_data_old
                wis = w_img.train_data_new
                wib = torch.unsqueeze(wib, dim=0)
                wis = torch.unsqueeze(wis, dim=0)
                win_img_big = torch.cat((win_img_big, wib), 0)
                win_img_small = torch.cat((win_img_small, wis), 0)
                '''
                tr = []
                for p_ in range(len(point_list[l])):                
                    trajs_x = random.uniform(-1, 1)
                    trajs_x = trajs_x * 540 + center[check_1][0]
                    trajs_x = int(trajs_x)
                    trajs_y = random.uniform(-1, 1)
                    trajs_y = trajs_y * 540 + center[check_1][1]
                    trajs_y = int(trajs_y)
                    w_img = data.get_img([trajs_x, trajs_y], [trajs_x, trajs_y], ndpi_list[l], level_list[l], cuda_1)
                    wib = w_img.train_data_old
                    wis = w_img.train_data_new
                    wib = torch.unsqueeze(wib, dim=0)
                    wis = torch.unsqueeze(wis, dim=0)
                    win_img_big = torch.cat((win_img_big, wib), 0)
                    win_img_small = torch.cat((win_img_small, wis), 0)            
                    if min([(trajs_x - point_list[l][x][0]) ** 2 + (trajs_y - point_list[l][x][1]) ** 2 for x in range(len(point_list[l]))]) < 64:
                        one_hot.append(one_hot[check_1])
                    else:
                        one_hot.append(0.0)
                    tr.append([trajs_x, trajs_y])
                    check_1 = check_1 + 1
                tra.append(tr)
            point_list = point_list + point_list + tra     

            point_new = []
            center = center + center + center
            for m in point_list:
                point_new = point_new + m 
            input_ = []
            for o in range(len(point_new)):
                if o < sum(len_list) or o >= 2 * sum(len_list):
                    input_.append([(point_new[o][0] - center[o][0]) / 540, (point_new[o][1] - center[o][1]) / 540])
                else:
                    input_.append([(point_new[o][0] - center[o][0]) / 540, (center[o][1] - point_new[o][1]) / 540])
            input__ = []
            one_hot_ = []
            for i in range(batch_size):
                input__ = input__ + input_[sum(len_list[: i + 1]) - len_list[i] : sum(len_list[: i + 1])]
                input__ = input__ + input_[sum(len_list) + sum(len_list[: i + 1]) - len_list[i] : sum(len_list) + sum(len_list[: i + 1])]
                input__ = input__ + input_[sum(len_list) + sum(len_list) + sum(len_list[: i + 1]) - len_list[i] : sum(len_list) + sum(len_list) + sum(len_list[: i + 1])]
                one_hot_ = one_hot_ + one_hot[sum(len_list[: i + 1]) - len_list[i] : sum(len_list[: i + 1])]
                one_hot_ = one_hot_ + one_hot[sum(len_list) + sum(len_list[: i + 1]) - len_list[i] : sum(len_list) + sum(len_list[: i + 1])]
                one_hot_ = one_hot_ + one_hot[sum(len_list) + sum(len_list) + sum(len_list[: i + 1]) - len_list[i] : sum(len_list) + sum(len_list) + sum(len_list[: i + 1])]
            one_hots = torch.tensor(one_hot_, dtype = torch.float).to(cuda_1)
            one_hots = torch.unsqueeze(one_hots, 1)
            input__ = torch.tensor(input__, dtype = torch.float).to(cuda_1)
            len_list_ = len_list + len_list + len_list
            '''
            l_list = copy.deepcopy(len_list_)
            l_list.sort(key=lambda x:-x)
            l_list2 = []
            checki = 0
            for i in range(len(l_list)):
                for j in range(l_list[i]):
                    if j != 0:
                        l_list2.append(checki)
                    checki = checki + 1
        #torch.cuda.empty_cache()
            one_hots = one_hots.index_select(0, torch.tensor(l_list2).to(cuda_1))
            '''
            #torch.cuda.empty_cache()
            with torch.no_grad():
                mse = torch.tensor([], dtype = torch.float).to(cuda_1)
                encoder = mae(win_img_big, win_img_small, len_list_)
                for en in range(len(encoder)):
                    mse = torch.cat((mse, encoder[en]), dim = 0)
            rebuild, reward= cost_f(img_screen_list, encoder, input__, len_list_, batch_size, cuda_1, padding = True)
            loss1 = F.mse_loss(reward, one_hots)
            loss2 = F.mse_loss(rebuild, mse)
            #cost_loss_list.append(loss1.item())
            test_loss_rebuild_list.append(loss2.item())
            test_loss_reward_list.append(loss1.item())
            del loss1
            del loss2
    print('train_result:')  
    print('rebuild:', np.mean(loss_rebuild_list), 'reward:', np.mean(loss_reward_list))
    #else:
        #print('cost:', np.mean(cost_loss_list))
    print('test_result:')
    print('rebuild:', np.mean(test_loss_rebuild_list), 'reward:', np.mean(test_loss_reward_list))
    del test_loss_rebuild_list
    del test_loss_reward_list
    del loss_rebuild_list
    del loss_reward_list
    torch.cuda.empty_cache()
#####################################################################################

