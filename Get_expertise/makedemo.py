#-*- coding:utf-8 –*-
import eprRead as olderEPRReader
import openslide
import math
import os
import numpy as np
import openslide
import torch
class DatumFrame:
    def __init__(self, timeStamp, level, X, Y, epr, ndpi, num, screen_x, screen_y, label, next_level):
        self.level = level
        self.X = X
        self.Y = Y
        self.time = timeStamp
        self.epr = epr
        self.ndpi = ndpi
        self.num = num
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.label = label
        self.next_level = next_level

class make_demo():
    def __init__(self, epr, ndpi, i):    
        self.data = []
        self.data2 = []
        self.the_demo = []
        level_list = []
        self.demo_ = []
        self.num = i
        self.windows = []
        self.the_demo_ = []
        slide = openslide.OpenSlide(ndpi)        
        a = olderEPRReader.EPRread(epr)            
        if slide.properties['openslide.quickhash-1'] == a.quickHash:
            if ndpi.split('/')[-3] != 'nevus' and ndpi.split('/')[-3] != 'cn' and ndpi.split('/')[-3] != 'jn' and ndpi.split('/')[-3] != 'in':
                label = 1
            else:
                label = 0
            sum_step = len(a.data_frame)
            slide = openslide.OpenSlide(ndpi)
            self.max_epr_level = a.firstLevel - a.minlevel
            check = 1
            for d in a.data_frame:
                if((d.eyeX - d.screenX) > 0 and (d.eyeY - d.screenY) > 0 and d.eyeX > 0 and d.eyeY > 0 and d.eyeX < 1920 and d.eyeY < 1080):
                    time = d.timeStamp
                    level = d.level - a.minlevel
                    X = d.eyeX - d.screenX
                    Y = d.eyeY - d.screenY
                    screen_x = - d.screenX
                    screen_y = - d.screenY
                    if len(a.data_frame) > check + 5:
                        next_level = a.data_frame[check + 5].level - a.minlevel
                    else:
                        next_level = d.level - a.minlevel
                    level_list.append(level)
                    datumframe = DatumFrame(time, level, X, Y, epr, ndpi, self.num, screen_x, screen_y, label * (0.3 *(check / sum_step) + 0.7), next_level)
                        #print(label * (check / sum_step))
                        #print(label * 0.3)
                    self.data.append(datumframe)
                check = check + 1
            #self.min_epr_level = min(level_list)
            for p in range(len(self.data)):
                if self.data[p].level == 2 or self.data[p].level == 1:
                    self.data2.append(self.data[p])
            check = 0
            point = 0
            the_class = []
            while(point < len(self.data2)):
                if self.data2[point].level == 1:
                    if len(the_class) == 0:
                        the_class.append(self.data2[point])
                    else:
                        lenth_ = 0.002 * math.sqrt(((slide.level_dimensions[self.data2[point].level][0]) ** 2) + ((slide.level_dimensions[self.data2[point].level][1]) ** 2))
                        if lenth_ > 30:
                            lenth_ = 30
                        mean_x = np.mean([x.X for x in the_class])
                        mean_y = np.mean([x.Y for x in the_class])
                        if math.sqrt((self.data2[point].X - mean_x) ** 2 + (self.data2[point].Y - mean_y) ** 2) < lenth_ and math.sqrt((self.data2[point].X - the_class[0].X) ** 2 + (self.data2[point].Y - the_class[0].Y) ** 2) < lenth_:
                            the_class.append(self.data2[point])
                        else:
                            self.the_demo.append(the_class)
                            the_class = []
                            the_class.append(self.data2[point])
                point = point + 1
            self.the_demo.append(the_class)
            point = 0
            the_class = []
            while(point < len(self.data2)):
                if self.data2[point].level == 2:
                    if len(the_class) == 0:
                        the_class.append(self.data2[point])
                    else:
                        lenth_ = 0.002 * math.sqrt(((slide.level_dimensions[self.data2[point].level][0]) ** 2) + ((slide.level_dimensions[self.data2[point].level][1]) ** 2))
                        if lenth_ > 30:
                            lenth_ = 30
                        mean_x = np.mean([x.X for x in the_class])
                        mean_y = np.mean([x.Y for x in the_class])
                        if math.sqrt((self.data2[point].X - mean_x) ** 2 + (self.data2[point].Y - mean_y) ** 2) < lenth_ and math.sqrt((self.data2[point].X - the_class[0].X) ** 2 + (self.data2[point].Y - the_class[0].Y) ** 2) < lenth_:
                            the_class.append(self.data2[point])
                        else:
                            self.the_demo.append(the_class)
                            the_class = []
                            the_class.append(self.data2[point])
                point = point + 1
            self.the_demo.append(the_class)          
            the_class = []
            point = 0
            '''
            while(point < len(self.data2)):
                if self.data2[point].level == 3:
                    if len(the_class) == 0:
                        the_class.append(self.data2[point])
                    else:
                        lenth_ = 0.002 * math.sqrt(((slide.level_dimensions[self.data2[point].level][0]) ** 2) + ((slide.level_dimensions[self.data2[point].level][1]) ** 2))
                        if lenth_ > 30:
                            lenth_ = 30
                        mean_x = np.mean([x.X for x in the_class])
                        mean_y = np.mean([x.Y for x in the_class])
                        if math.sqrt((self.data2[point].X - mean_x) ** 2 + (self.data2[point].Y - mean_y) ** 2) < lenth_ and math.sqrt((self.data2[point].X - the_class[0].X) ** 2 + (self.data2[point].Y - the_class[0].Y) ** 2) < lenth_:
                            the_class.append(self.data2[point])
                        else:
                            self.the_demo.append(the_class)
                            the_class = []
                            the_class.append(self.data2[point])
                point = point + 1
            '''
            self.the_demo.append(the_class)
            for i in self.the_demo:
                if len(i) > 4:
                    self.the_demo_.append(i)   
            for d in self.the_demo_:
                if d[0].level != d[-1].level:
                    print('level_error')
            d = 0
            while d < len(self.the_demo_) - 1:
                screen1 = self.the_demo_[d][0].screen_x + self.the_demo_[d][0].screen_y
                for d_ in range(d + 1, len(self.the_demo_)):                 
                    screen2 = self.the_demo_[d_][0].screen_x + self.the_demo_[d_][0].screen_y
                    if screen1 != screen2 or d_ - d > 9:
                        break
                if len(self.the_demo_[d : d_]) > 2 and self.the_demo_[d][0].level == self.the_demo_[d_][0].level:
                    self.windows.append(self.the_demo_[d : d_])
                d = d_   
                #print('1')

            
        '''
        while(p < len(self.the_demo) - 1):
            ju_li = math.sqrt((self.the_demo[p].p4.X - self.the_demo[p + 1].p4.X) ** 2 + (self.the_demo[p].p4.Y - self.the_demo[p + 1].p4.Y) ** 2)
            point_juli = max([(self.the_demo[p].p4.X - self.the_demo[p + 1].p4.X) ** 2, (self.the_demo[p].p4.Y - self.the_demo[p + 1].p4.Y) ** 2])
            if(self.the_demo[p].p1.level == self.the_demo[p + 1].p1.level and ju_li < 314 and ju_li > 23 and point_juli < 40000):
                self.demo_.append([self.the_demo[p], self.the_demo[p + 1]])
            p = p + 1
        '''
def get_epr_filelist1(root_path):#输入路径
    epr_path = []
    file_name_list = []
    epr_sort = os.listdir(root_path)
    get_key1 = lambda i : int(i[0 : 3])
    epr_sort_ = sorted(epr_sort, key=get_key1)
    for class_name in epr_sort_:
        tumor_root_path = os.path.join(root_path, class_name)
        epr_sort2 = os.listdir(tumor_root_path)
        get_key2 = lambda i : int(i[0 : 7])
        epr_sort2_ = sorted(epr_sort2, key=get_key2)
        for file_name in epr_sort2_:
            tumor_path = os.path.join(tumor_root_path, file_name)  #epr数据的实际地址
            epr_dict = {"epr": tumor_path}
            epr_path.append(epr_dict)
            file_name_list.append(file_name)
            
    return epr_path, file_name_list



def get_ndpi_filelist1(root_path, epr_path, epr_name):#输入路径
    ndpi_path = []
    get_key1 = lambda i : int(i[0 : 3])
    get_key2 = lambda i : int(i[0 : 7])
    epr_num = 0
    while(epr_num < len(epr_path)):
    #for epr_num in range(len(epr_path)):
        true = 0
        ndpi_sort = os.listdir(root_path)
        ndpi_sort_ = sorted(ndpi_sort, key=get_key1)
        for class_name in ndpi_sort_:
            tumor_root_path = os.path.join(root_path, class_name)
            ndpi_sort2 = os.listdir(tumor_root_path)        
            ndpi_sort2_ = sorted(ndpi_sort2, key=get_key2)
            for file_name in ndpi_sort2_:
                if(epr_name[epr_num][: 7] == file_name[: 7]):
                    tumor_path = os.path.join(tumor_root_path, file_name)  #ndpi数据的实际地址
                    ndpi_dict = {"ndpi": tumor_path}
                    ndpi_path.append(ndpi_dict)
                    true = 1
                    break
            if true == 1:
                break
        if true == 0:
            del epr_path[epr_num]
            del epr_name[epr_num]
            epr_num = epr_num - 1
        epr_num = epr_num + 1
    return ndpi_path, epr_path

class demo_list():
    def __init__(self, path):
        self.demo__ = []
        epr = []
        ndpi = []
        for j in range(len(path)):
            if ('epr' in path[j][0].keys()) == True:
                epr.append(path[j][0]['epr'])
                ndpi.append(path[j][0]['slide'])
        if(len(epr) == len(ndpi)):
            for i in range(len(epr)):
                m = make_demo(epr[i], ndpi[i], i)
                #for j in range(len(m.demo_)):
                if len(m.windows) != 0:
                    self.demo__.append(m.windows)     #print('data ok')
        else:
            print('error')
'''
make_demo2 = demo_list('/home/omnisky/hdd_15T_sdc/NanTH/IRL_DATA/4th/epr', '/home/omnisky/hdd_15T_sdc/NanTH/IRL_DATA/4th/ndpi')
demo_trajs2 = make_demo2.demo__
make_demo3 = demo_list('/home/omnisky/hdd_15T_sdc/NanTH/IRL_DATA/Mr,Zheng/epr', '/home/omnisky/hdd_15T_sdc/NanTH/IRL_DATA/Mr,Zheng/2021.10.19')
demo_trajs3 = make_demo3.demo__
make_test_demo = demo_list('/home/omnisky/hdd_15T_sdc/NanTH/IRL_DATA/Mr,Zheng/epr_t', '/home/omnisky/hdd_15T_sdc/NanTH/IRL_DATA/Mr,Zheng/ndpi_t')
demo_tset = make_test_demo.demo__
demo_tset = demo_tset + demo_trajs2 + demo_trajs3
check_num = 0
test_demo = []
print("photo number:", len(demo_tset))
for m in range(0, len(demo_tset)):
    for n in range(len(demo_tset[m])):
        test_demo.append(demo_tset[m][n])
        check_num = check_num + 1
print('len_train:', len(test_demo))
lib = []
len_list = []
for i in range(len(test_demo)):
    if test_demo[i][0][0].level == 2:
        center = []
        for classes in range(len(test_demo[i])):
            len_list.append(len(test_demo[i][classes]))
            center.append([np.mean([x.X for x in test_demo[i][classes]]), np.mean([x.Y for x in test_demo[i][classes]])])
        for j in range(len(center)):
            if min([math.sqrt((center[j][0] - center[j - 1][0]) ** 2 + (center[j][1] - center[j - 1][1]) ** 2), math.sqrt((center[j][0] - center[(j + 1) % len(test_demo[i])][0]) ** 2 + (center[j][1] - center[(j + 1) % len(test_demo[i])][1]) ** 2)]) > 1400:
                print(test_demo[i][0][0].ndpi)
            lib.append(min([math.sqrt((center[j][0] - center[j - 1][0]) ** 2 + (center[j][1] - center[j - 1][1]) ** 2), math.sqrt((center[j][0] - center[(j + 1) % len(test_demo[i])][0]) ** 2 + (center[j][1] - center[(j + 1) % len(test_demo[i])][1]) ** 2)]))

print('avg juli:', np.mean(lib))
print('max juli:', max(lib))
print('min juli:', min(lib))
print('avg len:', np.mean(len_list))
print('max len:', max(len_list))
print('min len:', min(len_list))
lib = []
len_list = []

for i in range(len(test_demo)):
    if test_demo[i][0][0].level == 3:
        center = []
        for classes in range(len(test_demo[i])):
            len_list.append(len(test_demo[i][classes]))
            center.append([np.mean([x.X for x in test_demo[i][classes]]), np.mean([x.Y for x in test_demo[i][classes]])])
        for j in range(len(center)):
            if min([math.sqrt((center[j][0] - center[j - 1][0]) ** 2 + (center[j][1] - center[j - 1][1]) ** 2), math.sqrt((center[j][0] - center[(j + 1) % len(test_demo[i])][0]) ** 2 + (center[j][1] - center[(j + 1) % len(test_demo[i])][1]) ** 2)]) > 1500:
                print(test_demo[i][0][0].ndpi)
            lib.append(min([math.sqrt((center[j][0] - center[j - 1][0]) ** 2 + (center[j][1] - center[j - 1][1]) ** 2), math.sqrt((center[j][0] - center[(j + 1) % len(test_demo[i])][0]) ** 2 + (center[j][1] - center[(j + 1) % len(test_demo[i])][1]) ** 2)]))

print('avg juli:', np.mean(lib))
print('max juli:', max(lib))
print('min juli:', min(lib))
print('avg len:', np.mean(len_list))
print('max len:', max(len_list))
print('min len:', min(len_list))
'''