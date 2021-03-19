import argparse
import random
import pandas as pd
import numpy as np
from scipy import sparse
import pickle
import time
import re

import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import dgl
from sklearn.metrics import f1_score, roc_auc_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from summa.summarizer import summarize


def cleanlines(line):   #去除标点等无用的符号
    line = re.sub(r"can\'t", "can not", line)
    line = re.sub(r"cannot", "can not ", line)
    line = re.sub(r"what\'s", "what is", line)
    line = re.sub(r"What\'s", "what is", line)
    line = re.sub(r"\'ve ", " have ", line)
    line = re.sub(r"n\'t", " not ", line)
    line = re.sub(r"i\'m", "i am ", line)
    line = re.sub(r"I\'m", "i am ", line)
    line = re.sub(r"\'re", " are ", line)
    line = re.sub(r"\'d", " would ", line)
    line = re.sub(r"\'ll", " will ", line)
    line = re.sub(r" e mail ", " email ", line)
    line = re.sub(r" e \- mail ", " email ", line)
    line = re.sub(r" e\-mail ", " email ", line)
    
    p3=re.compile(r'[「『]')
    p4=re.compile(r'[\s+\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \-\ \[\ \]\ ]')
    line=p3.sub(r' ',line)
    line=p4.sub(r' ',line)
    
    line = line.strip()
    return line

def api_txt_pro(words, apica):
    api_info = pd.read_csv('./datao/api-info-all1.csv', encoding='utf-8')
    api_info['all-description'].fillna("None", inplace=True)
    api_info['name'].replace( 'API', '', regex=True, inplace=True)
    api_des = api_info['name']+'.'+api_info['all-description']
    api_des = api_des.apply(cleanlines)
    for i in range(len(api_des)):
        if i not in apica:
            apica[i] = []
        temp = summarize(text=api_des[i], words=words, cid=apica[i])
        if len(temp)>0:
            api_des[i] = temp
    des_list = np.array(api_des).tolist()

    return des_list

def ms_txt_pro(words, msca):
    ms_info = pd.read_csv('./datao/ms-info-all1.csv', encoding='utf-8')
    ms_info['all-description'].fillna("None", inplace=True)
    ms_des = ms_info['name'] +'.'+ ms_info['all-description']
    ms_des = ms_des.apply(cleanlines)
    for i in range(len(ms_des)):
        if i not in msca:
            msca[i] = []
        temp = summarize(text=ms_des[i], words=words, cid=msca[i])
        if len(temp)>0:
            ms_des[i] = temp
    #keywords(api_des[10], words=5)
    des_list = np.array(ms_des).tolist()

    return des_list


class HNet:
    g = None
    api_t0 = []
    api_t = None
    ms_t0 = []
    ms_t = None
    api_o = None
    ms_o = None

    def __init__(self, ch_ms, ch_api, words):
        # 筛选节点，创建异构网络
        datas = {}
        # 读入MS-CA边
        msca = {}
        edge = pd.read_csv('./datao/ms-ca.csv', encoding='utf-8')
        for i in range(len(edge['mid'])):
            mid = edge['mid'][i]-22967
            cid = edge['cid'][i]
            if mid in msca:
                msca[mid].append(cid)
            else:
                msca[mid] = [cid]
        ms = np.array(edge['mid']-22967).tolist()
        ca = np.array(edge['cid']-22481).tolist()
        data = [1]*edge.shape[0]
        datas['MvsC'] = sparse.coo_matrix((data, (ms, ca)), shape=(7954, 486)).tocsc()

        # 读入API-CA边
        apica = {}
        edge = pd.read_csv('./datao/api-ca.csv', encoding='utf-8')
        for i in range(len(edge['aid'])):
            aid = edge['aid'][i]
            cid = edge['cid'][i]
            if aid in apica:
                apica[aid].append(cid)
            else:
                apica[aid] = [cid]
        api = np.array(edge['aid']).tolist()
        ca = np.array(edge['cid']-22481).tolist()
        data = [1]*edge.shape[0]
        datas['AvsC'] = sparse.coo_matrix((data, (api, ca)), shape=(22481, 486)).tocsc()

        #读入API-PRO边
        edge = pd.read_csv('./datao/api-pro.csv', encoding='utf-8')
        api = np.array(edge['aid']).tolist()
        pro = np.array(edge['pid']-30921).tolist()
        data = [1]*edge.shape[0]
        datas['AvsP'] = sparse.coo_matrix((data, (api, pro)), shape=(22481, 14281)).tocsc()

        #读入MS-AP边
        edge = pd.read_csv('./datao/ms-api.csv', encoding='utf-8')
        ms = np.array(edge['mid']-22967).tolist()
        api = np.array(edge['aid']).tolist()
        data = [1]*edge.shape[0]
        datas['MvsA'] = sparse.coo_matrix((data, (ms, api)), shape=(7954, 22481)).tocsc()

        # 读入dev-dms边
        edge = pd.read_csv('./datao/usr-dms.csv', encoding='utf-8')
        dev = np.array(edge['uid']-45202).tolist()
        ms = np.array(edge['mid']-22967).tolist()
        data = [1]*edge.shape[0]
        datas['DvsdM'] = sparse.coo_matrix((data, (dev, ms)), shape=(2842, 7954)).tocsc()

        edge = pd.read_csv('./datao/usr-fms.csv', encoding='utf-8')
        dev = np.array(edge['uid']-45202).tolist()
        ms = np.array(edge['mid']-22967).tolist()
        data = [1]*edge.shape[0]
        datas['DvsfM'] = sparse.coo_matrix((data, (dev, ms)), shape=(2842, 7954)).tocsc()

        # 读入dev-ap边
        edge = pd.read_csv('./datao/usr-api.csv', encoding='utf-8')
        dev = np.array(edge['uid']-45202).tolist()
        api = np.array(edge['aid']).tolist()
        data = [1]*edge.shape[0]
        datas['DvsA'] = sparse.coo_matrix((data, (dev, api)), shape=(2842, 22481)).tocsc()

        # 选出用于计算的内容
        datas['MvsC'] = datas['MvsC'][ch_ms]
        datas['AvsC'] = datas['AvsC'][ch_api]
        datas['AvsP'] = datas['AvsP'][ch_api]
        datas['MvsA'] = datas['MvsA'][ch_ms]
        datas['MvsA'] = datas['MvsA'][:, ch_api]
        datas['DvsdM'] = datas['DvsdM'][:, ch_ms]
        datas['DvsfM'] = datas['DvsfM'][:, ch_ms]
        datas['DvsA'] = datas['DvsA'][:, ch_api]

        self.g = dgl.heterograph({
            ('mashup', 'ma', 'api') : datas['MvsA'],
            ('api', 'am', 'mashup') : datas['MvsA'].transpose(),
            ('mashup', 'mc', 'category') : datas['MvsC'],
            ('category', 'cm', 'mashup') : datas['MvsC'].transpose(),
            ('api', 'ac', 'category') : datas['AvsC'],
            ('category', 'ca', 'api') : datas['AvsC'].transpose(),
            ('api', 'ap', 'provider') : datas['AvsP'],
            ('provider', 'pa', 'api') : datas['AvsP'].transpose(),
            ('developer', 'ddm', 'mashup') : datas['DvsdM'],
            ('mashup', 'mdd', 'developer') : datas['DvsdM'].transpose(),
            ('developer', 'dfm', 'mashup') : datas['DvsfM'],
            ('mashup', 'mfd', 'developer') : datas['DvsfM'].transpose(),
            ('developer', 'da', 'api') : datas['DvsA'],
            ('api', 'ad', 'developer') : datas['DvsA'].transpose()
        })

        api_textdata = api_txt_pro(words, apica)
        n = 0
        for i in range(len(api_textdata)):
            if i not in ch_api:
                api_textdata.pop(i-n)
                n += 1
        self.api_t0 = api_textdata

        ms_textdata = ms_txt_pro(words, msca)
        n = 0
        for i in range(len(ms_textdata)):
            if i not in ch_ms:
                ms_textdata.pop(i-n)
                n += 1
        self.ms_t0 = ms_textdata

        #读入其他数据
        with open('./datao/api_otherdata.pkl', 'rb') as f:
            api_o = np.array(pickle.load(f))
        self.api_o = torch.FloatTensor(api_o[ch_api])
        with open('./datao/ms_otherdata.pkl', 'rb') as f:
            ms_o = np.array(pickle.load(f))
        self.ms_o = torch.FloatTensor(ms_o[ch_ms])
        #self.edge_o = torch.cat((api_o.repeat(1, ms_o.shape[0]).view(-1, api_o.shape[1]), ms_o.repeat(api_o.shape[0],1)), 1)
        #print(self.edge_o.shape)

    def extendDes(self):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.api_t0)]
        model = Doc2Vec(documents, vector_size=500, window=5, min_count=3, workers=4, epochs=25)
        docvecs = model.docvecs.vectors_docs
        data = []
        for i in range(len(self.api_t0)):
            data.append(docvecs[i])
        self.api_t = torch.FloatTensor(np.array(data))

        ms_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.ms_t0)]
        ms_model = Doc2Vec(ms_documents, vector_size=500, window=5, min_count=3, workers=4, epochs=25)
        ms_docvecs = ms_model.docvecs.vectors_docs
        ms_data = []
        for i in range(len(self.ms_t0)):
            ms_data.append(ms_docvecs[i])
        self.ms_t = torch.FloatTensor(np.array(ms_data))
    
    def calSim(self, type, sour, tar, nums):
        # 相似度计算，参考 https://blog.csdn.net/qq_41487299/article/details/106299882
        if type == 'ms':
            temp = (self.ms_t[sour]*self.ms_t[tar]).sum(-1)
        else:
            #print((self.api_t[sour]*self.api_t[tar]).shape)
            temp = (self.api_t[sour]*self.api_t[tar]).sum(-1)

        temp = temp.numpy()
        temp = dict(zip([i for i in range(len(temp))],temp))
        # 获取按 value 排序后的元组列表
        items = sorted(temp.items(), key=lambda obj: obj[1], reverse=True)
        ret = []
        for i in range(nums):
            ret.append(items[i][0])
        return ret
    
    def getTxt(self, type, sour, tar, nums):
        """获取Mashup和API的文本特征

        Args:
            type (str): 节点类型
            sour (int): 目标节点id
            tar (int[]): 目标节点邻居id
            nums (int): 需求数量

        Returns:
            tensor: 文本特征
        """
        if len(tar) > nums:
            tar = self.calSim(type, sour, tar, nums)
        temp = torch.zeros(nums, 500)
        if type == 'ms':
            temp[:len(tar)] = self.ms_t[tar]
        else:
            temp[:len(tar)] = self.api_t[tar]

        if type == 'ms':
            w = torch.softmax(self.ms_t[sour]*temp, dim=1)
        else:
            w = torch.softmax(self.api_t[sour]*temp, dim=1)
        datan = (temp*w).sum(0)

        return datan

    def getOth(self, type, sour, tar, nums):
        """获取Mashup和API的其他特征

        Args:
            type (str): 节点类型
            sour (int): 目标节点id
            tar (int[]): 目标节点邻居id
            nums (int): 需求数量

        Returns:
            tensor: 文本特征
        """
        if len(tar) > nums:
            tar = self.calSim(type, sour, tar, nums)
    
        if type == 'ms':
            temp = torch.zeros(nums, 89)
            temp[:len(tar)] = self.ms_o[tar]
        else:
            temp = torch.zeros(nums, 255)
            temp[:len(tar)] = self.api_o[tar]

        if type == 'ms':
            w = torch.softmax(self.ms_o[sour]*temp, dim=1)
        else:
            w = torch.softmax(self.api_o[sour]*temp, dim=1)
        datan = (temp*w).sum(0)

        return datan


class SemanticAttention(nn.Module):
    def __init__(self, in_size):
        super(SemanticAttention, self).__init__()
        self.sa = nn.Sequential(#SemanticAttention(in_size=500)
            nn.Linear(in_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False))

    def forward(self, x):
        w = self.sa(x) 
        beta = torch.softmax(w, dim=1)
        x = (beta * x).sum(1)
        return x


class NETA(nn.Module):
    def __init__(self):
        super(NETA, self).__init__()
        self.sa1 = SemanticAttention(in_size=500)
        self.sa2 = SemanticAttention(in_size=255)
        self.sa3 = SemanticAttention(in_size=89)
        self.mlp = nn.Sequential(
            nn.Linear(1344, 512),
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1))

    def forward(self, x):#, t_others
        # ms_tmp = self.conv1(ms_tmp).view(-1, 300)
        x1 = self.sa1(x[:, :2,:]) 
        x2 = self.sa2(x[:, 3:9, :255])
        x3 = self.sa3(x[:, 9:, :89])
        x = torch.cat((x1, x[:, 2, :].view(-1,500), x2, x3), dim=1)#x1,
        x = self.mlp(x)
        return x


def chooseTrainNode(time):
    ms_time = pd.read_csv('./datao/ms_time.csv', encoding='utf-8')
    api_time = pd.read_csv('./datao/api_time.csv', encoding='utf-8')

    ms_train = []
    for i in range(ms_time.shape[0]):
        if 60-time < int(ms_time['time'][i]) <= 64:
            ms_train.append(i)
    api_train = []
    for i in range(api_time.shape[0]):
        if int(api_time['time'][i]) <= 70:
            api_train.append(i)

    return api_train, ms_train


def chooseTestNode(time):
    ms_time = pd.read_csv('./datao/ms_time.csv', encoding='utf-8')
    api_time = pd.read_csv('./datao/api_time.csv', encoding='utf-8')

    ms_test = []
    for i in range(ms_time.shape[0]):
        if time < int(ms_time['time'][i]) <= time + 9:
            ms_test.append(i)
    api_test = []
    for i in range(api_time.shape[0]):
        if int(api_time['time'][i]) <= time + 9:#0 < 
            api_test.append(i)

    return api_test, ms_test


def getMaskLabel(ch_api, ch_ms, p, train=True):
    edge = pd.read_csv('./datao/ms-api.csv', encoding='utf-8')
    ms = np.array(edge['mid']-22967).tolist()
    api = np.array(edge['aid']).tolist()

    #创建ms-api字典
    msapi = {}
    for i in range(edge.shape[0]):
        if ms[i] not in msapi:
            msapi[ms[i]] = [api[i]]
        else:
            msapi[ms[i]].append(api[i])

    train_labels = []
    train_mask = []
    n = 0
    if train:
        for i in range(len(ch_api)):
            for j in range(len(ch_ms)):
                nj = ch_ms[j]
                if nj in ms and ch_api[i] in msapi[nj]:
                    train_labels.append(n)
                    train_mask.append(n)
                elif random.random() < p:
                    train_mask.append(n)
                n += 1
    else:
        for i in range(len(ch_api)):
            for j in range(len(ch_ms)):
                nj = ch_ms[j]
                if nj in ms and ch_api[i] in msapi[nj]:
                    train_labels.append(n)
                n += 1
    print('Mask Count:', len(train_mask), 'has edge:', len(train_labels))
    train_labels = get_binary_mask(len(ch_api)*len(ch_ms), train_labels).view(-1,1)
    return train_mask, train_labels

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask

def createTFeature(hg, ch_ms, ch_api, mask=False):
    """直接返回所有特征
    Args:
        g (HNnet): 网络数据
        ch_ms (int[]): 选择的ms节点
        ch_api (int[]): 选择的api节点

    Returns:
        [type]: [description]
    """
    msn, apin = len(ch_ms), len(ch_api)

    if mask:
        features = torch.zeros(len(mask), 1000)
        n = 0
        k = 0
        for i in range(len(ch_api)):
            for j in range(len(ch_ms)):
                if n in mask:
                    features[k] = torch.cat((hg.ms_t[j], hg.api_t[i]), 1)
                    k += 1
                n += 1
    else:
        features = torch.zeros(len(ch_api)*len(ch_ms), 1000)
        n = 0
        for i in range(len(ch_api)):
            for j in range(len(ch_ms)):
                features[n] = torch.cat((hg.ms_t[j], hg.api_t[i]), 1)
                n += 1

    return features

def createFeature(hg, ch_ms, ch_api, mask=False):
    """根据元路径和所需邻居个数，返回对应的特征
    Args:
        g (HNnet): 网络数据
        ch_ms (int[]): 选择的ms节点
        ch_api (int[]): 选择的api节点

    Returns:
        [type]: [description]
    """
    msn, apin = len(ch_ms), len(ch_api)

    def dumax(temp, nums):
        """[summary]获取temp中前nums大的节点id

        Args:
            temp ([type]): [description]数组
            nums ([type]): [description]数量
        """
        temp = dict(zip([i for i in range(len(temp))],temp))
        # 获取按 value 排序后的元组列表
        items = sorted(temp.items(), key=lambda obj: obj[1], reverse=True)
        ret = []
        for i in range(nums):
            ret.append(items[i][0])

    def getNeiBor(metapath, nums):
        # 得到元路径邻居， nums为该类型节点总数
        tempg = dgl.metapath_reachable_graph(hg.g, metapath)
        ret = {}
        for i in range(nums):
            ret[i] = tempg.successors(i).numpy()
        return ret

    # am邻居可以直接获取

    temp1 = getNeiBor(('mdd', 'da'), msn) # mda
    temp2 = getNeiBor(('mdd', 'dfm', 'ma'), msn) # mdma
    temp3 = getNeiBor(('mdd', 'ddm', 'ma'), msn)
    mda_all = {} # mashup开发者关注的api
    for i in range(msn):
        mda_all[i] = list(set(temp1[i]) | set(temp2[i]) | set(temp3[i]))

    mca = getNeiBor(('mc', 'ca'), msn) # mca邻居
    aca = getNeiBor(('ac', 'ca'), apin) # aca

    mcma = getNeiBor(('mc', 'cm', 'ma'), msn) # mcma邻居, 3956597个，平均500个
    amca = getNeiBor(('am', 'mc', 'ca'), apin) # amca

    apa = getNeiBor(('ap', 'pa'), apin) # apa

    if type(mask)==type(False):
        features = torch.zeros(len(ch_api)*len(ch_ms), 10, 500)
        n = 0
        for i in range(len(ch_api)):
            for j in range(len(ch_ms)):      
                features[n][0] = hg.api_t[i]
                features[n][1] = hg.getTxt('ms', j, hg.g.successors(i, 'am').numpy(), 10)

                features[n][2] = hg.ms_t[j]

                features[n][3][:255] = hg.getOth('api', i, mda_all[j], 5)
                maca = set(mca[j]) & set(aca[i])
                features[n][4][:255] = hg.getOth('api', i, list(maca), 10)
                maca = list(set(mcma[j]) & set(amca[i]) - maca)
                features[n][5][:255] = hg.getOth('api', i, maca, 10)
                features[n][6][:255] = hg.getOth('api', i, apa[i], 3)
                features[n][7][:255] = hg.api_o[i]

                features[n][8][:89] = hg.getOth('ms', j, hg.g.successors(i, 'am').numpy(), 10)
                features[n][9][:89]= hg.ms_o[j]
                #print(features[n].shape)
                n += 1       
    else:
        features = torch.zeros(len(mask), 10, 500)
        n = 0
        k = 0
        for i in range(len(ch_api)):
            for j in range(len(ch_ms)):
                if n in mask:
                    features[k][0] = hg.api_t[i]
                    features[k][1] = hg.getTxt('ms', j, hg.g.successors(i, 'am').numpy(), 10)

                    features[k][2] = hg.ms_t[j]

                    features[k][3][:255] = hg.getOth('api', i, mda_all[j], 5)
                    maca = set(mca[j]) & set(aca[i])
                    features[k][4][:255] = hg.getOth('api', i, list(maca), 10)
                    maca = list(set(mcma[j]) & set(amca[i]) - maca)
                    features[k][5][:255] = hg.getOth('api', i, maca, 10)
                    features[k][6][:255] = hg.getOth('api', i, apa[i], 3)
                    features[k][7][:255] = hg.api_o[i]

                    features[k][8][:89] = hg.getOth('ms', j, hg.g.successors(i, 'am').numpy(), 10)
                    features[k][9][:89] = hg.ms_o[j]
                    k += 1
                n += 1

    return features

def score(prediction, labels):
    prediction = torch.sigmoid(prediction).detach().numpy()
    auc = roc_auc_score(labels, prediction)

    prediction[prediction <= 0.5] = 0
    prediction[prediction > 0.5] = 1
    f1 = f1_score(labels, prediction, average='macro')
    
    return auc, f1

def main(args):
    print(args)

    _, train_ms = chooseTrainNode(args['time'])
    train_mask, train_labels = getMaskLabel(train_api, train_ms, args['savep'])
    test_api, test_ms = chooseTestNode(args['time'])
    _, test_labels = getMaskLabel(test_api, test_ms, args['savep'],False)
    print(type(train_mask),type(test_labels))
       
    print('train api:', len(train_api), 'train ms:', len(train_ms))
    #print(sum(test_labels.numpy()==1))
 
    train_hg = HNet(train_ms, train_api, args['words'])
    train_hg.extendDes()
    
    train_features = createFeature(train_hg, train_ms, train_api, train_mask)
    train_mask = get_binary_mask(len(train_api)*len(train_ms), train_mask).bool()

    deal_dataset = TensorDataset(train_features, train_labels[train_mask])
    train_loader = DataLoader(dataset=deal_dataset, batch_size=1024, shuffle=True)

    model = NETA()
    #model = torch.load("BiAtNet2-"+str(args['time'])+".pkl")
    #model.eval()
    loss_fcn = torch.nn.BCEWithLogitsLoss()

    print('test api:', len(test_api), 'test ms:', len(test_ms))
    test_hg = HNet(test_ms, test_api, args['words'])
    test_hg.extendDes()
    test_features = createFeature(test_hg, test_ms, test_api)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    for epoch in range(200):
        for i, data in enumerate(train_loader):
            # 将数据从 train_loader 中读出t_inputs, t_others, t_labels = data
            t_inputs, t_labels = data
            # 将这些数据转换成Variable类型t_inputs, t_others, t_labels = Variable(t_inputs), Variable(t_others), Variable(t_labels)
            t_inputs, t_labels = Variable(t_inputs), Variable(t_labels)
            
            logits = model(t_inputs) #, t_others
            loss = loss_fcn(logits, t_labels)#long())

            optimizer.zero_grad()
            #with torch.autograd.detect_anomaly():   
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            train_auc, acc, rec, f1 = score(logits, t_labels.numpy())
            # train_auc = score(logits, train_labels[train_mask])
            print('GNN Epoch {:d} | Train Loss {:.4f} | AUC {:.4f} | Macro F1 {:.4f}'.format(
                epoch + 1, loss.item(), train_auc, f1))

    logits = model(test_features)
    test_bce = loss_fcn(logits, test_labels)
    test_auc, acc, rec, f1 = score(logits, test_labels.numpy())
    print('GNN Test BCE {:.4f} | AUC {:.4f} | Macro F1 {:.4f}'.format(
        test_bce.item(), test_auc, f1))

#说明0是文本先叠加再doc2vec，1是文本先doc2vec再加并且一起元路径聚合，2文本不聚合了

if __name__ == '__main__':
    # 程序开始时的时间
    time_start=time.time()

    parser = argparse.ArgumentParser()
    #以2006年1月1日为0，每个月时间段+1
    parser.add_argument('-t', '--time', type=int, default=12,
                        help='Train data until time.')
    # 目前测试集默认为2011年4-12月
    parser.add_argument('-w', '--words', type=int, default=150,
                        help='Train text save.')
    parser.add_argument('-p', '--savep', type=float, default=0.02,
                        help='Train data save p.')
    args = parser.parse_args().__dict__

    main(args)
    # 程序结束时系统时间
    time_end=time.time()
    #两者相减
    print('totally cost',time_end-time_start)
