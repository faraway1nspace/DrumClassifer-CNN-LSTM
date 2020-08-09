import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from sklearn.utils.class_weight import compute_class_weight

class DataLoader:
    """ for model training, loads data during gradient descent"""    
    def __init__(self,batch_size,permute=False,sort=True, maxseqlen = None, minseqlen=None,pad=None, do_random_truncation_above=None, do_random_pad = None,nsamp=None):
        self.bs = batch_size
        self.permute = permute
        self.sort = sort
        self.num_iterations = 10**9
        self.order = None
        if maxseqlen is None:
            maxseqlen = 300
        self.maxseqlen = maxseqlen
        if minseqlen is None:
            minseqlen = 10
        self.minseqlen = minseqlen
        if pad is None:
            pad = 0
        self.pad_token = pad
        # whether to augment the data by truncating it above X
        if do_random_truncation_above is None:
            do_random_truncation_above = 10**9
        self.do_random_truncation_above = do_random_truncation_above
        if do_random_pad is None:
            do_random_pad = 0
        else:
            self.do_random_pad = do_random_pad
        self.nsamp =nsamp
    
    def data_iterator(self,Y,X):
        size = len(Y)
        # calculate number of iterations
        self.num_iterations = round(size/self.bs)
        o = np.arange(size)
        #if self.permute:
        #    o = np.random.permutation(o)
        if self.nsamp is not None:
            np.random.RandomState(seed=(hash(time.time()) % 100000))
            o = np.random.choice(o,self.nsamp)
            Y = [Y[i] for i in o]
            X = [X[i] for i in o]
            size = len(Y)
            self.num_iterations = round(size/self.bs)
        if self.sort:
            # sort
            if self.order is None:
                self.order = np.argsort([x.shape[1] for x in X])
            o = self.order
        
        for i in range(self.num_iterations):
            # batch order
            if i == (self.num_iterations-1):
                batch_order = self.order[(i*self.bs):]
            else:
                batch_order = self.order[(i*self.bs):((i+1)*self.bs)]
            
            batch_y = np.array([Y[b] for b in batch_order])
            batch_x1 = [X[b] for b in batch_order]
            # maximum length
            maxlength_natural = max([x.shape[1] for x in batch_x1])
            if maxlength_natural>=self.do_random_truncation_above:
                maxlen = max(self.minseqlen,random.randint(int(self.maxseqlen*0.66), self.maxseqlen))
            else:
                maxlen = max(self.minseqlen,min(self.maxseqlen, maxlength_natural))
            # feature length
            xdim = batch_x1[0].shape[0]
            # container
            batchxseq = self.pad_token*np.ones((len(batch_order), maxlen, xdim))
            for j in range(len(batch_order)):
                curlen = min(batch_x1[j].shape[1], maxlen)
                batchxseq[j][:curlen] = batch_x1[j][:,:curlen].T
            
            if self.do_random_pad>0:
                random_pad = random.randint(0,self.do_random_pad)
                if random_pad>0:
                    pad = self.pad_token*np.ones((batchxseq.shape[0], random_pad, xdim))
                    batchxseq = np.concatenate((pad, batchxseq),axis=1)
            batch_y, batchxseq = torch.from_numpy(batch_y), torch.from_numpy(batchxseq)
            batch_y, batchxseq = Variable(batch_y), Variable(batchxseq)
            yield batch_y, batchxseq.float()
    
    def get_val(self, Y,X):
        size = len(Y)
        xdim = X[0].shape[0]
        # calculate number of iterations
        num_iterations = round(size/self.bs)
        o = np.arange(size)
        for i in range(num_iterations):
            if i == (num_iterations-1):
                batch_order = o[(i*self.bs):]
            else:
                batch_order = o[(i*self.bs):((i+1)*self.bs)]
            
            batch_y = np.array([Y[b] for b in batch_order])
            batch_x1 = [X[b] for b in batch_order]
            # maximum length
            maxlength_natural = max([x.shape[1] for x in batch_x1])
            maxlen = max(self.minseqlen,min(self.maxseqlen, maxlength_natural))
            # feature length
            # container
            batchxseq = self.pad_token*np.ones((len(batch_order), maxlen, xdim))
            for j in range(len(batch_order)):
                curlen = min(batch_x1[j].shape[1], maxlen)
                batchxseq[j][:curlen] = batch_x1[j][:,:curlen].T
            
            batch_y, batchxseq = torch.from_numpy(np.array(batch_y)), torch.from_numpy(batchxseq)
            batch_y, batchxseq = Variable(batch_y), Variable(batchxseq)
            yield batch_y, batchxseq.float()
    
    def predict(self,mod,Y,X):
        size = len(Y)
        valiterator = self.get_val(Y,X)
        for i in range(round(size/self.bs)):
            by,bx = next(valiterator)
            pred = mod(bx)
            if i==0:
                valpred = pred.detach().numpy()
            else:
                valpred = np.concatenate((valpred,pred.detach().numpy()),axis=0)
            del pred,by,bx
        return valpred

def make_pos_encodings(bs, seqlen, band_dim):
    """ simple positional encoding; x & y coordinate"""
    # y direction (bands)
    ypos = (np.arange(band_dim)-63.5)/36.94928957368463
    # x direction (seqlen)
    xpos = np.log(np.arange(seqlen)+1)/4.38202663 -0.33
    ypos_extend = np.tile(ypos,(bs,len(xpos),1))
    # unsqueeze
    ypos_extend = ypos_extend.reshape(ypos_extend.shape[0],1, ypos_extend.shape[1], ypos_extend.shape[2])
    xpos_extend = np.tile(xpos,(bs,len(ypos),1))
    xpos_extend = xpos_extend.transpose(0,-1,-2)
    # unsqueeze
    xpos_extend = xpos_extend.reshape(xpos_extend.shape[0],1,xpos_extend.shape[1], xpos_extend.shape[2])
    posenc = np.concatenate((xpos_extend, ypos_extend),axis=1)
    posenc = Variable(torch.from_numpy(posenc))
    return posenc

class HyperParams:
    """ organizes hyperparameters-in-dictionary into a class"""
    def __init__(self, params_dict):
        for k,v in params_dict.items():
            setattr(self, k,v)

class Classifier(nn.Module):   
    def __init__(self,params):
        super(Classifier, self).__init__()
        
        # convolution
        self.convl11 = nn.Sequential(nn.Dropout(params.dropout_11),
                        nn.Conv2d(params.n_channels, params.num_filters_11, kernel_size=params.kernel_size_11, stride=params.stride_11),
                        nn.BatchNorm2d(params.num_filters_11),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=params.maxpool_size11, stride=params.maxpool_stride11))
        
        # conv2
        self.convl12 = nn.Sequential(nn.Dropout(params.dropout_12),
                        nn.Conv2d(params.num_filters_11, params.num_filters_12, kernel_size=params.kernel_size_12, stride=params.stride_12),
                        nn.BatchNorm2d(params.num_filters_12),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=params.maxpool_size12, stride=params.maxpool_stride12))
        
        # conv3
        self.convl13 = nn.Sequential(nn.Dropout(params.dropout_13),
                        nn.Conv2d(params.num_filters_12, params.num_filters_13, kernel_size=params.kernel_size_13, stride=params.stride_13),
                        nn.BatchNorm2d(params.num_filters_13),
                        nn.ReLU())
        
        # LSTM
        self.lstm = nn.LSTM(input_size=params.num_filters_13, hidden_size=params.lstm_dim,batch_first=True, bidirectional = True)
        
        # final 2 layer MLP
        self.fc1_dropout = nn.Dropout(params.fc1_dropout)
        self.fc1 = nn.Linear(params.lstm_dim*2, params.fc1_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(params.fc1_dim, params.outdim)
        
    def forward(self,bx):
        bs, seqlen, band_dim = bx.shape
        bx = bx.unsqueeze(1)
        # 
        x_pos_enc = self.make_posenc(bs, seqlen=seqlen, band_dim=band_dim)
        # concatenate the positional encodings with the mel
        bx_concat = torch.cat((x_pos_enc, bx),axis=1)
        o1 = self.convl11(bx_concat)
        o2 = self.convl12(o1) # torch.Size([16, 128, 8, 7])
        o3 = self.convl13(o2) # torch.Size([16, 128, 7, 5])
        #o4 = self.convl14(o3) # torch.Size([16, 128, 6, 3])
        #o5 = self.convl15(o4) # torch.Size([16, 128, 5, 1])
        o3 = o3.squeeze(-1)
        o = o3.transpose(-2,-1) # torch.Size([16, 5, 128])
        _,(hs,_) = self.lstm(o) # torch.Size([2, 16, 128])
        # concatenate both directions of LSTM
        s = torch.cat((hs[0],hs[1]),axis=1)
        # final LSTM
        out = self.fc2(self.relu(self.fc1(s)))
        return out
    
    def make_posenc(self, bs, seqlen, band_dim):
        posenc = make_pos_encodings(bs, seqlen, band_dim)
        return posenc.float()
