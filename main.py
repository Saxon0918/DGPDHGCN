import os
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch import nn, optim
from model import Model
from dataprocess import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from numpy import *
from utils import *
import copy
import torch
import pandas as pd


class Config(object):
    def __init__(self):
        self.epoch = 2000
        self.beta = 0.05


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, target, target_pdi, target_pdr, target_gdi, target_gdr, input, input_pdi, input_pdr, input_gdi, input_gdr):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        loss_pdi = loss(input_pdi, target_pdi)
        loss_pdr = loss(input_pdr, target_pdr)
        loss_gdi = loss(input_gdi, target_gdi)
        loss_gdr = loss(input_gdr, target_gdr)
        return loss_sum.sum() + opt.beta * (loss_pdi.sum() + loss_pdr.sum() + loss_gdi.sum() + loss_gdr.sum())


class Sizes(object):
    def __init__(self, dataset):
        self.drdip = dataset['didi']['data'].size(0) + dataset['drdr']['data'].size(0) + dataset['pp']['data'].size(0)
        self.drdig = dataset['didi']['data'].size(0) + dataset['drdr']['data'].size(0) + dataset['gg']['data'].size(0)
        self.di = dataset['didi']['data'].size(0)
        self.dr = dataset['drdr']['data'].size(0)
        self.p = dataset['pp']['data'].size(0)
        self.g = dataset['gg']['data'].size(0)
        self.k1 = 256
        self.k2 = 128
        self.k3 = 64


def train(model, train_data, optimizer, opt):
    model.train()
    regression_crit = Myloss()

    def train_epoch():
        model.zero_grad()
        score, score_pdi, score_pdr, score_gdi, score_gdr = model(train_data)
        loss = regression_crit(train_data[7].cuda(), train_data[3]['data'].cuda(),
                               train_data[4]['data'].cuda(), train_data[10]['data'].cuda(),
                               train_data[11]['data'].cuda(), score, score_pdi, score_pdr, score_gdi, score_gdr)
        loss.backward()
        optimizer.step()
        return loss, score

    for epoch in range(1, opt.epoch + 1):
        train_reg_loss, predict = train_epoch()
        if epoch % 100 == 0:
            print("epoch {}, training loss {}".format(epoch, train_reg_loss.float()))
        elif epoch == 1:
            print("epoch {}, training loss {}".format(epoch, train_reg_loss.float()))
    return predict


if __name__ == "__main__":
    opt = Config()
    # --------------------read csv----------------------
    Disease = np.genfromtxt('data/mat_disease_disease.csv', delimiter=',')  # dim=[394,394]
    Drug = np.genfromtxt('data/mat_drug_drug.csv', delimiter=',')  # dim=[542,542]
    DD = np.genfromtxt('data/mat_disease_drug.csv', delimiter=',')
    DD = DD.T  # dim=[542,394]

    gg = np.genfromtxt('data/mat_gene_gene.csv', delimiter=',')  # dim=[11153, 11153]
    diseaseg = np.genfromtxt('data/mat_disease_gene.csv', delimiter=',')  # dim=[394,11153]
    drugg = np.genfromtxt('data/mat_drug_gene.csv', delimiter=',')  # dim=[542,11153]

    pp = np.genfromtxt('data/mat_protein_protein.csv', delimiter=',')  # dim=[1512, 1512]
    diseasep = np.genfromtxt('data/mat_disease_protein.csv', delimiter=',')  # dim=[394,1512]
    drugp = np.genfromtxt('data/mat_drug_protein.csv', delimiter=',')  # dim=[542,1512]

    # --------------------cross validation setting----------------------
    [row, col] = np.shape(DD)
    indexn = np.argwhere(DD == 0)
    Index_zeroRow = indexn[:, 0]
    Index_zeroCol = indexn[:, 1]
    indexp = np.argwhere(DD == 1)
    Index_PositiveRow = indexp[:, 0]
    Index_PositiveCol = indexp[:, 1]
    totalassociation = np.size(Index_PositiveRow)
    cv_num = 5
    fold = int(totalassociation / cv_num)
    zero_length = np.size(Index_zeroRow)
    n = 5

    # --------------------metrices----------------------
    var_Auc = []
    Auc_list = []
    var_Aupr = []
    Aupr_list = []
    var_F1_score = []
    F1_score_list = []
    var_precision = []
    precision_list = []
    var_recall = []
    recall_list = []

    # --------------------model training and testing----------------------
    for time in range(1, n + 1):
        Auc_per = []
        Aupr_per = []
        F1_score_per = []
        precision_per = []
        recall_per = []
        p = np.random.permutation(totalassociation)
        for f in range(1, cv_num + 1):
            print("cross_validation:", '%01d' % (f))
            if f == cv_num:
                testset = p[((f - 1) * fold): totalassociation + 1]
            else:
                testset = p[((f - 1) * fold): f * fold]
            # positive and negative testing data
            all_f = np.random.permutation(np.size(Index_zeroRow))
            test_p = list(testset)
            test_f = all_f[0: len(test_p)]
            difference_set_f = list(set(all_f).difference(set(test_f)))
            train_p = list(set(p).difference(set(testset)))
            train_f = difference_set_f

            DD_tmp = copy.deepcopy(DD)
            zero_index = []
            for ii in range(len(train_f)):
                zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])
            true_list = zeros((len(test_p) + len(test_f), 1))
            for ii in range(len(test_p)):
                DD_tmp[Index_PositiveRow[testset[ii]], Index_PositiveCol[
                    testset[ii]]] = 0
                true_list[ii, 0] = 1
            DD_train = copy.deepcopy(DD_tmp)  # drug-disease training data

            # --------------------model training----------------------
            # dataset = process_data(opt, DD_train, Drug, Disease, pp, diseasep, drugp, pp, diseasep, drugp, zero_index)  # 测试使用
            dataset = process_data(opt, DD_train, Drug, Disease, pp, diseasep, drugp, gg, diseaseg, drugg, zero_index)
            sizes = Sizes(dataset)
            train_data = Dataset(opt, dataset)
            model = Model(sizes)
            model = model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            gc.collect()
            t.cuda.empty_cache()
            predict = train(model, train_data[f], optimizer, opt)

            # --------------------model testing----------------------
            predict = predict.data.cpu().numpy()
            # pred_tocsv = pd.DataFrame(predict)
            # pred_tocsv.to_csv('data/case_study.csv', index=False)
            test_predict = create_resultlist(predict, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,
                                             Index_zeroCol, len(test_p), zero_length, test_f)
            label = true_list
            test_auc = roc_auc_score(label, test_predict)
            Auc_per.append(test_auc)
            print("Auc: " + str(test_auc))
            pr, re, thresholds = precision_recall_curve(label, test_predict)
            test_aupr = auc(re, pr)
            Aupr_per.append(test_aupr)
            print("Aupr", test_aupr)
            test_F1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),
                                                      torch.from_numpy(test_predict).float())
            F1_score_per.append(test_F1_score)
            print("F1_score:", test_F1_score)
            print("threshold:", threshold)
            test_precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),
                                         threshold)
            precision_per.append(test_precision)
            print("precision:", test_precision)
            test_recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
            recall_per.append(test_recall)
            print("recall:", test_recall)

            var_Auc.append(test_auc)
            var_Aupr.append(test_aupr)
            var_F1_score.append(test_F1_score)
            var_precision.append(test_precision)
            var_recall.append(test_recall)

        Auc_list.append(np.mean(Auc_per))
        print("average Auc: " + str(Auc_list))
        Aupr_list.append(np.mean(Aupr_per))
        print("average Aupr: " + str(Aupr_list))
        F1_score_list.append(np.mean(F1_score_per))
        print("average F1_score: " + str(F1_score_list))
        precision_list.append(np.mean(precision_per))
        print("average precision: " + str(precision_list))
        recall_list.append(np.mean(recall_per))
        print("average recall: " + str(recall_list))

    vAuc = np.var(var_Auc)
    vAupr = np.var(var_Aupr)
    vF1_score = np.var(var_F1_score)
    vprecision = np.var(var_precision)
    vrecall = np.var(var_recall)
    print("whole Auc = %f±%f\n" % (float(np.mean(Auc_list)), vAuc))
    print("whole Aupr = %f±%f\n" % (float(np.mean(Aupr_list)), vAupr))
    print("whole F1_score = %f±%f\n" % (float(np.mean(F1_score_list)), vF1_score))
    print("whole precision = %f±%f\n" % (float(np.mean(precision_list)), vprecision))
    print("whole recall = %f±%f\n" % (float(np.mean(recall_list)), vrecall))
