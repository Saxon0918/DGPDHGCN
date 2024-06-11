import os
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch import nn, optim
from model import Model
from trainData import Dataset
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
        # --------------------普通loss----------------------
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


def train(model, train_data, optimizer, opt, f):
    model.train()
    regression_crit = Myloss()
    one_index = train_data[5][0].cuda().t().tolist()  # 正样本行列号
    zero_index = train_data[5][1].cuda().t().tolist()  # train集drug-disease负样本行列号

    def train_epoch():
        model.zero_grad()
        score, score_pdi, score_pdr, score_gdi, score_gdr = model(train_data)  # [542,394][1512,394][1512,542][11153,394][11153,542]

        loss = regression_crit(train_data[7].cuda(), train_data[3]['data'].cuda(),
                               train_data[4]['data'].cuda(), train_data[10]['data'].cuda(),
                               train_data[11]['data'].cuda(), score, score_pdi, score_pdr, score_gdi, score_gdr)  # 计算loss
        loss.backward()
        optimizer.step()
        return loss, score

    for epoch in range(1, opt.epoch + 1):
        train_reg_loss, predict = train_epoch()
    return predict  # dim=[542,394]


if __name__ == "__main__":
    opt = Config()
    # --------------------读取文件----------------------
    Disease = np.genfromtxt('data/original/mat_disease_disease.csv', delimiter=',')  # dim=[394,394]
    Drug = np.genfromtxt('data/original/mat_drug_drug.csv', delimiter=',')  # dim=[542,542]
    DD = np.genfromtxt('data/original/mat_disease_drug.csv', delimiter=',')
    DD = DD.T  # dim=[542,394]

    gg = np.genfromtxt('data/original/mat_gene_gene.csv', delimiter=',')  # dim=[11153, 11153]
    diseaseg = np.genfromtxt('data/original/mat_disease_gene.csv', delimiter=',')  # dim=[394,11153]
    drugg = np.genfromtxt('data/original/mat_drug_gene.csv', delimiter=',')  # dim=[542,11153]

    pp = np.genfromtxt('data/original/mat_protein_protein.csv', delimiter=',')  # dim=[1512, 1512]
    diseasep = np.genfromtxt('data/original/mat_disease_protein.csv', delimiter=',')  # dim=[394,1512]
    drugp = np.genfromtxt('data/original/mat_drug_protein.csv', delimiter=',')  # dim=[542,1512]

    # --------------------交叉验证配置----------------------
    [row, col] = np.shape(DD)  # row=542, col=394
    indexn = np.argwhere(DD == 0)  # 取出不相关的drug和disease的行列号,例如[[0,1]] dim=[166879,2]
    Index_zeroRow = indexn[:, 0]  # 行号dim=[166879]
    Index_zeroCol = indexn[:, 1]  # 列号dim=[166879]
    indexp = np.argwhere(DD == 1)  # 取出相关的drug和disease的行列号,dim=[466669,2]
    Index_PositiveRow = indexp[:, 0]  # 行号dim=[466669]
    Index_PositiveCol = indexp[:, 1]  # 列号dim=[466669]
    totalassociation = np.size(Index_PositiveRow)  # drug和disease相关数量=46669
    cv_num = 5  # 分train和test
    fold = int(totalassociation / cv_num)  # 5-fold， fold=9333
    zero_length = np.size(Index_zeroRow)  # 不相关的数量=166879
    n = 5  # 训练2轮

    # --------------------建立指标数组----------------------
    varauc = []
    AAuc_list1 = []
    varf1_score = []
    f1_score_list1 = []
    varprecision = []
    precision_list1 = []
    varacc = []
    acc_list1 = []
    varrecall = []
    recall_list1 = []
    varaupr = []
    aupr_list1 = []

    # --------------------模型训练----------------------
    for time in range(1, n + 1):
        Auc_per = []
        f1_score_per = []
        acc_per = []
        precision_per = []
        recall_per = []
        aupr_per = []
        p = np.random.permutation(totalassociation)  # 随机重排打乱dim=[46669], 0-46668
        for f in range(1, cv_num + 1):
            # --------------------划分训练集和测试集----------------------
            print("cross_validation:", '%01d' % (f))
            if f == cv_num:  # 判断是否是最后一个fold,防止无法均分的情况
                testset = p[((f - 1) * fold): totalassociation + 1]
            else:
                testset = p[((f - 1) * fold): f * fold]  # 从p中取9333个数据

            # test pos and neg
            all_f = np.random.permutation(np.size(Index_zeroRow))  # 随机重排打乱dim=[166879], 0-166878
            test_p = list(testset)  # 转成list, 9333个数据正样本test集
            test_f = all_f[0:len(test_p)]  # 从166879中取出前9333个负样本，当负样本test
            difference_set_f = list(set(all_f).difference(set(test_f)))  # 取出剩下的157546个样本id
            train_p = list(set(p).difference(set(testset)))  # 取出剩下的37336个正样本id当train集
            train_f = difference_set_f  # 取出剩下的157546个负样本id当train集

            X = copy.deepcopy(DD)  # drug和disease的邻接矩阵
            Xn = copy.deepcopy(X)  # drug和disease的邻接矩阵
            # test_length = len(test_p)
            zero_index = []  # 构造负样本的train行列号157546个, 前面是drug,后面disease
            for ii in range(len(train_f)):
                zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])
            true_list = zeros((len(test_p) + len(test_f), 1))  # 构造test全零集合dim=[18666,1]
            for ii in range(len(test_p)):
                Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[
                    testset[ii]]] = 0  # 将test集中的正样本行列号在邻接矩阵的值设为0，可能是防止test集对train集的影响
                true_list[ii, 0] = 1  # test集合的前面9333个数据为1，后面9333为0
            DD_train = copy.deepcopy(Xn)  # 除去测试集正样本影响的drug和disease邻接矩阵

            # --------------------disease-protein-gene-drug模型训练----------------------
            # dataset = prepare_alldata(opt, DD, DD_train, Drug, Disease, pp, diseasep, drugp, pp, diseasep, drugp, zero_index)  # 测试使用
            dataset = prepare_alldata(opt, DD, DD_train, Drug, Disease, pp, diseasep, drugp, gg, diseaseg, drugg, zero_index)  # protein and gene
            sizes = Sizes(dataset)
            train_data = Dataset(opt, dataset)
            model = Model(sizes)
            model = model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            gc.collect()
            t.cuda.empty_cache()
            predict = train(model, train_data[f], optimizer, opt, f)

            # --------------------验证测试数据----------------------
            predict = predict.data.cpu().numpy()
            # 使用pandas将ndarray转换为DataFrame
            pred_tocsv = pd.DataFrame(predict)
            # 将DataFrame存储到CSV文件中
            pred_tocsv.to_csv('data/case_study.csv', index=False)

            test_predict = create_resultlist(predict, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,
                                             Index_zeroCol, len(test_p), zero_length, test_f)
            label = true_list
            test_auc = roc_auc_score(label, test_predict)
            Auc_per.append(test_auc)
            print("//////////每一次auc: " + str(test_auc))
            varauc.append(test_auc)

            # # 新的测评方式
            # result = get_metrics(label, test_predict)
            # print(result)

            ####
            max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),
                                                      torch.from_numpy(test_predict).float())
            f1_score_per.append(max_f1_score)
            print("//////////max_f1_score:", max_f1_score)
            print("//////////threshold:", threshold)
            acc = accuracy_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
            acc_per.append(acc)
            print("//////////acc:", acc)
            precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),
                                         threshold)
            precision_per.append(precision)
            print("//////////precision:", precision)
            recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
            recall_per.append(recall)
            print("//////////recall:", recall)
            # mcc_score = mcc_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),threshold)
            # print("mcc_score:", mcc_score)
            pr, re, thresholds = precision_recall_curve(label, test_predict)
            aupr = auc(re, pr)
            aupr_per.append(aupr)
            print("//////////aupr", aupr)

            varf1_score.append(max_f1_score)
            varacc.append(acc)
            varprecision.append(precision)
            varrecall.append(recall)
            varaupr.append(aupr)

        AAuc_list1.append(np.mean(Auc_per))
        f1_score_list1.append(np.mean(f1_score_per))
        acc_list1.append(np.mean(acc_per))
        precision_list1.append(np.mean(precision_per))
        recall_list1.append(np.mean(recall_per))
        aupr_list1.append(np.mean(aupr_per))

        print("//////////Aucaverage: " + str(AAuc_list1))
        print("//////////f1_scoreaverage: " + str(f1_score_list1))
        print("//////////accaverage: " + str(acc_list1))
        print("//////////precisionaverage: " + str(precision_list1))
        print("//////////recallaverage: " + str(recall_list1))
        print("//////////aupraverage: " + str(aupr_list1))

    vauc = np.var(varauc)
    vf1_score = np.var(varf1_score)
    vacc = np.var(varacc)
    vprecision = np.var(varprecision)
    vrecall = np.var(varrecall)
    vaupr = np.var(varaupr)

    print("sumauc = %f±%f\n" % (float(np.mean(AAuc_list1)), vauc))
    print("sumf1_score = %f±%f\n" % (float(np.mean(f1_score_list1)), vf1_score))
    print("sumacc = %f±%f\n" % (float(np.mean(acc_list1)), vacc))
    print("sumprecision = %f±%f\n" % (float(np.mean(precision_list1)), vprecision))
    print("sumrecall = %f±%f\n" % (float(np.mean(recall_list1)), vrecall))
    print("sumaupr = %f±%f\n" % (float(np.mean(aupr_list1)), vaupr))
