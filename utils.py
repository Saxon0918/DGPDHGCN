import numpy as np
import csv
import torch as t
import random
from numpy import *
import torch

def gaussian_normalization(x: t.Tensor, dim: None or 0 or 1):
    """
    Gaussian normalization.
    :param x: matrix
    :param dim: dimension, global normalization if None, column normalization if 0, else row normalization
    :return: After normalized matrix
    """
    if dim is None:
        mean = t.mean(x)
        std = t.std(x)
        x = t.div(t.sub(x, mean), std)
    else:
        mean = t.mean(x, dim=dim)
        std = t.std(x, dim=dim)
        if dim:
            x = t.div(t.sub(x, mean.view([-1, 1])), std.view([-1, 1]))
        else:
            x = t.div(t.sub(x, mean.view([1, -1])), std.view([1, -1]))
    return x

def create_resultlist(result,testset,Index_PositiveRow,Index_PositiveCol,Index_zeroRow,Index_zeroCol,test_length_p,zero_length,test_f):
    #result_list = zeros((test_length+zero_length, 1))
    result_list = zeros((test_length_p+len(test_f), 1))  # dim=[3076,1]全零
    for i in range(test_length_p):
        result_list[i,0] = result[Index_PositiveRow[testset[i]], Index_PositiveCol[testset[i]]]  # 提取test集正样本的相似度
    for i in range(len(test_f)):
        result_list[i+test_length_p, 0] = result[Index_zeroRow[test_f[i]], Index_zeroCol[test_f[i]]]  # 提取test集负样本的相似度
    # for i in range(zero_length):
    #     result_list[i+test_length, 0] = result[Index_zeroRow[i],Index_zeroCol[i]]
    return result_list  # 前1538个是正样本，后1538负样本

def create_resultmatrix(result,testset,prolist):
    leave_col = prolist[testset]
    result = result[:,leave_col]
    return result

def computer_GIP(intMat,gamadd = 1,gamall = 1):
    num_miRNAs, num_diseasesss = intMat.shape
    sd = []
    sl = []

    for i in range(num_miRNAs):
        sd.append((np.linalg.norm(intMat[i, :])) ** 2)
    gamad = num_miRNAs / sum(sd) * gamadd
    for j in range(num_diseasesss):
        sl.append((np.linalg.norm(intMat[:, j])) ** 2)
    gamal = num_diseasesss / sum(sl) * gamall

    kd = np.ones([num_miRNAs, num_miRNAs],dtype=np.float64)
    for i in range(num_miRNAs):
        for j in range(i+1,num_miRNAs):
            kd[i, j] = np.e ** (-gamad * pow(np.linalg.norm(intMat[i, :] - intMat[j, :]),2))
            kd[j, i] = kd[i, j]

    kt = np.ones([num_diseasesss, num_diseasesss],dtype=np.float64)
    for i in range(num_diseasesss):
        for j in range(i+1,num_diseasesss):
            kt[i, j] = np.e ** (-gamal * pow(np.linalg.norm(intMat[:, i] - intMat[:, j]),2))
            kt[j, i] = kt[i, j]
    return kd,kt

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)

# dis  m   g
def get_edge_index(matrix,i_offset,j_offset):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i+i_offset)
                edge_index[1].append(j+j_offset)
    return t.LongTensor(edge_index)

def f1_score_binary(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: true data,torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: max F1 score and threshold
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    with torch.no_grad():
        thresholds = torch.unique(predict_data)
    size = torch.tensor([thresholds.size()[0], true_data.size()[0]], dtype=torch.int32, device=true_data.device)
    ones = torch.ones([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros([size[0].item(), size[1].item()], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.view([1, -1]).ge(thresholds.view([-1, 1])), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data.view([1, -1])), ones, zeros), dim=1)
    tp = torch.sum(torch.mul(predict_value, true_data.view([1, -1])), dim=1)
    two = torch.tensor(2, dtype=torch.float32, device=true_data.device)
    n = torch.tensor(size[1].item(), dtype=torch.float32, device=true_data.device)
    scores = torch.div(torch.mul(two, tp), torch.add(n, torch.sub(torch.mul(two, tp), tpn)))
    max_f1_score = torch.max(scores)
    threshold = thresholds[torch.argmax(scores)]
    return max_f1_score, threshold


def accuracy_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: acc
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    n = true_data.size()[0]
    ones = torch.tensor([1.0], dtype=torch.float32, device=true_data.device)
    zeros = torch.tensor([0.0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tpn = torch.sum(torch.where(predict_value.eq(true_data), ones, zeros))
    score = torch.div(tpn, n)
    return score


def precision_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    true_neg = torch.sub(ones, true_data)
    tf = torch.sum(torch.mul(true_neg, predict_value))
    score = torch.div(tp, torch.add(tp, tf))
    return score


def recall_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: recall
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    tp = torch.sum(torch.mul(true_data, predict_value))
    predict_neg = torch.sub(ones, predict_value)
    fn = torch.sum(torch.mul(predict_neg, true_data))
    score = torch.div(tp, torch.add(tp, fn))
    return score


def mcc_binary(true_data: torch.Tensor, predict_data: torch.Tensor, threshold: float or torch.Tensor):
    """
    :param true_data: true data, 1D torch Tensor
    :param predict_data: predict data , 1D torch Tensor
    :param threshold: threshold, float or torch Tensor
    :return: precision
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    ones = torch.ones(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    zeros = torch.zeros(true_data.size()[0], dtype=torch.float32, device=true_data.device)
    predict_value = torch.where(predict_data.ge(threshold), ones, zeros)
    predict_neg = torch.sub(ones, predict_value)
    true_neg = torch.sub(ones, true_data)
    tp = torch.sum(torch.mul(true_data, predict_value))
    tn = torch.sum(torch.mul(true_neg, predict_neg))
    fp = torch.sum(torch.mul(true_neg, predict_value))
    fn = torch.sum(torch.mul(true_data, predict_neg))
    delta = torch.tensor(0.00001, dtype=torch.float32, device=true_data.device)
    score = torch.div((tp*tn-fp*fn), torch.add(delta, torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
    return score

def get_metrics(real_score, predict_score):
    import gc
    gc.collect()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    thresholds = sorted_predict_score

    negative_index = np.where(predict_score < thresholds.T)
    positive_index = np.where(predict_score >= thresholds.T)
    predict_score[negative_index] = 0
    predict_score[positive_index] = 1
    TP = predict_score.dot(real_score.T)
    FP = predict_score.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]

# 传的参数顺序(opt, DD, DD_train, Drug, Disease, pp, diseasep, drugp, gg, diseaseg, drugg, zero_index)
# DD_train是1/5清零之后的drug和disease关联数据
# ddrug是drug相似性数据
# ddisease是disease相似性数据
# pp是protein矩阵
# dip是disease和protein
# drp是drug和protein
# gg是gene矩阵
# dig是disease和gene
# drg是drug和gene
def prepare_alldata(opt, DD_all, DD_train, ddrug, ddisease, pp, dip, drp, gg, dig, drg, zero_index ):
    [row_dr, col_di] = np.shape(DD_train)  # col_di=394, row_dr=542
    [row_p, col_di] = np.shape(dip.T)  # row_p = 1512
    [row_g, col_di] = np.shape(dig.T)  # row_g = 11153

    dataset = dict()
    dataset['drdi_p'] = t.FloatTensor(DD_train)  # dim=[542, 394]
    dataset['drdi_true'] = t.FloatTensor(DD_train)  # dim=[542, 394]

    one_index = []  # 正样本行列号位置，37336个
    # 读取正样本的坐标位置
    for i in range(dataset['drdi_p'].size(0)):
        for j in range(dataset['drdi_p'].size(1)):
            if dataset['drdi_p'][i][j] >= 1:
                # 读取正样本坐标
                one_index.append([i, j])

    random.shuffle(one_index)  # 打乱位置
    random.shuffle(zero_index)  # 打乱位置
    zero_tensor = t.LongTensor(zero_index)  # 转成tensor格式dim=[157546,2], 前面是drug,后面disease
    one_tensor = t.LongTensor(one_index)  # 转成tensor格式dim=[37336,2], 前面是drug,后面disease
    dataset['drdi'] = dict()  # disease*drug
    # md,train中放正样本和负样本
    dataset['drdi']['train'] = [one_tensor, zero_tensor]  # dataset['md']['train'][0]是1,dataset['md']['train'][1]是0

    # --------------------disease-drug网络----------------------
    # disease相似性数据
    didi_matrix = t.FloatTensor(ddisease)  # disease相似度转tensor
    # 这里是返回疾病-疾病关联矩阵中数值不是0的坐标位置,形如[[1,2],[1,4]]表示第一行第一列、第二行第四列不等于0
    didi_edge_index = get_edge_index(didi_matrix, 0, 0)  # 29294条
    # dd中的数据就是所有的相似性值和值大于0的坐标位置
    dataset['didi'] = {'data': didi_matrix, 'edge_index': didi_edge_index}

    # drug相似性数据
    drdr_matrix = t.FloatTensor(ddrug)
    drdr_edge_index = get_edge_index(drdr_matrix, 0, 0)  # 12290条
    dataset['drdr'] = {'data': drdr_matrix, 'edge_index': drdr_edge_index}

    # disease和drug边
    didr_matrix = t.FloatTensor(DD_train.T)
    didr_edge_index = get_edge_index(didr_matrix, 0, col_di)  # 顺序是disease[0-394],miRNA[395-937]
    dataset['didr'] = {'data': didr_matrix, 'edge_index': didr_edge_index}

    # --------------------disease-protein-drug网络----------------------
    # protein相似性数据
    pp_matrix = t.FloatTensor(pp)
    pp_edge_index = get_edge_index(pp_matrix, 0, 0)  # pp是8284
    dataset['pp'] = {'data': pp_matrix, 'edge_index': pp_edge_index}

    # 编号顺序为disease[0-394],drug[395-937],protein[938-2450]
    # protein和disease边
    pdi_matrix = t.FloatTensor(dip.T)
    pdi_edge_index = get_edge_index(pdi_matrix, col_di + row_dr, 0)  # 传入的col_di+row_dr是为了编号
    dip_edge1 = get_edge_index(t.FloatTensor(dip), 0, col_di)  # 顺序是disease[0-394], protein[395-1907]
    dataset['pdi'] = {'data': pdi_matrix, 'edge_index': pdi_edge_index, 'dip_edge_gcn': dip_edge1}

    # protein和drug边
    pdr_matrix = t.FloatTensor(drp.T)
    pdr_edge_index = get_edge_index(pdr_matrix, col_di + row_dr, col_di)
    drp_edge1 = get_edge_index(t.FloatTensor(drp), 0, row_dr)  # 顺序是drug[0-542], protein[543-2055]
    dataset['pdr'] = {'data': pdr_matrix, 'edge_index': pdr_edge_index, 'drp_edge_gcn': drp_edge1}

    # --------------------disease-gene-drug网络----------------------
    # protein相似性数据
    gg_matrix = t.FloatTensor(gg)
    gg_edge_index = get_edge_index(gg_matrix, 0, 0)  # gg是1023439条
    dataset['gg'] = {'data': gg_matrix, 'edge_index': gg_edge_index}

    # 编号顺序为disease[0-394],drug[395-937],gene[938-12091]
    # gene和disease边
    gdi_matrix = t.FloatTensor(dig.T)
    gdi_edge_index = get_edge_index(gdi_matrix, col_di + row_dr, 0)  # 传入的col_di+row_dr是为了编号
    dig_edge1 = get_edge_index(t.FloatTensor(dig), 0, col_di)  # 顺序是disease[0-394],gene[395-11548]
    dataset['gdi'] = {'data': gdi_matrix, 'edge_index': gdi_edge_index, 'dig_edge_gcn': dig_edge1}

    # gene和drug边
    gdr_matrix = t.FloatTensor(drg.T)
    gdr_edge_index = get_edge_index(gdr_matrix, col_di + row_dr, col_di)
    drg_edge1 = get_edge_index(t.FloatTensor(drg), 0, row_dr)  # 顺序是drug[0-542],gene[543-11695]
    dataset['gdr'] = {'data': gdr_matrix, 'edge_index': gdr_edge_index, 'drg_edge_gcn': drg_edge1}

    # --------------------rgcn edge----------------------
    di1di3_edge = t.FloatTensor(ddisease)  # ddisease是disease相似性数据
    di1di3_edge_rgcn = get_edge_index(di1di3_edge, 0, col_di + col_di)

    di1dr1_edge = t.FloatTensor(DD_train.T)
    di1dr1_edge_rgcn = get_edge_index(di1dr1_edge, 0, col_di + col_di + col_di)

    di1dr2_edge = t.FloatTensor(DD_train.T)
    di1dr2_edge_rgcn = get_edge_index(di1dr2_edge, 0, col_di + col_di + col_di + row_dr)

    di2dr1_edge = t.FloatTensor(DD_train.T)
    di2dr1_edge_rgcn = get_edge_index(di2dr1_edge, col_di, col_di + col_di + col_di)

    dr1dr3_edge = t.FloatTensor(ddrug)
    dr1dr3_edge_rgcn = get_edge_index(dr1dr3_edge, col_di + col_di + col_di, col_di + col_di + col_di + row_dr + row_dr)

    p3di1_edge = t.FloatTensor(dip.T)
    p3di1_edge_rgcn = get_edge_index(p3di1_edge, col_di + col_di + col_di + row_dr + row_dr + row_dr + row_p, 0)

    p2dr1_edge = t.FloatTensor(drp.T)
    p2dr1_edge_rgcn = get_edge_index(p2dr1_edge, col_di + col_di + col_di + row_dr + row_dr + row_dr, col_di + col_di + col_di)

    g3di1_edge = t.FloatTensor(dig.T)
    g3di1_edge_rgcn = get_edge_index(g3di1_edge, col_di + col_di + col_di + row_dr + row_dr + row_dr + row_p + row_p + row_g, 0)

    g2dr1_edge = t.FloatTensor(drg.T)
    g2dr1_edge_rgcn = get_edge_index(g2dr1_edge, col_di + col_di + col_di + row_dr + row_dr + row_dr + row_p + row_p, col_di + col_di + col_di)

    dataset['rgcn_edge'] = {'di1di3_edge_rgcn': di1di3_edge_rgcn, 'di1dr1_edge_rgcn': di1dr1_edge_rgcn,
                         'di1dr2_edge_rgcn': di1dr2_edge_rgcn, 'di2dr1_edge_rgcn': di2dr1_edge_rgcn,
                         'dr1dr3_edge_rgcn': dr1dr3_edge_rgcn, 'p3di1_edge_rgcn': p3di1_edge_rgcn,
                         'p2dr1_edge_rgcn': p2dr1_edge_rgcn, 'g3di1_edge_rgcn': g3di1_edge_rgcn,
                         'g2dr1_edge_rgcn': g2dr1_edge_rgcn}

    return dataset