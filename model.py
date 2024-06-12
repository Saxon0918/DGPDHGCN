from utils import *
import torch as t
from torch import nn
from torch_geometric.nn import conv
from torch_geometric.nn import dense, norm
import copy


class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()
        self.k1 = sizes.k1    # num=256
        self.k2 = sizes.k2    # num=128
        self.k3 = sizes.k3  # num=64
        self.drdip = sizes.drdip  # num=2448
        self.drdig = sizes.drdig  # num=12089
        self.di = sizes.di  # num=394
        self.dr = sizes.dr  # num=542
        self.p = sizes.p  # num=1512
        self.g = sizes.g  # num=11153

        # --------------------FC----------------------
        self.linear_di = nn.Linear(self.di, self.k1)
        self.linear_dr = nn.Linear(self.dr, self.k1)
        self.linear_p = nn.Linear(self.p, self.k1)
        self.linear_g = nn.Linear(self.g, self.k1)

        # --------------------disease-drug----------------------
        self.gcn_drdr = conv.GCNConv(self.dr, self.k1)
        self.gcn_didi = conv.GCNConv(self.di, self.k1)
        self.gcn_drdi = conv.GCNConv(self.k1, self.k1)

        # --------------------disease-protein-drug----------------------
        self.gcn_dip = conv.GCNConv(self.k1, self.k1)
        self.gcn_drp = conv.GCNConv(self.k1, self.k1)

        # --------------------disease-gene-drug----------------------
        self.gcn_dig = conv.GCNConv(self.k1, self.k1)
        self.gcn_drg = conv.GCNConv(self.k1, self.k1)

        # --------------------disease-gene-protein-drug----------------------
        self.rgcn_didrpg = conv.RGCNConv(self.k1, self.k1, 18)

        # --------------------MLP----------------------
        self.linear_mlp_di1 = nn.Linear(self.k1, self.k1)
        self.linear_mlp_di2 = nn.Linear(self.k1, self.k2)
        self.linear_mlp_di3 = nn.Linear(self.k2, self.k3)

        self.linear_mlp_dr1 = nn.Linear(self.k1, self.k1)
        self.linear_mlp_dr2 = nn.Linear(self.k1, self.k2)
        self.linear_mlp_dr3 = nn.Linear(self.k2, self.k3)

        self.linear_mlp_p1 = nn.Linear(self.k1, self.k1)
        self.linear_mlp_p2 = nn.Linear(self.k1, self.k2)
        self.linear_mlp_p3 = nn.Linear(self.k2, self.k3)

        self.linear_mlp_g1 = nn.Linear(self.k1, self.k1)
        self.linear_mlp_g2 = nn.Linear(self.k1, self.k2)
        self.linear_mlp_g3 = nn.Linear(self.k2, self.k3)

    def forward(self, input):
        t.manual_seed(2)
        di_dim = input[0]['data'].size(0)  # dim=394
        dr_dim = input[1]['data'].size(0)  # dim=542
        p_dim = input[2]['data'].size(0)  # dim=1512
        g_dim = input[9]['data'].size(0)  # dim=11153

        didr_edge = input[8]['edge_index']
        drdi_edge = t.cat((input[8]['edge_index'][1, :].reshape(1, -1), input[8]['edge_index'][0, :].reshape(1, -1)))

        # --------------------similarity GCN----------------------
        di1 = t.relu(self.gcn_didi(input[0]['data'].cuda(), input[0]['edge_index'].cuda()))  # dim=[394,256]
        dr1 = t.relu(self.gcn_drdr(input[1]['data'].cuda(), input[1]['edge_index'].cuda()))  # dim=[542,256]

        # --------------------FC layer----------------------
        di2 = t.relu(self.linear_di(input[0]['data'].cuda()))  # dim=[394,256]
        dr2 = t.relu(self.linear_dr(input[1]['data'].cuda()))  # dim=[542,256]
        p1 = t.relu(self.linear_p(input[2]['data'].cuda()))  # dim=[1512,256]
        g1 = t.relu(self.linear_g(input[9]['data'].cuda()))  # dim=[11153,256]

        # --------------------disease-drug GCN----------------------
        didr_drdi_edge = t.cat((didr_edge, drdi_edge), 1)
        didr_drdi = t.cat((di2, dr2))
        didr_drdi_gcn = t.relu(self.gcn_drdi(didr_drdi.cuda(), didr_drdi_edge.cuda()))
        di3, dr3 = t.split(didr_drdi_gcn.cuda(), (di_dim, dr_dim))  # dim_di3=[394,256], dim_dr3=[542,256]

        # --------------------disease-protein-drug GCN----------------------
        dip_gcn_edge = input[3]['dip_edge_gcn']
        pdi_gcn_edge = t.cat((input[3]['dip_edge_gcn'][1, :].reshape(1, -1), input[3]['dip_edge_gcn'][0, :].reshape(1, -1)))
        dip_pdi_edge = t.cat((dip_gcn_edge, pdi_gcn_edge), 1)
        dip_pdi = t.cat((di2, p1))
        dip_pdi_gcn = t.relu(self.gcn_dip(dip_pdi.cuda(), dip_pdi_edge.cuda()))
        di5, p3 = t.split(dip_pdi_gcn.cuda(), (di_dim, p_dim))  # dim_di5=[394,256], dim_p3=[1512,256]

        drp_gcn_edge = input[4]['drp_edge_gcn']
        pdr_gcn_edge = t.cat((input[4]['drp_edge_gcn'][1, :].reshape(1, -1), input[4]['drp_edge_gcn'][0, :].reshape(1, -1)))
        drp_pdr_edge = t.cat((drp_gcn_edge, pdr_gcn_edge), 1)
        drp_pdr = t.cat((dr2, p1))
        drp_pdr_gcn = t.relu(self.gcn_drp(drp_pdr.cuda(), drp_pdr_edge.cuda()))
        dr5, p2 = t.split(drp_pdr_gcn.cuda(), (dr_dim, p_dim))  # dim_dr5=[542,256], dim_p2=[1512,256]

        # --------------------disease-gene-drug GCN----------------------
        dig_gcn_edge = input[10]['dig_edge_gcn']
        gdi_gcn_edge = t.cat((input[10]['dig_edge_gcn'][1, :].reshape(1, -1), input[10]['dig_edge_gcn'][0, :].reshape(1, -1)))
        dig_gdi_edge = t.cat((dig_gcn_edge, gdi_gcn_edge), 1)
        dig_gdi = t.cat((di2, g1))
        dig_gdi_gcn = t.relu(self.gcn_dig(dig_gdi.cuda(), dig_gdi_edge.cuda()))
        di4, g3 = t.split(dig_gdi_gcn.cuda(), (di_dim, g_dim))  # dim_di4=[394,256], dim_g3=[11153,256]

        drg_gcn_edge = input[11]['drg_edge_gcn']
        gdr_gcn_edge = t.cat((input[11]['drg_edge_gcn'][1, :].reshape(1, -1), input[11]['drg_edge_gcn'][0, :].reshape(1, -1)))
        drg_gdr_edge = t.cat((drg_gcn_edge, gdr_gcn_edge), 1)
        drg_gdr = t.cat((dr2, g1))
        drg_gdr_gcn = t.relu(self.gcn_drg(drg_gdr.cuda(), drg_gdr_edge.cuda()))
        dr4, g2 = t.split(drg_gdr_gcn.cuda(), (dr_dim, g_dim))  # dim_dr4=[542,256], dim_g2=[11153,256]

        # --------------------disease-protein-gene-drug RGCN----------------------
        di1di3_mp = input[12]['di1di3_edge_rgcn']
        di3di1_mp = t.cat(
            (input[12]['di1di3_edge_rgcn'][1, :].reshape(1, -1), input[12]['di1di3_edge_rgcn'][0, :].reshape(1, -1)))
        di1dr1_mp = input[12]['di1dr1_edge_rgcn']
        dr1di1_mp = t.cat(
            (input[12]['di1dr1_edge_rgcn'][1, :].reshape(1, -1), input[12]['di1dr1_edge_rgcn'][0, :].reshape(1, -1)))
        di1dr2_mp = input[12]['di1dr2_edge_rgcn']
        dr2di1_mp = t.cat(
            (input[12]['di1dr2_edge_rgcn'][1, :].reshape(1, -1), input[12]['di1dr2_edge_rgcn'][0, :].reshape(1, -1)))
        di2dr1_mp = input[12]['di2dr1_edge_rgcn']
        dr1di2_mp = t.cat(
            (input[12]['di2dr1_edge_rgcn'][1, :].reshape(1, -1), input[12]['di2dr1_edge_rgcn'][0, :].reshape(1, -1)))
        dr1dr3_mp = input[12]['dr1dr3_edge_rgcn']
        dr3dr1_mp = t.cat(
            (input[12]['dr1dr3_edge_rgcn'][1, :].reshape(1, -1), input[12]['dr1dr3_edge_rgcn'][0, :].reshape(1, -1)))
        p3di1_mp = input[12]['p3di1_edge_rgcn']
        di1p3_mp = t.cat(
            (input[12]['p3di1_edge_rgcn'][1, :].reshape(1, -1), input[12]['p3di1_edge_rgcn'][0, :].reshape(1, -1)))
        p2dr1_mp = input[12]['p2dr1_edge_rgcn']
        dr1p2_mp = t.cat(
            (input[12]['p2dr1_edge_rgcn'][1, :].reshape(1, -1), input[12]['p2dr1_edge_rgcn'][0, :].reshape(1, -1)))
        g3di1_mp = input[12]['g3di1_edge_rgcn']
        di1g3_mp = t.cat(
            (input[12]['g3di1_edge_rgcn'][1, :].reshape(1, -1), input[12]['g3di1_edge_rgcn'][0, :].reshape(1, -1)))
        g2dr1_mp = input[12]['g2dr1_edge_rgcn']
        dr1g2_mp = t.cat(
            (input[12]['g2dr1_edge_rgcn'][1, :].reshape(1, -1), input[12]['g2dr1_edge_rgcn'][0, :].reshape(1, -1)))

        di1di3_num = input[12]['di1di3_edge_rgcn'].size(1)  # num=29294
        di1dr1_num = input[12]['di1dr1_edge_rgcn'].size(1)  # num=37336
        di1dr2_num = input[12]['di1dr2_edge_rgcn'].size(1)  # num=37336
        di2dr1_num = input[12]['di2dr1_edge_rgcn'].size(1)  # num=37336
        dr1dr3_num = input[12]['dr1dr3_edge_rgcn'].size(1)  # num=12290
        p3di1_num = input[12]['p3di1_edge_rgcn'].size(1)  # num=295070
        p2dr1_num = input[12]['p2dr1_edge_rgcn'].size(1)  # num=1516
        g3di1_num = input[12]['g3di1_edge_rgcn'].size(1)  # num=148903
        g2dr1_num = input[12]['g2dr1_edge_rgcn'].size(1)  # num=6716

        rgcn_edge_index = t.cat((di1di3_mp, di3di1_mp, di1dr1_mp, dr1di1_mp, di1dr2_mp, dr2di1_mp, p3di1_mp, di1p3_mp, di2dr1_mp, dr1di2_mp,
                               dr1dr3_mp, dr3dr1_mp, p2dr1_mp, dr1p2_mp, g3di1_mp, di1g3_mp, g2dr1_mp, dr1g2_mp), 1)
        rgcn_edge_type = t.cat((t.zeros(1, di1di3_num, dtype=t.int64), t.ones(1, di1di3_num, dtype=t.int64),
                              t.full((1, di1dr1_num), 2.0, dtype=t.int64), t.full((1, di1dr1_num), 3.0, dtype=t.int64),
                              t.full((1, di1dr2_num), 4.0, dtype=t.int64), t.full((1, di1dr2_num), 5.0, dtype=t.int64),
                              t.full((1, p3di1_num), 6.0, dtype=t.int64), t.full((1, p3di1_num), 7.0, dtype=t.int64),
                              t.full((1, di2dr1_num), 8.0, dtype=t.int64), t.full((1, di2dr1_num), 9.0, dtype=t.int64),
                              t.full((1, dr1dr3_num), 10.0, dtype=t.int64), t.full((1, dr1dr3_num), 11.0, dtype=t.int64),
                              t.full((1, p2dr1_num), 12.0, dtype=t.int64), t.full((1, p2dr1_num), 13.0, dtype=t.int64),
                              t.full((1, g3di1_num), 14.0, dtype=t.int64), t.full((1, g3di1_num), 15.0, dtype=t.int64),
                              t.full((1, g2dr1_num), 16.0, dtype=t.int64), t.full((1, g2dr1_num), 17.0, dtype=t.int64)),
                             1)
        rgcn_edge_type = t.reshape(rgcn_edge_type, [-1, ])

        rgcn_didrpg = t.cat((di2, di1, di3, dr2, dr1, dr3, p3, p2, g3, g2))
        rgcn_final_didrpg = t.relu(self.rgcn_didrpg(rgcn_didrpg.cuda(), rgcn_edge_index.cuda(), rgcn_edge_type.cuda()))
        di1_rgcn, di2_rgcn, di3_rgcn, dr1_rgcn, dr2_rgcn, dr3_rgcn, p3_rgcn, p2_rgcn, g3_rgcn, g2_rgcn = t.split(
            rgcn_final_didrpg.cuda(), (di_dim, di_dim, di_dim, dr_dim, dr_dim, dr_dim, p_dim, p_dim, g_dim, g_dim))

        # --------------------output layer----------------------
        tmp_didi = t.add(di1_rgcn, di2_rgcn)
        di_di = t.add(tmp_didi, di3_rgcn)  # dim=[394,256]
        tmp_drdr = t.add(dr1_rgcn, dr2_rgcn)
        dr_dr = t.add(tmp_drdr, dr3_rgcn)  # dim=[542,256]
        p_p = t.add(p3_rgcn, p2_rgcn)  # dim=[1512,256]
        g_g = t.add(g3_rgcn, g2_rgcn)  # dim=[11153,256]

        mlp_di1 = t.relu(self.linear_mlp_di1(di_di))
        mlp_di2 = t.relu(self.linear_mlp_di2(mlp_di1))
        mlp_di3 = t.relu(self.linear_mlp_di3(mlp_di2))  # dim=[394,64]

        mlp_dr1 = t.relu(self.linear_mlp_dr1(dr_dr))
        mlp_dr2 = t.relu(self.linear_mlp_dr2(mlp_dr1))
        mlp_dr3 = t.relu(self.linear_mlp_dr3(mlp_dr2))  # dim=[542,64]

        mlp_p1 = t.relu(self.linear_mlp_p1(p_p))
        mlp_p2 = t.relu(self.linear_mlp_p2(mlp_p1))
        mlp_p3 = t.relu(self.linear_mlp_p3(mlp_p2))  # dim=[1512,64]

        mlp_g1 = t.relu(self.linear_mlp_g1(g_g))
        mlp_g2 = t.relu(self.linear_mlp_g2(mlp_g1))
        mlp_g3 = t.relu(self.linear_mlp_g3(mlp_g2))  # dim=[11153,64]

        # --------------------calculate similarity----------------------
        drdi = mlp_dr3.mm(mlp_di3.t())  # dim=[542,394]
        pdi = mlp_p3.mm(mlp_di3.t())  # dim=[1512,394]
        pdr = mlp_p3.mm(mlp_dr3.t())  # dim=[1512,542]
        gdi = mlp_g3.mm(mlp_di3.t())  # dim=[11153,394]
        gdr = mlp_g3.mm(mlp_dr3.t())  # dim=[11153,542]

        return drdi, pdi, pdr, gdi, gdr

