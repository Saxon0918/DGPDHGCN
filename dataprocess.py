from __future__ import division


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset

    def __getitem__(self, index):
        return (self.data_set['didi'], self.data_set['drdr'],
                self.data_set['pp'], self.data_set['pdi'],
                self.data_set['pdr'], self.data_set['drdi']['train'],
                self.data_set['drdi_p'], self.data_set['drdi_true'],
                self.data_set['didr'], self.data_set['gg'],
                self.data_set['gdi'], self.data_set['gdr'],
                self.data_set['rgcn_edge'])