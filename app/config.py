
import argparse
import json
import torch

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='ScribbleSeg expert annotation pipeline')
        parser.add_argument('--params', help="The input parameters json file path", required=True)

        args = parser.parse_args()

        with open(args.params) as f:
            params = json.load(f)
        self.test_folder_base_name = params['test_folder_base_name']
        self.dataset = params['dataset']
        self.n_pcs = params['n_pcs']
        self.scribble = params['scribble']
        self.expert_scribble = params['expert_scribble']
        self.nChannel = params['nChannel']
        self.max_iter = params['max_iter']
        self.nConv = params['nConv']
        self.visualize = params['visualize']
        self.use_background_scribble = params['use_background_scribble']
        self.added_layers = params['added_layers']
        self.last_layer_channel_count = params['last_layer_channel_count']
        self.hyper_sum_division = params['hyper_sum_division']
        self.seed_options = params['seed_options']
        self.sim_options = params['sim_options']
        self.miu_options = params['miu_options']
        self.niu_options = params['niu_options']
        self.lr_options = params['lr_options']
        self.samples = params['samples']

        self.use_cuda = torch.cuda.is_available()

        self.mclust_scribble = not self.expert_scribble

        self.minLabels = -1 # will be assigned to the number of different scribbles used

        self.intermediate_channels = n_pcs # was n_pcs

        self.meta_data_index = ['test_name', 'seed', 'dataset', 'sample', 'n_pcs', 'scribble', 'max_iter', 'sim', 'miu', 'niu', 'scheme', 'lr', 'nConv', 'no_of_scribble_layers', 'intermediate_channels', 'added_layers', 'last_layer_channel_count', 'hyper_sum_division']

        self.test_name = f'{self.test_folder_base_name}_itr_{self.max_iter}'
