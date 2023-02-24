
import argparse
import json
import torch

class Config:
    test_folder_base_name = None
    dataset = None
    n_pcs = None
    scribble = None
    expert_scribble = None
    nChannel = None
    max_iter = None
    nConv = None
    visualize = None
    use_background_scribble = None
    added_layers = None
    last_layer_channel_count = None
    hyper_sum_division = None
    seed_options = None
    sim_options = None
    miu_options = None
    niu_options = None
    lr_options = None
    samples = None
    use_cuda = None
    mclust_scribble = None
    minLabels = None
    intermediate_channels = None
    meta_data_index = None
    test_name = None
    scheme = None

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

        self.intermediate_channels = self.n_pcs # was n_pcs

        self.meta_data_index = ['test_name', 'seed', 'dataset', 'sample', 'n_pcs', 'scribble', 'max_iter', 'sim', 'miu', 'niu', 'scheme', 'lr', 'nConv', 'no_of_scribble_layers', 'intermediate_channels', 'added_layers', 'last_layer_channel_count', 'hyper_sum_division']

        self.test_name = f'{self.test_folder_base_name}_itr_{self.max_iter}'

        if self.scribble:
            if self.expert_scribble: self.scheme = 'Expert_scribble'
            elif self.mclust_scribble: self.scheme = 'Mclust_scribble'
            else: self.scheme = 'Other_scribble'
        else: self.scheme = 'No_scribble'
