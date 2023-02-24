
from config import Config
from helpers import make_directory_if_not_exist
from modelbuilder import Model


class PathBuilder:
    def __init__(self,model:Model) -> None:
          self.model = model
    
    def buildPath(self):
        npz_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{Config.dataset}/{self.model.sample}/Npzs'
        npy_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{Config.dataset}/{self.model.sample}/Npys'
        pickle_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{Config.dataset}/{self.model.sample}/Pickles'
        coordinates_file_name = 'coordinates.csv'

        scribble_img = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{Config.dataset}/{self.model.sample}/Scribble/manual_scribble_1.npy'
        if Config.mclust_scribble:
            scribble_img = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{Config.dataset}/{self.model.sample}/Scribble/Config.mclust_scribble.npy'
        local_data_folder_path = './Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data'

        input = f'{npy_path}/mapped_{Config.n_pcs}.npy'
        inv_xy = f'{pickle_path}/inv_spot_xy.pickle'
        border = npz_path+'/borders.npz'
        background = npy_path+'/backgrounds.npy'
        foreground = npy_path+'/foregrounds.npy'
        indices_arg = npy_path+'/indices.npy'
        pixel_barcode_map_path = pickle_path+'/pixel_barcode_map.pickle'
        coordinate_file = f'Data/{Config.dataset}/{self.model.sample}/{coordinates_file_name}'
        map_pixel_to_grid_spot_file_path = f'{local_data_folder_path}/{Config.dataset}/{self.model.sample}/Jsons/map_pixel_to_grid_spot.json'
        pixel_barcode_file_path = f'{local_data_folder_path}/{Config.dataset}/{self.model.sample}/Npys/pixel_barcode.npy'
        manual_annotation_file_path = f'./Data/{Config.dataset}/{self.model.sample}/manual_annotations.csv'

        output_folder_path = f'./Outputs/{Config.test_name}/{Config.dataset}/{self.model.sample}'
        leaf_output_folder_path = f'{output_folder_path}/{Config.scheme}/{Config.n_pcs}_pcs/Seed_{self.model.seed}/Lr_{self.model.lr}/Hyper_{self.model.stepsize_sim}_{self.model.stepsize_con}_{self.model.stepsize_scr}'
        labels_per_itr_folder_path = f'{leaf_output_folder_path}/Labels_per_itr/'
        image_per_itr_folder_path = f'{leaf_output_folder_path}/Image_per_itr/'
        meta_data_file_path = f'{leaf_output_folder_path}/meta_data.csv'

        make_directory_if_not_exist(output_folder_path)
        make_directory_if_not_exist(labels_per_itr_folder_path)
        make_directory_if_not_exist(image_per_itr_folder_path)