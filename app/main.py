from cnnmodel import MyNet
from helpers import *
from init import *
from config import *
from modelbuilder import BruitForceModelBuilder
from pathbuilder import PathBuilder

# Load Config
config = Config()

if config.use_cuda:
    print("GPU available")
else:
    print("GPU not available")

# Build Models
modelBuilder = BruitForceModelBuilder()
models = modelBuilder.buildModel()


for model in tqdm(models):

    print(model)

    pathBuilder = PathBuilder(model)
    paths = pathBuilder.buildPath()

    # %%
    pixel_barcode = np.load(paths.pixel_barcode_file_path)
    pixel_rows_cols = np.argwhere(pixel_barcode != '')
    df_man = pd.read_csv(paths.manual_annotation_file_path, index_col=0)
    manual_annotation_labels = df_man['label'].values
    ari_per_itr = []
    loss_per_itr = []
    df_barcode_labels_per_itr = pd.DataFrame(index = pixel_barcode[pixel_barcode != ''])
    backgrounds = np.load(paths.background)
    foregrounds = np.load(paths.foreground)
    
    # %%
    with open(paths.map_pixel_to_grid_spot_file_path, 'r') as f:
        map_pixel_to_grid_spot = json.load(f)


    # %%
    torch.manual_seed(model.seed)
    np.random.seed(model.seed)

    no_of_scribble_layers = 0

    # %%
    im = np.load(input)
    im.shape

    # %%
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')]) ) # z, y, x
    data.shape

    # %%
    if Config.use_cuda:
        data = data.cuda()
    data = Variable(data)
    data.shape

    # %%
    

    # %%
    # load scribble
    if Config.scribble:
        mask = np.load(paths.scribble_img)
        foreground_val = 1000
        background_val = 255
        mask = relabel_mask(mask.copy(), background_val)
        if len(mask[mask != background_val]) == 0:
            print('Expecting some scribbles, but no scribbles are found!')
            last_layer_channel_count = 100 + Config.added_layers
            nChannel = last_layer_channel_count
        else:
            
            mask_foreground = mask.copy()
            mask_foreground[foregrounds[:, 0], foregrounds[:, 1]] = foreground_val
            
            mx_label_num = mask[mask != background_val].max()
            if Config.use_background_scribble:
                mask[backgrounds[:, 0], backgrounds[:, 1]] = mx_label_num + 1 # Assuming that scribble labels increase by 1
            mask = mask.reshape(-1)
            scr_idx = np.where(mask != 255)[0]
            mask_foreground = mask_foreground.reshape(-1)

            mask_inds = np.unique(mask)
            mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==background_val) )

            for i in range(1, len(mask_inds)):
                if mask_inds[i] - mask_inds[i-1] != 1:
                    print("Problem in scribble labels. Not increasing by 1.")

            # # Take all of the foreground into similarity component
            # inds_sim = torch.from_numpy( np.where( mask_foreground == foreground_val )[ 0 ] ) # Big change done!

            # # Take the non-scribbled foreground into similarity component
            mask_foreground[scr_idx] = background_val
            inds_sim = torch.from_numpy( np.where( mask_foreground == foreground_val )[ 0 ] ) # Big change done!

            # inds_sim = torch.from_numpy( np.where( mask == background_val )[ 0 ] )
            inds_scr = torch.from_numpy( np.where( mask != background_val )[ 0 ] )
            inds_scr_array = [None for _ in range(mask_inds.shape[0])]

            for i in range(mask_inds.shape[0]):
                inds_scr_array[i] = torch.from_numpy( np.where( mask == mask_inds[i] )[ 0 ] )

            target_scr = torch.from_numpy( mask.astype(np.int64) )

            if Config.use_cuda:
                inds_sim = inds_sim.cuda()
                inds_scr = inds_scr.cuda()
                target_scr = target_scr.cuda()


            target_scr = Variable( target_scr ) # *************** Why? **************

            minLabels = len(mask_inds)
            # nChannel = minLabels + 1
            nChannel = minLabels + Config.added_layers # ************ Change ************ 

            no_of_scribble_layers = minLabels # **************** Addition *****************
            last_layer_channel_count = no_of_scribble_layers + Config.added_layers
    else:
        last_layer_channel_count = 100 + Config.added_layers
        nChannel = last_layer_channel_count



    # %%
    # train
    model = MyNet( data.size(1) , last_layer_channel_count)
    if Config.use_cuda:
        model.cuda()
    model.train()

    # %%
    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # scribble loss definition
    loss_fn_scr = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(reduction='mean')
    loss_hpz = torch.nn.L1Loss(reduction='mean')
    # loss for the diagonal neighbour
    loss_hp_diag = torch.nn.L1Loss(reduction='mean')

    HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, nChannel)
    HP_diag_target = torch.zeros(im.shape[0] - 1, im.shape[1] - 1, nChannel)
    if Config.use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()
        HP_diag_target = HP_diag_target.cuda()
        
    optimizer = optim.SGD(model.parameters(), lr=model.lr, momentum=0.9)
    label_colours = np.random.randint(255,size=(255,3))

    label_colours[0,:] = [255,255,255]
    label_colours[1,:] = [0,255,0]
    label_colours[2,:] = [255,0,0]
    label_colours[3,:] = [255,255,0]
    label_colours[4,:] = [0,255,255]
    label_colours[5,:] = [255,0,255]
    label_colours[6,:] = [0,0,0]
    label_colours[7,:] = [73,182,255]

    loss_comparison = 0

    # %%
    borders = np.load(paths.border)

    right_border = borders['right_border']
    left_border = borders['left_border']
    up_border = borders['up_border']
    down_border = borders['down_border']
    nw_border = borders['nw_border']
    se_border = borders['se_border']

    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    loss_list = []
    loss_without_hyperparam_list = []

    for batch_idx in (range(Config.max_iter)):

        # forwarding
        optimizer.zero_grad()   # ******************** check ********************

        output = model( data )[ 0 ]
        output[:, backgrounds[:, 0], backgrounds[:, 1]] = 0 # Big problem, as all these 1s will be normalized

        output = output.permute( 1, 2, 0 )
        output = output.contiguous().view( -1, nChannel )

        outputHP = output.reshape( (im.shape[0], im.shape[1], nChannel) )


        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPy[up_border[:, 0] - 1, up_border[:, 1], :] = 0
        HPy[down_border[:, 0], down_border[:, 1], :] = 0

        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        HPz[left_border[:, 0], left_border[:, 1] - 1, :] = 0
        HPz[right_border[:, 0], right_border[:, 1], :] = 0
        
        HP_diag = outputHP[1:,1:, :] - outputHP[0:-1, 0:-1, :]
        HP_diag[nw_border[:, 0] - 1, nw_border[:, 1] - 1, :] = 0
        HP_diag[se_border[:, 0], se_border[:, 1], :] = 0


        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)
        lhp_diag = 0
        lhp_diag = loss_hp_diag(HP_diag, HP_diag_target)
        

        ignore, target = torch.max( output, 1 )


        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        im_ari = im_target.reshape(im.shape[0], im.shape[1])

        
        im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
        labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
        df_labels = pd.DataFrame({'label': labels}, index=pixel_barcode[pixel_barcode != ''])
        ari_per_itr.append(calc_ari(df_man, df_labels))
        df_barcode_labels_per_itr[f'itr_{batch_idx}'] = labels

        if Config.visualize and (batch_idx<10 or batch_idx%10 == 0):
        
            im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
            labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
            grid_spots, colors = get_grid_spots_from_pixels(pixel_rows_cols, labels, map_pixel_to_grid_spot)
            if Config.dataset == 'Custom': rad = 700
            else: rad = 10
            plt.figure(figsize=(5.5,5))
            plt.scatter(grid_spots[:, 1], 1000 - grid_spots[:, 0], c=colors, s=rad)
            plt.axis('off')
            plt.savefig(f'{paths.image_per_itr_folder_path}/itr_{batch_idx}.png',format='png',dpi=1200,bbox_inches='tight',pad_inches=0)
            plt.close('all')

        # loss 
        if Config.scribble:        

    
            loss_lr = 0
            for i in range(mask_inds.shape[0]):
                loss_lr += loss_fn_scr(output[ inds_scr_array[i] ], target_scr[ inds_scr_array[i] ])

            loss_sim = loss_fn(output[ inds_sim ], target[ inds_sim ])
            hyper_sum = model.stepsize_sim + model.stepsize_scr + model.stepsize_con

            sim_multiplier = 1
            con_multiplier = 1
            scr_multiplier = 1
            L_sim = model.stepsize_sim * loss_sim * sim_multiplier
            L_scr = model.stepsize_scr * loss_lr * scr_multiplier

            L_con = model.stepsize_con * (lhpy + lhpz + lhp_diag) * con_multiplier
            loss_without_hyperparam = loss_sim + loss_lr + (lhpy + lhpz + lhp_diag)

            if Config.hyper_sum_division:
                loss = (L_sim + L_con + L_scr) / hyper_sum
            else:
                loss = (L_sim + L_con + L_scr)

        else:
            loss = (model.stepsize_sim * loss_fn(output, target) + model.stepsize_con * (lhpy + lhpz + lhp_diag)) # consider hyperparameter sum division later

        loss_without_hyperparam_list.append(loss_without_hyperparam.data.cpu().numpy())
        loss_per_itr.append(loss.data.cpu().numpy())
        

        loss.backward()
        optimizer.step()

    # %%
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ (c + 10) % nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( np.array([im.shape[0],im.shape[1],3]).astype( np.uint8 ))
    im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
    f = im_cluster_num
    s = np.argwhere(f != 110) # not a good way
    colors = f.flatten()
    plt.figure(figsize = (4, 4))
    if Config.dataset == 'Custom': rad = 1500
    else: rad = 10
    plt.scatter(s[:, 1], 1000 - s[:, 0], c=colors, s = rad)

    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( np.array([im.shape[0],im.shape[1],3]).astype( np.uint8 ))
    im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
    labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
    grid_spots, colors = get_grid_spots_from_pixels(pixel_rows_cols, labels, map_pixel_to_grid_spot)

    df_ari_per_itr = pd.DataFrame({'ARI': ari_per_itr})
    df_ari_per_itr.to_csv(f'{paths.leaf_output_folder_path}/ari_per_itr.csv')

    df_loss_per_itr = pd.DataFrame({'Loss': loss_per_itr})
    df_loss_per_itr.to_csv(f'{paths.leaf_output_folder_path}/loss_per_itr.csv')

    df_loss_without_hyperparam_per_itr = pd.DataFrame({'Loss_without_hyperparam': loss_without_hyperparam_list})
    df_loss_without_hyperparam_per_itr.to_csv(f'{paths.leaf_output_folder_path}/loss_without_hyperparam_per_itr.csv')

    df_labels = pd.DataFrame({'label': labels}, index=pixel_barcode[pixel_barcode != ''])
    df_labels.to_csv(f'{paths.leaf_output_folder_path}/final_barcode_labels.csv')

    df_final_metrics = pd.DataFrame({'ARI': df_ari_per_itr['ARI'].values[-1:], 'Loss': df_loss_per_itr['Loss'].values[-1:], 'Loss_without_hyperparam': df_loss_without_hyperparam_per_itr['Loss_without_hyperparam'].values[-1:]})
    df_final_metrics.to_csv(f'{paths.leaf_output_folder_path}/final_metrics.csv')

    df_barcode_labels_per_itr.to_csv(f'{paths.leaf_output_folder_path}/barcode_labels_per_itr.csv')

    print("ARI:", calc_ari(df_man, df_labels))
    print(f"L_sim: {L_sim}, L_con: {L_con}, L_scr: {L_scr}")
    print(f"L_sim + L_con + L_scr: {L_sim + L_con + L_scr}")
    print(f"Total loss: {loss}")
    print(f"Loss without hyperparam: {loss_without_hyperparam}")

    meta_data_value = [Config.test_name, model.seed, Config.dataset, model.sample, Config.n_pcs, Config.scribble, Config.max_iter, model.stepsize_sim, model.stepsize_con, model.stepsize_scr, Config.scheme, model.lr, Config.nConv, no_of_scribble_layers, Config.intermediate_channels, Config.added_layers, last_layer_channel_count, Config.hyper_sum_division]
    df_meta_data = pd.DataFrame(index=Config.meta_data_index, columns=['value'])
    df_meta_data['value'][Config.meta_data_index] = meta_data_value
    df_meta_data.to_csv(paths.meta_data_file_path)

    if Config.dataset == 'Custom': rad = 700
    else: rad = 10
    plt.figure(figsize=(5.5,5))
    plt.axis('off')
    plt.scatter(grid_spots[:, 1], 1000 - grid_spots[:, 0], c=colors, s=rad)
    plt.savefig(f'{paths.leaf_output_folder_path}/seg_{model.stepsize_sim}_{model.stepsize_con}_{model.stepsize_scr}_seed_{model.seed}_pcs_{Config.n_pcs}.png',format='png',dpi=1200,bbox_inches='tight',pad_inches=0)
    plt.savefig(f'{paths.leaf_output_folder_path}/seg_{model.stepsize_sim}_{model.stepsize_con}_{model.stepsize_scr}_seed_{model.seed}_pcs_{Config.n_pcs}.eps',format='eps',dpi=1200,bbox_inches='tight',pad_inches=0)

    plt.close('all')

