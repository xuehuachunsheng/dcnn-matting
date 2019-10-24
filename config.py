

class Config:

    ###################
    # Data paths
    ###################

    data_type = 'lowres' # or highres

    trimap_type = 'Trimap1' # or Trimap2, Trimap3

    ROOT_PATH = '/home/wuyanxue/Data/alphamatting.com'

    gt_train_alpha_path = 'gt_training_{}'.format(data_type)

    train_trimap_path = 'trimap_training_{}/{}'.format(data_type, trimap_type)

    train_input_path = 'input_training_{}'.format(data_type)

    test_input_path = 'input_{}'.format(data_type)

    test_trimap_path = 'trimap_{}/{}'.format(data_type, trimap_type)

    # Used to composite the background and foreground
    bg_image_path = '/home/wuyanxue/Data/MSCOCO_val2017'

    # How many background images used to composite with foreground images
    comp_ratio = 10

    # Composite images path
    comp_images_path = 'comp_images_{}'.format(data_type)
    comp_matte_path = 'comp_gt_mattes_{}'.format(data_type)
    comp_knn_matte_path = 'comp_knn_mattes_{}'.format(data_type)
    comp_closed_form_matte_path = 'comp_closed_form_mattes_{}'.format(data_type)
    comp_trimap_path = 'comp_trimaps_{}'.format(data_type)

    # How many 27x27 patches generated of each image
    n_patches_one_image = 1000

    # Trimap unknown region the gray level
    unknown_code = 128

    train_patch_size = 27

    patch_path = 'patch_data'

    train_patch_path = 'Train'
    validation_patch_path = 'Validation'
    test_patch_path = 'Test'

    train_val_test_ratio = [0.9, 0.08, 0.02]

    # Image patches path for training
    image_patch_path = 'image_patches_{}'.format(data_type)
    matte_patch_path = 'matte_patches_{}'.format(data_type)
    knn_matte_patch_path = 'knn_matte_patches_{}'.format(data_type)
    closed_form_matte_patch_path = 'closed_form_matte_patches_{}'.format(data_type)

    ###### End of data preparation


    ###################
    # Training parameters
    ###################
    batch_size = 32
    n_epochs = 50
    epoch_decays = (10, 20, 30, 40, 50)
    epoch_decay_rate = 0.2
    learning_rate = 1e-4
    early_stop = True