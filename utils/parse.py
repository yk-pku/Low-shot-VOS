import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path')
    parser.add_argument('--image_info_root')
    parser.add_argument('--mask_root')
    parser.add_argument('--batch_size', default = 4, type = int)
    parser.add_argument('--epoch_num', default = 200, type = int)
    parser.add_argument('--lr', default = 0.001, type = float)
    parser.add_argument('--min_lr', default = 0.0001, type = float)
    parser.add_argument('--point_num', default = 16, type = int)
    parser.add_argument('--multimask_output', action = 'store_true')
    parser.add_argument('--aug_mask', action='store_true')
    parser.add_argument('--start_aug', default = -1, type = int)
    parser.add_argument('--point_prompt_fix', action = 'store_true')
    parser.add_argument('--decoder_fix', action = 'store_true')

    # PointDisturb
    parser.add_argument('--point_disturb', action = 'store_true')
    parser.add_argument('--point_max_off', default = 20, type = int)

    # BootstrappedBCE
    parser.add_argument('--bsbce', action = 'store_true')
    parser.add_argument('--start_warm', default = 40, type = int)
    parser.add_argument('--end_warm', default = 100, type = int)
    parser.add_argument('--top_p', default = 0.15, type = float)

    return parser.parse_args()

def arg_parse_iou():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path')
    # dataset
    parser.add_argument('--image_info_root')
    parser.add_argument('--mask_root')
    parser.add_argument('--pgt1_root')
    parser.add_argument('--pgt2_root')
    # dataloader
    parser.add_argument('--batch_size', default = 20, type = int)
    parser.add_argument('--epoch_num', default = 200, type = int)
    parser.add_argument('--lr', default = 0.001, type = float)
    parser.add_argument('--min_lr', default = 0.0001, type = float)

    parser.add_argument('--point_num', default = 20, type = int)
    parser.add_argument('--multimask_output', action = 'store_true')

    # use decoder feature
    parser.add_argument('--feature256', action = 'store_true')
    
    # infer related 
    parser.add_argument('--phase1_root')
    parser.add_argument('--pre1_root')
    parser.add_argument('--union_root')
    parser.add_argument('--save_ckpt')


    return parser.parse_args()
