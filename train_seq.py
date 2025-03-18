# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Train the trajectory-based motion segmentation network.
"""
import os
import shutil
import argparse
import torch
import torchvision
from core.utils.utils import load_config_file, save_model, cls_iou, cal_roc, get_feat
from core.dataset.kubric import Kubric_dataset, find_traj_label
from core.dataset.dynamic_stereo import Stereo_dataset
from core.dataset.waymo import Waymo_dataset
from core.dataset.hoi4d import HOI_dataset
from core.dataset.base import ProbabilisticDataset
from core.network.traj_oa_depth import traj_oa_depth
from core.network.transfomer import traj_seg
from core.network import loss_func
from core.utils.visualize import Visualizer,read_imgs_from_path
from core.dataset.data_utils import normalize_point_traj_torch
from glob import glob
import numpy as np
from PIL import Image
import json
import wandb
from itertools import zip_longest

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.autograd.set_detect_anomaly(True)

def setup_wandb(cfg):
    wandb.init(
    project="paratrain",
    name=f"{os.path.basename(cfg.log_dir)}",
    save_code=False,
    config={
        "log_dir": cfg.log_dir,
        "extra_info": cfg.extra_info,
        "pos_embed": cfg.pos_embed,
        "time_embed": cfg.time_embed,
        "model_name": cfg.model_name,
        "efficient": cfg.efficient,
        "lr": cfg.lr,
        "out_dim": cfg.out_dim,
        "oanet":cfg.oanet
    },
    # mode="offline"
)

def setup_para_dataset(cfg):
    train_transform = torchvision.transforms.ToTensor()
    test_transform = torchvision.transforms.ToTensor()

    # 遍历 train_dataset 列表，创建每个对应的 DataLoader
    waymo_train_dataset = None
    for i, dataset_name in enumerate(cfg.train_dataset):
        if dataset_name == 'kubric':
            kubric_train_dataset = Kubric_dataset(cfg.train_root[i], transform=train_transform, split='train', 
                                                  track_method=cfg.track_method, depth_type=cfg.depth_type,
                                                  load_dino=cfg.dino)

        elif dataset_name == 'dynamic_stereo':
            stereo_train_dataset = Stereo_dataset(cfg.train_root[i], transform=train_transform, split='train', 
                                                  track_method=cfg.track_method, depth_type=cfg.depth_type,
                                                  load_dino=cfg.dino)
        
        elif dataset_name == 'waymo':
            waymo_train_dataset = Waymo_dataset(cfg.train_root[i], transform=train_transform, split='train', 
                                                  track_method=cfg.track_method, depth_type=cfg.depth_type,
                                                  load_dino=cfg.dino)
        elif dataset_name == 'hoi4d':
            hoi_train_dataset = HOI_dataset(cfg.train_root[i], transform=train_transform, split='train', 
                                                  track_method=cfg.track_method, depth_type=cfg.depth_type,
                                                  load_dino=cfg.dino)
        else:
            raise NotImplementedError(f"Train dataset {dataset_name} not supported")

    combined_dataset = ProbabilisticDataset(stereo_train_dataset, kubric_train_dataset, 
                                            waymo_train_dataset, hoi_train_dataset,
                                            prob1=cfg.prob1, prob2=cfg.prob2,
                                            prob3=cfg.prob3, prob4=cfg.prob4)
    train_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    test_loaders = []
    for i, dataset_name in enumerate(cfg.test_dataset):
        if dataset_name == 'kubric':
            test_dataset = Kubric_dataset(cfg.test_root[i], transform=test_transform, split='validation', 
                                          track_method=cfg.track_method, depth_type=cfg.depth_type,
                                          load_dino=cfg.dino)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=16)
            test_loaders.append(test_loader)

        elif dataset_name == 'dynamic_stereo':
            test_dataset = Stereo_dataset(cfg.test_root[i], transform=test_transform, split='valid', 
                                          track_method=cfg.track_method, depth_type=cfg.depth_type,
                                          load_dino=cfg.dino)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=3)
            test_loaders.append(test_loader)

        else:
            raise NotImplementedError(f"Test dataset {dataset_name} not supported")
    
    # 返回多个 train_loader 和一个 test_loader
    return train_loader, test_loaders

def setup_model(cfg):
    if cfg.model_name == 'traj_oa_depth':
        model = traj_oa_depth(cfg.extra_info, cfg.oanet, cfg.pos_embed, cfg.dino, cfg.target_feature_dim, cfg.dino_later,cfg.dino_woatt, cfg.time_att, cfg.tracks, cfg.depths)
    elif cfg.model_name == 'traj_attention':
        model = traj_seg(cfg.window_size, cfg.resolution, cfg.extra_info, cfg.out_dim, cfg.time_embed ,cfg.pos_embed, cfg.efficient)
    else:
        raise NotImplementedError
    return model

def train_para_epoch(cfg, model, optimizer, train_loader, epoch, test_loaders):
    model.train()

    for idx, sample in enumerate(train_loader):
        # load batch, kubric: 11s; stereo: 12s - 18s
        depth, traj = sample["depths"], sample["track_2d"]
        depth = depth.permute(0,2,3,1).unsqueeze(0)  # [B, 1, H, W, L]
        visible_mask = sample["visible_mask"]

        # vis
        if cfg.vis:
            seq_name = sample['case_name'][0]
            img_dir = sample['video_dir'][0]
            imgs = read_imgs_from_path(img_dir)            
            vis = Visualizer(cfg.log_dir, pad_value=120, linewidth=3)
            vis.visualize(imgs, traj.permute(0,3,2,1), visible_mask.permute(0,2,1).unsqueeze(-1), filename=seq_name)
        
        B, _, N, L = traj.shape
        _, _, H, W, _ = depth.shape
        # mask: padding = invisible
        mask = (~visible_mask).unsqueeze(0)
        # feat
        # dino_list = []
        # for path in sample['dinos']: 
        #     # [103, 183, 768], [num_patches_h, num_patches_w, C]
        #     features = np.load(path, mmap_mode="r").squeeze()
        #     features = torch.from_numpy(features).float().unsqueeze(0)
        #     dino_list.append(features)

        if cfg.dino:
            dino_list = sample["dinos"]
            featmap_downscale_factor = (
                dino_list[0].shape[1]/ H,
                dino_list[0].shape[2]/ W,
            )
            dino = get_feat(featmap_downscale_factor, traj.squeeze(0)[..., sample['q_t']], dino_list)
            if not cfg.dino_later:
                dino = dino.repeat(1, 1, 1, L)
                
            dino_t = dino.float().cuda()
            # 降维
            if cfg.target_feature_dim is not None:
                features = dino_t.permute(0,2,3,1) # [B,C,N,L]->[B,N,L,C]
                # compute feature visualization matrix
                C = features.shape[-1]
                # no need to compute PCA on the entire set of features, we randomly sample 100k features
                temp_feats = features.reshape(-1, C)
                # compute PCA to reduce the feature dimension to target_feature_dim
                U, S, reduce_to_target_dim_mat = torch.pca_lowrank(
                    temp_feats, q=cfg.target_feature_dim, niter=20
                )

                # reduce the features to target_feature_dim [C,target_dim]
                features =  features @ reduce_to_target_dim_mat
                C = features.shape[-1]

                # normalize the reduced features to [0, 1] along each dimension
                feat_min = features.reshape(-1, C).min(dim=0)[0]
                feat_max = features.reshape(-1, C).max(dim=0)[0]
                features = (features - feat_min) / (feat_max - feat_min)
                # final features are of shape (B, target_feature_dim, N, L)
                dino_t = features.permute(0,3,1,2)

        # gt
        dynamic_mask = sample["dynamic_mask"]
        if cfg.two_label:
            traj_label = find_traj_label(traj, mask, dynamic_mask, sample['q_t'])
            traj_label_t = torch.from_numpy(traj_label).unsqueeze(0).float().cuda()

        # normalize traj
        traj = normalize_point_traj_torch(traj, [H, W])
        depth_t, traj_t, mask_t = depth.float().cuda(), traj.float().cuda(), mask.float().cuda()

        if cfg.extra_info:
            visib_value = sample['visib_value'].unsqueeze(0)
            confi_value = sample['confi_value'].unsqueeze(0)
            visib_value_t, confi_value_t = visib_value.float().cuda(), confi_value.float().cuda()
            input_batch = {"traj": traj_t, "mask": mask_t, "depth": depth_t,
                            "visib_value": visib_value_t, "confi_value": confi_value_t}
        else:
            input_batch = {"traj": traj_t, "mask": mask_t, "depth": depth_t}
        
        if cfg.dino:
            input_batch["dino"] = dino_t
        
        # forward pass
        pred = model(input_batch)
        traj_label_t = traj_label_t.unsqueeze(1)

        # weighted BCEloss
        N = traj_label_t.shape[-1]
        scale = (N - traj_label_t.sum((1,2))) / (traj_label_t.sum((1,2)) + 1e-6)
        scale = scale.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, N)
        weight = scale * traj_label_t + (1.0 - traj_label_t)

        loss = loss_func.BCELoss(pred, traj_label_t, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % cfg.print_freq == 0:
            log_dict = {"loss": loss.item(), "epoch": epoch}
                        
            wandb.log(log_dict, step=epoch * len(train_loader) + idx)

            print(f"Train iter {idx}/{len(train_loader)}, Total Loss: {loss.item()}")

        if idx % 3000 == 0 and idx > 0:
            # dynamic_stereo
            test_iou_1 = test_epoch(cfg, epoch, model, test_loaders[0])
            # kubric
            test_iou_2 = 0
            if len(test_loaders)>1:
                test_iou_2 = test_epoch(cfg, epoch, model, test_loaders[1])
            
            save_model(model, cfg.log_dir, epoch, test_iou_1, test_iou_2)

    return loss

@torch.no_grad()
def test_epoch(cfg, epoch, model, test_loader):
    model.eval()
    sum_iou = 0
    all_preds = []
    all_labels = []
    for idx, sample in enumerate(test_loader):
        # load batch
        depth, traj = sample["depths"], sample["track_2d"]
        depth = depth.permute(0,2,3,1).unsqueeze(0) # [B, 1, H, W, L]
        visible_mask = sample["visible_mask"]
        # no padding points
        B,_,N,L = traj.shape
        _,_,H,W,_ = depth.shape
        # mask = torch.zeros(B, 1, N, L, dtype=torch.float32) 
        mask = (~visible_mask).unsqueeze(0)
        # label
        if cfg.dino:
            dino_list = sample["dinos"]
            featmap_downscale_factor = (
                dino_list[0].shape[1]/ H,
                dino_list[0].shape[2]/ W,
            )
            dino = get_feat(featmap_downscale_factor, traj.squeeze(0)[..., sample['q_t']], dino_list)
            if not cfg.dino_later:
                dino = dino.repeat(1, 1, 1, L)

            dino_t = dino.float().cuda()
            # 降维
            if cfg.target_feature_dim is not None:
                features = dino_t.permute(0,2,3,1) # [B,C,N,L]->[B,N,L,C]
                # compute feature visualization matrix
                C = features.shape[-1]
                # no need to compute PCA on the entire set of features, we randomly sample 100k features
                temp_feats = features.reshape(-1, C)
                # compute PCA to reduce the feature dimension to target_feature_dim
                U, S, reduce_to_target_dim_mat = torch.pca_lowrank(
                    temp_feats, q=cfg.target_feature_dim, niter=20
                )

                # reduce the features to target_feature_dim [C,target_dim]
                features =  features @ reduce_to_target_dim_mat
                C = features.shape[-1]

                # normalize the reduced features to [0, 1] along each dimension
                feat_min = features.reshape(-1, C).min(dim=0)[0]
                feat_max = features.reshape(-1, C).max(dim=0)[0]
                features = (features - feat_min) / (feat_max - feat_min)
                # final features are of shape (B, target_feature_dim, N, L)
                dino_t = features.permute(0,3,1,2)

        # gt
        dynamic_mask = sample["dynamic_mask"]
        if cfg.two_label:
            traj_label = find_traj_label(traj,mask,dynamic_mask,sample['q_t'])
            traj_label_t = torch.from_numpy(traj_label).unsqueeze(0).float().cuda()
        # normilize traj
        traj = normalize_point_traj_torch(traj,[H,W])
        depth_t, traj_t, mask_t = depth.float().cuda(), traj.float().cuda(), mask.float().cuda()

        if cfg.extra_info:
            visib_value = sample['visib_value'].unsqueeze(0)
            confi_value = sample['confi_value'].unsqueeze(0)
            visib_value_t, confi_value_t = visib_value.float().cuda(), confi_value.float().cuda()
            input_batch = {"traj": traj_t, "mask": mask_t, "depth": depth_t,
                            "visib_value": visib_value_t, "confi_value": confi_value_t,}
        else:
            input_batch = {"traj": traj_t, "mask": mask_t, "depth": depth_t}
        
        if cfg.dino:
            input_batch["dino"] = dino_t
            
        # inference
        pred = model(input_batch)
        traj_label_t = traj_label_t.unsqueeze(1)
        
        # evaluate
        cur_iou = cls_iou(pred, traj_label_t)
        sum_iou += cur_iou
        
        all_preds.append(pred.cpu().numpy())
        all_labels.append(traj_label_t.cpu().numpy())
    
    metric_dir = os.path.join(cfg.log_dir, "metrics")
    os.makedirs(metric_dir, exist_ok=True)
    
    mean_iou = sum_iou / len(test_loader)
    mean_iou = mean_iou.cpu().numpy()
    print("Test epoch {}, mean iou {}".format(epoch, mean_iou))
    all_preds_flattened = [pred.flatten() for pred in all_preds]
    all_labels_flattened = [label.flatten() for label in all_labels]
    all_preds = np.concatenate(all_preds_flattened)
    all_labels = np.concatenate(all_labels_flattened)
    fprs, tprs, recalls, precisions, auc_res = cal_roc(all_preds, all_labels, metric_dir, epoch)
            
    return mean_iou

def main(cfg):
    setup_wandb(cfg)
    # initialize dataloader
    # train_loader, test_loader = setup_dataset(cfg)
    # print('Data loader ready...Training samples number %d' % (len(train_loader)))
    train_loader, test_loaders = setup_para_dataset(cfg)
    print('Data loader ready...')
    # initialize model
    model = setup_model(cfg)
    model.cuda()
    # initialize training
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(cfg.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Load model from {}'.format(cfg.resume_path))
    else:
        print('training from scratch')
    
    for epoch in range(cfg.max_epochs):
        train_loss = train_para_epoch(cfg, model, optimizer, train_loader, epoch, test_loaders)        
        # dynamic_stereo
        test_iou_1 = test_epoch(cfg, epoch, model, test_loaders[0])
        # kubric
        test_iou_2 = 0
        if len(test_loaders)>1:
            test_iou_2 = test_epoch(cfg, epoch, model, test_loaders[1])
        
        save_model(model, cfg.log_dir, epoch, test_iou_1, test_iou_2)

    wandb.finish()

def debug_bceloss(pred, traj_label_t, weight, mask, depth, traj):
    pred = pred.to('cpu')  # 确保数据在 CPU 上
    traj_label_t = traj_label_t.to('cpu')
    weight = weight.to('cpu')
    mask = mask.to('cpu') 
    depth = depth.to('cpu')
    traj = traj.to('cpu')
    
    mask_info = {
        "all_true": torch.all(mask).item(),  # 检查 mask 是否全为 True
        "min": mask.min().item(),
        "max": mask.max().item(),
        "contains_nan": torch.isnan(mask).any().item(),
        "contains_inf": torch.isinf(mask).any().item(),
        "shape": list(mask.shape)
    }

    # 获取信息
    pred_info = {
        "min": pred.min().item(),
        "max": pred.max().item(),
        "contains_nan": torch.isnan(pred).any().item(),
        "contains_inf": torch.isinf(pred).any().item(),
        "shape": list(pred.shape)
    }

    traj_label_t_info = {
        "min": traj_label_t.min().item(),
        "max": traj_label_t.max().item(),
        "contains_nan": torch.isnan(traj_label_t).any().item(),
        "contains_inf": torch.isinf(traj_label_t).any().item(),
        "shape": list(traj_label_t.shape)
    }

    traj_info = {
        "min": traj.min().item(),
        "max": traj.max().item(),
        "contains_nan": torch.isnan(traj).any().item(),
        "contains_inf": torch.isinf(traj).any().item(),
        "shape": list(traj.shape)
    }

    depth_info = {
        "min": depth.min().item(),
        "max": depth.max().item(),
        "contains_nan": torch.isnan(depth).any().item(),
        "contains_inf": torch.isinf(depth).any().item(),
        "shape": list(depth.shape)
    }
    
    weight_info = {
        "min": weight.min().item(),
        "max": weight.max().item(),
        "contains_nan": torch.isnan(weight).any().item(),
        "contains_inf": torch.isinf(weight).any().item(),
        "shape": list(weight.shape)
    }
    
    # 写入到文件
    with open("tensor_info.txt", "w") as f:
        f.write("Pred Tensor Info:\n")
        f.write(str(pred_info) + "\n\n")

        f.write("Traj Label Tensor Info:\n")
        f.write(str(traj_label_t_info) + "\n\n")

        f.write("Weight Tensor Info:\n")
        f.write(str(weight_info) + "\n\n")
        
        f.write("Mask Tensor Info:\n")
        f.write(str(mask_info) + "\n\n")

        f.write("Depth Tensor Info:\n")
        f.write(str(depth_info) + "\n\n")
        
        f.write("Traj Tensor Info:\n")
        f.write(str(traj_info) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trajectory-based motion segmentation network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config_file', metavar='DIR',help='path to config file')
    args = parser.parse_args()
    cfg = load_config_file(args.config_file)

    # copy the config file into log dir
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    shutil.copy(args.config_file, cfg.log_dir)
        
    main(cfg)

# CUDA_VISIBLE_DEVICES=3 python train_seq.py ./configs/kubric_train.yaml