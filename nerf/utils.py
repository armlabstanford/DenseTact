import os
import glob
import tqdm
import math
import random
import warnings
import tensorboardX
import GPy

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
from itertools import cycle
import json

import sys

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# Generates GP from densetact data.
def get_gp(data):
    lines = data['lookuptable']

    # Build GP using data.
    X = np.atleast_2d(lines[:,0]).T
    y = np.atleast_2d(lines[:,1]).T
    kernel = GPy.kern.RBF(input_dim = 1, variance = 3.0, lengthscale = 40.0)
    model_gp = GPy.models.GPRegression(X,y,kernel)
    model_gp.optimize()

    return model_gp

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, camera_model='pinhole', MAX_TOUCH_ANGLE = 72.5):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    print("Inside the GET_RAYS function....")

    #print("Checking the argument B in get_rays definition: ",)

    # print(camera_model)
    # print("INTRINICS")
    # print(H,W)
    
    
    results = {}

    if camera_model == "pinhole":
        #print("Using the pinhole camera....")
        i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
        # print("MESH INDS")
        # print(i)
        i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
        j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

        if N > 0:
            N = min(N, H*W)

            if error_map is None:
                inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
                inds = inds.expand([B, N])
                # print("CHECKING INDS")
                # print(inds)
            else:
                # weighted sample on a low-reso grid
                inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

                # map to the original resolution with random perturb.
                inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
                sx, sy = H / 128, W / 128
                inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
                inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
                inds = inds_x * W + inds_y

                results['inds_coarse'] = inds_coarse # need this when updating error_map

            i = torch.gather(i, -1, inds)
            j = torch.gather(j, -1, inds)

            results['inds'] = inds

        else:
            inds = torch.arange(H*W, device=device).expand([B, H*W])

        zs = torch.ones_like(i)
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        directions = torch.stack((xs, ys, zs), dim=-1)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        # print("RGB")
        # print(directions.shape)
        rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

        rays_o = poses[..., :3, 3] # [B, 3]
        rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]
    
    elif camera_model == "touch":
        #print("Using TOUCH images based on the camera module...")
        if os.path.exists('./data/touch_rays.pt'):
            # If we've already saved the touch rays to disk, just load them.
            # dirs, mask = torch.load('data/touch_rays.pt')
            pass
        else:
            # print("HELLO 1")
            assert W, H == (800, 600)
            data = np.load('./Sensor_calibration/table.npz')
            c_x, c_y = float(data['centerx']), float(data['centery'])
            xx, yy = custom_meshgrid(c_x - torch.linspace(0, W-1, W, device=device), 
                                     c_y - torch.linspace(0, H-1, H, device=device))
            
            # print("HELLO 2")
            xx = xx.reshape([1, H*W]).expand([B, H*W])
            yy = yy.reshape([1, H*W]).expand([B, H*W])
            phi = torch.atan2(yy,xx)

            # If using real sensor, generate rays via GP.
            gp = get_gp(data)
            r = torch.sqrt((xx)**2 + (yy)**2)

            # Generate thetas via GP prediction.
            theta, _ = gp.predict(r.reshape(-1,1).cpu().numpy())
            theta = torch.from_numpy(theta).reshape(xx.shape)

            # print("HELLO 3")
            # Convert to radians.
            theta = (np.pi/180) * theta

            # Put onto correct device, as needed.
            theta = theta.to(phi.device, phi.dtype)

            # Generate mask by clipping to max angle of sensor.
            mask = theta <= (np.pi / 180) * MAX_TOUCH_ANGLE

            # print("THETA AND PHI")
            # print(theta.shape)
            # print(phi.shape)
            # print("HELLO 4")
            directions = torch.stack([torch.sin(theta)*torch.cos(phi),
                                      torch.sin(theta)*torch.sin(phi),
                                      -torch.cos(theta)], -1).reshape(-1,3)
            
            # print(directions.shape)
            # R = torch.Tensor([[np.cos(np.pi/2), -1*np.sin(np.pi/2), 0],
            #                   [np.sin(np.pi/2), np.cos(np.pi/2), 0],
            #                   [0, 0, 1]]).to(device)
            R = torch.Tensor([[1, 0, 0],
                              [0, np.cos(np.pi), -1*np.sin(np.pi)],
                              [0, np.sin(np.pi), np.cos(np.pi)]]).to(device)
            #R = torch.eye(3).to(device)
            # print(R.shape)
            # print(directions.t().shape)
            # print((R@directions.t()).shape)
            directions = ((R@directions.t()).t()).reshape(B,-1,3)
            # print(directions.shape)
            # print(mask.shape)
            # view_dirs = directions.squeeze().detach().cpu().numpy()
            # view_mask = mask.squeeze().detach().cpu().numpy()
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(view_dirs[:,0],view_dirs[:,1],view_dirs[:,2], c=view_mask)
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.savefig('test.png')
            # print("HELLO 5")
            if N > 0:
                # print("CHECK")
                N = min(N, H*W)
                inds = torch.multinomial(mask/torch.sum(mask), N, replacement=True)
                directions = directions[torch.arange(directions.size(0)),inds]

                view_dirs = directions.squeeze().detach().cpu().numpy()
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(view_dirs[:,0],view_dirs[:,1],view_dirs[:,2])
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')
                # plt.savefig('directions.png')
                # stop
                inds = inds.expand([B, N])
                results['inds'] = inds
                # print(N)
                # print(directions.shape)
                # print(inds)
            
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)
            rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)

            rays_o = poses[..., :3, 3] # [B, 3]
            rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

            results['mask'] = mask
            # print("HELLO 6")
            #print(N)
            #stop
            #plt.scatter()
    #         dirs = dirs.transpose(0,1)
    #         rays_d = torch.sum(dirs[..., np.newaxis, :] * poses[:3,:3], -1)
    #         results['mask'] = mask
    #     else:
    #         assert W, H == (800, 600)
    #         data = np.load('./Sensor_calibration/table.npz')

    #         c_x, c_y = float(data['centerx']), float(data['centery'])
            
    #         y = c_y - torch.arange(0,H)
    #         x = c_x - torch.arange(0,W)
            
    #         xx, yy = torch.meshgrid(x, y, indexing='ij')
    #         phi = torch.atan2(yy,xx)

    #         # If using real sensor, generate rays via GP.
    #         gp = get_gp(data)
    #         r = torch.sqrt((xx)**2 + (yy)**2)

    #         # Generate thetas via GP prediction.
    #         theta, _ = gp.predict(r.reshape(-1,1).cpu().numpy())
    #         theta = torch.from_numpy(theta).reshape(xx.shape)

    #         # Convert to radians.
    #         theta = (np.pi/180) * theta

    #         # Put onto correct device, as needed.
    #         theta = theta.to(phi.device, phi.dtype)

    #         # Generate mask by clipping to max angle of sensor.
    #         mask = theta <= (np.pi / 180) * MAX_TOUCH_ANGLE

    #         dirs = torch.stack([torch.sin(theta)*torch.cos(phi),
    #                             torch.sin(theta)*torch.sin(phi),
    #                             -torch.cos(theta)], -1)
    #         print("TOUCH")
    #         print(dirs.shape)
    #         #torch.save((dirs.transpose(0,1),mask.T), 'data/touch_rays.pt')


    #         R = torch.Tensor([[np.cos(-np.pi/2), -1*np.sin(-np.pi/2), 0],
    #                           [np.sin(-np.pi/2), np.cos(-np.pi/2), 0],
    #                           [0, 0, 1]])
            
    #         dirs = dirs@R
    #         dirs, mask = dirs.transpose(0,1).to(device), mask.T

    #         print(dirs.shape)
    #         print(dirs[..., np.newaxis, :].shape)
    #         print(poses[:3,:3].shape)
    #         print(poses[..., :3,:3].shape)
    #         rays_d = torch.sum(dirs[..., np.newaxis, :] * poses[...,:3,:3], -1)
    #         rays_o = poses[..., :3,-1].expand(rays_d.shape)

    #         results['mask'] = mask

    # print(rays_o.shape)
    # print(rays_d.shape)
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    # print("HELLO 7")
    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        print()
        print("============All the print statement from now on are from inside the TRAINER fn in the UTILS FILE============")
        print()
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        print("The device that I am using is: ", self.device)
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            print("Instance is true")
            criterion.to(self.device)
            print("Printing the criterion: ", criterion)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            print("Self.model: ", self.model)
            self.optimizer = optimizer(self.model)
            print("Self.optimizer: ", self.optimizer)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)
            print("Learning rate is already declared....")

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
            print("have set the decay value to 0.95...")
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {        # declaring a dictionary to keep a track of the model
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            print("Preparing the workspace.......")
            os.makedirs(self.workspace, exist_ok=True)    # to make the workspace directory       
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)  # to make the checkpoint directory
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        # to store the log files in the workspace folder
        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")

            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()   # just initialized a class variable
                print("Printing load_checkpoint(): ", self.load_checkpoint())
                print("Loading latest checkpoint and jumping into the load_checkpoint definition...")
                #sys.exit()

            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)

            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()

            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):


        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, 
                                            perturb=True, force_all_rays=True, datatype=data['type'],
                                            max_far=data['far'], min_near=data['near'], **vars(self.opt))
            # if data['type'] == 'rgb':
            #     outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, 
            #                                 perturb=True, force_all_rays=True, datatype=data['type'],
            #                                 max_depth=data['type'], **vars(self.opt))
            # elif data['type'] == 'depth':
            #     outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, 
            #                                 perturb=True, force_all_rays=True, datatype=data['type'],
            #                                 max_depth=data['type'], **vars(self.opt))
            # elif data['type'] == 'touch':
            #     outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, 
            #                                 perturb=True, force_all_rays=True, datatype=data['type'],
            #                                 max_depth=data['type'], **vars(self.opt))



            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            #TODO: figure out if this needs to be changed to handle depth and touch!
            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        #TODO: probably will have to change these lines for depth and touch
        # ignore artifacts that say depth is zero
        #if data['type'] == 'depth':
        #    continue

        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape
        print()
        print("PRINTING THE IMAGE SHAPE IN ORDER B, N, C: ", B, N, C)
        print()
        # print("C")
        # print(C)
        
        if data['type'] == 'rgb':
            if self.opt.color_space == 'linear':
                images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            if data['type'] == 'rgb':
                bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.
            elif data['type'] == 'depth' or data['type'] == 'touch':
                bg_color = torch.rand_like(images[..., :1]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images


        #print("WHAT TYPE AM I?")
        #print(data['type'])

        #print(data['type'])
        #print(data['far'])
        #prirays_ont(data['near'])

        

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, 
                                        perturb=True, force_all_rays=False, datatype=data['type'],
                                         max_far=data['far'], min_near=data['near'], **vars(self.opt))

        if data['type'] == 'rgb':
            pred_rgb = outputs['image']
            
            l2 = torch.nn.MSELoss()
            #var_loss = torch.mean(torch.max(torch.abs(pred_rgb-gt_rgb),dim=-1)[0].unsqueeze(-1)/(outputs['image_var']+1e-10))
            # print("predicted image")
            # print(outputs['image'])
            # print("gt image")
            # print(gt_rgb)
            loss = torch.max(torch.abs(pred_rgb-gt_rgb),dim=-1)[0].mean() #l2(pred_rgb,gt_rgb) #+ 1e-4*var_loss#torch.max(torch.abs(pred_rgb-gt_rgb),dim=-1)[0].mean()
            #print("CHECKING ELEMENTS")
            #print(torch.any(torch.isnan(pred_rgb)))
            #print(torch.any(torch.isinf(pred_rgb)))
            #print(torch.any(torch.isnan(gt_rgb)))
            #print(torch.any(torch.isinf(gt_rgb)))
            #stop
            # loss = torch.max(torch.abs(pred_rgb-gt_rgb),dim=-1)[0].mean()

        elif data['type'] == 'depth':
            gt_rgb = torch.squeeze(gt_rgb,axis=-1)
            pred_depth = outputs['depth']
            # print("CHECKING NETWORK RESULT")
            # print(gt_rgb)
            # print(" ")
            # print(pred_depth)
            l2 = torch.nn.MSELoss()
            l1 = torch.nn.L1Loss()
            var = torch.mean(torch.abs(pred_depth-gt_rgb)/(torch.sqrt(outputs['depth_var']+1e-12)))
            depth_loss = l1(pred_depth,gt_rgb) #+ var + torch.max(torch.abs(pred_depth-gt_rgb),dim=-1)[0].mean()
            loss = depth_loss

            pred_rgb = outputs['image']

        elif data['type'] == 'touch':
            gt_rgb = torch.squeeze(gt_rgb,axis=-1)
            pred_depth = outputs['depth']
            l2 = torch.nn.MSELoss()
            l1 = torch.nn.L1Loss()
            #print("vals")
            #print(pred_depth[gt_rgb<self.opt.touch_far].shape)
            touch_loss = l1(pred_depth[gt_rgb<data['far']],gt_rgb[gt_rgb<data['far']])
            #print("CHECK VALUES")
            #print(touch_loss)
            #print(pred_depth)
            #print(gt_rgb)
            loss = touch_loss

            pred_rgb = outputs['image']
            # print("not implemented touch yet!")
           
        # print("loss")
        # print(loss)
        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map[data['type']] is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[data['type']][index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()
        #print(loss)

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if data['type'] == 'rgb':
            if self.opt.color_space == 'linear':
                images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, 
                                        datatype=data['type'], max_far=data['far'], min_near=data['near'], **vars(self.opt))
        # if data['type'] == 'rgb':
        #     outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, 
        #                                 datatype=data['type'], max_depth=data['type'], **vars(self.opt))
        # elif data['type'] == 'depth':
        #     outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, 
        #                                 datatype=data['type'], max_depth=data['type'], **vars(self.opt))
        # elif data['type'] == 'touch':
        #     outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, 
        #                                 datatype=data['type'], max_depth=data['type'], **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  

        # print(data)

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        #print(data['type'])
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, 
                                        perturb=perturb, datatype=data['type'], 
                                        max_far = data['far'], min_near = data['near'], **vars(self.opt))
        # if data['type'] == 'rgb':
        #     outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, 
        #                                 perturb=perturb, datatype=data['type'], 
        #                                 max_depth = data['far'], **vars(self.opt))
        # elif data['type'] == 'depth':
        #     outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, 
        #                                 perturb=perturb, datatype=data['type'], 
        #                                 max_depth = data['far'], **vars(self.opt))
        # elif data['type'] == 'touch':
        #     outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, 
        #                                 perturb=perturb, datatype=data['type'], 
        #                                 max_depth = data['far'], **vars(self.opt))
        # elif data['type'] == 'viewer':
        #     outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, 
        #                                 perturb=perturb, datatype=data['type'], 
        #                                 max_depth = data['far'], **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        training_poses = [train_loader[i]._data.poses for i in range(len(train_loader))]
        training_intrinsics = [train_loader[i]._data.intrinsics for i in range(len(train_loader))]
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(training_poses, training_intrinsics)

        
        # get a ref to error_map
        err_dict = {}
        for i in range(len(train_loader)):
            err_dict[train_loader[i]._data.datatype] = train_loader[i]._data.error_map
        self.error_map = err_dict

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()
    
    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = []
        for i in range(len(loader)):
            pbar.append(tqdm.tqdm(total=len(loader[i]) * loader[i].batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'))

        self.model.eval()
        with torch.no_grad():

            # allow for multiple datasets to be trained at once!
            # ensure only the smaller datasets are cycled through!
            loader_lens = [len(loader[i]) for i in range(len(loader))]
            index = loader_lens.index(max(loader_lens))
            zipper = [loader[index]] + [cycle(loader[i]) for i in range(len(loader)) if i!=index]
            for data in zip(*zipper):
            #for i, data in enumerate(loader):
                
                for d in data:
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth = self.test_step(d)                
                
                    path = os.path.join(save_path, f'{name}_{i:04d}.png')
                    path_depth = os.path.join(save_path, f'{name}_{i:04d}_depth.png')

                    #self.log(f"[INFO] saving test image to {path}")

                    if d['type'] == 'rgb':
                        if self.opt.color_space == 'linear':
                            preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()

                    cv2.imwrite(path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(path_depth, (pred_depth * 255).astype(np.uint8))

                    for i in range(len(loader)):
                        pbar[i].update(loader[i].batch_size)

        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()  # to start training the model...
        print("==================Inside the TRAIN_GUI function where the actual model training happens================")

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)  # declaring a tensor

        print("Printing range of train_loader: ", len(train_loader))

        # allow for multiple datasets to be trained at once!
        # ensure only the smaller datasets are cycled through!
        loader_lens = [len(train_loader[i]) for i in range(len(train_loader))]
        print("Printing the legnth of the loader lens: ", loader_lens)
        print("Printing loader lens: ", loader_lens)

        print("Printing max of loader lens: ", max(loader_lens))
        index = loader_lens.index(max(loader_lens))
        print("Printing index: ", index)
        #sys.exit()
        zipper = [cycle(train_loader[i]) for i in range(len(train_loader))]
        loader = zip(*zipper)


        #loader = iter(zipper)

        # mark untrained grid
        # mark untrained region (i.e., not covered by any camera from the training dataset)
        training_poses = [train_loader[i]._data.poses for i in range(len(train_loader))]
        training_intrinsics = [train_loader[i]._data.intrinsics for i in range(len(train_loader))]
        if self.global_step == 0:
            self.model.mark_untrained_grid(training_poses, training_intrinsics)
            #self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):
            
            # print(loader)
            data = next(loader)
            # print("HMMMMMMM")
            # print(data)
            # print("HI")
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            #try:
            #    data = next(loader)
            #except StopIteration:
            #    loader = iter(train_loader)
            #    data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1
            for d in data:
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.train_step(d)
         
                self.scaler.scale(loss).backward()
                #print("MODEL")
                #print(self.model.sigma_net[0].weight.grad)
                #print(self.model.sigma_net[1].weight.grad)
                #print(self.model.color_net[0].weight.grad)
                #print(self.model.color_net[1].weight.grad)
                #print(self.model.color_net[2].weight.grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            
                if self.scheduler_update_every_step:
                    self.lr_scheduler.step()

                total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, datatype, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        # print(pose)
        # print(intrinsics)
        # print(W)
        # print(H)
        # print(datatype)
        # print(bg_color)
        # print(spp)
        # print(downscale)


        data = {
            'type': datatype,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'near': 1e-9,
            'far': 1e9
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            [loader[i].sampler.set_epoch(self.epoch) for i in range(len(loader))]
        
        if self.local_rank == 0:
            pbar = []
            for i in range(len(loader)):
                pbar.append(tqdm.tqdm(total=len(loader[i]) * loader[i].batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'))

        self.local_step = 0

        # allow for multiple datasets to be trained at once!
        # ensure only the smaller datasets are cycled through!
        loader_lens = [len(loader[i]) for i in range(len(loader))]
        index = loader_lens.index(max(loader_lens))
        zipper = [loader[index]] + [cycle(loader[i]) for i in range(len(loader)) if i!=index]
        for data in zip(*zipper):
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    

            self.global_step += 1
            self.local_step += 1
            for d in data:
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.train_step(d)
         
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler_update_every_step:
                    self.lr_scheduler.step()

                loss_val = loss.item()
                total_loss += loss_val

                if self.local_rank == 0:
                    if self.report_metric_at_train:
                        for metric in self.metrics:
                            metric.update(preds, truths)
                        
                    if self.use_tensorboardX:
                        self.writer.add_scalar("train/loss", loss_val, self.global_step)
                        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                    for i in range(len(loader)):
                        if self.scheduler_update_every_step:
                            pbar[i].set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                        else:
                            pbar[i].set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                        pbar[i].update(loader[i].batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / (len(data)*self.local_step)
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            for i in range(len(loader)):
                pbar[i].close()

            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = []
            for i in range(len(loader)):
                pbar.append(tqdm.tqdm(total=len(loader[i]) * loader[i].batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'))

        with torch.no_grad():
            self.local_step = 0

            # allow for multiple datasets to be trained at once!
            # ensure only the smaller datasets are cycled through!
            loader_lens = [len(loader[i]) for i in range(len(loader))]
            index = loader_lens.index(max(loader_lens))
            zipper = [loader[index]] + [cycle(loader[i]) for i in range(len(loader)) if i!=index]
            
            
            for data in zip(*zipper):
                #print(" ")
                #print("HEY!")
                self.local_step += 1
                
                #print(data)
                #stop
              
                
                for d in data:
                    #out_data = {} 
                    #out_data['camera_angle_x'] = d['camera_angle_x']
                    #out_data['frames'] = []
                    #print(d)
                    #stop
                    
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth, truths, loss = self.eval_step(d)

                    # all_gather/reduce the statistics (NCCL only support all_*)
                    if self.world_size > 1:
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss = loss / self.world_size
                    
                        preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_list, preds)
                        preds = torch.cat(preds_list, dim=0)

                        preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_depth_list, preds_depth)
                        preds_depth = torch.cat(preds_depth_list, dim=0)

                        truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(truths_list, truths)
                        truths = torch.cat(truths_list, dim=0)
                
                    loss_val = loss.item()
                    total_loss += loss_val

                    # only rank = 0 will perform evaluation.
                    if self.local_rank == 0:

                        for metric in self.metrics:
                            metric.update(preds, truths)

                        # save image
                        save_path = os.path.join(self.workspace, 'validation', f'r_{self.local_step}.png')
                        save_path_depth = os.path.join(self.workspace, 'validation/depth', f'r_{self.local_step}.png')
                        #save_path_gt = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_gt.png')

                        #self.log(f"==> Saving validation image to {save_path}")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        os.makedirs(os.path.dirname(save_path_depth), exist_ok=True)

                        if self.opt.color_space == 'linear':
                            preds = linear_to_srgb(preds)

                        pred = preds[0].detach().cpu().numpy()
                        pred_depth = preds_depth[0].detach().cpu().numpy()
                    
                        cv2.imwrite(save_path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_depth, (pred_depth * 255).astype(np.uint8))
                        
                        #mat = []
                        #print(" ")
                        #print("val poses")
                        #print(d['poses'].reshape(4,4))
                        #print(d['poses'].reshape(4,4).shape)
                        #for row in d['poses'].detach().cpu().numpy().reshape(4,4):
                        #    mat.append(list(row))
                        #    
                        #print(mat)
                        #frame = {'file_path':f'./train/r_{self.local_step:d}.png',
                        #         'transform_matrix': d['poses'][0].detach().cpu().numpy().tolist(),
                        #}
                        #with open(os.path.join(self.workspace, 'validation', "train.json"), "a+") as file_object:   
                        #    json.dump(frame, file_object,indent=4)
                        #    file_object.write(',\n')
                        #
                        #out_data['frames'].append(frame)
                        #print(out_data)
                        #cv2.imwrite(save_path_gt, cv2.cvtColor((linear_to_srgb(truths[0].detach().cpu().numpy()) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                        for i in range(len(loader)):
                            pbar[i].set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                            pbar[i].update(loader[i].batch_size)


        average_loss = total_loss / (len(data)*self.local_step)
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            for i in range(len(loader)):
                pbar[i].close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    
    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        print()
        print("============INSIDE THE SAVE CHECKPOINT FUNCTION IN THE UTILS FILE============")
        print()

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        print()
        print("INSIDE THE LOAD CHECK POINTS DEFINITION IN THE UTILS FILE....")
        print()
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
