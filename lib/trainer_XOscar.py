import glob
import random
import tqdm
import imageio
import tensorboardX
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from pytorch3d.ops.knn import knn_points
# from pytorch3d.loss import chamfer_distance
import torch.distributed as dist
from rich.console import Console
from torch_ema import ExponentialMovingAverage
import torchvision

from lib.common.utils import *
from lib.common.visual import draw_landmarks, draw_mediapipe_landmarks
from lib.dpt import DepthNormalEstimation

class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 text, negative, dir_text,
                 opt,  # extra conf
                 model,  # network
                 guidance,  # guidance network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 max_keep_ckpt=1,  # max num of saved ckpts in disk
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.dpt = DepthNormalEstimation(use_depth=False) if opt.use_dpt else None
        self.default_view_data = None
        self.name = name
        self.text = text
        self.negative = negative
        self.dir_text = dir_text
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size

        self.workspace = os.path.join(opt.workspace, self.name, self.text)
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = opt.eval_interval
        self.use_checkpoint = opt.ckpt
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        self.model = model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3]).module
        if self.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

        # guide model
        self.guidance = guidance
        # self.guidance = torch.nn.DataParallel(self.guidance, device_ids=[0, 1, 2, 3]).module
        # text prompt
        self.text_embeds = None
        if self.guidance is not None:
            for p in self.guidance.parameters():
                p.requires_grad = False
            self.prepare_text_embeddings()

        # try out torch 2.0
        if torch.__version__[0] == '2':
            self.model = torch.compile(self.model)
            self.guidance = torch.compile(self.guidance)

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.optimizer_geo = optimizer[0](self.model)
        self.optimizer_app = optimizer[1](self.model)
        self.optimizer_ani = optimizer[2](self.model)


        self.lr_scheduler_geo = lr_scheduler[0](self.optimizer_geo)
        self.lr_scheduler_app = lr_scheduler[1](self.optimizer_app)
        self.lr_scheduler_ani = lr_scheduler[2](self.optimizer_ani)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(self.workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
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
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embeddings.
    def prepare_text_embeddings(self):
        if self.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            return

        self.text_embeds = {
            'uncond': self.guidance.get_text_embeds([self.negative]),
            'default': self.guidance.get_text_embeds([f"a 3D rendering of {self.text}, full-body"]),
        }

        if self.opt.train_face_ratio < 1:
            self.text_embeds['body'] = {
                d: self.guidance.get_text_embeds([f"a {d} view 3D rendering of {self.text}, full-body"])
                for d in ['front', 'side', 'back', "overhead"]
            }

        if self.opt.train_face_ratio > 0:
            id_text = self.text.split("wearing")[0]
            self.text_embeds['face'] = {
                d: self.guidance.get_text_embeds([f"a {d} view 3D rendering of {id_text}, face"])
                for d in ['front', 'side', 'back']
            }

    def __del__(self):
        pass
        # if self.log_ptr:
        #     self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def train_step(self, data, is_full_body,is_animate=False):
        do_rgbd_loss = self.default_view_data is not None and (self.global_step % self.opt.known_view_interval == 0)

        if do_rgbd_loss:
            data = self.default_view_data

        H, W = data['H'], data['W']
        mvp = data['mvp']  # [B, 4, 4]
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        # TEST: progressive training resolution
        if self.opt.anneal_tex_reso:
            if(self.global_step <= self.opt.iters_geo):
                scale = min(1, (self.global_step) / (0.8 * self.opt.iters_geo))
            else:
                scale = min(1, (self.global_step-self.opt.iters_geo) / (0.8 * (self.opt.iters_app+self.opt.iters_ani)))
            def make_divisible(x, y): return x + (y - x % y)
            H = max(make_divisible(int(H * scale), 16), 32)
            W = max(make_divisible(int(W * scale), 16), 32)

        if do_rgbd_loss and self.opt.known_view_noise_scale > 0:
            noise_scale = self.opt.known_view_noise_scale  # * (1 - self.global_step / self.opt.iters)
            rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
            rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================

        dir_text_z = [self.text_embeds['uncond'], self.text_embeds[data['camera_type'][0]][data['dirkey'][0]]]
        dir_text_z = torch.cat(dir_text_z)

        out = self.model(rays_o, rays_d, mvp, data['H'], data['W'], shading='albedo',is_animate=is_animate)
        image = out['image'].permute(0, 3, 1, 2)
        normal = out['normal'].permute(0, 3, 1, 2)
        alpha = out['alpha'].permute(0, 3, 1, 2)

        out_annel = self.model(rays_o, rays_d, mvp, H, W, shading='albedo',is_animate=is_animate)
        image_annel = out_annel['image'].permute(0, 3, 1, 2)
        normal_annel = out_annel['normal'].permute(0, 3, 1, 2)
        alpha_annel = out_annel['alpha'].permute(0, 3, 1, 2)

        pred = torch.cat([out['image'], out['normal']], dim=2)
        pred = (pred[0].detach().cpu().numpy() * 255).astype(np.uint8)

        p_iter = self.global_step / (self.opt.iters_geo+self.opt.iters_app+self.opt.iters_ani)


        if(p_iter<(self.opt.iters_geo)/(self.opt.iters_geo+self.opt.iters_app+self.opt.iters_ani)):
            loss = self.guidance.train_step(dir_text_z, normal_annel, guidance_scale=10+90*self.global_step/self.opt.iters_geo,geo_std=torch.std(self.model.v_offsets),app_std=torch.std(self.model.raw_albedo)).mean()
        elif(p_iter<(self.opt.iters_geo+self.opt.iters_app)/(self.opt.iters_geo+self.opt.iters_app+self.opt.iters_ani)):
            loss = self.guidance.train_step(dir_text_z, image_annel, guidance_scale=10+90*(self.global_step-self.opt.iters_geo)/self.opt.iters_app,geo_std=torch.std(self.model.v_offsets),app_std=torch.std(self.model.raw_albedo)).mean()
            # loss += self.guidance.train_step(dir_text_z, normal_annel, guidance_scale=100,geo_std=torch.std(self.model.v_offsets),app_std=torch.std(self.model.raw_albedo)).mean()
        else:
            loss = self.guidance.train_step(dir_text_z, image_annel, guidance_scale=100,geo_std=torch.std(self.model.v_offsets),app_std=torch.std(self.model.raw_albedo)).mean()
            loss += self.guidance.train_step(dir_text_z, normal_annel, guidance_scale=100,geo_std=torch.std(self.model.v_offsets),app_std=torch.std(self.model.raw_albedo)).mean()

           
        return pred, loss

    def eval_step(self, data):
        H, W = data['H'].item(), data['W'].item()
        mvp = data['mvp']
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        out = self.model(rays_o, rays_d, mvp, H, W, shading='albedo', is_train=False)
        w = out['normal'].shape[2]
        pred = torch.cat([out['normal'], out['image'],
                          torch.cat([out['normal'][:, :, :w // 2], out['image'][:, :, w // 2:]], dim=2)], dim=1)

        # dummy 
        loss = torch.zeros([1], device=pred.device, dtype=pred.dtype)

        return pred, loss

    def test_step(self, data):
        H, W = data['H'].item(), data['W'].item()
        mvp = data['mvp']
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        out = self.model(rays_o, rays_d, mvp, H, W, shading='albedo', is_train=False)
        w = out['normal'].shape[2]
        pred = torch.cat([out['normal'], out['image'],
                          torch.cat([out['normal'][:, :, :w // 2], out['image'][:, :, w // 2:]], dim=2)], dim=2)

        return pred, None

    def save_mesh(self, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, "mesh")

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path)

        self.log(f"==> Finished saving mesh.")

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            with torch.no_grad():
                if random.random() < self.opt.train_face_ratio:
                    train_loader.dataset.full_body = False
                    face_center, face_scale = self.model.get_mesh_center_scale("face")
                    train_loader.dataset.face_center = face_center
                    train_loader.dataset.face_scale = face_scale.item() * 10

                else:
                    train_loader.dataset.full_body = True

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, _ = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                else:
                    os.makedirs(os.path.join(save_path, "image"), exist_ok=True)
                    cv2.imwrite(os.path.join(save_path, "image", f'{i:04d}.png'),
                                cv2.cvtColor(pred[..., :3], cv2.COLOR_RGB2BGRA))

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}.mp4'), all_preds, fps=25, quality=9,
                             macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr_geo={self.optimizer_geo.param_groups[0]['lr']:.6f} ...")
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr_app={self.optimizer_app.param_groups[0]['lr']:.6f} ...")
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr_ani={self.optimizer_ani.param_groups[0]['lr']:.6f} ...")
        
        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.optimizer_geo.zero_grad()
            self.optimizer_app.zero_grad()
            # self.optimizer_ani.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if(self.global_step<self.opt.iters_geo+self.opt.iters_app):
                    pred_rgbs, loss = self.train_step(data, loader.dataset.full_body)
                else:
                    pred_rgbs, loss = self.train_step(data, loader.dataset.full_body,is_animate=True)

            if self.global_step % 20 == 0:
                pred = cv2.cvtColor(pred_rgbs, cv2.COLOR_RGB2BGR)
                #save_path = os.path.join(self.workspace, 'train-vis', f'{self.name}/{self.global_step:04d}.png')
                #os.makedirs(os.path.dirname(save_path), exist_ok=True)
                #cv2.imwrite(save_path, pred)

            self.scaler.scale(loss).backward()

            if self.global_step < self.opt.iters_geo:
                self.scaler.step(self.optimizer_geo)
                # self.scaler.step(self.optimizer_ani)
                self.lr_scheduler_geo.step()
                # self.lr_scheduler_ani.step()
            else:
                self.scaler.step(self.optimizer_geo)
                self.scaler.step(self.optimizer_app)
                # self.scaler.step(self.optimizer_ani)
                self.lr_scheduler_geo.step()
                self.lr_scheduler_app.step()
                # self.lr_scheduler_ani.step()
            self.scaler.update()

            # if self.scheduler_update_every_step:
            #     if self.global_step < self.opt.iters_geo:
            #         self.lr_scheduler_geo.step()
            #     elif self.global_step < (self.opt.iters_geo+self.opt.iters_app):
            #         self.lr_scheduler_app.step()
            #     else:
            #         self.lr_scheduler_ani.step()


            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr_geo", self.optimizer_geo.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar("train/lr_app", self.optimizer_app.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar("train/lr_ani", self.optimizer_ani.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), "
                        f"lr_geo={self.optimizer_geo.param_groups[0]['lr']:.6f}, ")
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), "
                        f"lr_app={self.optimizer_app.param_groups[0]['lr']:.6f}, ")
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), "
                        f"lr_ani={self.optimizer_ani.param_groups[0]['lr']:.6f}, ")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        # if not self.scheduler_update_every_step:
        #     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         self.lr_scheduler.step(average_loss)
        #     else:
        #         self.lr_scheduler.step()

        # if not self.scheduler_update_every_step:
        #     if isinstance(self.lr_scheduler_geo, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         if self.global_step < self.opt.iters_geo:
        #             self.lr_scheduler_geo.step(average_loss)
        #         elif self.global_step < (self.opt.iters_geo+self.opt.iters_app):
        #             self.lr_scheduler_app.step(average_loss)
        #         else:
        #             self.lr_scheduler_ani.step(average_loss)
        #     else:
        #         if self.global_step < self.opt.iters_geo:
        #             self.lr_scheduler_geo.step()
        #         elif self.global_step < (self.opt.iters_geo+self.opt.iters_app):
        #             self.lr_scheduler_app.step()
        #         else:
        #             self.lr_scheduler_ani.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        vis_frames = []
        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1 #total_step:4

                # with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, loss = self.eval_step(data) #preds.shape:1 2400 800 3

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    pred = (preds[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR) #preds.shape:2400 800 3
                    vis_frames.append(pred)
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)
        
        # save_path = os.path.join(self.workspace, 'validation', f'{name}.png')
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # cv2.imwrite(save_path, np.hstack(vis_frames)) #vis_frams:len: 4


        front = vis_frames[0]
        split_images = np.split(front, 3, axis=0)
        save_path_channel1 = os.path.join(self.workspace, 'val_nor', f'{name}.png')
        os.makedirs(os.path.dirname(save_path_channel1), exist_ok=True)
        cv2.imwrite(save_path_channel1, split_images[0])
        save_path_channel2 = os.path.join(self.workspace, 'val_app', f'{name}.png')
        os.makedirs(os.path.dirname(save_path_channel2), exist_ok=True)
        cv2.imwrite(save_path_channel2, split_images[1])
        #

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if full:
            state['optimizer_geo'] = self.optimizer_geo.state_dict()
            state['optimizer_app'] = self.optimizer_app.state_dict()
            state['optimizer_ani'] = self.optimizer_ani.state_dict()
            state['lr_scheduler_geo'] = self.lr_scheduler_geo.state_dict()
            state['lr_scheduler_app'] = self.lr_scheduler_app.state_dict()
            state['lr_scheduler_ani'] = self.lr_scheduler_ani.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}*.pth'))
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
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer_geo and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer_geo.load_state_dict(checkpoint_dict['optimizer_geo'])
                self.log("[INFO] loaded optimizer_geo.")
            except:
                self.log("[WARN] Failed to load optimizer_geo.")

        if self.optimizer_app and 'optimizer_app' in checkpoint_dict:
            try:
                self.optimizer_app.load_state_dict(checkpoint_dict['optimizer_app'])
                self.log("[INFO] loaded optimizer_app.")
            except:
                self.log("[WARN] Failed to load optimizer_geo.")

        if self.optimizer_ani and 'optimizer_ani' in checkpoint_dict:
            try:
                self.optimizer_ani.load_state_dict(checkpoint_dict['optimizer_ani'])
                self.log("[INFO] loaded optimizer_ani.")
            except:
                self.log("[WARN] Failed to load optimizer_ani.")

        if self.lr_scheduler_geo and 'lr_scheduler_geo' in checkpoint_dict:
            try:
                self.lr_scheduler_geo.load_state_dict(checkpoint_dict['lr_scheduler_geo'])
                self.log("[INFO] loaded scheduler_geo.")
            except:
                self.log("[WARN] Failed to load scheduler_geo.")


        if self.lr_scheduler_app and 'lr_scheduler_app' in checkpoint_dict:
            try:
                self.lr_scheduler_app.load_state_dict(checkpoint_dict['lr_scheduler_app'])
                self.log("[INFO] loaded scheduler_app.")
            except:
                self.log("[WARN] Failed to load scheduler_app.")



        if self.lr_scheduler_ani and 'lr_scheduler_ani' in checkpoint_dict:
            try:
                self.lr_scheduler_ani.load_state_dict(checkpoint_dict['lr_scheduler_ani'])
                self.log("[INFO] loaded scheduler_ani.")
            except:
                self.log("[WARN] Failed to load scheduler_ani.")


        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
