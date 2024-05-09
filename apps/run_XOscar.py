import os
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['DIFFUSERS_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1'
import sys
sys.path.append("")
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.provider import ViewDataset
from lib.trainer_XOscar import *
from lib.dlmesh_XOscar import DLMesh
from lib.common.utils import load_config

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/debug.yaml", help='Path to config file')
    parser.add_argument('--name', type=str, default="debug", help="file name")
    # parser.add_argument('--mesh', type=str, required=True, help="mesh template, must be obj format")
    parser.add_argument('--text', default="", help="text prompt")
    # parser.add_argument('--negative', default='', help="negative text prompt")
    args = parser.parse_args()

    cfg = load_config(args.config, 'configs/default.yaml')

    cfg.merge_from_list([
        'name', args.name,
        'text', args.text,
    ])
    # cfg.model.merge_from_list(['mesh', args.mesh])
    # cfg.training.merge_from_list(['workspace', args.workspace])
    cfg.freeze()

    seed_everything(cfg.seed)

    def build_dataloader(phase):
        """
        Args:
            phase: str one of ['train', 'test' 'val']
        Returns:
        """
        size = 4 if phase == 'val' else 100
        dataset = ViewDataset(cfg.data, device=device, type=phase, size=size)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def configure_guidance():
        opt = cfg.guidance
        if opt.name == 'sd':
            from lib.guidance.sd_XOscar import StableDiffusion
            return StableDiffusion(device, cfg.fp16, opt.vram_O, opt.sd_version)
        elif opt.name == 'if':
            from lib.guidance.deepfloyd import IF
            return IF(device, opt.vram_O)
        else:
            from lib.guidance.clip import CLIP
            return CLIP(device)

    def configure_optimizer():
        opt = cfg.training
        if opt.optim == 'adan':
            from lib.common.optimizer import Adan
            optimizer_geo = lambda model: Adan(model.get_geo_params(0.1 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
            optimizer_app = lambda model: Adan(model.get_app_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
            optimizer_ani = lambda model: Adan(model.get_ani_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            optimizer_geo = lambda model: torch.optim.Adam(model.get_geo_params(0.1 * opt.lr), betas=(0.9, 0.99), eps=1e-15)
            optimizer_app = lambda model: torch.optim.Adam(model.get_app_params(5 * opt.lr), betas=(0.9, 0.99), eps=1e-15)
            optimizer_ani = lambda model: torch.optim.Adam(model.get_ani_params(5 * opt.lr), betas=(0.9, 0.99), eps=1e-15)

        scheduler_geo = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x / opt.iters_geo, 1))
        scheduler_app = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.3 ** min(x / opt.iters_app, 1))
        scheduler_ani = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x / (opt.iters_geo+opt.iters_app+opt.iters_ani), 1))
        return optimizer_geo, optimizer_app, optimizer_ani, scheduler_geo, scheduler_app, scheduler_ani

    model = DLMesh(cfg.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.test:
        trainer = Trainer(cfg.name,
                          text=cfg.text,
                          negative=cfg.negative,
                          dir_text=cfg.data.dir_text,
                          opt=cfg.training,
                          model=model,
                          guidance=None,
                          device=device,
                          fp16=cfg.fp16
                          )
        
        model.body_pose = model.body_pose_static.clone()
        test_loader = build_dataloader('test')

        trainer.test(test_loader)

        if cfg.save_mesh:
            trainer.save_mesh()

    else:
        train_loader = build_dataloader('train')

        optimizer_geo, optimizer_app, optimizer_ani, scheduler_geo, scheduler_app, scheduler_ani= configure_optimizer()
        try:
            guidance = configure_guidance()
        except:
            guidance = configure_guidance()
        trainer = Trainer(cfg.name,
                          text=cfg.text,
                          negative=cfg.negative,
                          dir_text=cfg.data.dir_text,
                          opt=cfg.training,
                          model=model,
                          guidance=guidance,
                          device=device,
                          optimizer=[optimizer_geo, optimizer_app, optimizer_ani],
                          fp16=cfg.fp16,
                          lr_scheduler=[scheduler_geo, scheduler_app, scheduler_ani],
                          scheduler_update_every_step=True
                          )
        if os.path.exists(cfg.data.image):
            trainer.default_view_data = train_loader.dataset.get_default_view_data()

        valid_loader = build_dataloader('val')
        max_epoch = np.ceil(cfg.training.iters / (len(train_loader) * train_loader.batch_size)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        # test
        model.body_pose = model.body_pose_static.clone()
        test_loader = build_dataloader('test')
        trainer.test(test_loader)
        trainer.save_mesh()
