import gc
import os
from os.path import join, exists

import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from featup.datasets.JitteredImage import apply_jitter, sample_transform
from featup.datasets.util import get_dataset, SingleImageDataset
from featup.downsamplers import SimpleDownsampler, AttentionDownsampler
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.losses import TVLoss, SampledCRFLoss, entropy
from featup.upsamplers import get_upsampler
from featup.util import pca, RollingAvg, unnorm, prep_image
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, JaccardIndex
from featup.util import get_transform
from PIL import Image
import os

torch.multiprocessing.set_sharing_strategy('file_system')


class ScaleNet(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = torch.nn.Conv2d(dim, 1, 1)
        with torch.no_grad():
            self.net.weight.copy_(self.net.weight * .1)
            self.net.bias.copy_(self.net.bias * .1)

    def forward(self, x):
        return torch.exp(self.net(x) + .1).clamp_min(.0001)


class JBUFeatUp(pl.LightningModule):
    def __init__(self,
                 model_type,
                 activation_type,
                 n_jitters,
                 max_pad,
                 max_zoom,
                 kernel_size,
                 final_size,
                 lr,
                 random_projection,
                 predicted_uncertainty,
                 crf_weight,
                 filter_ent_weight,
                 tv_weight,
                 upsampler,
                 downsampler,
                 chkpt_dir,
                 is_norm,
                 res,
                 validate_image
                 ):
        super().__init__()
        self.model_type = model_type
        self.activation_type = activation_type
        self.n_jitters = n_jitters
        self.max_pad = max_pad
        self.max_zoom = max_zoom
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.lr = lr
        self.random_projection = random_projection
        self.predicted_uncertainty = predicted_uncertainty
        self.crf_weight = crf_weight
        self.filter_ent_weight = filter_ent_weight
        self.tv_weight = tv_weight
        self.chkpt_dir = chkpt_dir
        self.is_norm = is_norm
        
        self.model, self.patch_size, self.dim = get_featurizer(model_type, activation_type, num_classes=1000)
        for p in self.model.parameters():
            p.requires_grad = False
        if self.is_norm:
            self.model = torch.nn.Sequential(self.model, ChannelNorm(self.dim))

        self.validation_model = self.model
        self.validation_model.eval()

        self.upsampler = get_upsampler(upsampler, self.dim)

        if downsampler == 'simple':
            self.downsampler = SimpleDownsampler(self.kernel_size, self.final_size)
        elif downsampler == 'attention':
            self.downsampler = AttentionDownsampler(self.dim, self.kernel_size, self.final_size, blur_attn=True)
            self.downsampler_2x = AttentionDownsampler(self.dim, self.kernel_size, self.final_size, blur_attn=True)
            #self.downsampler_4x = AttentionDownsampler(self.dim, self.kernel_size, self.final_size, blur_attn=True)
        else:
            raise ValueError(f"Unknown downsampler {downsampler}")

        if self.predicted_uncertainty:
            self.scale_net = ScaleNet(self.dim)
            self.scale_net_2x = ScaleNet(self.dim)
            #self.scale_net_4x = ScaleNet(self.dim)

        self.avg = RollingAvg(20)

        self.crf = SampledCRFLoss(
            alpha=.1,
            beta=.15,
            gamma=.005,
            w1=10.0,
            w2=3.0,
            shift=0.00,
            n_samples=1000)
        self.tv = TVLoss()

        self.automatic_optimization = False
        
        self.linear_acc_metric = Accuracy(num_classes=27, task="multiclass")
        self.linear_acc_buff = self.register_buffer("linear_acc", torch.tensor(0.0))
        self.linear_iou_metric = JaccardIndex(num_classes=27, task="multiclass")
        self.linear_iou_buff = self.register_buffer("linear_iou", torch.tensor(0.0))
        self.res = res
        self.validate_image = validate_image

    def forward(self, x):
        return self.upsampler(self.model(x))

    def project(self, feats, proj):
        if proj is None:
            return feats
        else:
            return torch.einsum("bchw,bcd->bdhw", feats, proj)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        with torch.no_grad():
            if type(batch) == dict:
                img = batch['img']
            else:
                img, _ = batch
            lr_feats = self.model(img)

        full_rec_loss = 0.0
        full_crf_loss = 0.0
        full_entropy_loss = 0.0
        full_tv_loss = 0.0
        full_total_loss = 0.0
        for i in range(self.n_jitters):
            hr_feats_2x, hr_feats = self.upsampler.forward_stageloss(lr_feats, img)

            if hr_feats.shape[2] != img.shape[2]:
                hr_feats = torch.nn.functional.interpolate(hr_feats, img.shape[2:], mode="bilinear")
                hr_feats_2x = torch.nn.functional.interpolate(hr_feats_2x, img.shape[2:], mode="bilinear")
                #hr_feats_4x = torch.nn.functional.interpolate(hr_feats_4x, img.shape[2:], mode="bilinear")

            with torch.no_grad():
                transform_params = sample_transform(
                    True, self.max_pad, self.max_zoom, img.shape[2], img.shape[3])
                jit_img = apply_jitter(img, self.max_pad, transform_params)
                lr_jit_feats = self.model(jit_img)

            if self.random_projection is not None:
                proj = torch.randn(lr_feats.shape[0],
                                   lr_feats.shape[1],
                                   self.random_projection, device=lr_feats.device)
                proj /= proj.square().sum(1, keepdim=True).sqrt()
            else:
                proj = None

            hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
            proj_hr_feats = self.project(hr_jit_feats, proj)

            hr_jit_feats_2x = apply_jitter(hr_feats_2x, self.max_pad, transform_params)
            proj_hr_feats_2x = self.project(hr_jit_feats_2x, proj)
            
            #hr_jit_feats_4x = apply_jitter(hr_feats_4x, self.max_pad, transform_params)
            #proj_hr_feats_4x = self.project(hr_jit_feats_4x, proj)

            down_jit_feats = self.project(self.downsampler(hr_jit_feats, jit_img), proj)
            down_jit_feats_2x = self.project(self.downsampler_2x(hr_jit_feats_2x, jit_img), proj)
            #down_jit_feats_4x = self.project(self.downsampler_4x(hr_jit_feats_4x, jit_img), proj)

            if self.predicted_uncertainty:
                scales = self.scale_net(lr_jit_feats)
                scale_factor = (1 / (2 * scales ** 2))
                mse = (down_jit_feats - self.project(lr_jit_feats, proj)).square()
                rec_loss = (scale_factor * mse + scales.log()).mean() / self.n_jitters

                scales_2x = self.scale_net_2x(lr_jit_feats)
                scale_factor_2x = (1 / (2 * scales_2x ** 2))
                mse_2x = (down_jit_feats_2x - self.project(lr_jit_feats, proj)).square()
                rec_loss_2x = (scale_factor_2x * mse_2x + scales_2x.log()).mean() / self.n_jitters

                # scales_4x = self.scale_net_4x(lr_jit_feats)
                # scale_factor_4x = (1 / (2 * scales_4x ** 2))
                # mse_4x = (down_jit_feats_4x - self.project(lr_jit_feats, proj)).square()
                # rec_loss_4x = (scale_factor_4x * mse_4x + scales_4x.log()).mean() / self.n_jitters
            else:
                rec_loss = (self.project(lr_jit_feats, proj) - down_jit_feats).square().mean() / self.n_jitters
                rec_loss_2x = (self.project(lr_jit_feats, proj) - down_jit_feats_2x).square().mean() / self.n_jitters
                #rec_loss_4x = (self.project(lr_jit_feats, proj) - down_jit_feats_4x).square().mean() / self.n_jitters

            full_rec_loss += rec_loss.item() + rec_loss_2x.item() #+ rec_loss_4x.item()

            if self.crf_weight > 0 and i == 0:
                crf_loss = self.crf(img, proj_hr_feats)
                crf_loss_2x = self.crf(img, proj_hr_feats_2x)
                #crf_loss_4x = self.crf(img, proj_hr_feats_4x)
                full_crf_loss += crf_loss.item() + crf_loss_2x.item() #+ crf_loss_4x.item()
            else:
                crf_loss = 0.0
                crf_loss_2x = 0.0
                #crf_loss_4x = 0.0

            if self.filter_ent_weight > 0.0:
                entropy_loss = entropy(self.downsampler.get_kernel())
                entropy_loss_2x = entropy(self.downsampler_2x.get_kernel())
                #entropy_loss_4x = entropy(self.downsampler_4x.get_kernel())
                full_entropy_loss += entropy_loss.item() + entropy_loss_2x.item() #+ entropy_loss_4x.item()
            else:
                entropy_loss = 0
                entropy_loss_2x = 0
                #entropy_loss_4x = 0

            if self.tv_weight > 0 and i == 0:
                tv_loss = self.tv(proj_hr_feats.square().sum(1, keepdim=True))
                tv_loss_2x = self.tv(proj_hr_feats_2x.square().sum(1, keepdim=True))
                #tv_loss_4x = self.tv(proj_hr_feats_4x.square().sum(1, keepdim=True))
                full_tv_loss += tv_loss.item() + tv_loss_2x.item() #+ tv_loss_4x.item()
            else:
                tv_loss_2x = 0.0
                #tv_loss_4x = 0.0
                tv_loss = 0.0
            #tv == 0, ent == 0
            loss = rec_loss + self.crf_weight * crf_loss + self.tv_weight * tv_loss - self.filter_ent_weight * entropy_loss
            loss_2x = rec_loss_2x  + self.crf_weight * crf_loss_2x + self.tv_weight * tv_loss_2x - self.filter_ent_weight * entropy_loss_2x
            #loss_4x = rec_loss_4x + self.crf_weight * crf_loss_4x + self.tv_weight * tv_loss_4x - self.filter_ent_weight * entropy_loss_4x
            loss_all = loss + loss_2x #+ loss_4x
            full_total_loss += loss_all.item()
            self.manual_backward(loss_all)


        self.avg.add("loss/ent", full_entropy_loss)
        self.avg.add("loss/tv", full_tv_loss)
        self.avg.add("loss/total", full_total_loss)

        self.avg.add("loss/crf_4x", full_crf_loss)
        self.avg.add("loss/crf_2x", crf_loss_2x)
        #self.avg.add("loss/crf_4x", crf_loss_4x)

        self.avg.add("loss/rec_4x", rec_loss.item())
        self.avg.add("loss/rec_2x", rec_loss_2x.item())
        #self.avg.add("loss/rec_4x", rec_loss_4x.item())

        if self.global_step % 500 == 0:
            self.trainer.save_checkpoint(self.chkpt_dir[:-5] + '/' + self.chkpt_dir[:-5] + f'_{self.global_step}.ckpt')

        self.avg.logall(self.log)
        if self.global_step < 10:
            self.clip_gradients(opt, gradient_clip_val=.0001, gradient_clip_algorithm="norm")

        opt.step()

        return None

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.global_step % 1 == 0:
                img = self.validate_image.to(self.device)
                lr_feats = self.validation_model(img)

                hr_feats_2x, hr_feats = self.upsampler.forward_stageloss(lr_feats, img)

                if hr_feats.shape[2] != img.shape[2]:
                    hr_feats = torch.nn.functional.interpolate(hr_feats, img.shape[2:], mode="bilinear")
                    hr_feats_2x = torch.nn.functional.interpolate(hr_feats_2x, img.shape[2:], mode="bilinear")
                    #hr_feats_4x = torch.nn.functional.interpolate(hr_feats_4x, img.shape[2:], mode="bilinear")

                transform_params = sample_transform(
                    True, self.max_pad, self.max_zoom, img.shape[2], img.shape[3])
                jit_img = apply_jitter(img, self.max_pad, transform_params)
                lr_jit_feats = self.model(jit_img)

                if self.random_projection is not None:
                    proj = torch.randn(lr_feats.shape[0],
                                       lr_feats.shape[1],
                                       self.random_projection, device=lr_feats.device)
                    proj /= proj.square().sum(1, keepdim=True).sqrt()
                else:
                    proj = None

                scales = self.scale_net(lr_jit_feats)

                writer = self.logger.experiment

                hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
                down_jit_feats = self.downsampler(hr_jit_feats, jit_img)

                [red_lr_feats], fit_pca = pca([lr_feats[0].unsqueeze(0)])
                [red_hr_feats], _ = pca([hr_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_hr_feats_2x], _ = pca([hr_feats_2x[0].unsqueeze(0)], fit_pca=fit_pca)
                #[red_hr_feats_4x], _ = pca([hr_feats_4x[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_lr_jit_feats], _ = pca([lr_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_hr_jit_feats], _ = pca([hr_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_down_jit_feats], _ = pca([down_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)

                writer.add_image("viz/image", unnorm(img[0].unsqueeze(0))[0], self.global_step)
                writer.add_image("viz/lr_feats", red_lr_feats[0], self.global_step)
                writer.add_image("viz/hr_feats", red_hr_feats[0], self.global_step)
                writer.add_image("viz/hr_feats_2x", red_hr_feats_2x[0], self.global_step)
                #writer.add_image("viz/hr_feats_4x", red_hr_feats_4x[0], self.global_step)
                writer.add_image("jit_viz/jit_image", unnorm(jit_img[0].unsqueeze(0))[0], self.global_step)
                writer.add_image("jit_viz/lr_jit_feats", red_lr_jit_feats[0], self.global_step)
                writer.add_image("jit_viz/hr_jit_feats", red_hr_jit_feats[0], self.global_step)
                writer.add_image("jit_viz/down_jit_feats", red_down_jit_feats[0], self.global_step)

                norm_scales = scales[0]
                norm_scales /= scales.max()
                writer.add_image("scales", norm_scales, self.global_step)
                writer.add_histogram("scales hist", scales, self.global_step)

                if isinstance(self.downsampler, SimpleDownsampler):
                    writer.add_image(
                        "down/filter",
                        prep_image(self.downsampler.get_kernel().squeeze(), subtract_min=False),
                        self.global_step)

                if isinstance(self.downsampler, AttentionDownsampler):
                    writer.add_image(
                        "down/att",
                        prep_image(self.downsampler.forward_attention(hr_feats, None)[0]),
                        self.global_step)
                    writer.add_image(
                        "down/w",
                        prep_image(self.downsampler.w.clone().squeeze()),
                        self.global_step)
                    writer.add_image(
                        "down/b",
                        prep_image(self.downsampler.b.clone().squeeze()),
                        self.global_step)
                        
                    writer.add_image(
                        "down_2x/att",
                        prep_image(self.downsampler_2x.forward_attention(hr_feats, None)[0]),
                        self.global_step)
                    writer.add_image(
                        "down_2x/w",
                        prep_image(self.downsampler_2x.w.clone().squeeze()),
                        self.global_step)
                    writer.add_image(
                        "down_2x/b",
                        prep_image(self.downsampler_2x.b.clone().squeeze()),
                        self.global_step)

                    # writer.add_image(
                    #     "down_4x/att",
                    #     prep_image(self.downsampler_4x.forward_attention(hr_feats, None)[0]),
                    #     self.global_step)
                    # writer.add_image(
                    #     "down_4x/w",
                    #     prep_image(self.downsampler_4x.w.clone().squeeze()),
                    #     self.global_step)
                    # writer.add_image(
                    #     "down_4x/b",
                    #     prep_image(self.downsampler_4x.b.clone().squeeze()),
                    #     self.global_step)

                writer.flush()

    def configure_optimizers(self):
        all_params = []
        all_params.extend(list(self.downsampler.parameters()))
        all_params.extend(list(self.downsampler_2x.parameters()))
        #all_params.extend(list(self.downsampler_4x.parameters()))
        all_params.extend(list(self.upsampler.parameters()))

        if self.predicted_uncertainty:
            all_params.extend(list(self.scale_net.parameters()))
            all_params.extend(list(self.scale_net_2x.parameters()))
            #all_params.extend(list(self.scale_net_4x.parameters()))

        return torch.optim.NAdam(all_params, lr=self.lr)


@hydra.main(config_path="configs", config_name="vdim_upsampler")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)

    if cfg.model_type == "dinov2":
        final_size = 16
        kernel_size = 14
    elif cfg.model_type == "clip-large":
        final_size = 24
        kernel_size = 14
    elif cfg.model_type == "siglip":
        final_size = 28
        kernel_size = 14
    else:
        final_size = 14
        kernel_size = 16

    name = (f"{cfg.model_type}_{cfg.upsampler_type}_"
            f"{cfg.train_dataset}_{cfg.downsampler_type}_"
            f"crf_{cfg.crf_weight}_tv_{cfg.tv_weight}"
            f"_ent_{cfg.filter_ent_weight}")

    log_dir = join(cfg.output_root, f"../logs/jbu/muti-stage-{name}-{cfg.lr}-{cfg.is_norm}-{cfg.max_pad}-{cfg.n_jitters}-{cfg.epochs}")
    chkpt = join(cfg.output_root, f"../checkpoints/muti-stage-{name}-{cfg.lr}-{cfg.is_norm}-{cfg.max_pad}-{cfg.n_jitters}-{cfg.epochs}.ckpt")
    os.makedirs(log_dir, exist_ok=True)

    image = Image.open(f"./sample-images/scene.png")
    transform = get_transform(cfg.res, False, "center")
    validate_image = transform(image.convert("RGB")).unsqueeze(0)

    model = JBUFeatUp(
        model_type=cfg.model_type,
        activation_type=cfg.activation_type,
        n_jitters=cfg.n_jitters,
        max_pad=cfg.max_pad,
        max_zoom=cfg.max_zoom,
        kernel_size=kernel_size,
        final_size=final_size,
        lr=cfg.lr,
        random_projection=cfg.random_projection,
        predicted_uncertainty=cfg.outlier_detection,
        crf_weight=cfg.crf_weight,
        filter_ent_weight=cfg.filter_ent_weight,
        tv_weight=cfg.tv_weight,
        upsampler=cfg.upsampler_type,
        downsampler=cfg.downsampler_type,
        is_norm=cfg.is_norm,
        chkpt_dir=chkpt,
        res = cfg.res,
        validate_image = validate_image
    )

    train_dataset= dataset_with_transform("train", cfg)
    val_dataset= dataset_with_transform("val", cfg)
    
    loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)
    callbacks = [ModelCheckpoint(chkpt[:-5], every_n_epochs=1)]

    trainer = Trainer(
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_true",
        devices= cfg.num_gpus,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        check_val_every_n_epoch=None,
        val_check_interval=50,
        log_every_n_steps=1,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    trainer.fit(model, loader, val_loader)
    trainer.save_checkpoint(chkpt)

def dataset_with_transform(split, cfg):
    dataset_split = cfg.train_dataset if split == "train" else cfg.val_dataset
    dataset = get_dataset(
        cfg.pytorch_data_dir,
        dataset_split,
        split,
        transform=get_transform(cfg.res, False, "center"),
        target_transform=get_transform(cfg.res, True, "center"),
        include_labels=True,
    )
    return dataset

if __name__ == "__main__":
    my_app()