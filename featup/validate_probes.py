from os.path import join, exists

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, JaccardIndex

from featup.datasets.COCO import Coco
from featup.datasets.EmbeddingFile import EmbeddingFile
from featup.losses import ScaleAndShiftInvariantLoss
from featup.util import pca
from featup.util import remove_axes
from memory_profiler import profile
from hubconf import clipLarge, vit
from featup.datasets.util import get_dataset
from featup.util import get_transform
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm

def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

class LitPrototypeEvaluator(pl.LightningModule):
    def __init__(self, task, res, n_dim):
        super().__init__()

        self.task = task
        self.n_dim = n_dim

        self.n_classes = 0

        if self.task == 'seg':
            self.n_classes = 27
        elif self.task == 'depth':
            self.n_classes = 1

            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').cuda()
            self.midas.eval()
            self.midas_loss = ScaleAndShiftInvariantLoss()

            self.mse = 0
            self.ssil = 0
            self.steps = 0

        self.prototypes_buff = self.register_buffer("prototypes", torch.zeros(self.n_classes, n_dim))
        self.classifier = torch.nn.Conv2d(n_dim, self.n_classes, 1)

        self.prot_acc_metric = Accuracy(num_classes=self.n_classes, task="multiclass")
        self.prot_acc_buff = self.register_buffer("prot_acc", torch.tensor(0.0))
        self.prot_iou_metric = JaccardIndex(num_classes=self.n_classes, task="multiclass")
        self.prot_iou_buff = self.register_buffer("prot_iou", torch.tensor(0.0))

        self.linear_acc_metric = Accuracy(num_classes=self.n_classes, task="multiclass")
        self.linear_acc_buff = self.register_buffer("linear_acc", torch.tensor(0.0))
        self.linear_iou_metric = JaccardIndex(num_classes=self.n_classes, task="multiclass")
        self.linear_iou_buff = self.register_buffer("linear_iou", torch.tensor(0.0))

        self.ce = torch.nn.CrossEntropyLoss()

        self.prot_acc_metric_steps = [] 
        self.prot_iou_metric_steps = []  

        self.linear_acc_metric_steps = [] 
        self.linear_iou_metric_steps = []

        self.upsampler = None
        self.res = res
        self.featurizer = None
        self.chkpt_dir = None
        self.epoch_size = 0

    def get_prototypes(self, feats):
        b, c, h, w = feats.shape
        k = self.prototypes.shape[0]
        matches = torch.einsum("kc,bchw->kbhw", F.normalize(self.prototypes, dim=1), F.normalize(feats, dim=1)) \
            .reshape(k, -1).argmax(0)
        return self.prototypes[matches].reshape(b, h, w, c).permute(0, 3, 1, 2)

    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        label = batch['label']
        with torch.no_grad():
            feats = self.featurizer(imgs.to("cuda"))
        
        b, c, h, w = feats.shape #torch.Size([2, 384, 14, 14]) torch.Size([2, 768, 24, 24])

        small_labels = F.interpolate(
            label.unsqueeze(1).to(torch.float32),
            size=(feats.shape[2], feats.shape[3])).to(torch.int64)

        linear_preds = self.classifier(feats)

        if self.task == 'seg':
            flat_labels = small_labels.permute(0, 2, 3, 1).reshape(b * h * w)
            flat_linear_preds = linear_preds.permute(0, 2, 3, 1).reshape(b * h * w, -1)

            selected = flat_labels > -1
            linear_loss = self.ce(
                flat_linear_preds[selected],
                flat_labels[selected])
            loss = linear_loss
            self.log("linear_loss", linear_loss)
            self.log("loss", loss)

            for l in range(self.n_classes):
                self.prototypes[l] += feats.permute(0, 2, 3, 1).reshape(b * h * w, -1)[flat_labels == l].sum(dim=0)

            if self.global_step % 10 == 1 and self.trainer.is_global_zero:
                with torch.no_grad():
                    prots = self.get_prototypes(feats)
                    prot_loss = -(F.normalize(feats, dim=1) * F.normalize(prots, dim=1)).sum(1).mean()
                self.logger.experiment.add_scalar("prot_loss", prot_loss, self.global_step)

        elif self.task == 'depth':
            loss = self.midas_loss(linear_preds.squeeze(), small_labels.squeeze(),
                                   torch.ones_like(linear_preds.squeeze()))
            self.log('loss', loss)

        if self.global_step % 200 == 0 and self.trainer.is_global_zero:
            n_images = 1
            fig, axes = plt.subplots(4, n_images, figsize=(4 * n_images, 5 * 5), squeeze=False)

            prot_preds = torch.einsum("bchw,kc->bkhw",
                                      F.normalize(feats, dim=1),
                                      F.normalize(self.prototypes, dim=1, eps=1e-10))

            colorize = Coco.colorize_label if self.task == 'seg' else lambda x: x.detach().cpu()
            for i in range(n_images):
                feats_pca = pca([feats.cpu()])[0][0][i]
                feats_pca = feats_pca.permute(1,2,0)
                axes[0, i].imshow(feats_pca)
                axes[1, i].imshow(colorize(label[i]))
                if self.task == 'depth':
                    axes[2, i].imshow(colorize(linear_preds[i][0]))
                    axes[3, i].imshow(colorize(prot_preds[i][0]))
                elif self.task == 'seg':
                    axes[2, i].imshow(colorize(linear_preds.argmax(1)[i]))
                    axes[3, i].imshow(colorize(prot_preds.argmax(1)[i]))

            plt.tight_layout()
            remove_axes(axes)
            self.logger.experiment.add_figure('predictions', fig, self.global_step)
        if self.global_step % (self.epoch_size * 10) == 0:
            self.trainer.save_checkpoint(self.chkpt_dir[:-5] + '/' + self.chkpt_dir[:-5] + f'_{self.global_step}.ckpt')
        

        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            image = batch['img']
            label = batch['label']
            feats = self.featurizer(image.to("cuda"))

            if self.task == 'seg':
                if self.upsampler is not None:
                    feats = self.upsampler(image) #torch.Size([2, 1024, 384, 384])
                    if feats.shape[2] != self.res:
                        feats = torch.nn.functional.interpolate(feats, self.res, mode="bilinear") #torch.Size([2, 1024, 336, 336])
                label = F.interpolate(
                        label.to(torch.float32).unsqueeze(1), size=(self.res, self.res)).to(torch.int64).squeeze(1)
                b, h, w = label.shape
                prot_preds = torch.einsum(
                    "bchw,kc->bkhw",
                    F.normalize(feats, dim=1),
                    F.normalize(self.prototypes, dim=1, eps=1e-10)).argmax(1, keepdim=True)

                linear_preds = self.classifier(feats).argmax(1, keepdim=True)

                
                flat_labels = label.flatten()
                selected = flat_labels > -1
                flat_labels = flat_labels[selected]

                flat_prot_preds = F.interpolate(
                    prot_preds.to(torch.float32), (h, w)).to(torch.int64).flatten()[selected]
                self.prot_acc_metric.update(flat_prot_preds, flat_labels)
                self.prot_iou_metric.update(flat_prot_preds, flat_labels)

                flat_linear_preds = F.interpolate(
                    linear_preds.to(torch.float32), (h, w)).to(torch.int64).flatten()[selected]
                self.linear_acc_metric.update(flat_linear_preds, flat_labels)
                self.linear_iou_metric.update(flat_linear_preds, flat_labels)

            elif self.task == 'depth':
                linear_preds = self.classifier(feats)
                small_labels = F.interpolate(
                    label.unsqueeze(1).to(torch.float32),
                    size=(feats.shape[2], feats.shape[3])).to(torch.int64)
                mse = (small_labels - linear_preds).pow(2).mean()
                midas_l = self.midas_loss(linear_preds.squeeze(), small_labels.squeeze(),
                                          torch.ones_like(linear_preds.squeeze()))
                self.mse += mse.item()
                self.ssil += midas_l.item()

                self.steps += 1

        return None
    # NotImplementedError: Support for `validation_epoch_end` has been removed in v2.0.0. `LitPrototypeEvaluator` implements this method. You can use the `on_validation_epoch_end` hook instead. To access outputs, save them in-memory as instance attributes. You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520.
    def on_validation_epoch_end(self):
        self.prot_acc = self.prot_acc_metric.compute()
        self.prot_iou = self.prot_iou_metric.compute()
        self.linear_acc = self.linear_acc_metric.compute()
        self.linear_iou = self.linear_iou_metric.compute()
        
        # free up the memory
        # --> HERE STEP 3 <--
        self.prot_acc_metric_steps.clear()
        self.prot_iou_metric_steps.clear()
        self.linear_acc_metric_steps.clear()
        self.linear_iou_metric_steps.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=5e-3)

@hydra.main(config_path="configs", config_name="validate_train_probe.yaml")
def validate_linear_probes(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)
    log_dir = f"/home/jeeves/LowResCV/probes/0801-{cfg.task}-probe-{cfg.model_type}-validation"
    #chkpt_dir = f"/home/jeeves/LowResCV/probes/unnorm_{cfg.task}-probe-{cfg.model_type}.ckpt"
    chkptFile = 'vit-True-token-0.005-10-512_464.ckpt' #250

    chkpt_dir = "/home/jeeves/probes/vit-True-token-0.005-10-512/home/jeeves/probes/" + chkptFile
    val_dataset= dataset_with_transform("val", cfg)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    evaluator = LitPrototypeEvaluator(cfg.task, cfg.res, cfg.dim)
    
    # Load pretrained checkpoint if available
    if exists(chkpt_dir):
        checkpoint = torch.load(chkpt_dir)
        evaluator.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded checkpoint from {chkpt_dir}")
    else:
        print(f"No checkpoint found at {chkpt_dir}. Training from scratch.")
        return
    
    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)

    evaluator.featurizer = load_featurizer(cfg.model_type, cfg.activation_type, cfg.is_norm)
    
    upsample = True
    if upsample :
        if cfg.model_type == 'clip-large':
            upsampler = clipLarge(use_norm=cfg.is_norm)
        elif cfg.model_type == 'vit':
            #upsampler = torch.hub.load("mhamilton723/FeatUp", 'vit', use_norm=cfg.is_norm).to('cuda')
            upsampler = vit(use_norm=cfg.is_norm)
        upsampler.eval()
        evaluator.upsampler = upsampler

    trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=cfg.epochs,
        logger=tb_logger,
        log_every_n_steps=100,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=10,
    )

    # Set the evaluator for validation
    trainer.validate(evaluator, val_loader)
    
    result = {
        "Prototype Accuracy": float(evaluator.prot_acc),
        "Prototype mIoU": float(evaluator.prot_iou),
        "Linear Accuracy": float(evaluator.linear_acc),
        "Linear mIoU": float(evaluator.linear_iou),
        "Model": cfg.model_type
    }
    print(f"{chkptFile} probes validation:")

    print(result)


def load_featurizer(model_type, activation="token", is_norm=False):
    featurizer, patch_size, dim = get_featurizer(model_type, activation, num_classes=27)
    if is_norm:
        featurizer = torch.nn.Sequential(featurizer, ChannelNorm(dim))
    featurizer = featurizer.eval()
    return featurizer


def dataset_with_transform(split, cfg):
    dataset = get_dataset(
        cfg.pytorch_data_dir,
        cfg.dataset,
        split,
        transform=get_transform(cfg.res, False, "center"),
        target_transform=get_transform(cfg.res, True, "center"),
        include_labels=True,
    )
    return dataset

if __name__ == "__main__":
    validate_linear_probes()
