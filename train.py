import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


from utils.dataset_utils import PromptTrainDataset
from utils.image_utils import crop_img
from net.model import UniRain
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt


# =========================================================

# ==========================================
# 1. 验证集 Dataset
# ==========================================
class PromptValDataset(Dataset):
    def __init__(self, args, val_type='haze'):
        super().__init__()
        self.args = args
        self.val_type = val_type

        # 映射字典
        self.de_dict = {
            'drs': 0, 'drd': 1,
            'nrs': 2, 'nrd': 3
        }
        self.de_id = self.de_dict[val_type]

        self.input_root = os.path.join(args.val_dir, val_type, 'input')
        self.target_root = os.path.join(args.val_dir, val_type, 'target')

        if not os.path.exists(self.input_root):
            raise FileNotFoundError(f"Validation dir not found: {self.input_root}")

        self.sample_ids = []
        valid_exts = {'.jpg', '.png', '.jpeg', '.bmp'}

        files = sorted([f for f in os.listdir(self.input_root) if os.path.splitext(f)[-1].lower() in valid_exts])
        for fname in files:
            target_path = os.path.join(self.target_root, fname)
            if os.path.exists(target_path):
                self.sample_ids.append({
                    "name": os.path.splitext(fname)[0],
                    "input": os.path.join(self.input_root, fname),
                    "target": target_path
                })

        self.toTensor = ToTensor()
        print(f"✅ [Val] Loaded {len(self.sample_ids)} images for {val_type}")

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        try:
            inp = crop_img(np.array(Image.open(sample["input"]).convert('RGB')), base=16)
            tar = crop_img(np.array(Image.open(sample["target"]).convert('RGB')), base=16)

            h, w, _ = inp.shape
            ps = self.args.patch_size
            if h >= ps and w >= ps:
                top = (h - ps) // 2
                left = (w - ps) // 2
                inp = inp[top:top + ps, left:left + ps, :]
                tar = tar[top:top + ps, left:left + ps, :]

            return [sample["name"], self.de_id], self.toTensor(inp), self.toTensor(tar)
        except Exception as e:
            print(f"Error loading {sample['input']}: {e}")
            return self.__getitem__((idx + 1) % len(self.sample_ids))

    def __len__(self):
        return len(self.sample_ids)


# ==========================================
# 2. CoBaStatus
# ==========================================
class CoBaStatus:
    def __init__(self, num_tasks, history_length=10, tau=5, device="cpu"):
        self.num_tasks = num_tasks
        self.history_length = history_length
        self.tau = tau
        self.device = device
        self.history_valid_loss = None
        self.per_task_slope_history = None
        self.total_slope_history = None

    def update_valid_loss(self, valid_loss_per_task: torch.Tensor):
        valid_loss_per_task = valid_loss_per_task.detach().to(self.device, dtype=torch.float64)
        if self.history_valid_loss is None:
            self.history_valid_loss = valid_loss_per_task.unsqueeze(1)
        else:
            self.history_valid_loss = torch.cat([self.history_valid_loss, valid_loss_per_task.unsqueeze(1)], dim=1)

    def fit_slope(self, y: torch.Tensor) -> torch.Tensor:
        L = y.shape[0]
        if L < 2: return torch.tensor(0.0, device=self.device, dtype=torch.float64)
        x = torch.arange(L, device=self.device, dtype=torch.float64)
        X = torch.stack([x, torch.ones_like(x)], dim=1)
        try:
            w = torch.linalg.solve(X.T @ X, X.T @ y)
            slope = w[0]
        except RuntimeError:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)
        return slope.clamp(-1e3, 1e3)

    def compute_task_weight(self):
        loss_hist = self.history_valid_loss
        if loss_hist is None: return torch.ones(self.num_tasks, device=self.device) / self.num_tasks

        T = loss_hist.shape[1]
        W = min(self.history_length, T)
        loss_window = loss_hist[:, -W:]

        slopes = torch.zeros(self.num_tasks, dtype=torch.float64, device=self.device)
        for i in range(self.num_tasks):
            slopes[i] = self.fit_slope(loss_window[i])

        # RCS
        denom = slopes.abs().sum() + 1e-8
        rcs_logits = self.num_tasks * slopes / denom
        RCS = F.softmax(rcs_logits, dim=-1)

        # ACS
        if self.per_task_slope_history is None:
            self.per_task_slope_history = slopes.unsqueeze(1)
        else:
            self.per_task_slope_history = torch.cat([self.per_task_slope_history, slopes.unsqueeze(1)], dim=1)
        K = min(self.history_length, self.per_task_slope_history.shape[1])
        slope_window = self.per_task_slope_history[:, -K:]
        acs_logits = -K * slope_window / (slope_window.abs().sum(dim=1, keepdim=True) + 1e-8)
        ACS = F.softmax(acs_logits[:, -1], dim=-1)

        # DF
        max_loss = loss_window.max(dim=0).values
        total_slope = self.fit_slope(max_loss)
        if self.total_slope_history is None:
            self.total_slope_history = total_slope.unsqueeze(0)
        else:
            self.total_slope_history = torch.cat([self.total_slope_history, total_slope.unsqueeze(0)], dim=0)
        total_K = min(self.history_length, self.total_slope_history.shape[0])
        total_window = self.total_slope_history[-total_K:]
        df_logits = -total_K * total_window / (total_window.abs().sum() + 1e-8)
        DF = F.softmax(df_logits * self.tau, dim=-1)[-1]

        weight_logits = DF * RCS + (1.0 - DF) * ACS
        weight = F.softmax(weight_logits * self.num_tasks, dim=-1)
        return weight


# ==========================================
# 3. Model
# ==========================================
class UniRainModel(pl.LightningModule):
    def __init__(self, num_tasks=4, warmup_epochs=20, min_weight=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])

        self.net = UniRain()
        self.task_loss_fns = nn.ModuleDict({str(i): nn.L1Loss() for i in range(num_tasks)})

        self.num_tasks = num_tasks
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight

        # 任务名称与简写
        self.task_names = ['drs', 'drd', 'nrs', 'nrd']


        self.register_buffer("current_task_weights", torch.ones(num_tasks) / num_tasks)

        self.coba_inst = CoBaStatus(num_tasks=4)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        total_loss = 0.0
        batch_size = restored.shape[0]

        for task_id in range(self.num_tasks):
            mask = (de_id == task_id)
            if mask.sum() > 0:
                loss = self.task_loss_fns[str(task_id)](restored[mask], clean_patch[mask])
                weight = self.current_task_weights[task_id]
                total_loss += weight * loss * (mask.sum() / batch_size)

        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # 1. Loss for CoBa
        loss = F.l1_loss(restored, clean_patch)
        self.log(f"val_loss", loss, add_dataloader_idx=True, sync_dist=True)

        # 2. PSNR for Checkpoint
        mse = F.mse_loss(restored, clean_patch)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        self.log(f"val_psnr", psnr, add_dataloader_idx=True, sync_dist=True)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        valid_losses = []
        valid_psnrs = []

        # --- 收集 Metrics ---
        for i in range(self.num_tasks):
            loss_key = f"val_loss/dataloader_idx_{i}"
            psnr_key = f"val_psnr/dataloader_idx_{i}"

            if loss_key in metrics:
                valid_losses.append(metrics[loss_key])
                valid_psnrs.append(metrics.get(psnr_key, torch.tensor(0.0, device=self.device)))

                # 记录单独的 PSNR，方便 main 函数引用
                self.log(f"val_psnr_{self.task_names[i]}", metrics[psnr_key], sync_dist=True)
            else:
                return  # Skip if data incomplete

        v_loss_inst = torch.stack(valid_losses)

        # --- 计算并记录平均 PSNR (Avg) ---
        avg_psnr = torch.stack(valid_psnrs).mean()
        # 这个 val_avg_psnr 就是我们要放到文件名最后的东西
        self.log("val_avg_psnr", avg_psnr, prog_bar=True, sync_dist=True)

        # --- CoBa Update ---
        device = self.device
        self.coba_inst.device = device
        self.coba_inst.update_valid_loss(v_loss_inst)


        if self.current_epoch >= self.warmup_epochs:
            w_inst = self.coba_inst.compute_task_weight()


            final_w = F.softmax(w_inst * self.num_tasks, dim=-1)
            final_w = final_w * (1.0 - self.min_weight * self.num_tasks) + self.min_weight

            self.current_task_weights = final_w.float()

            for i in range(self.num_tasks):
                self.log(f"weights/{self.task_names[i]}", final_w[i], sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)
        return [optimizer], [scheduler]


# ==========================================
# 4. Main Function
# ==========================================
def main():
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="UniRain-Full")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers, pin_memory=True)

    # 验证集
    val_keys = ['drs', 'drd', 'nrs', 'nrd']
    val_loaders = []
    print("\n--- Preparing Validation Loaders ---")
    for k in val_keys:
        val_set = PromptValDataset(opt, val_type=k)
        val_loaders.append(DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1))

    model = UniRainModel(num_tasks=4, warmup_epochs=20)

    # --- Checkpoint 文件名模板 ---
    # {val_avg_psnr:.2f} 放在了字符串的最后面
    filename_fmt = (
        "epoch={epoch:02d}-"
        "drs={val_psnr_drs:.2f}-drd={val_psnr_drd:.2f}-"
        "nrs={val_psnr_nrs:.2f}-nrd={val_psnr_nrd:.2f}-"
        "Avg={val_avg_psnr:.2f}"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename=filename_fmt,  # 使用上面的格式
        monitor="val_avg_psnr",  # 依然根据平均 PSNR 选最好的
        mode="max",
        save_top_k=5,
        save_last=True,
        auto_insert_metric_name=False  # 必须设为 False
    )

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1
    )

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=val_loaders)


if __name__ == '__main__':
    main()