import argparse
import subprocess
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
import lightning.pytorch as pl


from net.model import UniRain
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from utils.image_utils import crop_img


# ==========================================
# 1. 通用 Dataset (适配 4 种任务)
# ==========================================
class PromptTestDataset(Dataset):
    def __init__(self, root_dir, task_name):
        super().__init__()
        self.task_name = task_name

        # 假设你的数据结构是:
        # /root/autodl-tmp/dataset/Test/haze/input
        # /root/autodl-tmp/dataset/Test/haze/target
        self.input_root = os.path.join(root_dir, task_name, 'input')
        self.target_root = os.path.join(root_dir, task_name, 'target')

        if not os.path.exists(self.input_root):
            # 如果某个任务文件夹不存在，抛出异常或打印警告
            print(f"⚠️ Warning: Directory not found: {self.input_root}")
            self.sample_ids = []
        else:
            self.sample_ids = []
            valid_exts = {'.jpg', '.png', '.jpeg', '.bmp'}

            files = sorted([f for f in os.listdir(self.input_root) if os.path.splitext(f)[-1].lower() in valid_exts])
            for fname in files:
                target_path = os.path.join(self.target_root, fname)
                if not os.path.exists(target_path):
                    target_path = None  # 没有GT只做推理

                self.sample_ids.append({
                    "name": os.path.splitext(fname)[0],
                    "input": os.path.join(self.input_root, fname),
                    "target": target_path
                })
            print(f"✅ [{task_name.upper()}] Loaded {len(self.sample_ids)} images.")

        self.toTensor = ToTensor()

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]

        # 读取并预处理 Input
        inp_img = np.array(Image.open(sample["input"]).convert('RGB'))
        inp_img = crop_img(inp_img, base=16)
        inp_tensor = self.toTensor(inp_img)

        # 读取 Target (如果有)
        if sample["target"] is not None:
            tar_img = np.array(Image.open(sample["target"]).convert('RGB'))
            tar_img = crop_img(tar_img, base=16)
            tar_tensor = self.toTensor(tar_img)
        else:
            tar_tensor = torch.zeros_like(inp_tensor)

        return [sample["name"]], inp_tensor, tar_tensor

    def __len__(self):
        return len(self.sample_ids)


# ==========================================
# 2. 模型定义 (推理模式)
# ==========================================
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 确保这里参数与训练时一致 (例如 decoder=True)
        self.net = UniRain()

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. 单个任务的测试函数
# ==========================================
def test_single_task(net, dataset, output_path):
    # 创建保存目录: results/haze/
    os.makedirs(output_path, exist_ok=True)

    if len(dataset) == 0:
        return 0.0, 0.0  # 空数据集返回0

    testloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        # 使用 tqdm 显示进度条
        for ([name], degrad_patch, clean_patch) in tqdm(testloader, desc=f"Testing {dataset.task_name}"):
            degrad_patch = degrad_patch.cuda()
            clean_patch = clean_patch.cuda()

            # 推理
            restored = net(degrad_patch)
            restored = torch.clamp(restored, 0, 1)

            # 计算指标 (仅当有GT时)
            if clean_patch.sum() > 0:
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr_meter.update(temp_psnr, N)
                ssim_meter.update(temp_ssim, N)

            # 保存图片
            save_path = os.path.join(output_path, name[0] + '.png')
            save_image_tensor(restored, save_path)

    return psnr_meter.avg, ssim_meter.avg


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集根目录 (下面应该包含 haze/input, rain/input 等)
    parser.add_argument('--data_root', type=str, default="/root/autodl-tmp/dataset/Test/", help='Dataset Root')
    # 权重路径
    parser.add_argument('--ckpt_path', type=str,
                        default="ckpt/",
                        help='Checkpoint path')
    # 输出根目录
    parser.add_argument('--output_root', type=str, default="results_all/", help='Output Root')
    parser.add_argument('--cuda', type=int, default=0)

    opt = parser.parse_args()

    torch.cuda.set_device(opt.cuda)

    # --- 加载模型 (只需加载一次) ---
    print(f"Loading Model from: {opt.ckpt_path}")
    # 注意: strict=False 用于忽略 CoBa 等多余参数
    model = PromptIRModel.load_from_checkpoint(opt.ckpt_path, strict=False)
    model.cuda()
    model.eval()

    # --- 定义 4 个任务 ---
    TASKS = ['drs', 'drd', 'nrs', 'nrd']

    # 用于存储最终结果
    results = {}

    print("\n" + "=" * 50)
    print("STARTING BATCH EVALUATION")
    print("=" * 50 + "\n")

    # --- 循环测试 ---
    for task in TASKS:
        print(f"Processing Task: {task} ...")

        # 1. 准备 Dataset
        dataset = PromptTestDataset(opt.data_root, task)

        # 2. 设置输出路径 (例如: results_all/haze/)
        task_out_dir = os.path.join(opt.output_root, task)

        # 3. 执行测试
        psnr, ssim = test_single_task(model, dataset, task_out_dir)

        # 4. 记录结果
        results[task] = {'PSNR': psnr, 'SSIM': ssim}
        print(f"👉 {task}: PSNR={psnr:.2f}, SSIM={ssim:.4f}\n")

    # --- 打印最终汇总报告 ---
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Task':<15} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 40)

    avg_psnr = 0.0
    avg_ssim = 0.0
    count = 0

    for task in TASKS:
        res = results[task]
        print(f"{task:<15} | {res['PSNR']:<10.2f} | {res['SSIM']:<10.4f}")

        if res['PSNR'] > 0:  # 排除空数据集
            avg_psnr += res['PSNR']
            avg_ssim += res['SSIM']
            count += 1

    print("-" * 40)
    if count > 0:
        print(f"{'AVERAGE':<15} | {avg_psnr / count:<10.2f} | {avg_ssim / count:<10.4f}")
    print("=" * 50)
    print(f"All images saved to: {opt.output_root}")