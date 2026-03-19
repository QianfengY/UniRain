import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.sample_ids = []

        # 1. 定义你的6种退化类型及其对应的ID
        self.de_dict = {
            'drs': 0,
            'drd': 1,
            'nrs': 2,
            'nrd': 3
        }

        # 获取传入参数中需要训练的类型 (例如 args.de_type = ['rain', 'snow'])
        self.de_type = self.args.de_type
        print(f"Training with degradation types: {self.de_type}")

        # 2. 初始化数据列表
        self._init_ids()

        # 定义裁剪变换
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        self.sample_ids = []

        # 遍历参数中指定的所有退化类型
        for de_name in self.de_type:
            if de_name not in self.de_dict:
                continue

            de_id = self.de_dict[de_name]

            # 构造路径： data_dir/type_name/input 和 data_dir/type_name/target
            # 例如: ./data/rain/input/
            input_root = os.path.join(self.args.data_file_dir, de_name, 'input')
            target_root = os.path.join(self.args.data_file_dir, de_name, 'target')

            if not os.path.exists(input_root) or not os.path.exists(target_root):
                print(f"Warning: Directory not found for {de_name}: {input_root}")
                continue

            # 获取所有图片文件
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
            file_names = sorted(
                [f for f in os.listdir(input_root) if os.path.splitext(f)[-1].lower() in valid_extensions])

            print(f"Found {len(file_names)} images for {de_name}")

            # 将每一对图片的信息存入列表
            for fname in file_names:
                self.sample_ids.append({
                    "input_path": os.path.join(input_root, fname),
                    "target_path": os.path.join(target_root, fname),  # 假设target文件名和input一致
                    "clean_name": os.path.splitext(fname)[0],
                    "de_type": de_id
                })

        # 打乱数据
        random.shuffle(self.sample_ids)
        print(f"Total merged samples: {len(self.sample_ids)}")

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]

        # 确保图片尺寸大于 patch_size，否则不裁剪直接缩放或报错（这里假设图片够大）
        if H < self.args.patch_size or W < self.args.patch_size:
            # 如果图片比patch小，通常需要Resize，或者Padding。这里简单处理：随机选个起始点(虽然会越界，依赖random.randint的范围)
            # 更好的做法是Resize或者Padding，这里沿用原逻辑，但加上保护
            ind_H = 0 if H <= self.args.patch_size else random.randint(0, H - self.args.patch_size)
            ind_W = 0 if W <= self.args.patch_size else random.randint(0, W - self.args.patch_size)
        else:
            ind_H = random.randint(0, H - self.args.patch_size)
            ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        clean_name = sample["clean_name"]

        # 1. 读取图片
        # 这是一个成对训练 (Paired Training)
        degrad_img = crop_img(np.array(Image.open(sample["input_path"]).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(sample["target_path"]).convert('RGB')), base=16)

        # 2. 随机裁剪 (Crop)
        # 注意：原代码的crop_transform是基于PIL的，但这里读入的是numpy
        # 原代码逻辑中 _crop_patch 是手写的numpy裁剪，我们沿用它以保证 input/target 位置对应
        degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)

        # 3. 随机增强 (Augmentation - Flip/Rotate)
        # 传入的 random_augmentation 应该能同时处理两个patch以保持变换一致
        degrad_patch, clean_patch = random_augmentation(degrad_patch, clean_patch)

        # 4. 转为 Tensor
        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        # 返回格式: [文件名, 退化ID], 退化图, GT图
        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class PromptValDataset(Dataset):
    def __init__(self, args, val_type='haze'):
        super(PromptValDataset, self).__init__()
        self.args = args
        self.val_type = val_type

        # 1. 定义映射字典 (请确保这里与训练集的 ID 一致)
        # 注意：这里修正了拼写 (night 而不是 neight)，请根据你实际文件夹名调整
        self.de_dict = {
            'drs': 0,
            'drd': 1,
            'nrs': 2,
            'nrd': 3
        }

        # 检查输入的类型是否有效
        if val_type not in self.de_dict:
            raise ValueError(f"Invalid val_type: {val_type}. Must be one of {list(self.de_dict.keys())}")

        self.de_id = self.de_dict[val_type]

        # 2. 自动构建路径
        # 结构假设: args.val_dir / type_name / input
        #           args.val_dir / type_name / target
        self.input_root = os.path.join(args.val_dir, val_type, 'input')
        self.target_root = os.path.join(args.val_dir, val_type, 'target')

        if not os.path.exists(self.input_root) or not os.path.exists(self.target_root):
            raise FileNotFoundError(f"Validation directory not found: {self.input_root} or {self.target_root}")

        # 3. 扫描文件夹中的图片
        self.sample_ids = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}

        # 获取 input 文件夹下的所有有效图片
        file_names = sorted(
            [f for f in os.listdir(self.input_root) if os.path.splitext(f)[-1].lower() in valid_extensions])

        for fname in file_names:
            # 检查 target 文件夹里有没有同名文件
            if os.path.exists(os.path.join(self.target_root, fname)):
                self.sample_ids.append({
                    "name": os.path.splitext(fname)[0],
                    "input_path": os.path.join(self.input_root, fname),
                    "target_path": os.path.join(self.target_root, fname)
                })
            else:
                print(f"Warning: Missing target for {fname}, skipping.")

        print(f"✅ [Validation] Loaded {len(self.sample_ids)} samples for type: {val_type}")

        self.toTensor = ToTensor()

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]

        try:
            # 读取图片并转为 RGB
            # crop_img 通常用于确保尺寸是 16 的倍数，防止网络下采样报错
            inp_img = crop_img(np.array(Image.open(sample["input_path"]).convert('RGB')), base=16)
            tar_img = crop_img(np.array(Image.open(sample["target_path"]).convert('RGB')), base=16)

            # --- Center Crop (中心裁剪) ---
            # 验证集通常不做随机裁剪，而是取中间部分，保证结果可复现
            H, W, _ = inp_img.shape
            ps = self.args.patch_size

            # 如果图片比 patch_size 大，就裁剪中间；否则保留原图
            if H >= ps and W >= ps:
                top = (H - ps) // 2
                left = (W - ps) // 2
                inp_patch = inp_img[top:top + ps, left:left + ps, :]
                tar_patch = tar_img[top:top + ps, left:left + ps, :]
            else:
                inp_patch = inp_img
                tar_patch = tar_img

            # 转为 Tensor
            inp_tensor = self.toTensor(inp_patch)
            tar_tensor = self.toTensor(tar_patch)

            # 返回格式: [文件名, 任务ID], 输入图, 标签图
            return [sample["name"], self.de_id], inp_tensor, tar_tensor

        except Exception as e:
            print(f"Error loading validation sample {sample['input_path']}: {e}")
            # 如果出错，为了不中断训练，返回下一个样本
            return self.__getitem__((idx + 1) % len(self.sample_ids))

    def __len__(self):
        return len(self.sample_ids)


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            # print(name_list)
            print(self.args.derain_path)
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


