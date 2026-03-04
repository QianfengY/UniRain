#  <center>  [CVPR 2026] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization 

> [Paper] &emsp; [Supplemental Material]

> [Qianfeng Yang](https://qianfengy.github.io/) <sup>1</sup>, [Qiyuan Guan](https://guanqiyuan.github.io/) <sup>1</sup>, [Xiang Chen](https://cschenxiang.github.io/) <sup>2</sup>, Jiyu Jin* <sup>1</sup>,  Guiyue Jin <sup>1</sup>, [Jiangxin Dong](https://scholar.google.com/citations?user=ruebFVEAAAAJ&hl=zh-CN&oi=ao) <sup>2</sup>
>
> Dalian Polytechnic University<sup>1</sup>, Nanjing University of Science and Technology<sup>2</sup>

 **👉️ Welcome to visit our website (专注底层视觉领域的信息服务平台) for low-level vision:[https://lowlevelcv.com/](https://lowlevelcv.com/)**
 
---

## :hammer: Overview
![image](https://github.com/QianfengY/UniRain/blob/main/figs/UniRain.png)
*Overall framework of UniRain. (Left) The RAG-based dataset distillation pipeline retrieves real rainy references consistent with the query image via multi-level similarity search and employs vision language models to evaluate its quality, thereby distilling reliable samples from public datasets. (Right) The asymmetric MoE architecture consists of soft-MoE encoder and hard-MoE decoder, optimized via the multi-objective reweighted strategy to achieve balanced learning and robust performance across multiple rain degradation types.*

---
### 🚩 **New Features/Updates**
- ✅ February 21, 2026. Our UniRain was accepted by **CVPR 2026**!
  
### Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{UniRain,
      title={UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization},
      author={Yang, Qainfeng and Guan, Qiyuan and Chen, Xiang and Jin, Jiyu and Jin, Guiyue and Dong, Jiangxin},
      journal={CVPR},
      year={2026}
}
 ```

### Contact
If you have any questions, please feel free to reach us out at <a href="mailto:csqianfengyang@163.com">csqianfengyang@163.com</a>.

## 📘 Performance Evaluation
![image](https://github.com/QianfengY/UniRain/blob/main/figs/Table1.png)
![image](https://github.com/QianfengY/UniRain/blob/main/figs/Table2.png)
---
## 📷️ Visual Results

| Method    |  DRS |DRD | NRS |NRD |
|--------|------|------|------|------|
| PReNet    |  [Baidu Netdisk](https://pan.baidu.com/s/10qUU7ukkB0g5Lyfi78J2UA?pwd=1234) (1234) | [Baidu Netdisk]( https://pan.baidu.com/s/1U8vI_2y4bDNfwDPMEOgHrg?pwd=1234) (1234)| [Baidu Netdisk](https://pan.baidu.com/s/1OkORVlu7lPcl8GzFCRBmZQ?pwd=1234) (1234)| [Baidu Netdisk]() (6666)|
| RCDNet    |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| MPRNet    |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| Restormer |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| IDT       |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| DRSformer |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| RLP       |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| MSDT      |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| NeRD-Rain |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| URIR      |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|
| UniRain (Ours) |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|

| Method    |  RealRain-1k-L | RealRain-1k-H | RainDS-real-RD |RainDS-real-RDS | WeatherBench-rain |
|--------|------|------|------|------|------|
| PReNet    |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| RCDNet    |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| MPRNet    |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| Restormer |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| IDT       |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| DRSformer |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| RLP       |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| MSDT      |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| NeRD-Rain |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| URIR      |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|
| UniRain (Ours) |  [Baidu Netdisk]() (6666) | [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)| [Baidu Netdisk]() (6666)|[Baidu Netdisk]() (6666)|

---


