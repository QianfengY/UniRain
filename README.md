#  <center>  [CVPR 2026] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization 

> [Paper] &emsp; [Supplemental Material]

> [Qianfeng Yang](https://qianfengy.github.io/) <sup>1</sup>, [Qiyuan Guan](https://guanqiyuan.github.io/) <sup>1</sup>, [Xiang Chen](https://cschenxiang.github.io/) <sup>2</sup>, Jiyu Jin* <sup>1</sup>,  Guiyue Jin <sup>1</sup>, [Jiangxin Dong](https://scholar.google.com/citations?user=ruebFVEAAAAJ&hl=zh-CN&oi=ao) <sup>2</sup>
>
> Dalian Polytechnic University<sup>1</sup>, Nanjing University of Science and Technology<sup>2</sup>

 **👉️ Welcome to visit our website (专注底层视觉领域的信息服务平台) for low-level vision:[https://lowlevelcv.com/](https://lowlevelcv.com/)**
 
---

### 🛠️ Overview
![image](https://github.com/QianfengY/UniRain/blob/main/figs/UniRain.png)
*Overall framework of UniRain. (Left) The RAG-based dataset distillation pipeline retrieves real rainy references consistent with the query image via multi-level similarity search and employs vision language models to evaluate its quality, thereby distilling reliable samples from public datasets. (Right) The asymmetric MoE architecture consists of soft-MoE encoder and hard-MoE decoder, optimized via the multi-objective reweighted strategy to achieve balanced learning and robust performance across multiple rain degradation types.*

---
### 🚩 **New Features/Updates**
- ✅ March 03, 2026. Release the dataset, visual results, and testing code.
- ✅ February 21, 2026. 🎉 Our UniRain was accepted by **CVPR 2026**!
### ⚡ **To Do**
- [x] Release training code and training set
---
### 📥 Dataset Download
| Download Link | Description |
|---------|------|
|  [Baidu Netdisk]( ) (1111)  | Tran: 0,000 pairs.    Test: 000 pairs. |
---

### 📈 Performance Evaluation
![image](https://github.com/QianfengY/UniRain/blob/main/figs/Table1.png)

---

### 📷️ Visual Results

| Method    |  DRS |DRD | NRS |NRD |
|--------|------|------|------|------|
| PReNet    |  [Baidu Netdisk](https://pan.baidu.com/s/10qUU7ukkB0g5Lyfi78J2UA?pwd=1234)| [Baidu Netdisk]( https://pan.baidu.com/s/1U8vI_2y4bDNfwDPMEOgHrg?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1OkORVlu7lPcl8GzFCRBmZQ?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1UFn4bjlWUbfuJCIgy0WRZQ?pwd=1234) |
| RCDNet    |  [Baidu Netdisk](https://pan.baidu.com/s/1An2ACzEWCzJECbxoUZBTxg?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1jLjlfKONNf9MMtx2EDWh_A?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1Y_aszPj_Wwca0xW3o0QB-g?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/19VapnHoH8ryIlLh73Lsw-Q?pwd=1234)|
| MPRNet    |  [Baidu Netdisk](https://pan.baidu.com/s/1FPzFtxinV0y8t0dt8X2p2w?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1B5ZXhfdzYYhCYkXLzHVj7A?pwd=1234)| [Baidu Netdisk]( https://pan.baidu.com/s/1XnopLPOdIQVysLHangAAeA?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1QFs3l4ykopln8Ed8fzRE0A?pwd=1234) |
| Restormer |  [Baidu Netdisk](https://pan.baidu.com/s/1R7q0qUL9f4j7QJ9bD4SqqQ?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1U-f7LI6rlhsPf_PHZB3m8A?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1aTVPzAg3wsSR5NsfIeA1Cg?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1bHD0ZgabM3wtfTEpTVsKhA?pwd=1234)|
| IDT       |  [Baidu Netdisk](https://pan.baidu.com/s/1ULuqzWNkv49Tl91srK02nA?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/16YIUZX94ANuc1yhSO23yFA?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1XPNt7TUH4nMMXt0cKwfWPA?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1RVGFaJbksRUUE_7gdU2YuQ?pwd=1234)|
| DRSformer |  [Baidu Netdisk](https://pan.baidu.com/s/1tlfojpO98mDqZZ4nf_Pb5Q?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1f2U0DIorhsNJ1ftDccRBvw?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1xHeVnbyiwMGD92wvTZGWog?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1kH9tYeKYISjLauLm7KOCHQ?pwd=1234)|
| RLP       |  [Baidu Netdisk](https://pan.baidu.com/s/1EfigBVpwBtE7f-9_mRfgPg?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1uAgzUxha__B5kS1H7oc9Ow?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1M0ISP7wLKy0fKr57ryNdcA?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/15Wf-oTKUP7Nh8-YyJhIq-g?pwd=1234)|
| MSDT      |  [Baidu Netdisk](https://pan.baidu.com/s/1SGm1-OI0EtUzic7Q3gm1LA?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1P5KvpD9v9zbPaJQfHcMTCg?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1SsF3CUdMVx1sFf7D9Z6Q4g?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1S8JZ8INLUUcSTnSN3oZSwQ?pwd=1234)|
| NeRD-Rain |  [Baidu Netdisk](https://pan.baidu.com/s/1Uu-0Lxtjps264wO657Xing?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1ZFqOCVK0xbdOMMtOCbHrUw?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1lHbTeuxdsU9-mN511ROlhg?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1cSJCAGSA7BQ7cz0J_pixlw?pwd=1234) |
| URIR      |  [Baidu Netdisk](https://pan.baidu.com/s/1pJqs5qaezGRYqs3ynL99Rw?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1_JJ8-PB3B-i-wlZS0dJASA?pwd=1234)| [Baidu Netdisk](https://pan.baidu.com/s/1WECGrY0PKBB-C5-BmBHXbg?pwd=1234) | [Baidu Netdisk]( https://pan.baidu.com/s/16w1SL1CX628wV2bgRLCwnw?pwd=1234)|
| UniRain (Ours) |  [Baidu Netdisk](https://pan.baidu.com/s/19nbPXixwBsTfRSDilt-KMQ?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1tlTdDLMdSKa-QNRvUkEwEA?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1xkN0XitcMRLOosDA-NR00g?pwd=1234) | [Baidu Netdisk]( https://pan.baidu.com/s/1zjtkfWu7t-YLsiE98AaRfA?pwd=1234)|



--
### 💖 Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{UniRain,
      title={UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization},
      author={Yang, Qainfeng and Guan, Qiyuan and Chen, Xiang and Jin, Jiyu and Jin, Guiyue and Dong, Jiangxin},
      journal={CVPR},
      year={2026}
}
 ```
---

### Contact
If you have any questions, please feel free to reach us out at <a href="mailto:csqianfengyang@163.com">csqianfengyang@163.com</a>.

