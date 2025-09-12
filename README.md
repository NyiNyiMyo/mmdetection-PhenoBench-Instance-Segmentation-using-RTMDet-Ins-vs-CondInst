# mmdetection PhenoBench Instance Segmentation using RTMDet-Ins vs CondInst

## 🌱🍀 PhenoBench Dataset 

[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.1+cu117-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
![Static Badge](https://img.shields.io/badge/Instance-Segmentation-cyan)
![Static Badge](https://img.shields.io/badge/mmdetection-blue)

This repository contains a instance segmentation project focused on **interpretation in the agricultural domain** across **2 categories** using **RTMDet-Ins** Vs. **CondInst**.

---

## 🧭 Dataset Overview

The final dataset includes the following 2 classes for instance annotations:

| Class ID | Class Name             |
|--------- | ---------------------- |
| 0        | crop + partial-crop    |
| 1        | weed + partial-weed    |

Total train images: 1,407 / Total val images: 772

✅ Already instance masks for crop and weed for training and validation.  
✅ Converted semantic to instance for partial-crop and partial-weed for final combined dataset.

---

## 🏗️ Model Architecture

- 💎 Model1: **RTMDet-Ins m size**
- 💎 Model2: **CondInst with r50**
- 💎 Framework: **PyTorch + mmdetection**
- 💎 Input Size: **512, 512**
- 💎 Normalization: **ImageNet Mean/Std**

---

## 📊 Final Performance
```
🔸RTMDet-Ins with epochs 20
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.250
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.487
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.245
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.248
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.054
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.097
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570

🔸CondInst with iters 704
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.178
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.391
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.168
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.007
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.133
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.222
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.222
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.222
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.471
```

### 📈 Per-Class Evaluation Metrics:
```
🔸RTMDet-Ins with epochs 20
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| crop     | 0.423 | 0.7    | 0.47   | 0.018 | 0.37  | 0.699 |
| weed     | 0.077 | 0.274  | 0.02   | 0.007 | 0.126 | 0.372 |
+----------+-------+--------+--------+-------+-------+-------+

🔸CondInst with iters 704
+----------+-------+--------+--------+-------+-------+-------+
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
+----------+-------+--------+--------+-------+-------+-------+
| crop     | 0.314 | 0.613  | 0.325  | 0.013 | 0.21  | 0.58  |
| weed     | 0.043 | 0.17   | 0.012  | 0.002 | 0.055 | 0.281 |
+----------+-------+--------+--------+-------+-------+-------+
```

---

## 🎨 Visualization Samples

The model outputs of **validation set** are visualized:

📌 Example of RTMDet-Ins:
![Visualization Model1](PhenoBench-val-seg-RTMDet-Ins-results.png)  

📌 Example of CondInst:
![Visualization Model2](PhenoBench-val-seg-CondInst-results.png)  

---

## 🚀 How to Run Inference
```python
!python mmdetection/tools/test.py \
  mmdetection/configs/rtmdet/rtmdet_ins_m_pheno.py \
  mmdetection/mmdet_outputs/rtmdet_ins_m_pheno/epoch_20.pth \
  --show-dir pheno_seg_exp1
```

```python
!python mmdetection/tools/test.py \
  mmdetection/configs/condinst/condinst_pheno.py \
  mmdetection/mmdet_outputs/condinst_crop_weed/iter_704.pth \
  --show-dir pheno_seg_exp1
```

---

## 🔑 Summary

✅ Applied mostly default configs  
✅ Implemented minimally  
✅ **Note** RTMDet-Ins outperformed CondInst.

---

## 📄 License

This project is intended for **academic research and educational use** only. Please cite **original dataset paper** or **appropriately to this repo** if used in publications.

---

## ⭐ Acknowledgements

- RTMDet-Ins and CondInst powered by `mmdetection`
- Based on Popular instance segmentation benchmarking dataset `PhenoBench`

---

