## Real-time Domain Adaptation in Semantic Segmentation

## Authors  
- Andrea Roffinella -  s349557@studenti.polito.it  
- Giovanni Stinà - s332085@studenti.polito.it  
- Leonardo Superno Falco - s338685@studenti.polito.it  
- Nicola Biagioli - s344677@studenti.polito.it  

## Abstract  
This work focuses on domain adaptation techniques for real-time semantic segmentation. We trained and evaluated a classical model (DeepLabV2) and a real-time model (BiSeNet) on the real-world Cityscapes dataset to establish baseline performance. We then trained BiSeNet on the synthetic GTA5 dataset and evaluated its generalization to Cityscapes, highlighting the impact of domain shift.

## Introduction  
Semantic segmentation of real-world images typically requires large-scale datasets with dense, pixel-level annotations, which are costly and time-consuming to obtain. To mitigate this limitation, synthetic datasets such as GTA5 provide automatic, accurate, and large-scale semantic annotations. This work applies domain adaptation techniques to improve model performance when transferring from synthetic to real-world data.

## Methods  
- **DeepLabV2**: A classical semantic segmentation network.  
- **BiSeNet**: A real-time segmentation network.  
- **Domain Adaptation Techniques**: Data augmentation, single-level adversarial learning, and multi-level adversarial learning.  
- **Alternative Segmentation Loss Functions**: Focal and Dice loss, combined with cross-entropy loss.

## Results  
- **Baseline Performance**: BiSeNet shows a significant drop in performance when trained on synthetic data and tested on real-world data due to domain shift: from 46.80% to 17.41% mIoU.  
- **Data Augmentation**: Improved mIoU from 17.41% to 17.82%.  
- **Single-level Adversarial Learning**: Further increased mIoU to 28.11%.  
- **Multi-level Adversarial Learning**: Did not improve performance; the best result using spatial features reached 26.88% mIoU.  
- **Focal Loss**:  
- **Dice Loss**:  

## Conclusion  
Domain adaptation techniques significantly improve the performance of real-time semantic segmentation models in real-world applications. While notable improvements were achieved, further research is needed to fully bridge the performance gap.
