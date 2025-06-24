
## Real-time Domain Adaptation in Semantic Segmentation

## Authors  
- Andrea Roffinella - s349557@studenti.polito.it  
- Giovanni Stinà - s332085@studenti.polito.it  
- Leonardo Superno Falco - s338685@studenti.polito.it  
- Nicola Biagioli - s344677@studenti.polito.it  

## Abstract  
This work focuses on domain adaptation techniques for real-time semantic segmentation. We trained and evaluated a classical model (DeepLabV2) and a real-time model (BiSeNet) on the real-world Cityscapes dataset to establish baseline performance. We then trained BiSeNet on the synthetic GTA5 dataset and evaluated its generalization to Cityscapes, highlighting the impact of domain shift.

## GTA5 Dataset Preprocessing
Before training or evaluating models using the GTA5 dataset, it is necessary to preprocess the label images by converting them to grayscale format. This can be done by running the `makeGTA5mask.py` script located in the `GTA5masks` folder. This step ensures that the semantic labels are properly formatted for compatibility with the training pipeline.

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
- **Multi-level Adversarial Learning**: The best result using spatial features achieved a mean mIoU of 26.89%. In comparison, the mIoU scores for Context1 and Context2 were respectively 24.55% and 26.64%.  
- **Focal Loss and Dice Loss**: Using the Focal Loss we reached a mIoU of 26.00%, and with the Dice Loss a mIoU of 26.73%.

## Conclusion  
Domain adaptation techniques significantly improve the performance of real-time semantic segmentation models. Although the multi-level adversarial approach and the use of additional loss functions did not improve overall performance, they provided valuable insights into the model's behavior across different classes.
