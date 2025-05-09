from utils.visualization import plot_segmented_images
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# deepLab on the same domain 
# BiSeNet on the same domain
# BiSeNet on different domain
# BiSeNet domain adaptation

model_roots = ['/kaggle/input/checkpoint/checkpoint/checkpoint.pth']

model_types = [
    #('DeepLabV2','DeepLabV2'),
    #('BiSeNet', 'BiSeNet'),
    #('BiSeNet', 'BiSeNet Domain Shift'),
    ('BiSeNet', 'BiSeNet Domain Adaptation')
]

plot_segmented_images(model_roots=model_roots, model_types=model_types,n_images=5, device=device)