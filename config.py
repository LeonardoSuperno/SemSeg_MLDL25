MODE = 'output'                     # ['train', 'output']

# Core
EPOCHS = 5
DEVICE = 'cuda'
PARALLELIZE = True
PROJECT_STEP = 'Step5'              # [Step2_1, Step2_2, Step3_1, Step3_2, Step4, Step5]
VERBOSE = True
EVAL_ITERATIONS = 100
ADVERSARIAL = True
LAMBDA_CE = 1                       #[0.1, 0.5, 1.0]
LAMBDA_EXTRA = 0.5                  #[0.1, 0.5, 1.0, 2.0]
EXTRA_LOSS_NAME = 'DiceLoss'        # [FocalLoss, DiceLoss, None]
MULTI_LEVEL = False
FEATURE = 'spatial'                 # ['spatial', 'context1, 'context2']

# Model
MODEL_NAME = 'BiSeNet'              # [DeepLabV2, BiSeNet]

# Optimizer
LOSS_FN_NAME = 'CrossEntropyLoss'   # [CrossEntropyLoss]
LR = 2.5e-4
POWER = 0.9                         # for poly_lr_scheduler 
IGNORE_INDEX = 255

# Datasets
N_CLASSES = 19
TRAIN_DATASET_NAME = 'GTA5'         # [CityScapes, GTA5]
VAL_DATASET_NAME = 'CityScapes'     # [CityScapes]
TEST_DATASET_NAME = 'CityScapes'    # [CityScapes]
AUGMENTED = True
AUGMENTED_TYPE = 'aug3'             # ['aug1', 'aug2', 'aug3', 'aug4']
BATCH_SIZE = 4                      # [2, 4, 6]
N_WORKERS = 2                       # [0, 2, 4]
CITYSCAPES = {
    'width': 1024,
    'height': 512
}
GTA = {
    'width': 1280,
    'height': 720
}

# Paths
CITYSCAPES_PATH = '/kaggle/input/cityscapes/Cityscapes/Cityspaces'
GTA5_PATH = '/kaggle/input/gta5-withmask/GTA5'
DEEPLABV2_PATH = '/kaggle/input/deeplab-resnet-pretrained-imagenet/deeplab_resnet_pretrained_imagenet.pth'
OUTPUT_ROOT = '/kaggle/working/'
CHECKPOINT_ROOT = '/kaggle/input/checkpoint/checkpoint' #['/kaggle/input/checkpoint/checkpoint', None]
