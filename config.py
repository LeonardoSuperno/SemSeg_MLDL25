# Core
EPOCHS = 50
DEVICE = 'cuda'
PARALLELIZE = True
PROJECT_STEP = 'Step2_1'  # [Step2_1, Step2_2, Step3_1, Step3_2, Step4]
VERBOSE = True
EVAL_ITERATIONS = 100
ADVERSARIAL = False

# Model
MODEL_NAME = 'DeepLabV2'  # [DeepLabV2, BiSeNet]

# Optimizer
OPTIMIZER_NAME = 'Adam'  # [SGD, Adam]
LOSS_FN_NAME = 'CrossEntropyLoss'  # [CrossEntropyLoss]
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LR = 2.5e-4
POWER = 0.9  # for poly_lr_scheduler 
IGNORE_INDEX = 255

# Datasets
N_CLASSES = 19
TRAIN_DATASET_NAME = 'CityScapes'  # [CityScapes, GTA5]
VAL_DATASET_NAME = 'CityScapes'  # [CityScapes]
AUGMENTED = False
AUGMENTED_TYPE = 'transform4'  # [transform1, transform2, transform3, transform4]
BATCH_SIZE = 2  # [2, 4, 8]
N_WORKERS = 0 # [0, 2, 4]
CITYSCAPES = {
    'width': 1024,
    'height': 512
}
GTA = {
    'width': 1280,
    'height': 720
}

# Paths
CITYSCAPES_PATH = 'data/Cityscapes'
GTA5_PATH = 'data/GTA5_with_mask'
DEEPLABV2_PATH = 'models/Deeplab_resnet_pretrained_imagenet.pth'
CHECKPOINT_ROOT = 'checkpoints'