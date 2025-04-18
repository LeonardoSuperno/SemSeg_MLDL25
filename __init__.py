from pipeline import pipeline
from config import *


if __name__ == '__main__':
    
    pipeline(
        model_name=MODEL_NAME, 
        train_dataset_name=TRAIN_DATASET_NAME, 
        val_dataset_name=VAL_DATASET_NAME,
        n_classes=N_CLASSES,
        epochs=EPOCHS,
        augmented=AUGMENTED,
        augmentedType=AUGMENTED_TYPE,
        optimizer_name=OPTIMIZER_NAME,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        loss_fn_name=LOSS_FN_NAME,
        ignore_index=IGNORE_INDEX,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS,
        device=DEVICE,
        parallelize=PARALLELIZE,
        project_step=PROJECT_STEP,
        verbose=VERBOSE,
        checkpoint_root=CHECKPOINT_ROOT,
        power=POWER,
        evalIterations=EVAL_ITERATIONS,
        adversarial=ADVERSARIAL
    ) 