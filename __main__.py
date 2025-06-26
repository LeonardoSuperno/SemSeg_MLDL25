from pipeline import pipeline
from save_output import save_output
from config import *



if __name__ == '__main__':
    if MODE == 'train':
        pipeline(
            model_name=MODEL_NAME, 
            train_dataset_name=TRAIN_DATASET_NAME, 
            val_dataset_name=VAL_DATASET_NAME,
            n_classes=N_CLASSES,
            epochs=EPOCHS,
            augmented=AUGMENTED,
            multi_level=MULTI_LEVEL,
            feature=FEATURE,
            augmentedType=AUGMENTED_TYPE,
            lr=LR,
            loss_fn_name=LOSS_FN_NAME,
            ignore_index=IGNORE_INDEX,
            batch_size=BATCH_SIZE,
            n_workers=N_WORKERS,
            device=DEVICE,
            parallelize=PARALLELIZE,
            project_step=PROJECT_STEP,
            verbose=VERBOSE,
            output_root=OUTPUT_ROOT,
            checkpoint_root=CHECKPOINT_ROOT,
            power=POWER,
            evalIterations=EVAL_ITERATIONS,
            adversarial=ADVERSARIAL,
            lambda_ce=LAMBDA_CE,
            lambda_extra=LAMBDA_EXTRA,
            extra_loss_name=EXTRA_LOSS_NAME
        ) 
    elif MODE == 'output':
        save_output(
            model_name=MODEL_NAME, 
            test_dataset_name=TEST_DATASET_NAME, 
            n_classes=N_CLASSES,
            multi_level=MULTI_LEVEL,
            feature=FEATURE,
            lr=LR,
            loss_fn_name=LOSS_FN_NAME,
            ignore_index=IGNORE_INDEX,
            batch_size=BATCH_SIZE,
            n_workers=N_WORKERS,
            device=DEVICE,
            parallelize=PARALLELIZE,
            project_step=PROJECT_STEP,
            verbose=VERBOSE,
            output_root=OUTPUT_ROOT,
            checkpoint_root=CHECKPOINT_ROOT,
            adversarial=ADVERSARIAL
        )
