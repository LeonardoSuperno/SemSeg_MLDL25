from utils.metrics import *
from builder import *
from utils.checkpoints import *
from utils.data_processing import label_to_rgb
import torch
import os
from PIL import Image
import time

# This function given the final checkpoint generate the test image, label and the output prediction of the trained model

def save_output(model_name, 
            test_dataset_name, 
            n_classes,
            multi_level,
            feature,
            lr,
            loss_fn_name,
            ignore_index,
            batch_size,
            n_workers,
            device,
            parallelize,
            project_step,
            verbose,
            output_root,
            checkpoint_root,
            adversarial):
    
    model, optimizer, _, model_D, optimizer_D, _ = build_model(model_name, 
                                                                       n_classes,
                                                                       device,
                                                                       parallelize, 
                                                                       lr,
                                                                       loss_fn_name,
                                                                       ignore_index,
                                                                       adversarial,
                                                                       multi_level,
                                                                       feature)

    # get loader
    test_loader, _, _ = build_test_loader(test_dataset_name, 
                                                                    batch_size,
                                                                    n_workers)
    
    no_checkpoint, _, _, _, _, _, _, _ = load_checkpoint(checkpoint_root=checkpoint_root, adversarial=adversarial, model=model, model_D=model_D, optimizer=optimizer, optimizer_D=optimizer_D, multi_level=multi_level)

    if no_checkpoint:
        print('No checkpoint found. Please train the model first.')
        return
    
    total_miou = 0
    total_iou = np.zeros(n_classes)
    
    images, labels = next(iter(test_loader))
    _, outputs = model(images)
    predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

    hist = fast_hist(labels.cpu().numpy(), predictions.cpu().numpy(), n_classes)
    running_iou = np.array(per_class_iou(hist)).flatten()
    total_miou += running_iou.sum()
    total_iou += running_iou
    total_miou += running_iou.sum()
    total_iou += running_iou

    save_dir = os.path.join(output_root, "test", project_step)
    os.makedirs(save_dir, exist_ok=True)

    for idx, (image, label, output) in enumerate(zip(images, labels, predictions)):
        image = (image.permute(1, 2, 0)*255).cpu().numpy().astype(np.uint8)
        label = label.cpu().numpy()
        output = output.cpu().numpy()

        rgb_image = Image.fromarray(image)
        rgb_label = label_to_rgb(label)
        rgb_output = label_to_rgb(output)

        # Save the test image and label, output prediction
        rgb_image.save(os.path.join(save_dir, f"{idx:02d}_image.png"))
        rgb_label.save(os.path.join(save_dir, f"{idx:02d}_label.png"))
        rgb_output.save(os.path.join(save_dir, f"{idx:02d}_prediction.png"))
        


    