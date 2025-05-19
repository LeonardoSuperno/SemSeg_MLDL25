import numpy as np

'''
def fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(
        num_classes * true[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)


def evaluate_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: str,
                   n_classes: int,
                   ignore_index: int = 255) -> Dict[str, float]:
    """
    Evaluate a trained segmentation model on a validation set.
    """
    model.eval()
    hist = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            for pred, label in zip(preds, labels):
                mask = (label != ignore_index)
                hist += fast_hist(label[mask], pred[mask], n_classes)

    iou = per_class_iou(hist)
    miou = np.nanmean(iou)
    acc = np.diag(hist).sum() / hist.sum()

    results = {f"IoU_class_{i}": round(iou[i], 4) for i in range(n_classes)}
    results["mIoU"] = round(miou, 4)
    results["accuracy"] = round(acc, 4)

    return results
'''


