import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from neuroscan.config_loader import ConfigLoader
from neuroscan.data_pipeline import DataPipeline
from neuroscan.trainer import dice_coef, bce_dice_loss


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classification(config, pipeline):

    print("\n" + "=" * 60)
    print("CLASSIFICATION EVALUATION")
    print("=" * 60)

    ckpt_path = config["model"]["classification"]["checkpoint_path"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Classification model not found:\n{ckpt_path}"
        )

    print("\nLoading classification model...")
    model = tf.keras.models.load_model(ckpt_path)

    print("Loading testing dataset...")
    test_ds = pipeline.build_cls_dataset(
        subset="Testing"
    )

    y_true = []
    y_pred = []

    print("\nRunning inference on test set...")

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0
        )

        true_labels = np.argmax(
            labels.numpy(),
            axis=1
        )

        predicted_labels = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(true_labels)
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n" + "-" * 60)
    print(
        f"Overall Test Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )
    print("-" * 60)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    classes = config["model"]["classification"]["classes"]

    print("\nClassification Report:")

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClass order:")
    for index, class_name in enumerate(classes):
        print(f"{index} = {class_name}")

    return accuracy, cm, report


# ============================================================
# SEGMENTATION EVALUATION
# ============================================================

def evaluate_segmentation(config, pipeline):
    print("\n" + "="*60)
    print("SEGMENTATION EVALUATION")
    print("="*60)

    ckpt_path = config['model']['segmentation']['checkpoint_path']

    if not os.path.exists(ckpt_path):
        print(f"Model not found at {ckpt_path}")
        return None

    print("Loading segmentation model...")

    model = tf.keras.models.load_model(
        ckpt_path,
        custom_objects={
            "bce_dice": bce_dice_loss,
            "bce_dice_loss": bce_dice_loss,
            "dice_coef": dice_coef
        }
    )

    print("Loading validation dataset...")
    train_ds, val_ds = pipeline.build_seg_dataset()

    print("\nEvaluating segmentation model...")
    results = model.evaluate(val_ds, verbose=1, return_dict=True)

    print("\nSegmentation Metrics:")

    for name, value in results.items():
        print(f"  {name}: {value:.4f}")

    dice = results.get("dice_coef", None)

    accuracy = results.get("accuracy", None)

    mean_iou = results.get("mean_io_u", None)

    print("\n" + "-"*60)
    print("SEGMENTATION FINAL RESULTS")
    print("-"*60)

    if accuracy is not None:
        print(f"Pixel Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")

    if dice is not None:
        print(f"Dice Score     : {dice:.4f} ({dice * 100:.2f}%)")

    if mean_iou is not None:
        print(f"Mean IoU       : {mean_iou:.4f} ({mean_iou * 100:.2f}%)")

    return dice

    # --------------------------------------------------------
    # Extract Dice coefficient
    # --------------------------------------------------------

    if "dice_coef" in metrics_names:

        dice_index = metrics_names.index(
            "dice_coef"
        )

        dice_score = results[dice_index]

    else:

        dice_score = None

    return dice_score


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    print("Loading configuration...")

    config = ConfigLoader(
        "config.yaml"
    ).config

    pipeline = DataPipeline(
        config
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    cls_accuracy, confusion_mat, report = evaluate_classification(
        config,
        pipeline
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    seg_dice = evaluate_segmentation(
        config,
        pipeline
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Classification Test Accuracy: "
        f"{cls_accuracy:.4f} "
        f"({cls_accuracy * 100:.2f}%)"
    )

    if seg_dice is not None:
        print(
            f"Segmentation Validation Dice: "
            f"{seg_dice:.4f}"
        )
    else:
        print(
            "Segmentation Dice: Not available"
        )

    print("=" * 60)