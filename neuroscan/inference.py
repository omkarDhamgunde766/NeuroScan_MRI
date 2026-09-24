import cv2
import numpy as np
import tensorflow as tf
from .config_loader import config


# Confidence threshold below which results are flagged as unreliable
CONFIDENCE_THRESHOLD = 0.70


class NeuroScanPipeline:
    def __init__(self, seg_model_path=None, cls_model_path=None):
        self.config = config

        self.classes = self.config['model']['classification']['classes']

        self.seg_shape = tuple(
            self.config['model']['segmentation']['input_shape'][:2]
        )

        self.cls_shape = tuple(
            self.config['model']['classification']['input_shape'][:2]
        )

        seg_path = (
            seg_model_path
            or self.config['model']['segmentation']['checkpoint_path']
        )

        cls_path = (
            cls_model_path
            or self.config['model']['classification']['checkpoint_path']
        )

        self.seg_model = None
        self.cls_model = None

        try:
            self.seg_model = tf.keras.models.load_model(
                seg_path,
                compile=False
            )
            print("Loaded segmentation model.")
        except Exception as e:
            print(f"Warning: Could not load segmentation model: {e}")

        try:
            self.cls_model = tf.keras.models.load_model(
                cls_path,
                compile=False
            )
            print("Loaded classification model.")
        except Exception as e:
            print(f"Warning: Could not load classification model: {e}")

    @staticmethod
    def validate_mri_image(image_bgr):
        """
        Lightweight Out-of-Distribution (OOD) detector.

        Checks if an image has visual characteristics
        consistent with brain MRI scans.
        """

        failed_checks = []
        scores = []

        # ---------------------------------------------------------
        # Check 1: Low color saturation
        # ---------------------------------------------------------
        hsv = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2HSV
        )

        mean_saturation = float(
            np.mean(hsv[:, :, 1])
        )

        sat_ok = mean_saturation < 30

        scores.append(
            1.0
            if sat_ok
            else max(
                0.0,
                1.0 - (mean_saturation - 30) / 120
            )
        )

        if not sat_ok:
            failed_checks.append(
                f"High color saturation ({mean_saturation:.0f}/255) "
                f"— MRI scans are grayscale"
            )

        # ---------------------------------------------------------
        # Check 2: Grayscale channel uniformity
        # ---------------------------------------------------------
        b, g, r = cv2.split(
            image_bgr.astype(np.float32)
        )

        rg_diff = float(
            np.mean(np.abs(r - g))
        )

        gb_diff = float(
            np.mean(np.abs(g - b))
        )

        channel_diff = (
            rg_diff + gb_diff
        ) / 2

        gray_ok = channel_diff < 15

        scores.append(
            1.0
            if gray_ok
            else max(
                0.0,
                1.0 - (channel_diff - 15) / 60
            )
        )

        if not gray_ok:
            failed_checks.append(
                f"Non-uniform color channels "
                f"(avg diff={channel_diff:.1f}) "
                f"— MRI scans have equal R/G/B channels"
            )

        # ---------------------------------------------------------
        # Check 3: Dark background prevalence
        # ---------------------------------------------------------
        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        dark_pixel_ratio = float(
            np.sum(gray < 30) / gray.size
        )

        dark_ok = dark_pixel_ratio > 0.20

        scores.append(
            1.0
            if dark_ok
            else dark_pixel_ratio / 0.20
        )

        if not dark_ok:
            failed_checks.append(
                f"Insufficient dark background "
                f"({dark_pixel_ratio * 100:.1f}%) "
                f"— MRI scans have dark surrounds"
            )

        # ---------------------------------------------------------
        # Check 4: Brightness and contrast
        # ---------------------------------------------------------
        mean_brightness = float(
            np.mean(gray)
        )

        std_brightness = float(
            np.std(gray)
        )

        brightness_ok = (
            mean_brightness < 160
            and std_brightness > 15
        )

        scores.append(
            1.0
            if brightness_ok
            else 0.3
        )

        if not brightness_ok:

            if mean_brightness >= 160:
                failed_checks.append(
                    f"Image too bright "
                    f"(mean={mean_brightness:.0f}) "
                    f"— MRI scans are predominantly dark"
                )

            if std_brightness <= 15:
                failed_checks.append(
                    f"Low contrast "
                    f"(std={std_brightness:.1f}) "
                    f"— MRI scans have high contrast tissue boundaries"
                )

        # ---------------------------------------------------------
        # Final OOD score
        # ---------------------------------------------------------
        ood_score = float(
            np.mean(scores)
        )

        is_valid = (
            len(failed_checks) <= 1
            and ood_score >= 0.55
        )

        return {
            "is_valid_mri": is_valid,
            "ood_score": ood_score,
            "failed_checks": failed_checks
        }

    def process_image(self, image_bgr):
        """
        Runs the full NeuroScan pipeline:

        1. Segment
        2. Create tumor mask / ROI
        3. Classify
        4. Generate Grad-CAM
        """

        if (
            self.seg_model is None
            or self.cls_model is None
        ):
            raise ValueError(
                "Models are not fully loaded."
            )

        # =========================================================
        # 1. SEGMENTATION
        # =========================================================

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        img_seg = cv2.resize(
            gray,
            self.seg_shape
        )

        img_seg_norm = (
            img_seg.astype(np.float32) / 255.0
        )

        img_seg_input = np.expand_dims(
            img_seg_norm,
            axis=(0, -1)
        )

        pred_mask = self.seg_model.predict(
            img_seg_input,
            verbose=0
        )[0]

        binary_mask = (
            pred_mask > 0.40
        ).astype(np.uint8).squeeze()

        # ---------------------------------------------------------
        # Segmentation overlay
        # ---------------------------------------------------------
        overlay = cv2.cvtColor(
            img_seg,
            cv2.COLOR_GRAY2BGR
        )

        overlay[
            binary_mask == 1
        ] = [0, 0, 255]

        # =========================================================
        # 2. CROP LARGEST ROI
        # =========================================================

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cropped_roi = img_seg

        if contours:

            largest_contour = max(
                contours,
                key=cv2.contourArea
            )

            x, y, w, h = cv2.boundingRect(
                largest_contour
            )

            if w > 10 and h > 10:
                cropped_roi = img_seg[
                    y:y + h,
                    x:x + w
                ]

        # =========================================================
        # 3. CLASSIFICATION
        # =========================================================
        #
        # IMPORTANT:
        # The classification model was trained using
        # the full MRI image.
        #
        # Therefore, classification inference also uses
        # the full MRI image instead of the cropped ROI.
        # =========================================================

        # Convert original BGR image to RGB
        full_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        # Resize exactly to classifier input size
        cls_input = cv2.resize(
            full_rgb,
            self.cls_shape
        )

        # Keep pixel values in [0, 255],
        # matching the training pipeline.
        cls_input_batch = np.expand_dims(
            cls_input.astype(np.float32),
            axis=0
        )

        pred_probs = self.cls_model.predict(
            cls_input_batch,
            verbose=0
        )[0]

        class_idx = int(
            np.argmax(pred_probs)
        )

        confidence = float(
            pred_probs[class_idx]
        )

        pred_class = self.classes[
            class_idx
        ]

        # =========================================================
        # 4. GRAD-CAM
        # =========================================================

        from .explainability import (
            generate_gradcam_heatmap,
            superimpose_heatmap
        )

        try:

            heatmap = generate_gradcam_heatmap(
                self.cls_model,
                cls_input_batch,
                target_class_idx=class_idx
            )

            # Grad-CAM is also displayed on the
            # same full MRI image used by classifier.
            heatmap_overlay = superimpose_heatmap(
                cls_input,
                heatmap
            )

            heatmap_overlay = cv2.cvtColor(
                heatmap_overlay,
                cv2.COLOR_BGR2RGB
            )

        except Exception as e:

            print(
                f"Failed to generate Grad-CAM: {e}"
            )

            heatmap = None
            heatmap_overlay = None

        # =========================================================
        # 5. CONFIDENCE THRESHOLD
        # =========================================================

        below_threshold = (
            confidence < CONFIDENCE_THRESHOLD
        )

        # =========================================================
        # FINAL RESULT
        # =========================================================

        return {
            "mask": binary_mask,

            "overlay": cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            ),

            "cropped_roi": cropped_roi,

            "pred_class": pred_class,

            "confidence": confidence,

            "probabilities": {
                cls: float(prob)
                for cls, prob in zip(
                    self.classes,
                    pred_probs
                )
            },

            "gradcam_heatmap": heatmap,

            "gradcam_overlay": heatmap_overlay,

            "below_threshold": below_threshold,

            "confidence_threshold": (
                CONFIDENCE_THRESHOLD
            ),
        }