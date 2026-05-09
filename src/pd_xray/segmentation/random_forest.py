from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from pd_xray.processing import ImageProcessor
from pd_xray.segmentation.features import extract_features, FEATURE_NAMES
from pd_xray.core.logging import get_logger

logger = get_logger(__name__)

# Preprocessing steps applied by default when none are specified.
# Normalises intensity and removes high-frequency noise at two scales.
DEFAULT_PREPROCESSING: list[dict] = [
    {"name": "normalise"},
    {"name": "gaussian_blur", "sigma": 1.0},
]


class RFSegmenter:
    """Pixel-wise random forest segmenter for 2D images.

    Preprocessing (via ImageProcessor) is integrated at three levels:

    1. **Saved** — set at ``__init__`` or at ``fit()``.  Persisted with the
       model (``save`` / ``load``) and applied automatically to every
       ``predict``, ``predict_proba``, and ``evaluate`` call.

    2. **Per-call override** — pass ``preprocessing_steps`` directly to
       ``predict``, ``predict_proba``, or ``evaluate``.  Used for that call
       only; never overwrites the saved steps.  Useful when test slices were
       acquired differently and need a different filter chain.

    Precedence (highest → lowest):  per-call override > saved > none.

    Example::

        from pd_xray.segmentation import RFSegmenter, LabelledDataset
        from pd_xray.processing import ImageProcessor

        ds    = LabelledDataset("data/labelled_data")
        pairs = ds.load_pairs()

        # Steps saved with the model and used for training + default inference.
        train_steps = [
            {"name": "normalise"},
            {"name": "gaussian_blur", "sigma": 1.5},
            {"name": "median_filter", "size": 3},
        ]

        model = RFSegmenter(
            preprocessing_steps=train_steps,
            class_names=ds.load_classes(),
        )
        model.fit(
            [img for img, _ in pairs[:3]],
            [mask for _, mask in pairs[:3]],
        )

        # Validate with the saved preprocessing.
        metrics = model.evaluate(
            [img for img, _ in pairs[3:]],
            [mask for _, mask in pairs[3:]],
        )

        # Test slice acquired differently — use a temporary override.
        test_steps = [{"name": "normalise"}, {"name": "clahe", "clip_limit": 2.0}]
        mask = model.predict(new_image, preprocessing_steps=test_steps)

        model.save("models/rf_segmenter.joblib")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
        random_state: int = 42,
        class_names: Optional[dict[int, str]] = None,
        preprocessing_steps: Optional[list[dict]] = None,
    ) -> None:
        """
        Args:
            n_estimators:        Number of trees in the random forest.
            max_depth:           Maximum tree depth (None = unlimited).
            n_jobs:              Parallel jobs for sklearn (-1 = all cores).
            random_state:        Reproducibility seed.
            class_names:         Mapping ``{class_index: name}`` for reports.
            preprocessing_steps: ImageProcessor step dicts applied before
                                 feature extraction.  Saved with the model.
                                 Defaults to normalise + light Gaussian blur.
        """
        self._rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight="balanced",
        )
        self.class_names: dict[int, str] = class_names or {}
        self._preprocessing_steps: list[dict] = (
            preprocessing_steps if preprocessing_steps is not None
            else DEFAULT_PREPROCESSING
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    @property
    def preprocessing_steps(self) -> list[dict]:
        """The saved preprocessing pipeline (applied by default)."""
        return self._preprocessing_steps

    def _resolve_processor(
        self, override: Optional[list[dict]]
    ) -> Optional[ImageProcessor]:
        """Return the processor to use for a single operation.

        Args:
            override: Per-call steps (not saved).  If None the saved steps
                      are used.  Pass an empty list ``[]`` to skip preprocessing.

        Returns:
            ``ImageProcessor`` instance, or ``None`` when there are no steps.
        """
        steps = override if override is not None else self._preprocessing_steps
        return ImageProcessor(steps=steps) if steps else None

    def _preprocess(
        self,
        image: NDArray[np.float32],
        processor: Optional[ImageProcessor],
    ) -> NDArray[np.float32]:
        if processor is None:
            return image
        processed, result = processor.process(image)
        if result.status != "success":
            logger.warning("Preprocessing warnings: %s", result.warnings)
        return processed

    def _features_and_mask_for(
        self,
        image: NDArray[np.float32],
        processor: Optional[ImageProcessor],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Preprocess, derive valid-pixel mask, then extract features.

        NaN pixels (e.g. from circular_mask) are filled with 0.0 before feature
        extraction so filters don't propagate NaN. The boolean mask returned
        is True where the pixel is valid and should be used for training /
        evaluation / included in predictions.

        Returns:
            features  : (H*W, N_FEATURES) float32 array.
            valid_mask: (H*W,) bool array, True for non-NaN pixels.
        """
        preprocessed = self._preprocess(image, processor)
        if preprocessed.shape != image.shape:
            raise ValueError(
                f"Preprocessing changed image shape from {image.shape} to "
                f"{preprocessed.shape}. Shape-changing steps (crop_to_circle=True, "
                "crop, resize) cannot be used in RFSegmenter — features and labels "
                "must share the same spatial layout. Use crop_to_circle=False to "
                "mask pixels outside the FOV with NaN instead."
            )
        valid_mask = ~np.isnan(preprocessed)
        filled = np.where(valid_mask, preprocessed, 0.0).astype(np.float32)
        return extract_features(filled), valid_mask.ravel()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        images: list[NDArray[np.float32]],
        masks: list[NDArray[np.int32]],
        max_samples_per_image: int = 100_000,
        preprocessing_steps: Optional[list[dict]] = None,
    ) -> RFSegmenter:
        """Train on a list of (image, mask) pairs.

        Args:
            images:               2D float32 arrays.
            masks:                2D int32 label arrays matching spatial dims.
            max_samples_per_image: Random pixel subsample per image to cap memory.
                                   Set to 0 to use all pixels.
            preprocessing_steps:  If provided, **overwrites** the saved steps so
                                   the same pipeline is used for all future
                                   inference calls (unless overridden per-call).

        Returns:
            self
        """
        if preprocessing_steps is not None:
            self._preprocessing_steps = preprocessing_steps
            logger.info("Saved preprocessing updated at fit() (%d steps)", len(preprocessing_steps))

        processor = self._resolve_processor(None)  # use saved steps
        if processor:
            logger.info(
                "Preprocessing pipeline: %s",
                [s.get("name") for s in self._preprocessing_steps],
            )

        X_parts, y_parts = [], []
        for i, (image, mask) in enumerate(zip(images, masks)):
            logger.info("Extracting features: image %d/%d", i + 1, len(images))
            feats, valid = self._features_and_mask_for(image, processor)
            labels = mask.ravel()

            # Exclude pixels that were masked out (e.g. outside circular FOV).
            feats, labels = feats[valid], labels[valid]

            if max_samples_per_image and feats.shape[0] > max_samples_per_image:
                rng = np.random.default_rng(42 + i)
                idx = rng.choice(feats.shape[0], size=max_samples_per_image, replace=False)
                feats = feats[idx]
                labels = labels[idx]

            X_parts.append(feats)
            y_parts.append(labels)

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        logger.info("Fitting RF on %d samples, %d features", X.shape[0], X.shape[1])
        self._rf.fit(X, y)
        self._fitted = True
        logger.info("Training complete")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: NDArray[np.float32],
        preprocessing_steps: Optional[list[dict]] = None,
    ) -> NDArray[np.int32]:
        """Predict a segmentation mask for a single 2D image.

        Args:
            image:               2D float32 array.
            preprocessing_steps: Temporary override for this call only.
                                 Does not update the saved pipeline.
                                 Pass ``[]`` to skip preprocessing entirely.

        Returns:
            Integer label array of the same spatial shape as ``image``.
        """
        self._check_fitted()
        processor = self._resolve_processor(preprocessing_steps)
        feats, valid = self._features_and_mask_for(image, processor)
        pred = np.zeros(image.size, dtype=np.int32)
        pred[valid] = self._rf.predict(feats[valid]).astype(np.int32)
        return pred.reshape(image.shape)

    def predict_proba(
        self,
        image: NDArray[np.float32],
        preprocessing_steps: Optional[list[dict]] = None,
    ) -> NDArray[np.float32]:
        """Return per-class probability maps.

        Args:
            image:               2D float32 array.
            preprocessing_steps: Temporary override for this call only.

        Returns:
            Float32 array of shape (H, W, n_classes).
        """
        self._check_fitted()
        h, w = image.shape
        n_classes = len(self._rf.classes_)
        processor = self._resolve_processor(preprocessing_steps)
        feats, valid = self._features_and_mask_for(image, processor)
        proba = np.zeros((image.size, n_classes), dtype=np.float32)
        proba[valid] = self._rf.predict_proba(feats[valid]).astype(np.float32)
        return proba.reshape(h, w, n_classes)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        images: list[NDArray[np.float32]],
        masks: list[NDArray[np.int32]],
        preprocessing_steps: Optional[list[dict]] = None,
    ) -> dict:
        """Compute per-class metrics over a set of images.

        Args:
            images:              2D float32 arrays.
            masks:               2D int32 label arrays.
            preprocessing_steps: Temporary override applied to every image in
                                 this evaluation only.  Does not update the
                                 saved pipeline.

        Returns:
            Dict with keys:

            - ``classification_report``: sklearn report (nested dict)
            - ``iou``:                   per-class Intersection-over-Union
            - ``confusion_matrix``:      confusion matrix as nested list
        """
        self._check_fitted()
        processor = self._resolve_processor(preprocessing_steps)

        all_true, all_pred = [], []
        for image, mask in zip(images, masks):
            feats, valid = self._features_and_mask_for(image, processor)
            pred = np.zeros(image.size, dtype=np.int32)
            pred[valid] = self._rf.predict(feats[valid]).astype(np.int32)
            # Evaluate only on valid (non-masked) pixels.
            all_true.append(mask.ravel()[valid])
            all_pred.append(pred[valid])

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        target_names = [
            self.class_names.get(int(c), str(c)) for c in sorted(self._rf.classes_)
        ]
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=sorted(self._rf.classes_))

        iou: dict[str, float] = {}
        for i, cls in enumerate(sorted(self._rf.classes_)):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum()) - tp
            fn = int(cm[i, :].sum()) - tp
            denom = tp + fp + fn
            name = self.class_names.get(int(cls), str(cls))
            iou[name] = float(tp / denom) if denom > 0 else 0.0

        return {
            "classification_report": report,
            "iou": iou,
            "confusion_matrix": cm.tolist(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist model and saved preprocessing pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "rf": self._rf,
                "class_names": self.class_names,
                "preprocessing_steps": self._preprocessing_steps,
            },
            path,
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> RFSegmenter:
        """Load a previously saved model, restoring its preprocessing pipeline."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._rf = data["rf"]
        obj.class_names = data["class_names"]
        obj._preprocessing_steps = data.get("preprocessing_steps", DEFAULT_PREPROCESSING)
        obj._fitted = True
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> dict[str, float]:
        """Feature importance scores keyed by feature name."""
        self._check_fitted()
        return dict(zip(FEATURE_NAMES, self._rf.feature_importances_.tolist()))

    @property
    def classes_(self) -> list[int]:
        self._check_fitted()
        return [int(c) for c in self._rf.classes_]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        steps = [s.get("name") for s in self._preprocessing_steps]
        return (
            f"RFSegmenter("
            f"n_estimators={self._rf.n_estimators}, "
            f"status={status}, "
            f"preprocessing={steps})"
        )
