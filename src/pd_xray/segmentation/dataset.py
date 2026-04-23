import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pd_xray.data.backends.local import LocalBackend
from pd_xray.data.formats.tiff import TIFFReader
from pd_xray.core.logging import get_logger

logger = get_logger(__name__)


class LabelledDataset:
    """Loads matched image/mask pairs from a labelled_data directory.

    Expected layout (nnUNet-compatible)::

        data_dir/
            imagesTr/   <case>_0000.tif   (raw float images)
            labelsTr/   <case>.tif        (integer label masks 0..N)
            classes.json

    The ``_0000`` channel suffix in imagesTr is stripped to find the
    corresponding mask in labelsTr.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._backend = LocalBackend(self.data_dir)
        self._backend.connect()
        self._reader = TIFFReader()

    def load_classes(self) -> dict[int, str]:
        """Return {class_index: class_name} from classes.json."""
        classes_path = self.data_dir / "classes.json"
        with open(classes_path) as f:
            data = json.load(f)
        return {int(k): v for k, v in data["label_map"].items()}

    def _find_pairs(self) -> list[tuple[Path, Path]]:
        images_dir = self.data_dir / "imagesTr"
        labels_dir = self.data_dir / "labelsTr"
        pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(images_dir.glob("*.tif")):
            stem = img_path.stem
            mask_stem = stem[:-5] if stem.endswith("_0000") else stem
            mask_path = labels_dir / f"{mask_stem}.tif"
            if mask_path.exists():
                pairs.append((img_path, mask_path))
            else:
                logger.warning("No mask found for %s, skipping.", img_path.name)
        return pairs

    def load_pairs(
        self,
    ) -> list[tuple[NDArray[np.float32], NDArray[np.int32]]]:
        """Load all image/mask pairs as (float32 image, int32 mask) tuples."""
        result = []
        for img_path, mask_path in self._find_pairs():
            image = self._reader.read(img_path)
            mask = self._reader.read_single_file(mask_path).astype(np.int32)
            logger.info(
                "Loaded %s: shape=%s, classes=%s",
                img_path.name,
                image.shape,
                np.unique(mask).tolist(),
            )
            result.append((image, mask))
        return result

    def __len__(self) -> int:
        return len(self._find_pairs())

    def __repr__(self) -> str:
        return f"LabelledDataset(data_dir={self.data_dir!r}, n_pairs={len(self)})"
