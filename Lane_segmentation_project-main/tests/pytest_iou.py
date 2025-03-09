from evaluation.iou import compute_iou
import pytest
from PIL import Image
import numpy as np

true_mask = Image.open("true_000037_10.png").convert("RGB")
true_mask_array = np.array(true_mask)
black_rgb_array = np.zeros((375, 1242, 3), dtype=np.uint8)
falseH_rgb_array = np.ones((300, 1242, 3), dtype=np.uint8)
falseW_rgb_array = np.ones((375, 1000, 3), dtype=np.uint8)

def test_compute_iou():
    assert compute_iou(true_mask_array, black_rgb_array) == 0, "black mask and true mask should give 0"
    assert compute_iou(true_mask_array, true_mask_array) == 1, "true mask and true mask should give 1"
    assert compute_iou(black_rgb_array, black_rgb_array) == 0, 'black mask and black mask should give union 0'

    with pytest.raises(AssertionError, match='Mask shapes should be the same'):
        compute_iou(falseH_rgb_array, true_mask_array)
    
    with pytest.raises(AssertionError, match='Mask shapes should be the same'):
        compute_iou(falseW_rgb_array, true_mask_array)

    with pytest.raises(AssertionError, match='class_ID should be defined in color to class mapping'):
        compute_iou(true_mask_array, black_rgb_array, 20)

    

