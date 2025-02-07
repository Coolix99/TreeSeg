import numpy as np
import pytest
from tree_seg.metrices.score import calculate_scores

def test_calculate_scores():
    # Simple synthetic dataset
    y_true = np.array([
        [1, 1, 0],
        [2, 2, 0],
        [2, 2, 0]
    ])
    y_pred = np.array([
        [1, 1, 0],
        [1, 2, 2],
        [0, 2, 2]
    ])

    # Define thresholds and expected values for testing
    thresholds = [0.45, 0.75]
    expected_results = {
        0.45: {
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'panoptic_quality': 0.5,
            'tp': 1,
            'fp': 1,
            'fn': 1
        },
        0.75: {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'panoptic_quality': 0.0,
            'tp': 0,
            'fp': 2,
            'fn': 2
        }
    }

    # Calculate scores using the function
    scores = calculate_scores(y_true, y_pred, thresholds=thresholds, criterion='iou', method='1:1')
    print(scores)
    # Verify each metric for each threshold
    for thresh in thresholds:
        result = scores[thresh]
        expected = expected_results[thresh]

        assert np.isclose(result['precision'], expected['precision']), f"Precision mismatch for threshold {thresh}"
        assert np.isclose(result['recall'], expected['recall']), f"Recall mismatch for threshold {thresh}"
        assert np.isclose(result['f1'], expected['f1']), f"F1 mismatch for threshold {thresh}"
        assert np.isclose(result['panoptic_quality'], expected['panoptic_quality']), f"Panoptic Quality mismatch for threshold {thresh}"
        assert result['tp'] == expected['tp'], f"True Positives mismatch for threshold {thresh}"
        assert result['fp'] == expected['fp'], f"False Positives mismatch for threshold {thresh}"
        assert result['fn'] == expected['fn'], f"False Negatives mismatch for threshold {thresh}"

if __name__ == "__main__":
    pytest.main()
