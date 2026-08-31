import unittest

import numpy as np

from evaluation_scripts.evaluate_calibrated_dataset import (
    binary_mask_metrics,
    bbox_iou,
    normalized_center_distance,
    point_inside_bbox,
    summarize,
)


class CalibratedEvaluationTest(unittest.TestCase):
    def test_geometry_metrics(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 5, 5], [5, 5, 10, 10]), 0.0)
        self.assertTrue(point_inside_bbox([5, 5], [0, 0, 10, 10]))
        self.assertFalse(point_inside_bbox([11, 5], [0, 0, 10, 10]))
        self.assertEqual(normalized_center_distance([0, 0, 10, 10], [0, 0, 10, 10], 10, 10), 0.0)

    def test_proxy_mask_metrics_and_strict_paper_bbox_threshold(self):
        first = np.array([[1, 1], [0, 0]], dtype=bool)
        second = np.array([[1, 0], [1, 0]], dtype=bool)
        metrics = binary_mask_metrics(first, second)
        self.assertAlmostEqual(metrics["proxy_mask_iou"], 1 / 3)
        self.assertAlmostEqual(metrics["proxy_mask_dice"], 0.5)
        records = [
            {
                "parse_success": True,
                "bbox_iou": 0.5,
                "points_inside_gt_bbox": 1.0,
                "normalized_center_distance": 0.0,
                "sam2_success": False,
            }
        ]
        self.assertEqual(summarize(records, dataset_rows=1)["bbox_accuracy_iou_gt_0_5"], 0.0)

    def test_summary_penalizes_unparseable_outputs(self):
        records = [
            {
                "parse_success": True,
                "bbox_iou": 1.0,
                "points_inside_gt_bbox": 1.0,
                "normalized_center_distance": 0.0,
                "sam2_success": True,
                "sam2_score": 0.8,
            },
            {
                "parse_success": False,
                "bbox_iou": 0.0,
                "points_inside_gt_bbox": 0.0,
                "normalized_center_distance": 1.0,
                "sam2_success": False,
            },
        ]
        summary = summarize(records, dataset_rows=2)
        self.assertEqual(summary["evaluated_rows"], 2)
        self.assertEqual(summary["parse_success_rate"], 0.5)
        self.assertEqual(summary["mean_bbox_iou"], 0.5)
        self.assertEqual(summary["bbox_iou_at_0_5"], 0.5)
        self.assertEqual(summary["sam2_success_rate"], 0.5)
        self.assertAlmostEqual(summary["paper_reference_delta"]["bbox_accuracy_iou_gt_0_5"], -0.4012)
        self.assertIsNone(summary["paper_reference_delta"]["dice"])


if __name__ == "__main__":
    unittest.main()
