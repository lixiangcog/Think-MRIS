import unittest

import numpy as np

from think_mris.data_calibration import (
    assign_group_split,
    describe_bbox,
    derive_two_interior_points,
    hard_quality_reasons,
    infer_sample_metadata,
    quality_reasons,
    rewrite_problem,
)


class DataCalibrationTest(unittest.TestCase):
    def test_source_and_modality_inference(self):
        adam = infer_sample_metadata("10055B_10055B_aneurysms_slice_57", "The aneurysm is round.")
        self.assertEqual((adam.source_dataset, adam.modality), ("ADAM", "MR"))
        prostate = infer_sample_metadata("Case05_segmentation_slice_32", "The prostate is visible.")
        self.assertEqual((prostate.source_dataset, prostate.modality), ("PROMISE12_centers_mixed", "MR"))
        brats = infer_sample_metadata("BraTS20_Training_343_seg_slice_44", "The whole tumor is irregular.")
        self.assertEqual(brats.group_id, "BraTS20_Training_343")

    def test_failed_generation_filters(self):
        metadata = infer_sample_metadata("10055B_10055B_aneurysms_slice_57", "The aneurysm is round.")
        refusal = "I cannot provide details without more information about the aneurysm in this image."
        self.assertIn("generation_refusal", hard_quality_reasons(refusal, metadata))
        negative = "There are no intracranial aneurysms visible in the provided image."
        self.assertIn("target_negated_despite_positive_geometry", hard_quality_reasons(negative, metadata))
        multiple = "The image shows several intracranial aneurysms in different arteries."
        self.assertIn("multiple_targets_with_single_geometry", hard_quality_reasons(multiple, metadata))
        hounsfield = "The intracranial aneurysm measures 17 Hounsfield units."
        self.assertIn("modality_contradiction", hard_quality_reasons(hounsfield, metadata))
        wrong_sequence = "The intracranial aneurysm is hyperintense on a T2-weighted image."
        self.assertIn("source_sequence_contradiction", hard_quality_reasons(wrong_sequence, metadata))
        bilateral = "The intracranial aneurysm is between the left and right communicating arteries."
        self.assertIn("bilateral_location_for_single_target", hard_quality_reasons(bilateral, metadata))
        medical_dimensions = "The aneurysm measures 3.2 × 4.1 mm and is round."
        self.assertNotIn("non_mris_content", hard_quality_reasons(medical_dimensions, metadata))
        self.assertIn("unsupported_numeric_measurement", quality_reasons(medical_dimensions, metadata))
        self.assertNotIn("unsupported_numeric_measurement", hard_quality_reasons(medical_dimensions, metadata))
        vague = "The exact shape is not provided and would depend on the specific image."
        self.assertIn("non_visual_or_missing_description", quality_reasons(vague, metadata))
        duplicate = "The aneurysm is a round round vascular abnormality in the image."
        self.assertIn("duplicated_generation_tokens", quality_reasons(duplicate, metadata))
        diagnostic = "This aneurysm is likely the source of the patient's subarachnoid hemorrhage."
        self.assertIn("unsupported_diagnostic_claim", quality_reasons(diagnostic, metadata))
        plural = "Intracranial aneurysms are vascular outpouchings in the brain."
        self.assertIn("plural_target_with_single_geometry", hard_quality_reasons(plural, metadata))
        spelled_unit = "The aneurysm measures approximately 4 millimeters and is round."
        self.assertIn("unsupported_numeric_measurement", quality_reasons(spelled_unit, metadata))
        meta = "This information is important for accurate medical segmentation."
        self.assertIn("generation_meta_text", quality_reasons(meta, metadata))
        context_dependent = "It resembles the previously identified aneurysm in the prior image."
        self.assertIn("non_visual_or_missing_description", quality_reasons(context_dependent, metadata))

    def test_repairable_text_is_cleaned_before_rewrite(self):
        metadata = infer_sample_metadata("10055B_10055B_aneurysms_slice_57", "The aneurysm is round.")
        original = (
            "The aneurysm measures approximately 4 millimeters and has a rounded contour. "
            "This information is important for medical segmentation and treatment planning."
        )
        rewritten, _ = rewrite_problem("repair", original, metadata)
        lower = rewritten.lower()
        self.assertIn("visible extent", lower)
        self.assertIn("rounded contour", lower)
        self.assertNotIn("millimeter", lower)
        self.assertNotIn("treatment", lower)
        self.assertNotIn("aneurysm", lower)

        brats = infer_sample_metadata(
            "BraTS20_Training_119_seg_slice_70",
            "The whole tumor is irregular.",
        )
        brats_query, _ = rewrite_problem(
            "brats-repair",
            "The whole tumor (WT) segmentation in the image is large, irregular, and near the left ventricle.",
            brats,
        )
        self.assertIn("large, irregular", brats_query.lower())
        self.assertNotIn("segmentation", brats_query.lower())
        self.assertNotIn("tumor", brats_query.lower())

        regular_query, _ = rewrite_problem(
            "regular-retina",
            "The optic disc appears to be normal in size, shape, color, and position within the image.",
            infer_sample_metadata("034", "The optic disc is visible."),
        )
        self.assertIn("regular overall appearance", regular_query.lower())
        self.assertNotIn("optic disc", regular_query.lower())

    def test_rewrite_is_deterministic_diverse_and_implicit(self):
        metadata = infer_sample_metadata("10055B_10055B_aneurysms_slice_57", "The aneurysm is round.")
        original = "In summary, the intracranial aneurysm is round, bright, and located on the left side."
        first = rewrite_problem("key", original, metadata)
        self.assertEqual(first, rewrite_problem("key", original, metadata))
        self.assertNotIn("aneurysm", first[0].lower())
        self.assertNotIn("the the", first[0].lower())
        self.assertNotIn("an the", first[0].lower())
        article = rewrite_problem("article", "The tumor shows a rounded shape.", infer_sample_metadata(
            "BraTS20_Training_343_seg_slice_44", "The whole tumor is irregular."
        ))[0]
        self.assertNotIn("a abnormal", article.lower())
        self.assertNotIn("(wt)", article.lower())
        modified = rewrite_problem("modified", "A large tumor occupies the frontal lobe.", infer_sample_metadata(
            "BraTS20_Training_343_seg_slice_44", "The whole tumor is irregular."
        ))[0]
        self.assertNotIn("an large", modified.lower())
        adjective = rewrite_problem("adjective", "An intracranial aneurysmal segment is visible.", metadata)[0]
        self.assertNotIn("aneurysm", adjective.lower())
        styles = {rewrite_problem(f"key-{index}", original, metadata)[1] for index in range(80)}
        self.assertGreaterEqual(len(styles), 7)

    def test_two_points_are_distinct_and_inside_mask(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 6:26] = True
        points = derive_two_interior_points(mask, [6, 8, 25, 23])
        self.assertEqual(len(points), 2)
        self.assertNotEqual(points[0], points[1])
        for x, y in points:
            self.assertTrue(mask[y, x])

    def test_group_split_is_stable(self):
        first = assign_group_split("BraTS2020", "BraTS20_Training_343")
        self.assertEqual(first, assign_group_split("BraTS2020", "BraTS20_Training_343"))
        self.assertIn(first, {"train", "validation", "test"})

    def test_bbox_description_is_grounded_and_deterministic(self):
        first = describe_bbox("sample", [4, 8, 14, 40], 100, 100)
        self.assertEqual(first, describe_bbox("sample", [4, 8, 14, 40], 100, 100))
        self.assertRegex(first, r"small|limited|subtle|tiny")
        self.assertIn("far-left", first)
        self.assertRegex(first, r"vertically elongated|taller-than-wide|longitudinally extended")

        metadata = infer_sample_metadata("10055B_10055B_aneurysms_slice_57", "The aneurysm is round.")
        varied = {
            describe_bbox(f"sample-{index}", [4, 8, 14, 40], 100, 100, metadata)
            for index in range(40)
        }
        self.assertGreaterEqual(len(varied), 4)
        self.assertTrue(all("aneurysm" not in value.lower() for value in varied))

    def test_repair_mode_uses_only_grounded_geometry(self):
        metadata = infer_sample_metadata("10055B_10055B_aneurysms_slice_57", "The aneurysm is round.")
        rewritten, _ = rewrite_problem(
            "geometry-only",
            "The aneurysm measures 99 mm and needs treatment.",
            metadata,
            bbox=[20, 20, 30, 30],
            image_width=100,
            image_height=100,
            use_original_description=False,
        )
        self.assertNotIn("99", rewritten)
        self.assertNotIn("treatment", rewritten.lower())
        self.assertNotIn("aneurysm", rewritten.lower())
        self.assertIn("small", rewritten.lower())
        self.assertNotRegex(rewritten.lower(), r"\b([a-z][a-z-]{2,})\s+\1\b")


if __name__ == "__main__":
    unittest.main()
