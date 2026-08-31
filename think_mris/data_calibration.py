"""Deterministic MRIS-Bench quality-control and query calibration helpers."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class SampleMetadata:
    source_dataset: str
    modality: str
    target_category: str
    target_kind: str
    group_id: str


_REFUSAL_PATTERN = re.compile(
    r"\b(?:i cannot|i can't|cannot provide|unable to|cannot be determined|"
    r"not possible to determine|difficult to provide|without (?:more|specific) "
    r"information|would need to|consult a healthcare professional|"
    r"if you could provide|as a text-based ai|i do not have access)\b",
    re.IGNORECASE,
)
_META_PATTERN = re.compile(
    r"\b(?:medical segmentation|segmentation task|provided as a reference|"
    r"help healthcare professionals|treatment planning|i would be happy|"
    r"appropriate treatment|further evaluation and consultation|"
    r"i(?:'|’)ll describe|i will describe|to provide more information|"
    r"would be helpful|would depend on|depending on the specific|"
    r"better appreciated by examining|would be better described|"
    r"assessment and management|important for understanding|"
    r"potential impact on|the exact details|specific details of the image|"
    r"segmentation process|needs? to be segmented|helps? to identify and differentiate|"
    r"trained person|medical professionals?|expertise in|more detailed summary|"
    r"please describe|keep in mind|it is noteworthy|this segmentation helps|"
    r"important|segmentation|assessment|management|diagnos(?:is|tic)|"
    r"this information|noteworthy|serious medical condition|more accurate assessment)\b",
    re.IGNORECASE,
)
_MULTI_PATHOLOGY_PATTERN = re.compile(
    r"\b(?:two|three|multiple|several|various|numerous)\b.{0,60}"
    r"\b(?:aneurysm|tumou?r|lesion|polyp)s?\b",
    re.IGNORECASE,
)
_NEGATIVE_PATHOLOGY_PATTERN = re.compile(
    r"(?:\bthere (?:is|are) no\b|\bdoes not show any\b|\bno (?:visible |"
    r"evident |apparent )?(?:intracranial )?(?:aneurysm|tumou?r|polyp)s?\b)",
    re.IGNORECASE,
)
_NON_MEDICAL_PATTERN = re.compile(
    r"(?:等于几|答案是|^\s*\d+(?:\.\d+)?\s*[+×÷-]\s*\d+(?:\.\d+)?\s*[?？]?\s*$)",
    re.IGNORECASE,
)
_MEASUREMENT_PATTERN = re.compile(
    r"(?:(?:approximately\s+|about\s+|around\s+)?"
    r"\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?){0,2}\s*-?\s*"
    r"(?:(?:mm|cm|mL|ml|[µμ]m|um)(?:[²2³3])?|"
    r"millimet(?:er|re)s?|centimet(?:er|re)s?|micromet(?:er|re)s?|"
    r"millilit(?:er|re)s?|lit(?:er|re)s?|pixels?|grams?|g|"
    r"HU|Hounsfield units?|degrees?|°|%))(?![A-Za-z])",
    re.IGNORECASE,
)
_DUPLICATED_WORD_PATTERN = re.compile(r"\b([a-z][a-z-]{2,})\s+\1\b", re.IGNORECASE)
_UNSUPPORTED_DIAGNOSIS_PATTERN = re.compile(
    r"\b(?:may|might|could|likely|potentially|may (?:indicate|represent|suggest)|"
    r"patient(?:'s|’s)|proper diagnosis|diagnosed with|glioblastoma|astrocytoma|"
    r"metastatic disease|subarachnoid hemorrhage|wall motion abnormality|fibrosis|"
    r"medical condition|cardiomyopathy|amyloidosis|hypertrophy|thrombus|"
    r"necrotic|necrosis|hemorrhage|calcified|stroke|complications?|rupture|risk)\b",
    re.IGNORECASE,
)
_NON_VISUAL_META_PATTERN = re.compile(
    r"\b(?:not (?:directly |explicitly )?(?:mentioned|provided|described|shown|available)|"
    r"not clearly (?:visible|distinguishable|defined)|"
    r"exact (?:size|shape|color|position|dimensions?|details)|"
    r"more specific information|if more information|if available|"
    r"more detailed (?:analysis|examination|information)|"
    r"based on the image|can also be seen in the image|specific case|"
    r"specific image|imaging technique used|from the description|in the context|"
    r"previously (?:identified|described|seen|mentioned)|"
    r"normal (?:in |for )?(?:its )?size, shape, color, and position|"
    r"would|can vary|this means|(?:tumou?r|polyp|aneurysm) image)\b",
    re.IGNORECASE,
)
_PATHOLOGY_PATTERN = re.compile(r"\b(?:aneurysm|tumou?r|polyp|lesion)s?\b", re.IGNORECASE)
_PLURAL_TARGET_PATTERN = re.compile(r"\b(?:aneurysms|tumou?rs|polyps|lesions)\b", re.IGNORECASE)
_NON_PHOTOGRAPHIC_COLOR_PATTERN = re.compile(
    r"\b(?:red|blue|green|yellow|orange|purple|pink)(?:ish|-\w+)?\b",
    re.IGNORECASE,
)
_ADVICE_SENTENCE_PATTERN = re.compile(
    r"\b(?:healthcare professional|treatment|clinical history|proper diagnosis|"
    r"further evaluation|medical segmentation|reference for segmentation|"
    r"provided as a reference)\b",
    re.IGNORECASE,
)
_VISUAL_CUE_PATTERN = re.compile(
    r"\b(?:small|large|round|rounded|oval|irregular|circular|elongated|narrow|"
    r"well[- ]defined|poorly[- ]defined|heterogeneous|homogeneous|bright|dark|"
    r"hyperintense|hypointense|mixed|central|peripheral|superior|inferior|"
    r"medial|lateral|left|right|frontal|parietal|temporal|occipital|"
    r"located|position|contour|outline|margin|appearance|extent|mass)\b",
    re.IGNORECASE,
)
_FUNCTIONAL_OR_EXPLANATORY_PATTERN = re.compile(
    r"\b(?:responsible for|pumps? (?:oxygen|blood)|receives? (?:oxygen|blood)|"
    r"carries? oxygen|visual representation|retinal ganglion|transmit visual|"
    r"healthcare professionals?|medical professionals?|treatment|clinical setting|"
    r"which (?:helps?|can help|is useful|allows?)|helps? (?:to|in)|"
    r"is a type of|refers to|provided as a reference)\b",
    re.IGNORECASE,
)

_HARD_QUALITY_REASONS = frozenset(
    {
        "empty_or_too_short_problem",
        "non_mris_content",
        "generation_refusal",
        "target_negated_despite_positive_geometry",
        "multiple_targets_with_single_geometry",
        "plural_target_with_single_geometry",
        "unrelated_pathology_in_anatomy_query",
        "unrecoverable_source_or_target",
        "modality_contradiction",
        "source_sequence_contradiction",
        "bilateral_location_for_single_target",
    }
)


def _cardiac_target(problem: str) -> str:
    text = problem.lower()
    patterns = (
        ("ascending_aorta", "ascending aorta"),
        ("descending_aorta", "descending aorta"),
        ("pulmonary_artery", "pulmonary artery"),
        ("left_atrium", "left atrium"),
        ("right_atrium", "right atrium"),
        ("left_ventricle_myocardium", "left ventricle myocardium"),
        ("left_ventricle_myocardium", "left ventricular myocardium"),
        ("left_ventricle_cavity", "left ventricle blood cavity"),
        ("left_ventricle_cavity", "left ventricular cavity"),
        ("right_ventricle_cavity", "right ventricle blood cavity"),
        ("right_ventricle_cavity", "right ventricular cavity"),
    )
    for label, phrase in patterns:
        if phrase in text:
            return label
    return "cardiac_structures"


def infer_sample_metadata(sample_id: str, problem: str) -> SampleMetadata:
    """Recover only metadata that is supported by stable public identifiers."""
    sample_id = str(sample_id)
    lower_id = sample_id.lower()
    lower_problem = (problem or "").lower()

    if "aneurysms_slice_" in lower_id:
        group = sample_id.split("_", 1)[0]
        return SampleMetadata("ADAM", "MR", "intracranial_aneurysm", "pathology", group)
    if re.match(r"patient\d+_(?:2ch|4ch)_(?:ed|es)$", lower_id):
        group = sample_id.split("_", 1)[0]
        return SampleMetadata("CAMUS", "ultrasound", "cardiac_structures", "anatomy", group)
    if "optic cup" in lower_problem or "optic disc" in lower_problem:
        target = "optic_cup" if "optic cup" in lower_problem else "optic_disc"
        return SampleMetadata("CFP_sources_mixed", "CFP", target, "anatomy", sample_id)
    if re.match(r"case\d+_segmentation_slice_\d+$", lower_id):
        group = sample_id.rsplit("_slice_", 1)[0]
        return SampleMetadata("PROMISE12_centers_mixed", "MR", "prostate", "anatomy", group)
    if lower_id.startswith("roi_ct_train_"):
        group = sample_id.rsplit("_slice_", 1)[0]
        return SampleMetadata("MM-WHS-CT", "CT", _cardiac_target(problem), "anatomy", group)
    if lower_id.startswith("roi_mr_train_"):
        group = sample_id.rsplit("_slice_", 1)[0]
        return SampleMetadata("MM-WHS-MRI", "MR", _cardiac_target(problem), "anatomy", group)
    if lower_id.startswith("brats20_training_"):
        match = re.match(r"(BraTS20_Training_\d+)", sample_id, re.IGNORECASE)
        group = match.group(1) if match else sample_id.rsplit("_slice_", 1)[0]
        return SampleMetadata("BraTS2020", "MR", "whole_brain_tumor", "pathology", group)
    if re.search(r"\b(?:colorectal )?polyp\b|\bmucosal lesion\b", lower_problem):
        return SampleMetadata("endoscopy_sources_mixed", "endoscopy", "colorectal_polyp", "pathology", sample_id)
    if "lesion" in lower_problem and not lower_id.startswith(("brats", "roi_")):
        return SampleMetadata("endoscopy_sources_mixed", "endoscopy", "colorectal_polyp", "pathology", sample_id)
    return SampleMetadata("unknown", "unknown", "unknown", "unknown", sample_id)


def validate_legacy_geometry(
    bbox: object,
    point: object,
    width: int,
    height: int,
) -> List[str]:
    reasons: List[str] = []
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
    ):
        return ["invalid_bbox_shape"]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        reasons.append("invalid_or_out_of_bounds_bbox")
    if not (
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
    ):
        reasons.append("invalid_legacy_point")
    else:
        x, y = (float(value) for value in point)
        if not (0 <= x < width and 0 <= y < height):
            reasons.append("legacy_point_out_of_image")
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            reasons.append("legacy_point_out_of_bbox")
    return reasons


def quality_reasons(
    problem: str,
    metadata: SampleMetadata,
    *,
    max_words: int = 140,
) -> List[str]:
    text = (problem or "").strip()
    lower = text.lower()
    reasons: List[str] = []
    word_count = len(text.split())
    if not text or word_count < 5:
        reasons.append("empty_or_too_short_problem")
    if word_count > max_words:
        reasons.append("excessively_long_problem")
    if _NON_MEDICAL_PATTERN.search(text):
        reasons.append("non_mris_content")
    if _REFUSAL_PATTERN.search(text):
        reasons.append("generation_refusal")
    if _META_PATTERN.search(text):
        reasons.append("generation_meta_text")
    if _NON_VISUAL_META_PATTERN.search(text):
        reasons.append("non_visual_or_missing_description")
    if _DUPLICATED_WORD_PATTERN.search(text):
        reasons.append("duplicated_generation_tokens")
    if _MEASUREMENT_PATTERN.search(text):
        reasons.append("unsupported_numeric_measurement")
    if _UNSUPPORTED_DIAGNOSIS_PATTERN.search(text):
        reasons.append("unsupported_diagnostic_claim")
    if metadata.target_kind == "pathology" and _NEGATIVE_PATHOLOGY_PATTERN.search(text):
        reasons.append("target_negated_despite_positive_geometry")
    if _MULTI_PATHOLOGY_PATTERN.search(text):
        reasons.append("multiple_targets_with_single_geometry")
    if _PLURAL_TARGET_PATTERN.search(text):
        reasons.append("plural_target_with_single_geometry")
    if metadata.target_kind == "anatomy" and _PATHOLOGY_PATTERN.search(text):
        reasons.append("unrelated_pathology_in_anatomy_query")
    if metadata.source_dataset == "unknown":
        reasons.append("unrecoverable_source_or_target")

    mentions_ct = bool(re.search(r"\bct(?: scan| image)?\b", lower))
    mentions_mr = bool(re.search(r"\b(?:mri|mra|mr image|t[12]-weighted)\b", lower))
    mentions_ultrasound = bool(re.search(r"\b(?:ultrasound|echocardiograph|sonograph)\w*\b", lower))
    mentions_oct = bool(re.search(r"\b(?:OCT|optical coherence tomography)\b", text, re.IGNORECASE))
    mentions_hounsfield = bool(re.search(r"\b(?:HU|Hounsfield units?)\b", text, re.IGNORECASE))
    if metadata.modality == "ultrasound" and (mentions_ct or mentions_mr):
        reasons.append("modality_contradiction")
    elif metadata.modality == "CFP" and (mentions_ct or mentions_mr or mentions_ultrasound or mentions_oct):
        reasons.append("modality_contradiction")
    elif metadata.modality == "endoscopy" and (mentions_ct or mentions_mr or mentions_ultrasound):
        reasons.append("modality_contradiction")
    elif metadata.modality == "MR" and (mentions_ct or mentions_hounsfield):
        reasons.append("modality_contradiction")
    elif metadata.modality == "CT" and (mentions_mr or mentions_ultrasound):
        reasons.append("modality_contradiction")
    if metadata.modality in {"MR", "CT", "ultrasound"} and _NON_PHOTOGRAPHIC_COLOR_PATTERN.search(text):
        reasons.append("non_photographic_color_claim")
    if metadata.source_dataset == "ADAM":
        if re.search(r"\b(?:T1|T2|diffusion-weighted|DWI)\b", text, re.IGNORECASE):
            reasons.append("source_sequence_contradiction")
        if re.search(r"\bleft\b", lower) and re.search(r"\bright\b", lower):
            reasons.append("bilateral_location_for_single_target")
    return sorted(set(reasons))


def hard_quality_reasons(
    problem: str,
    metadata: SampleMetadata,
    *,
    max_words: int = 140,
) -> List[str]:
    """Return only defects that cannot be repaired without inventing supervision."""
    return [
        reason
        for reason in quality_reasons(problem, metadata, max_words=max_words)
        if reason in _HARD_QUALITY_REASONS
    ]


def _replace_measurements(text: str) -> str:
    text = re.sub(
        rf"\bmeasuring\s+(?:approximately\s+)?{_MEASUREMENT_PATTERN.pattern}",
        "with a visible extent",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(?:measures?|measured)\s+(?:approximately\s+)?{_MEASUREMENT_PATTERN.pattern}",
        "has a visible extent",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(?:with|has)\s+(?:a\s+)?(?:size|diameter|dimensions?)\s+of\s+{_MEASUREMENT_PATTERN.pattern}",
        "has a visible extent",
        text,
        flags=re.IGNORECASE,
    )
    text = _MEASUREMENT_PATTERN.sub("visibly sized", text)
    text = re.sub(
        r"\b(?:cup-to-disc\s+)?ratio\s+of\s+\d+(?:\.\d+)?\b",
        "relative extent",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d+(?:\.\d+)?-visibly sized\b", "a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:visibly sized|a visible extent)(?:\s*(?:x|by)\s*(?:visibly sized|a visible extent))+\b",
        "a visible extent",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:and\s+)?color\s+of\s+(?:a\s+)?(?:visibly sized|visible extent)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:size|diameter|length|width|height|thickness|dimensions?)\s+of\s+(?:a\s+)?(?:visibly sized|visible extent)\b",
        "visible extent",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bvisibly sized\b", "a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:a visible extent)(?:\s+in\s+(?:size|diameter|length|width|height|thickness))\b",
        "a visible extent",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bvisibly sized\s+in size\b", "visibly sized", text, flags=re.IGNORECASE)
    return text


def _neutralize_target(
    text: str,
    metadata: SampleMetadata,
    *,
    selector_text: Optional[str] = None,
) -> str:
    replacements: Dict[str, Tuple[str, ...]] = {
        "intracranial_aneurysm": (r"(?:intracranial\s+)?aneurysm(?:s|al)?",),
        "optic_cup": (r"optic\s+cup",),
        "optic_disc": (r"optic\s+disc",),
        "prostate": (r"prostate(?:\s+gland)?",),
        "colorectal_polyp": (r"(?:colorectal\s+)?polyps?", r"(?:mucosal\s+)?lesions?"),
        "whole_brain_tumor": (
            r"whole\s+tumou?r(?:\s*\(WT\))?",
            r"entire\s+tumou?r\s+region",
            r"tumou?r\s+region",
            r"tumou?rs?",
        ),
        "ascending_aorta": (r"ascending\s+aorta",),
        "descending_aorta": (r"descending\s+aorta",),
        "pulmonary_artery": (r"pulmonary\s+artery",),
        "left_atrium": (r"left\s+atri(?:um|al)(?:\s+blood\s+cavity)?",),
        "right_atrium": (r"right\s+atri(?:um|al)(?:\s+blood\s+cavity)?",),
        "left_ventricle_myocardium": (r"left\s+ventric(?:le|ular)\s+myocardium",),
        "left_ventricle_cavity": (r"left\s+ventric(?:le|ular)(?:\s+blood)?\s+cavity", r"left\s+ventricle"),
        "right_ventricle_cavity": (r"right\s+ventric(?:le|ular)(?:\s+blood)?\s+cavity", r"right\s+ventricle"),
    }
    referents = {
        "intracranial_aneurysm": (
            "focal vascular abnormality",
            "localized vessel-associated bulge",
            "focal vascular outpouching",
            "localized arterial outpouching",
            "vessel-contour protrusion",
            "rounded vessel-wall change",
        ),
        "optic_cup": (
            "central retinal depression",
            "inner retinal depression",
            "vessel-bend-defined inner region",
            "central vessel-defined depression",
            "recessed central retinal zone",
            "inner circular retinal region",
        ),
        "optic_disc": (
            "larger circular retinal structure",
            "optic-nerve entry region",
            "outer retinal landmark",
            "vessel-convergence landmark",
            "circular fundus landmark",
            "retinal nerve-head region",
        ),
        "prostate": (
            "glandular pelvic structure",
            "pelvic glandular region",
            "gland beneath the bladder",
            "midline pelvic glandular tissue",
            "rounded sub-bladder region",
            "pelvic organ region",
        ),
        "colorectal_polyp": (
            "raised mucosal region",
            "focal mucosal protrusion",
            "surface-elevated mucosal area",
            "protruding luminal surface",
            "localized raised tissue",
            "discrete mucosal bump",
        ),
        "whole_brain_tumor": (
            "abnormal brain tissue",
            "heterogeneous intracranial abnormality",
            "abnormal parenchymal region",
            "focal intracranial signal abnormality",
            "space-occupying brain region",
            "irregular intra-axial area",
        ),
    }
    cardiac_referents = (
        "target cardiac structure",
        "referenced cardiovascular region",
        "indicated cardiac region",
        "outlined cardiovascular structure",
        "specified cardiac area",
        "selected heart region",
    )
    choices = referents.get(metadata.target_category, cardiac_referents)
    selector = hashlib.sha256(
        f"{metadata.target_category}\0{selector_text or text}".encode("utf-8")
    ).digest()[0]
    replacement = choices[selector % len(choices)]

    def substitute(match: re.Match) -> str:
        article_value = (match.group("article") or "").lower()
        modifier = (match.group("modifier") or "").lower()
        article_target = modifier or replacement
        starts_with_vowel = article_target.startswith(tuple("aeiou"))
        if article_value == "an" and not starts_with_vowel:
            article_value = "a"
        elif article_value == "a" and starts_with_vowel:
            article_value = "an"
        pieces = [value for value in (article_value, modifier, replacement) if value]
        return " ".join(pieces)

    for pattern in replacements.get(metadata.target_category, ()):
        qualified_pattern = (
            rf"\b(?:(?P<article>the|an?|these|those)\s+)?"
            rf"(?:(?P<modifier>possible|suspected|small|large|dominant)\s+)?"
            rf"(?:{pattern})\b"
        )
        text = re.sub(qualified_pattern, substitute, text, flags=re.IGNORECASE)
    if metadata.target_category == "cardiac_structures":
        text = re.sub(r"\bheart structures?\b", replacement, text, flags=re.IGNORECASE)
    return text


def _geometry_target_referent(sample_key: str, metadata: SampleMetadata) -> str:
    canonical_targets = {
        "intracranial_aneurysm": "intracranial aneurysm",
        "optic_cup": "optic cup",
        "optic_disc": "optic disc",
        "prostate": "prostate",
        "colorectal_polyp": "colorectal polyp",
        "whole_brain_tumor": "whole tumor",
        "ascending_aorta": "ascending aorta",
        "descending_aorta": "descending aorta",
        "pulmonary_artery": "pulmonary artery",
        "left_atrium": "left atrium",
        "right_atrium": "right atrium",
        "left_ventricle_myocardium": "left ventricular myocardium",
        "left_ventricle_cavity": "left ventricular cavity",
        "right_ventricle_cavity": "right ventricular cavity",
        "cardiac_structures": "heart structure",
    }
    canonical = canonical_targets.get(metadata.target_category, "heart structure")
    referent = _neutralize_target(
        f"the {canonical}",
        metadata,
        selector_text=sample_key,
    )
    return re.sub(r"^the\s+", "", referent, flags=re.IGNORECASE)


def clean_description(problem: str, metadata: SampleMetadata) -> str:
    text = re.sub(r"^\s*(?:in summary[,;:]?|sure[,!]?\s*)", "", problem.strip(), flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:appears? to (?:be|have)|is|has) (?:of )?normal (?:in )?size, shape, color, and position\b",
        "has a regular overall appearance and expected position",
        text,
        flags=re.IGNORECASE,
    )
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    unsupported = re.compile(
        r"\b(?:not explicitly (?:mentioned|specified)|not specified|not visible|"
        r"important to note|actual (?:size|shape|position)|cannot be seen|"
        r"Fazekas|(?:type|grade)\s+[IVX]+|is a type of)\b",
        re.IGNORECASE,
    )
    retained = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence = re.sub(
            r"\bsegmentation(?:\s+(?:in|on)\s+(?:the|this)\s+(?:MRI\s+)?image)?\b",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        sentence = re.sub(r"\bwell[- ]segmented\b", "well-defined", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\bsegmented(?:\s+in\s+\w+)?\b", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(
            r"\b(?:could|may|might) be described as\b",
            "appears",
            sentence,
            flags=re.IGNORECASE,
        )
        sentence = re.sub(
            r",?\s+(?:which|that) (?:may|might|could|likely).*?$",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        sentence = re.sub(
            r",?\s+(?:suggesting|indicating|consistent with)\b.*?$",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        sentence = re.sub(
            r",?\s+which (?:helps?|can help|is useful|allows?).*?$",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        if _ADVICE_SENTENCE_PATTERN.search(sentence) or unsupported.search(sentence):
            continue
        if _FUNCTIONAL_OR_EXPLANATORY_PATTERN.search(sentence):
            continue
        if _NON_VISUAL_META_PATTERN.search(sentence):
            continue
        if _UNSUPPORTED_DIAGNOSIS_PATTERN.search(sentence):
            continue
        if _META_PATTERN.search(sentence) and not _VISUAL_CUE_PATTERN.search(sentence):
            continue
        retained.append(sentence)
    text = " ".join(retained)
    text = _replace_measurements(text)
    if metadata.modality in {"MR", "CT", "ultrasound"}:
        text = _NON_PHOTOGRAPHIC_COLOR_PATTERN.sub("visually distinct", text)
    text = _neutralize_target(text, metadata)
    text = re.sub(r"\s*\((?:WT|LA|LV|RA|RV|AA|PA)\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:segmentation of (?:the )?|segmentation in (?:the )?)", "", text, flags=re.IGNORECASE)
    referent_pattern = "|".join(
        re.escape(value)
        for value in (
            "focal vascular abnormality",
            "localized vessel-associated bulge",
            "focal vascular outpouching",
            "central retinal depression",
            "inner retinal depression",
            "vessel-bend-defined inner region",
            "larger circular retinal structure",
            "optic-nerve entry region",
            "outer retinal landmark",
            "glandular pelvic structure",
            "pelvic glandular region",
            "gland beneath the bladder",
            "raised mucosal region",
            "focal mucosal protrusion",
            "surface-elevated mucosal area",
            "abnormal brain tissue",
            "heterogeneous intracranial abnormality",
            "abnormal parenchymal region",
            "target cardiac structure",
            "referenced cardiovascular region",
            "indicated cardiac region",
        )
    )
    text = re.sub(
        rf"\bthe (?:{referent_pattern}) (?P<modality>MRI |CT )?image\b",
        lambda match: f"the {match.group('modality') or ''}image",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b([a-z][a-z-]{2,})\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"^the image (?:shows|depicts|demonstrates|contains)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^in (?:this|the) image[, ]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r",?\s+which means.*?(?=\.|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r",?\s+meaning that.*?(?=\.|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:is described as being|appears to be) a visible extent in size\b", "has a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:has|with) a visible extent in size\b", "has a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:is described as being|appears to be|is) visibly sized(?: in diameter)?\b", "has a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:having|has) a size of visibly sized\b", "having a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvisibly sized in diameter\b", "with a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhas a visible extent,\s*has\b", "has a visible extent and", text, flags=re.IGNORECASE)
    text = re.sub(r"\bstructure has a visible extent\b", "structure with a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the )?size of ([^.]+?) is visibly sized\b", r"\1 is visibly sized", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:a|an)\s+(?:a|an)\b", "a", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the|a|an|these|those)\s+the\b", "the", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(has|with) a visible extent\s+(?:and\s+)?\1 a visible extent\b", r"\1 a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba visible extent\s+(?:x|by)\s+a visible extent\b", "a visible extent", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .?!:;")
    supported_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
        and not _META_PATTERN.search(sentence)
        and not _NON_VISUAL_META_PATTERN.search(sentence)
        and not _UNSUPPORTED_DIAGNOSIS_PATTERN.search(sentence)
        and not _FUNCTIONAL_OR_EXPLANATORY_PATTERN.search(sentence)
    ]
    text = " ".join(supported_sentences)
    words = text.split()
    if len(words) > 90:
        text = " ".join(words[:90]).rstrip(",;:")
    text = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        text,
    )
    return text


def describe_bbox(
    sample_key: str,
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    metadata: Optional[SampleMetadata] = None,
) -> str:
    """Create qualitative, annotation-grounded cues without exposing coordinates."""
    if image_width <= 0 or image_height <= 0 or len(bbox) != 4:
        raise ValueError("valid image dimensions and one bounding box are required")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    area_ratio = (box_width * box_height) / float(image_width * image_height)
    digest = hashlib.sha256(f"{sample_key}|{list(bbox)}".encode("utf-8")).digest()
    if area_ratio < 0.01:
        size_choices = ("very small", "tiny", "subtle")
    elif area_ratio < 0.04:
        size_choices = ("small", "limited", "relatively small")
    elif area_ratio < 0.15:
        size_choices = ("moderately sized", "medium-scale", "mid-sized")
    else:
        size_choices = ("broad", "wide-ranging", "large")
    size = size_choices[digest[1] % len(size_choices)]

    aspect_ratio = box_width / box_height
    if aspect_ratio > 1.6:
        shape_choices = ("horizontally elongated", "wider-than-tall", "laterally extended")
    elif aspect_ratio < 0.625:
        shape_choices = ("vertically elongated", "taller-than-wide", "longitudinally extended")
    else:
        shape_choices = ("roughly compact", "near-equiaxed", "proportionally balanced")
    shape = shape_choices[digest[2] % len(shape_choices)]

    center_x = (x1 + x2) / (2.0 * image_width)
    center_y = (y1 + y2) / (2.0 * image_height)
    horizontal = (
        "far-left"
        if center_x < 0.20
        else "left-of-center"
        if center_x < 0.40
        else "central"
        if center_x < 0.60
        else "right-of-center"
        if center_x < 0.80
        else "far-right"
    )
    vertical = (
        "upper"
        if center_y < 0.20
        else "upper-middle"
        if center_y < 0.40
        else "middle"
        if center_y < 0.60
        else "lower-middle"
        if center_y < 0.80
        else "lower"
    )
    if horizontal == "central":
        location = vertical
    elif vertical == "middle":
        location = f"middle {horizontal}"
    else:
        location = f"{vertical} {horizontal}"
    referent = _geometry_target_referent(sample_key, metadata) if metadata is not None else "region"

    templates = (
        "The {referent} occupies a {size}, {shape} area in the {location} part of the image",
        "This {referent} is {size} and {shape}, centered in the {location} image area",
        "The {referent} appears {size}, {shape}, and positioned in the {location} portion of the image",
        "The {referent} forms a {size}, {shape} region in the {location} part of the image",
        "The {referent}, with a {size} extent and a {shape} profile, appears in the {location} image area",
        "The {referent} has a {size}, {shape} footprint situated in the {location} portion of the image",
        "The {referent} is identifiable as a {size}, {shape} area toward the {location} part of the image",
        "Within the {location} image area lies the {size}, {shape} {referent}",
    )
    return templates[digest[0] % len(templates)].format(
        size=size,
        shape=shape,
        location=location,
        referent=referent,
    )


_QUESTION_TEMPLATES = (
    ("which_region", "Which region matches these visual and spatial cues: {description}?"),
    ("where_is_target", "Where is the region described by these cues: {description}?"),
    ("can_you_segment", "Can you segment the region that matches this description: {description}?"),
    ("could_you_outline", "Could you outline the target indicated by the following features: {description}?"),
    ("what_area", "What area best corresponds to these appearance and location cues: {description}?"),
    ("which_target", "Which target in the image is consistent with this description: {description}?"),
    ("please_delineate", "Please delineate the region matching these cues: {description}."),
    ("identify_and_segment", "Identify and segment the target described here: {description}."),
)


def rewrite_problem(
    sample_key: str,
    problem: str,
    metadata: SampleMetadata,
    *,
    bbox: Optional[Sequence[float]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    use_original_description: bool = True,
) -> Tuple[str, str]:
    description = clean_description(problem, metadata) if use_original_description else ""
    if bbox is not None and image_width is not None and image_height is not None:
        geometry_cue = describe_bbox(sample_key, bbox, image_width, image_height, metadata)
        description = f"{description}. {geometry_cue}" if description else geometry_cue
    digest = hashlib.sha256(f"{sample_key}\0{problem}".encode("utf-8")).digest()
    style, template = _QUESTION_TEMPLATES[digest[0] % len(_QUESTION_TEMPLATES)]
    query = template.format(description=description)
    query = re.sub(r"\s+", " ", query).replace(".?", "?").strip()
    return query, style


def stable_sample_id(
    metadata: SampleMetadata,
    original_id: str,
    bbox: Sequence[float],
    rewritten_problem: str,
) -> str:
    payload = f"{metadata.source_dataset}|{original_id}|{metadata.target_category}|{list(bbox)}|{rewritten_problem}"
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    source = re.sub(r"[^a-z0-9]+", "-", metadata.source_dataset.lower()).strip("-")
    return f"{source}:{original_id}:{metadata.target_category}:{suffix}"


def assign_group_split(source_dataset: str, group_id: str) -> str:
    bucket = int(hashlib.sha256(f"{source_dataset}|{group_id}".encode("utf-8")).hexdigest()[:8], 16) % 1000
    if bucket < 100:
        return "test"
    if bucket < 200:
        return "validation"
    return "train"


def mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
    rows, columns = np.where(mask.astype(bool))
    if columns.size == 0:
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def derive_two_interior_points(mask: np.ndarray, bbox: Sequence[float]) -> List[List[int]]:
    """Implement the supplement's deepest-point plus separated-point rule."""
    binary = np.asarray(mask, dtype=bool).copy()
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = binary.shape
    x1 = max(0, min(width - 1, int(math.floor(bbox[0]))))
    y1 = max(0, min(height - 1, int(math.floor(bbox[1]))))
    x2 = max(0, min(width - 1, int(math.ceil(bbox[2]))))
    y2 = max(0, min(height - 1, int(math.ceil(bbox[3]))))
    restricted = np.zeros_like(binary)
    restricted[y1 : y2 + 1, x1 : x2 + 1] = binary[y1 : y2 + 1, x1 : x2 + 1]
    if int(restricted.sum()) < 2:
        raise ValueError("at least two foreground pixels are required")

    distances = distance_transform_edt(restricted)
    first_y, first_x = np.unravel_index(int(np.argmax(distances)), distances.shape)
    rows, columns = np.where(restricted)
    separation = np.hypot(columns - first_x, rows - first_y)
    minimum_separation = max(2.0, 0.2 * math.hypot(x2 - x1, y2 - y1))
    candidates = separation >= minimum_separation
    if not np.any(candidates):
        candidates = separation > 0
    if not np.any(candidates):
        raise ValueError("could not find a distinct second foreground point")
    candidate_rows = rows[candidates]
    candidate_columns = columns[candidates]
    candidate_separation = separation[candidates]
    boundary_depth = distances[candidate_rows, candidate_columns]
    score = boundary_depth + 0.05 * candidate_separation
    second = int(np.argmax(score))
    second_x, second_y = int(candidate_columns[second]), int(candidate_rows[second])
    return [[int(first_x), int(first_y)], [second_x, second_y]]


def count_direct_target_mentions(texts: Iterable[Tuple[str, str]]) -> int:
    direct_patterns = {
        "intracranial_aneurysm": r"\baneurysm",
        "optic_cup": r"\boptic cup\b",
        "optic_disc": r"\boptic disc\b",
        "prostate": r"\bprostate\b",
        "colorectal_polyp": r"\bpolyp\b",
        "whole_brain_tumor": r"\b(?:whole )?tumou?r\b",
    }
    return sum(bool(re.search(direct_patterns.get(category, r"(?!x)x"), text, re.IGNORECASE)) for category, text in texts)
