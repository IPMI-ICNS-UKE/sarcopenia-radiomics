from config import FEATURE_LEVEL_LABELS, SCORE_LEVEL_LABELS


def make_feature_label(feature_name: str) -> str:
    """Return a human-readable label for a raw radiomics feature name."""
    # 1. Manual lookup
    if feature_name in FEATURE_LEVEL_LABELS:
        return FEATURE_LEVEL_LABELS[feature_name]
    # 2. Auto: split on "_", place each part on a new line
    # parts = feature_name.split("_")
    # return "\n".join(parts)
    return feature_name


def make_score_label(feature_name: str) -> str:
    """Return a human-readable label for a score-level feature name."""
    return SCORE_LEVEL_LABELS.get(feature_name, feature_name)
