"""
Cross-episode attribute-skew detector and mitigation, addressing the
spurious correlation confirmed in DEVIATIONS_LOG.md #27-29: when every
verified reflection sharing a given outcome label happens to share one
particular value of some OTHER attribute (e.g. all ANOMALY examples
being red objects), the model can spuriously treat that attribute as
necessary for the label, even though no individual reflection asserts
this and every reflection is independently correct.

This detector does NOT require knowing which attribute is causally
relevant -- it purely looks for coincidental within-label uniformity
against an attribute that varies elsewhere in the bank, which is
exactly the pattern that #27-29 showed causes the failure. It is
deliberately data-free: no additional consolidation or data collection
is needed, unlike a mitigation that tries to actively seek out more
diverse examples (which may not always be available in a real
deployment).
"""
from collections import defaultdict

ATTRS = ["shape", "color", "material", "size"]


def detect_attribute_skew(bank, min_examples: int = 2) -> dict:
    """
    Returns {label: {attribute: skewed_value}} for every (label, attribute)
    pair where ALL bank tuples carrying that label share one value of
    that attribute, while the attribute takes more than one value
    somewhere else in the bank (so flagging it is actionable -- an
    attribute that never varies at all isn't a meaningful skew to warn
    about).
    """
    by_label = defaultdict(list)
    for t in bank.tuples:
        by_label[t.combo["label"]].append(t.combo)

    all_attr_values = defaultdict(set)
    for t in bank.tuples:
        for attr in ATTRS:
            if attr in t.combo:
                all_attr_values[attr].add(t.combo[attr])

    skew = {}
    for label, combos in by_label.items():
        if len(combos) < min_examples:
            continue
        label_skew = {}
        for attr in ATTRS:
            if attr not in combos[0] or len(all_attr_values[attr]) < 2:
                continue  # doesn't vary anywhere in the bank -- not actionable to flag
            values_for_label = {c[attr] for c in combos if attr in c}
            if len(values_for_label) == 1:
                label_skew[attr] = values_for_label.pop()
        if label_skew:
            skew[label] = label_skew
    return skew


def format_skew_warning(skew: dict) -> str:
    """Turns a detect_attribute_skew() result into a natural-language
    warning to inject into the inference prompt. Empty string if no skew."""
    if not skew:
        return ""
    lines = []
    for label, attrs in skew.items():
        for attr, value in attrs.items():
            lines.append(
                f"Note: every '{label}' example shown happens to have {attr}={value}. "
                f"This may be a coincidence of which examples were available rather than "
                f"a real requirement -- do NOT assume {attr} is necessary for the {label} "
                f"condition. Judge the target image on its own visual content and on how "
                f"the examples' rules generalize, not on matching this incidental pattern."
            )
    return " ".join(lines)


def diversify_bank_keys(bank_keys, library: dict, attrs=ATTRS, min_diversity: int = 2) -> list:
    """
    STRUCTURAL mitigation (see DEVIATIONS_LOG.md #30 for why the
    prompt-warning approach failed and #31 for this alternative).
    Given a list of combo keys intended for a memory bank, and the full
    reflection library (as produced by
    experiments_visual/diagnose_controlled_bias_induction.py's Step A --
    {str(key): {"key", "label", "verified", "reflection", "image_path"}}),
    checks each label present in bank_keys for attribute-level skew and,
    where the library has additional VERIFIED entries of that label with
    a different value of the skewed attribute, adds them to the bank
    BEFORE inference ever runs -- no prompt-level warning is used; the
    model is simply never shown a skewed context in the first place.

    Returns the (possibly expanded) list of combo keys. Does not remove
    anything from the input list, only adds.
    """
    from collections import defaultdict

    by_label_keys = defaultdict(list)
    for k in bank_keys:
        entry = library.get(str(k))
        if entry and entry["verified"]:
            by_label_keys[entry["label"]].append(k)

    all_attr_values_in_library = defaultdict(set)
    for entry in library.values():
        if entry["verified"]:
            for i, attr in enumerate(attrs):
                all_attr_values_in_library[attr].add(entry["key"][i])

    expanded = list(bank_keys)
    for label, keys in by_label_keys.items():
        for attr_idx, attr in enumerate(attrs):
            if len(all_attr_values_in_library[attr]) < 2:
                continue  # doesn't vary anywhere in the library -- nothing to diversify with
            values_present = {library[str(k)]["key"][attr_idx] for k in keys}
            if len(values_present) >= min_diversity:
                continue
            needed_values = all_attr_values_in_library[attr] - values_present
            for entry in library.values():
                if not entry["verified"] or entry["label"] != label:
                    continue
                if entry["key"][attr_idx] in needed_values:
                    key_tuple = tuple(entry["key"])
                    if key_tuple not in expanded:
                        expanded.append(key_tuple)
                        values_present.add(entry["key"][attr_idx])
                    if len(values_present) >= min_diversity:
                        break
    return expanded
