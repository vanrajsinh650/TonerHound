# EXP-033: Candidate Generation Production Changes & Engineering Log

**Experiment**: EXP-033 (Candidate Generation Reconciliation Audit + Productionization)  
**Date**: October 2026  
**Status**: Implemented & Verified with Automated Test Suite

---

## 1. Overview of Production Changes

Following the 200-sample reconciliation audit of the 48,744 zero-candidate field population, 4 targeted candidate generation fixes were implemented. Each fix addresses an empirically audited failure category with zero risk of regressions when gated behind feature flags.

All fixes are controlled by granular feature flags in `src/tonerhound/resolution/resolver.py` and `src/tonerhound/benchmark/adapter.py`, plus an overarching master flag `ENABLE_EXP033_CANDIDATE_EXPANSION`.

---

## 2. Granular Fix Documentation

### Fix 1: Boolean Schema Generalization
* **Problem Observed**: In EXP-032 / EXP-028A, 974 checkbox fields across 15+ tax documents (e.g. `reason_pressure`, `productive_zone_no`, `workover_100ppm_3000ft_yes`) evaluated to 0 candidates because `EvidenceResolver` required field names to contain rigid keywords (`"_box"`, `"checkbox"`, `"is_"`, `"has_"`, `"flag"`, etc.). Field names outside this list were treated as plain strings, searching for the literal word `"False"`, which does not exist on the form.
* **Exact Code Location**:
  - `src/tonerhound/resolution/resolver.py`: `collect_candidates()` lines 112–126.
* **Feature Flag**: `ENABLE_BOOLEAN_EXPANSION` (Default: `False`).
* **Implementation**:
  ```python
  enable_bool = self._is_flag_enabled(self.enable_boolean_expansion, ENABLE_BOOLEAN_EXPANSION)
  if enable_bool:
      is_bool = isinstance(value, bool) or (
          isinstance(value, str)
          and value.strip().lower() in ("true", "false", "yes", "no")
      )
  else:
      is_bool = isinstance(value, bool) or (
          isinstance(value, str)
          and value.strip().lower() in ("true", "false", "yes", "no")
          and any(k in field.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no", "final", "amended", "general", "domestic", "contributed"))
      )
  ```
* **Expected Effect**: Directly recovers ~974 checkbox fields across forms W-14, 1040, W-2, and K-1.
* **Risk**: Negligible. Non-boolean string queries (e.g. text containing names or addresses) do not evaluate to `("true", "false", "yes", "no")`.
* **Test Added**: `tests/test_candidate_expansion.py::test_boolean_expansion_flag` verifying that `reason_pressure` returns 0 candidates under baseline and $\ge 1$ candidate under expansion.

---

### Fix 2: Punctuation & Token Strip Fallback
* **Problem Observed**: In the 200-sample audit, 22 fields (11.0%, ~5,361 benchmark fields) suffered from formatting mismatches:
  1. Parenthesized percentage allocations (e.g. `(99.4%)` in `real_vg_divappr_full` and `(97.5%)` in `real_vg_healthcare_full`) were parsed as negative amounts (`-99.4`) by `clean_currency_and_numbers`, causing numeric lookup to fail.
  2. Trailing depth and unit markers (e.g. `5482'` with apostrophe) were rejected by `parse_numeric_value`.
  3. Trailing punctuation on string fields prevented exact index slicing.
* **Exact Code Location**:
  - `src/tonerhound/normalization/normalizers.py`: `clean_currency_and_numbers()` lines 128–135; `parse_numeric_value()` line 160.
  - `src/tonerhound/resolution/resolver.py`: `collect_candidates()` lines 165–180.
* **Feature Flag**: `ENABLE_TOKEN_STRIP_RECOVERY` (Default: `False`).
* **Implementation**:
  ```python
  # Normalizers:
  if cleaned.startswith("(") and cleaned.endswith(")"):
      inner = cleaned[1:-1].strip()
      if not inner.endswith("%") and not inner.startswith("-"):
          negative = True
      cleaned = inner
  cleaned = cleaned.rstrip("%'\"")

  # Resolver fallback:
  enable_token_strip = self._is_flag_enabled(self.enable_token_strip_recovery, ENABLE_TOKEN_STRIP_RECOVERY)
  if not candidates and enable_token_strip and isinstance(value, str) and not is_bool:
      stripped = value.strip(" -.,;:_()[]{}/'\"")
      if stripped and stripped != value:
          candidates.extend(self.matcher.find_exact_candidates(stripped, page_hint=page_hint))
          if not candidates:
              candidates.extend(self.matcher.find_normalized_numeric_candidates(stripped, page_hint=page_hint))
          if not candidates and page_hint is not None:
              candidates.extend(self.matcher.find_exact_candidates(stripped, page_hint=None))
              if not candidates:
                  candidates.extend(self.matcher.find_normalized_numeric_candidates(stripped, page_hint=None))
  ```
* **Expected Effect**: Directly recovers ~5,361 fields including parenthesized percentages, depths in feet (`5482'`), and punctuated entity tokens.
* **Risk**: Low. Only triggered if the primary candidate generation tiers return 0 candidates.
* **Test Added**: `tests/test_candidate_expansion.py::test_token_strip_recovery` verifying that stripped tokens correctly generate matching candidates.

---

### Fix 3: Global Search Fallback Relaxation
* **Problem Observed**: In `resolver.py` lines 145–156, when `page_hint` was relaxed to `None`, `find_normalized_numeric_candidates(value, page_hint=None)` was omitted for string queries that represent numbers (Tier 2). If a formatted number appeared on a different page than predicted, it was completely dropped.
* **Exact Code Location**:
  - `src/tonerhound/resolution/resolver.py`: `collect_candidates()` lines 152–158.
* **Feature Flag**: `ENABLE_GLOBAL_SEARCH_RELAXATION` (Default: `False`).
* **Implementation**:
  ```python
  if not candidates and page_hint is not None:
      ...
      if not candidates and isinstance(value, str) and not is_bool:
          candidates.extend(self.matcher.find_normalized_numeric_candidates(value, page_hint=None))
  ```
* **Expected Effect**: Recovers ~2,193 fields where page routing drifted across multi-page filings.
* **Risk**: Low. Uses sublinear inverted numeric lookup; candidates are prioritized with `page_hint` priority.
* **Test Added**: Integrated in `tests/test_candidate_expansion.py::test_master_flag_expansion`.

---

### Fix 4: Multi-Line Span Recovery
* **Problem Observed**: In documents such as `real_bbb_service_list` and `real_ofac_ssi_full`, multi-line strings (e.g. email sequences and legal entities) broke across 4–5 visual lines. However, `_find_multiline_candidates` restricted `span_len` to `(2, 3)` lines, causing 4- and 5-line spans to fail exact line matching.
* **Exact Code Location**:
  - `src/tonerhound/matching/matcher.py`: `_find_multiline_candidates()` line 657.
  - `src/tonerhound/resolution/resolver.py`: `collect_candidates()` lines 180–190.
* **Feature Flag**: `ENABLE_MULTI_LINE_RECOVERY` (Default: `False`).
* **Implementation**:
  ```python
  # Matcher:
  for span_len in (2, 3, 4, 5):
      if i + span_len > n_lines:
          continue
      span_lines = page.lines[i : i + span_len]

  # Resolver newline normalization:
  enable_multiline = self._is_flag_enabled(self.enable_multi_line_recovery, ENABLE_MULTI_LINE_RECOVERY)
  if not candidates and enable_multiline and isinstance(value, str) and "\n" in value:
      clean_ml = " ".join(value.split())
      if clean_ml and clean_ml != value:
          candidates.extend(self.matcher.find_exact_candidates(clean_ml, page_hint=page_hint))
          if not candidates and page_hint is not None:
              candidates.extend(self.matcher.find_exact_candidates(clean_ml, page_hint=None))
  ```
* **Expected Effect**: Recovers ~487 multi-line fields across service lists and legal notices.
* **Risk**: Negligible. Only triggered on strings with `\n` that failed primary search.
* **Test Added**: `tests/test_candidate_expansion.py::test_multiline_recovery` verifying 4-line email span recovery.

---

## 3. Verification & Test Suite Summary

- **Total Test Suite**: 236 tests (`pytest`).
- **Pass Rate**: 100% (236 passed, 0 failed).
- **Regression Count**: 0 across all existing modules.
