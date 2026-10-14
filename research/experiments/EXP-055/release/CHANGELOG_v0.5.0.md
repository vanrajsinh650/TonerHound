## [0.5.0] - 2026-10-14

### Benchmark
- Word Grounding F1: 75.2099% (237-doc corpus)
- Page Grounding F1: 90.1733%
- Passing fields: 411,098 / 498,140
- Total experiments: EXP-001 through EXP-054
- Zero regressions across all 498,140 fields

### Added
- Cell padding normalizer (fix_01)
- Row pitch realigner (fix_02)
- Multi-column grid extractor (fix_03)
- Monotonic stream matcher (fix_04)
- OFAC entity row tracker (fix_05)
- Malformed bbox repair (fix_06)
- Memory-safe chunked benchmark evaluation
- Microscope V6 fixed field accounting

### Documentation
- Updated README.md with current benchmark
- Updated docs/benchmark-results.md
- Updated docs/how-tonerhound-reached-72.md to how-tonerhound-reached-75.md
- Regenerated all charts with current data
