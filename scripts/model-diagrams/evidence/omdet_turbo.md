# OMDet-Turbo

Rebuild with `python scripts/model-diagrams/builders/omdet_turbo.py --source /path/to/libreyolo`. All variants come from the in-tree source.

Route checks:

```
t-detect: {
  "findings": [
    {
      "kind": "coincident",
      "first": "lang4 / classout",
      "second": "lang4 / taskout",
      "axis": "v",
      "gap": 0.0,
      "overlap": 181.0,
      "same_source": true,
      "note": "Check whether this is a legitimate shared-tensor trunk."
    }
  ],
  "skipped": [],
  "segments": 250,
  "total_findings": 1
}
```
