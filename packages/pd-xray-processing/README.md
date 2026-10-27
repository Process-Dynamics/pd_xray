# pd-xray-processing

Image operations (flat/dark, filters, crop, rotate, rolling ball, tone, …) and pipelines saved as YAML.

```python
from pd_xray.processing import Pipeline, Step

out = Pipeline([Step("median", {"size": 3})])(frame)
```

Install from a GitHub release with `pip install --find-links …`: see the [main
README](../../README.md#2-python-packages).

Guide: [docs/processing.md](../../docs/processing.md).
