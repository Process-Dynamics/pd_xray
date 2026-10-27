# pd-xray-server

Web application server for pd_xray (FastAPI). API docs at `/docs` while running.

Install: see the [main README](../../README.md#1-web-app) (conda) or [installation](../../docs/installation.md).

```bash
pd-xray serve --root /data/scans --root /mnt/nas    # http://127.0.0.1:8000; --host 0.0.0.0 to share on the LAN
```

```python
from pathlib import Path

from pd_xray.server import Settings, create_app  # ASGI app for uvicorn or tests

app = create_app(Settings(roots=(Path("/data"),)))
```

Guides: [web app](../../docs/web_app.md), [installation and shared servers](../../docs/installation.md).
