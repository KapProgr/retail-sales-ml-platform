"""Entry point kept for existing deployments: `uvicorn main:app`."""

import os

from app.api import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
