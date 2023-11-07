import os
import uvicorn

from service.api.app import create_app
from service.settings import get_config

config = get_config()
app = create_app(config)

if __name__ == "__main__":
    # uvicorn main:app
    # cloudflared tunnel --url http://localhost:8000
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(app, host=host, port=port)
