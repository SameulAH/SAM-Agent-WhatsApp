from fastapi import FastAPI
from webhook.telegram_voice import voice_router

app = FastAPI()
app.include_router(voice_router)

# Print all routes
for route in app.routes:
    if hasattr(route, "path"):
        print(f"Path: {route.path}, Methods: {route.methods if hasattr(route, 'methods') else 'N/A'}")
