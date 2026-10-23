"""AWS Lambda entry point (Function URL): runs the same FastAPI app through Mangum."""
from mangum import Mangum

from .main import app

handler = Mangum(app, lifespan="auto")
