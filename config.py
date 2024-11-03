from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys and External Services
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
if not UNSTRUCTURED_API_KEY:
    raise ValueError("UNSTRUCTURED_API_KEY environment variable is not set")

# Database Configuration
COLLECTION_NAME = os.getenv("IRIS_VECTOR_COLLECTION_NAME")
if not COLLECTION_NAME:
    raise ValueError("IRIS_VECTOR_COLLECTION_NAME environment variable is not set")

CONNECTION_STRING = os.getenv("IRIS_CONNECTION_STRING")
if not CONNECTION_STRING:
    raise ValueError("IRIS_CONNECTION_STRING environment variable is not set")