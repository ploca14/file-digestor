from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Literal
import httpx
import logging
from chains.extraction import process_and_extract_document
from chains.suggestion import generate_suggestions
from langchain_core.vectorstores import VectorStore
from dependencies import get_embeddings, get_vector_store, get_retriever
from chains.suggestion import ChatMessage

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileUrlRequest(BaseModel):
    url: str
    callback_url: str
    patient_id: str

@app.post("/process-file", status_code=201, response_description="File processing started", response_model=Literal["OK"])
async def process_file_endpoint(
    file_request: FileUrlRequest,
    background_tasks: BackgroundTasks,
):
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings) 
    background_tasks.add_task(process_file_and_callback, file_request, vector_store)

    return "OK"

async def process_file_and_callback(file_request: FileUrlRequest, vector_store: VectorStore):
    async with httpx.AsyncClient() as client:
        try:
            # Download the file from the provided URL
            response = await client.get(file_request.url)
            response.raise_for_status()

            # Save the file locally
            file_path = f"tmp/{file_request.url.split('/')[-1]}"
            with open(file_path, "wb") as f:
                f.write(response.content)

            results = process_and_extract_document(file_path, file_request.patient_id, vector_store)
            
            logger.info(f"Extracted results for {file_request.url}")
            logger.info({
                "fhir": len(results["fhir"].model_dump_json()),
                "plain_text": len(results["plain_text"].text),
                "mortal_readable": len(results["plain_text"].text)
            })
            
            # Send results back to callback URL
            await client.post(
                file_request.callback_url,
                json={
                    "status": "success",
                    "hl7_fhir_data": results["fhir"].model_dump_json(),
                    "raw_text": results["plain_text"].text,
                    "mortal_readable": results["plain_text"].text
                }
            )

        except Exception as e:
            # Optionally notify callback URL about failure
            logger.error(f"Failed to process file {file_request.url}")
            logger.error(e)
            
            try: 
                await client.post(
                    file_request.callback_url,
                    json={
                        "status": "error",
                        "error": str(e)
                    }
                )
            except Exception as e:
                logger.error(f"Failed to notify callback URL {file_request.callback_url}")
                logger.error(e)

class SuggestionsRequest(BaseModel):
    patient_id: str
    chat_history: list[ChatMessage]
    
@app.post("/suggestions")
async def get_suggestions(request: SuggestionsRequest):
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    retriever = get_retriever(vector_store, request.patient_id)
    
    results = generate_suggestions(request.chat_history, retriever)
    
    return results
