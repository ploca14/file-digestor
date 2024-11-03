# Medical Record Extraction Service

A service for processing medical documents, extracting structured data in FHIR format, and providing AI-assisted medical suggestions. This system helps healthcare providers digitize, structure, and understand medical documents while maintaining compliance with healthcare data standards.

## Features

### Document Processing

- Processes medical documents using high-resolution document parsing
- Supports multiple languages (including Czech)
- Converts documents to structured formats while preserving original content
- Generates three different outputs for each document:
  - HL7 FHIR (Fast Healthcare Interoperability Resources)
  - Plain text summaries
  - Patient-friendly explanations ("Mortal Readable")

### AI-Assisted Suggestions

- Provides context-aware medical suggestions based on chat history
- Uses RAG (Retrieval Augmented Generation) to incorporate relevant medical documents
- Maintains conversation context between doctor and patient interactions

### Vector Storage

- Stores processed documents in IRIS Vector Store
- Enables semantic search capabilities
- Links documents to specific patient IDs
- Supports filtered retrieval based on patient context

## Technical Stack

- **Backend Framework**: FastAPI
- **Language Models**: OpenAI GPT-4
- **Document Processing**: Unstructured API
- **Vector Store**: IRIS Vector
- **LLM Framework**: LangChain
- **Data Validation**: Pydantic
- **Healthcare Standards**: HL7 FHIR R4

## Prerequisites

- Python 3.9+
- IRIS Database
- OpenAI API access
- Unstructured API access

## Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_key_here
UNSTRUCTURED_API_KEY=your_key_here
IRIS_VECTOR_COLLECTION_NAME=your_collection_name
IRIS_CONNECTION_STRING=your_connection_string
```

## Installation

1. Clone the repository

```bash
git clone git@github.com:ploca14/file-digestor.git
cd file-digestor
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables

```bash
cp .env.example .env
```

Edit .env file with your keys and configuration

## Development

1. Run the development server

```bash
fastapi dev main.py
```

2. Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
