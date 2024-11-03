from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel
from models.fhir import Bundle
from langchain_unstructured import UnstructuredLoader
from config import UNSTRUCTURED_API_KEY
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from typing import TypedDict

def process_file(file_path: str):
    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res",
        partition_via_api=True,
        languages=["ces"],
        api_key=UNSTRUCTURED_API_KEY,
    )
    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)

    return docs

def combine_docs(docs):
    """Combine document chunks into markdown formatted string based on element type"""
    formatted_chunks = []
    
    for doc in docs:
        content = doc.page_content
        category = doc.metadata.get("category", "UncategorizedText")
        
        # Format based on element type
        if category == "Title":
            formatted = f"## {content}"
        elif category == "Header":
            formatted = f"# {content}"
        elif category == "ListItem":
            formatted = f"* {content}"
        elif category == "FigureCaption":
            formatted = f"*Figure: {content}*"
        elif category == "Formula":
            formatted = f"```math\n{content}\n```"
        elif category == "CodeSnippet":
            formatted = f"```\n{content}\n```"
        elif category == "Table":
            # Tables might need more complex handling depending on structure
            formatted = f"| {content} |"
        elif category in ["Footer", "PageNumber"]:
            # Skip footer and page numbers
            continue
        else:
            # Default handling for NarrativeText and other types
            formatted = content
            
        formatted_chunks.append(formatted)
    
    return "\n\n".join(formatted_chunks)

def add_patient_id_to_docs(docs: list[Document], patient_id: str) -> list[Document]:
    """Add a patient ID to the metadata of each document"""
    for doc in docs:
        doc.metadata["patient_id"] = patient_id
    return docs

def store_docs_in_vector_store(
    docs: list[Document],
    patient_id: str,
    vector_store: VectorStore,
):
    """Store documents in vector store with provided embeddings"""
    documents = add_patient_id_to_docs(docs, patient_id)
    
    vector_store.add_documents(documents)

def create_fhir_extraction_chain():
    """Create an LLM chain for extracting FHIR data"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at converting medical documents into HL7 FHIR resources.
        Extract all relevant clinical information and structure it according to FHIR R4 standards.
        
        Important guidelines:
        - Ensure all required FHIR fields are populated
        - Use standard FHIR codings (LOINC, SNOMED CT, etc.) where applicable
        - Include proper status and category fields for observations
        - Link observations to the patient using proper references
        - Include dates and times when available
        - Only extract information explicitly stated in the source
        - Try to extract as much information as possible.
        - Try to fill in all optional fields.
        - Try to keep the format as close to the original as possible.
        - Keep the text as close to the original as possible.
        - Keep the original language of the document.
        """),
        ("human", "Medical Report:\n{text}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm.with_structured_output(schema=Bundle, method="json_schema", strict=True)

class PlainText(BaseModel):
    text: str

def create_plain_text_extraction_chain():
    """Create an LLM chain for extracting plain text medical information"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting relevant medical information from documents.
        Convert the markdown-formatted medical report into clean plaintext while:
        
        - Preserve all important medical information including:
          - Patient details
          - Diagnoses
          - Medications
          - Test results
          - Treatment plans
          - Clinical observations
        - Keep original medical terminology and exact values
        - Maintain the original sentence structure where possible
        - Remove formatting markers, headers, footers, page numbers
        - Skip administrative metadata
        - Present information in a clear, readable format
        - Preserve the logical flow of information
        - Keep the text as close to the original as possible.
        - Keep the original language of the document. 
        
        Return only the essential medical content in plain text format.
        """),
        ("human", "{text}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm.with_structured_output(schema=PlainText, method="json_schema", strict=True)

class MortalReadable(BaseModel):
    text: str

def create_mortal_readable_extraction_chain():
    """Create an LLM chain for converting medical text into simple, understandable language"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at explaining medical information in simple terms that anyone can understand.
        Convert the medical report into clear, everyday language while:

        - Replace medical jargon with simple explanations
        - Break down complex medical concepts into easy-to-understand terms
        - Use everyday analogies where helpful
        - Maintain all important medical information including:
          - What the doctor found
          - What the diagnosis means in plain terms
          - What medications are for
          - What test results mean
          - What the treatment plan involves and why
        - Structure information in a logical, easy-to-follow way
        - Use short, clear sentences
        - Add brief explanations for medical terms when needed
        - Keep a friendly, reassuring tone
        - Avoid oversimplifying critical medical details
        
        The goal is to help patients and family members clearly understand the medical information.
        """),
        ("human", "{text}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm.with_structured_output(schema=MortalReadable, method="json_schema", strict=True)

def create_parallel_extraction_chains():
    """Create parallel chains for both FHIR and plain text extraction"""
    fhir_extraction_chain = create_fhir_extraction_chain()
    plain_text_chain = create_plain_text_extraction_chain()
    mortal_readable_chain = create_mortal_readable_extraction_chain()

    return RunnableParallel(
        fhir=fhir_extraction_chain,
        plain_text=plain_text_chain,
        mortal_readable=mortal_readable_chain,
    ) 

class ExtractionResult(TypedDict):
    fhir: Bundle
    plain_text: PlainText
    mortal_readable: MortalReadable

def process_and_extract_document(file_path: str, patient_id: str, vector_store: VectorStore) -> ExtractionResult:
    """Process a document file and extract structured data
    
    Args:
        file_path: Path to the document file
        patient_id: The patient ID to link to the extracted FHIR data
    Returns:
        ExtractionResult containing FHIR and plain text extraction results as well as a mortal readable version of the plain text
    """
    # Process the file using Unstructured Ingest
    docs = process_file(file_path)

    # Store docs in vector store 
    store_docs_in_vector_store(docs, patient_id, vector_store)

    # Combine docs into markdown format
    combined_docs = combine_docs(docs)

    # Extract structured data using LLM chains
    extraction_chains = create_parallel_extraction_chains()
    return extraction_chains.invoke({"text": combined_docs})