# Import required libraries and modules
import neo4j
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveInt, PositiveFloat

from neo4j_graphrag.llm import AnthropicLLM as LLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings as Embeddings
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG

# Import evaluation module
try:
    from evaluation import (
        load_evaluation_dataset, run_evaluation_suite,
        save_evaluation_results, EvaluationQuery
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("⚠️ Evaluation module not available")

# Pydantic Models for Enhanced Data Validation

class GraphRAGConfig(BaseModel):
    """Immutable configuration with comprehensive validation"""
    neo4j_uri: str = Field(..., description="Neo4j database URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., min_length=1, description="Neo4j password")
    anthropic_api_key: str = Field(..., min_length=10, description="Anthropic API key")
    index_name: str = Field(default="text_embeddings", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    pdf_file_path: Path = Field(default=Path("data/sample_IT_compliance_document.pdf"))

    class Config:
        frozen = True  # Immutable like dataclass(frozen=True)
        validate_assignment = True
        use_enum_values = True

    @field_validator('neo4j_uri')
    @classmethod
    def validate_neo4j_uri(cls, v: str) -> str:
        """Validate Neo4j URI format"""
        if not v.startswith(('bolt://', 'bolt+s://', 'neo4j://', 'neo4j+s://')):
            raise ValueError('Neo4j URI must start with bolt://, bolt+s://, neo4j://, or neo4j+s://')
        return v

    @field_validator('pdf_file_path')
    @classmethod
    def validate_pdf_path(cls, v: Path) -> Path:
        """Validate PDF file path exists and is a PDF"""
        if not isinstance(v, Path):
            v = Path(v)
        if not v.suffix.lower() == '.pdf':
            raise ValueError('File must be a PDF (.pdf extension)')
        if not v.exists():
            raise ValueError(f'PDF file does not exist: {v}')
        return v

    @field_validator('anthropic_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate Anthropic API key format"""
        if not v.startswith('sk-'):
            raise ValueError('Anthropic API key must start with "sk-"')
        return v

    @classmethod
    def from_env(cls) -> 'GraphRAGConfig':
        """Create configuration from environment variables with validation"""
        load_dotenv()

        # Validate required environment variables
        required_vars = {
            'NEO4J_URI': os.getenv("NEO4J_URI"),
            'NEO4J_PASSWORD': os.getenv("NEO4J_PASSWORD"),
            'ANTHROPIC_API_KEY': os.getenv("ANTHROPIC_API_KEY")
        }

        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"❌ Missing required environment variables: {', '.join(missing_vars)}")

        try:
            return cls(
                neo4j_uri=required_vars['NEO4J_URI'],
                neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
                neo4j_password=required_vars['NEO4J_PASSWORD'],
                anthropic_api_key=required_vars['ANTHROPIC_API_KEY']
            )
        except Exception as e:
            raise ValueError(f"❌ Configuration validation failed: {e}")

class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing with validation"""
    chunk_size: PositiveInt = Field(default=1000, le=5000, description="Text chunk size in characters")
    chunk_overlap: PositiveInt = Field(default=200, le=1000, description="Overlap between chunks")
    max_entities_per_chunk: PositiveInt = Field(default=5, le=10, description="Maximum entities to extract per chunk")
    entity_prompt_max_chars: PositiveInt = Field(default=500, le=2000, description="Max characters for entity extraction")

    @model_validator(mode='after')
    def validate_chunk_overlap(self):
        """Ensure chunk overlap is less than chunk size"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('Chunk overlap must be less than chunk size')
        return self

class EmbeddingConfig(BaseModel):
    """Configuration for embeddings with validation"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dimensions: PositiveInt = Field(default=384, description="Embedding vector dimensions")
    similarity_function: str = Field(default="cosine", pattern=r"^(cosine|euclidean|dot)$")

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate embedding model name format"""
        if not v or len(v.strip()) == 0:
            raise ValueError('Model name cannot be empty')
        return v.strip()

# Neo4j Data Models with Pydantic

class DocumentModel(BaseModel):
    """Pydantic model for Document nodes"""
    path: Path = Field(..., description="Path to the source document")
    name: Optional[str] = Field(None, description="Document name")
    created_at: datetime = Field(default_factory=datetime.now)
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat()
        }

    @field_validator('path')
    @classmethod
    def validate_document_path(cls, v: Path) -> Path:
        """Validate document path exists"""
        if not isinstance(v, Path):
            v = Path(v)
        if not v.exists():
            raise ValueError(f'Document file does not exist: {v}')
        return v

class ChunkModel(BaseModel):
    """Pydantic model for Chunk nodes with validation"""
    index: int = Field(..., ge=0, description="Unique chunk index (zero-based)")
    text: str = Field(..., min_length=1, max_length=5000, description="Chunk text content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    document_path: Path = Field(..., description="Source document path")

    class Config:
        json_encoders = {
            Path: str
        }

    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v: str) -> str:
        """Validate and sanitize text content"""
        if not v or not v.strip():
            raise ValueError('Chunk text cannot be empty')
        # Remove null bytes and excessive whitespace
        sanitized = v.replace('\x00', '').strip()
        return sanitized

    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimensions(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding vector dimensions"""
        if v is not None:
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError('Embedding must be a list of numbers')
            if len(v) != 384:  # Default dimension for sentence-transformers/all-MiniLM-L6-v2
                raise ValueError(f'Embedding must have 384 dimensions, got {len(v)}')
        return v

class EntityModel(BaseModel):
    """Pydantic model for Entity nodes"""
    name: str = Field(..., min_length=1, max_length=200, description="Entity name")
    entity_type: Optional[str] = Field(None, pattern=r"^(PERSON|ORG|LOCATION|CONCEPT|OTHER)$")
    confidence: Optional[PositiveFloat] = Field(None, le=1.0, description="Extraction confidence score")
    mentions: List[int] = Field(default_factory=list, description="List of chunk indices where entity is mentioned")

    @field_validator('name')
    @classmethod
    def validate_entity_name(cls, v: str) -> str:
        """Validate and sanitize entity name"""
        if not v or not v.strip():
            raise ValueError('Entity name cannot be empty')
        # Remove special characters and normalize
        sanitized = v.strip().replace('\n', ' ').replace('\t', ' ')
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        return sanitized

class LLMResponseModel(BaseModel):
    """Pydantic model for LLM API responses"""
    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model used for generation")
    usage_tokens: Optional[int] = Field(None, ge=0, description="Tokens used")
    response_time: Optional[PositiveFloat] = Field(None, description="Response time in seconds")

    @field_validator('content')
    @classmethod
    def validate_response_content(cls, v: str) -> str:
        """Validate LLM response content"""
        if not v or not v.strip():
            raise ValueError('LLM response content cannot be empty')
        return v.strip()

class QueryResultModel(BaseModel):
    """Pydantic model for GraphRAG query results"""
    query: str = Field(..., min_length=1, description="Original query")
    answer: str = Field(..., min_length=1, description="Generated answer")
    chunks_used: List[int] = Field(default_factory=list, description="Chunk indices used for answer")
    entities_found: List[str] = Field(default_factory=list, description="Entities found in context")
    confidence: Optional[PositiveFloat] = Field(None, le=1.0, description="Answer confidence score")
    response_time: Optional[PositiveFloat] = Field(None, description="Query response time in seconds")

    @field_validator('query', 'answer')
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        """Validate query and answer text"""
        if not v or not v.strip():
            raise ValueError('Query and answer cannot be empty')
        return v.strip()

def create_neo4j_driver(config: GraphRAGConfig) -> neo4j.Driver:
    """Create and verify Neo4j driver connection"""
    print(f"Checking Neo4j connection to {config.neo4j_uri}...")

    try:
        driver = neo4j.GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password)
        )
        driver.verify_connectivity()
        print("✅ Neo4j connection successful")
        return driver
    except neo4j.exceptions.ServiceUnavailable as e:
        print(f"❌ Neo4j service unavailable: {e}")
        exit(1)
    except neo4j.exceptions.AuthError as e:
        print(f"❌ Authentication failed. Check username/password in .env: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        exit(1)

def create_llm(api_key: str) -> LLM:
    """Create Anthropic LLM instance"""
    try:
        model_name="claude-3-5-sonnet-20241022"
        llm = LLM(
            model_name=model_name,
            model_params={
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            api_key=api_key,
        )
        print(f"✅ Initialized Anthropic Extraction LLM {model_name}" )
        return llm
    except Exception as e:
        print(f"Error initializing Anthropic Extraction LLM model: {e}")
        exit(1)

def create_embedder() -> Embeddings:
    """Create sentence transformer embeddings"""
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedder = Embeddings(
            model=model_name,
        )
        print(f"✅ Initialized SentenceTransformer embeddings LLM {model_name}")
        return embedder
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        exit(1)

def create_vector_index(driver: neo4j.Driver, index_name: str) -> None:
    """Create vector index if it doesn't exist"""
    try:
        with driver.session() as session:
            result = session.run("SHOW INDEXES YIELD name WHERE name = $index_name", index_name=index_name)
            if not result.single():
                session.run(f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:Chunk) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                print(f"✅ Created vector index: {index_name}")
            else:
                print(f"✅ Vector index {index_name} already exists")
    except Exception as e:
        print(f"Error creating vector index: {e}")
        exit(1)

def create_retriever(driver: neo4j.Driver, index_name: str, embedder: Embeddings) -> VectorRetriever:
    """Create vector retriever"""
    try:
        retriever = VectorRetriever(driver, index_name, embedder)
        print("✅ Initialized VectorRetriever")
        return retriever
    except Exception as e:
        print(f"Error initializing VectorRetriever: {e}")
        exit(1)

# Enhanced Validation Functions with Pydantic

def validate_and_create_chunk(index: int, text: str, document_path: Path, embedding: Optional[List[float]] = None) -> ChunkModel:
    """Create and validate a chunk using Pydantic model"""
    try:
        chunk = ChunkModel(
            index=index,
            text=text,
            document_path=document_path,
            embedding=embedding
        )
        return chunk
    except Exception as e:
        raise ValueError(f"❌ Chunk validation failed for index {index}: {e}")

def validate_and_create_entity(name: str, entity_type: Optional[str] = None, confidence: Optional[float] = None) -> EntityModel:
    """Create and validate an entity using Pydantic model"""
    try:
        entity = EntityModel(
            name=name,
            entity_type=entity_type,
            confidence=confidence
        )
        return entity
    except Exception as e:
        raise ValueError(f"❌ Entity validation failed for '{name}': {e}")

def validate_llm_response(content: str, model: str, usage_tokens: Optional[int] = None) -> LLMResponseModel:
    """Validate LLM response using Pydantic model"""
    try:
        response = LLMResponseModel(
            content=content,
            model=model,
            usage_tokens=usage_tokens
        )
        return response
    except Exception as e:
        raise ValueError(f"❌ LLM response validation failed: {e}")

def validate_processing_config(chunk_size: int = 1000, chunk_overlap: int = 200) -> DocumentProcessingConfig:
    """Create and validate document processing configuration"""
    try:
        config = DocumentProcessingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return config
    except Exception as e:
        raise ValueError(f"❌ Processing configuration validation failed: {e}")

def handle_pydantic_validation_error(error: Exception, context: str) -> None:
    """Handle Pydantic validation errors with consistent formatting"""
    if hasattr(error, 'errors'):
        # Pydantic ValidationError
        error_details = []
        for err in error.errors():
            field = '.'.join(str(x) for x in err['loc'])
            message = err['msg']
            error_details.append(f"{field}: {message}")

        print(f"❌ {context} validation failed:")
        for detail in error_details:
            print(f"   • {detail}")
    else:
        print(f"❌ {context} validation failed: {error}")

    print("⚠️ Continuing with degraded functionality...")

async def process_pdf(config: GraphRAGConfig) -> Any:
    """Load and split PDF"""
    try:
        pdf_loader = PdfLoader()
        pdf_result = await pdf_loader.run(filepath=config.pdf_file_path)
        print(f"✅ Loaded PDF file from {config.pdf_file_path}")

        splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=200)
        split_result = await splitter.run(text=pdf_result.text)
        print(f"✅ Split PDF into {len(split_result.chunks)} chunks")

        return split_result
    except Exception as e:
        print(f"Error processing PDF: {e}")
        exit(1)

async def create_embeddings(embedder: Embeddings, chunks: Any) -> Any:
    """Create embeddings"""
    try:
        text_chunk_embedder = TextChunkEmbedder(embedder=embedder)
        return await text_chunk_embedder.run(text_chunks=chunks)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        exit(1)

def store_chunks_and_entities(driver: neo4j.Driver, embedded_chunks: Any, llm: LLM, pdf_path: str) -> None:
    """Store chunks and extract entities"""
    try:
        with driver.session() as session:
            for chunk in embedded_chunks.chunks:
                # Create basic chunk structure
                session.run("""
                    MERGE (d:Document {path: $document_path})
                    MERGE (c:Chunk {index: $index, text: $text})
                    MERGE (d)-[:CONTAINS]->(c)
                """,
                document_path=pdf_path,
                index=chunk.index,
                text=chunk.text[:1000]
                )

                try:
                    entity_prompt = f"Extract 3-5 main entities from: {chunk.text[:500]}"
                    # Remove asyncio.run() and just use the sync invoke method
                    entity_response = llm.invoke(entity_prompt)

                    entities_text = entity_response.content if hasattr(entity_response, 'content') else str(entity_response)
                    lines = entities_text.split('\n')

                    for line in lines[:5]:
                        if line.strip():
                            session.run("""
                                MERGE (e:Entity {name: $name})
                                WITH e
                                MATCH (c:Chunk {index: $index})
                                MERGE (e)-[:MENTIONED_IN]->(c)
                            """,
                            name=line.strip(),
                            index=chunk.index
                            )
                except Exception as e:
                    print(f"Warning: Could not extract entities from chunk {chunk.index}: {e}")
                    continue
        print("✅ Stored chunks and entities")
    except Exception as e:
        print(f"Error storing chunks and entities: {e}")
        exit(1)

def store_embeddings(driver: neo4j.Driver, embedded_chunks: Any, embedder: Embeddings) -> None:
    """Store embeddings"""
    try:
        with driver.session() as session:
            stored_count = 0
            for chunk in embedded_chunks.chunks:
                chunk_embedding = embedder.embed_query(chunk.text)
                embedding_list = chunk_embedding.tolist() if hasattr(chunk_embedding, 'tolist') else chunk_embedding

                session.run("""
                    MATCH (c:Chunk {index: $index})
                    SET c.embedding = $embedding
                """,
                index=chunk.index,
                embedding=embedding_list
                )
                stored_count += 1
            print(f"✅ Created and stored {stored_count} embeddings")
    except Exception as e:
        print(f"Error storing embeddings: {e}")

def query_graph(rag: GraphRAG, driver: neo4j.Driver, query: str) -> str:
    """Query the graph"""
    try:
        # Database statistics
        if driver:
            with driver.session() as session:
                chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as count").single()["count"]
                embedding_count = session.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) as count").single()["count"]
                print(f"📊 Found {chunk_count} chunks, {embedding_count} with embeddings")

        # Execute query
        response = rag.search(query)
        return response.answer
    except Exception as e:
        print(f"Error querying graph: {e}")
        return "Error occurred during query"

async def main():
    """Enhanced main function with Pydantic validation"""
    try:
        # Create and validate configuration
        config = GraphRAGConfig.from_env()
        print("✅ Configuration validated successfully")

        # Validate processing configuration
        processing_config = validate_processing_config()
        print("✅ Processing configuration validated")

    except Exception as e:
        handle_pydantic_validation_error(e, "Configuration")
        exit(1)

    # Create components using validated configuration
    driver = create_neo4j_driver(config)
    llm = create_llm(config.anthropic_api_key)
    embedder = create_embedder()
    create_vector_index(driver, config.index_name)
    retriever = create_retriever(driver, config.index_name, embedder)

    try:
        # Process document with validation
        split_result = await process_pdf(config)
        embedded_chunks = await create_embeddings(embedder, split_result)

        # Validate chunks before storing
        validated_chunks = []
        for chunk in embedded_chunks.chunks:
            try:
                validated_chunk = validate_and_create_chunk(
                    index=chunk.index,
                    text=chunk.text,
                    document_path=config.pdf_file_path
                )
                validated_chunks.append(validated_chunk)
            except Exception as e:
                handle_pydantic_validation_error(e, f"Chunk {chunk.index}")
                continue

        print(f"✅ Validated {len(validated_chunks)} chunks")

        # Store data with validation
        store_chunks_and_entities(driver, embedded_chunks, llm, str(config.pdf_file_path))
        store_embeddings(driver, embedded_chunks, embedder)
        rag = GraphRAG(llm=llm, retriever=retriever)

        # Validate and execute queries
        queries = [
            "What are the main topics in this document?",
            "How many levels is Company data classified?"
        ]

        for query_text in queries:
            try:
                print(f"\n** Query: ** {query_text}")
                answer = query_graph(rag, driver, query_text)

                # Validate query result
                query_result = QueryResultModel(
                    query=query_text,
                    answer=answer,
                    chunks_used=[],  # Would be populated in real implementation
                    entities_found=[]  # Would be populated in real implementation
                )
                print(f"** Answer: ** {query_result.answer}")

            except Exception as e:
                handle_pydantic_validation_error(e, f"Query '{query_text}'")
                continue

    except Exception as e:
        print(f"❌ Error in main: {e}")
    finally:
        try:
            if driver:
                driver.close()
                print("✅ Neo4j driver closed")
        except Exception as e:
            print(f"⚠️ Warning: Error closing driver: {e}")

async def run_evaluation_mode():
    """Run the system in evaluation mode"""
    if not EVALUATION_AVAILABLE:
        print("❌ Evaluation module not available. Please check evaluation.py")
        return

    try:
        # Create and validate configuration
        config = GraphRAGConfig.from_env()
        print("✅ Configuration validated for evaluation")

        # Create components
        driver = create_neo4j_driver(config)
        llm = create_llm(config.anthropic_api_key)
        embedder = create_embedder()
        create_vector_index(driver, config.index_name)
        retriever = create_retriever(driver, config.index_name, embedder)

        # Create GraphRAG instance
        rag = GraphRAG(llm=llm, retriever=retriever)
        print("✅ GraphRAG system ready for evaluation")

        # Load evaluation dataset
        eval_dataset_path = Path("data/evaluation_queries.json")
        if not eval_dataset_path.exists():
            print(f"❌ Evaluation dataset not found at {eval_dataset_path}")
            return

        queries = load_evaluation_dataset(eval_dataset_path)
        if not queries:
            print("❌ No evaluation queries loaded")
            return

        # Run evaluation
        print(f"🔄 Starting evaluation with {len(queries)} queries...")
        suite = await run_evaluation_suite(rag, queries)

        # Save results
        results_path = Path(f"data/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_evaluation_results(suite, results_path)

        # Print summary
        print("\n" + "="*50)
        print("📊 EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Queries: {suite.total_queries}")
        print(f"Avg Retrieval Precision: {suite.avg_retrieval_precision:.3f}")
        print(f"Avg Retrieval Recall: {suite.avg_retrieval_recall:.3f}")
        print(f"Avg Generation Accuracy: {suite.avg_generation_accuracy:.3f}")
        print(f"Avg Response Time: {suite.avg_response_time:.3f}s")
        print("="*50)

    except Exception as e:
        print(f"❌ Error in evaluation mode: {e}")
    finally:
        try:
            if 'driver' in locals() and driver:
                driver.close()
                print("✅ Neo4j driver closed")
        except Exception as e:
            print(f"⚠️ Warning: Error closing driver: {e}")

if __name__ == "__main__":
    import sys

    # Check for evaluation mode
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        print("🔬 Running in evaluation mode...")
        asyncio.run(run_evaluation_mode())
    else:
        print("🚀 Running in normal mode...")
        asyncio.run(main())
