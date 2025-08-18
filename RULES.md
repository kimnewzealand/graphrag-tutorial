# GraphRAG Project Coding Standards & Best Practices

## 📋 Table of Contents
1. [Code Style Guidelines](#code-style-guidelines)
2. [Architecture Principles](#architecture-principles)
3. [GraphRAG-Specific Rules](#graphrag-specific-rules)
4. [Error Handling](#error-handling)
5. [Testing Standards](#testing-standards)
6. [Security Guidelines](#security-guidelines)
7. [Performance Considerations](#performance-considerations)

---

## 1. Code Style Guidelines

### Python Formatting Standards
- **MUST** follow PEP 8 guidelines
- **MUST** use Black for automatic code formatting
- **MUST** remove unused imports and variables
- **MUST** keep code as short and readable as possible

### Naming Conventions
```python
# Functions: snake_case with descriptive verbs
def create_neo4j_driver(config: GraphRAGConfig) -> neo4j.Driver:

# Variables: snake_case with descriptive nouns
chunk_count = 42
embedding_list = [1.0, 2.0, 3.0]

# Constants: UPPER_SNAKE_CASE
INDEX_NAME = "text_embeddings"
MAX_CHUNK_SIZE = 1000

# Classes: PascalCase with descriptive nouns
class GraphRAGConfig:
```

### Documentation Requirements
- **MUST** include docstrings for all functions explaining purpose, arguments, and return values
- **MUST** use type hints for function parameters and return values
- **MUST** document complex business logic with inline comments
- **SHOULD** include usage examples for public APIs

```python
def process_pdf_functional(
    config: GraphRAGConfig, 
    driver: neo4j.Driver, 
    llm: LLM, 
    embedder: Embeddings
) -> Tuple[Any, Any]:
    """Process PDF using functional approach.
    
    Args:
        config: Immutable configuration object
        driver: Neo4j database driver
        llm: Language model for entity extraction
        embedder: Embedding model for vector creation
        
    Returns:
        Tuple of (pdf_result, split_result)
        
    Raises:
        SystemExit: If PDF processing fails
    """
```

---

## 2. Architecture Principles

### Functional Programming Approach
- **MUST** use pure functions with single responsibility
- **MUST** avoid global state and side effects
- **MUST** pass all dependencies as parameters
- **SHOULD** use function composition for complex workflows

### Immutable Data Structures
```python
@dataclass(frozen=True)  # Immutable configuration
class GraphRAGConfig:
    neo4j_uri: str
    neo4j_username: str
    # ... other fields
```

### Pure Functions
- **MUST** have predictable outputs for given inputs
- **MUST NOT** modify input parameters
- **MUST NOT** rely on external state
- **SHOULD** be easily testable in isolation

### Type Hints
- **MUST** use type hints for all function signatures
- **MUST** import types from `typing` module when needed
- **SHOULD** use `Optional[T]` for nullable parameters
- **SHOULD** use `Union[T, U]` for multiple possible types

---

## 3. GraphRAG-Specific Rules

### Neo4j Query Patterns
```python
# MUST use parameterized queries to prevent injection
session.run("""
    MATCH (c:Chunk {index: $index})
    SET c.embedding = $embedding
""", index=chunk.index, embedding=embedding_list)

# MUST use MERGE for upsert operations
session.run("""
    MERGE (e:Entity {name: $name})
    WITH e
    MATCH (c:Chunk {index: $index})
    MERGE (e)-[:MENTIONED_IN]->(c)
""", name=entity_name, index=chunk.index)
```

### Entity Extraction Guidelines
- **MUST** limit entity extraction to 3-5 main entities per chunk
- **MUST** truncate text input to LLM (max 500 chars for entity extraction)
- **SHOULD** handle entity extraction failures gracefully
- **SHOULD** validate entity names before storing

### Embedding Best Practices
- **MUST** convert numpy arrays to lists before Neo4j storage: `embedding.tolist()`
- **MUST** use consistent embedding dimensions (384 for sentence-transformers/all-MiniLM-L6-v2)
- **SHOULD** batch embedding operations for performance
- **SHOULD** cache embeddings when possible

---

## 4. Error Handling

### Exception Handling Patterns
```python
# MUST handle specific exceptions first
try:
    driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
except neo4j.exceptions.ServiceUnavailable as e:
    print(f"❌ Neo4j service unavailable: {e}")
    exit(1)
except neo4j.exceptions.AuthError as e:
    print(f"❌ Authentication failed: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)
```

### Logging Standards
- **MUST** use descriptive error messages with context
- **MUST** include emoji indicators: ✅ success, ❌ error, ⚠️ warning, 📊 info
- **SHOULD** log progress for long-running operations
- **SHOULD** include relevant data in error messages

### Graceful Degradation
- **MUST** continue execution when non-critical operations fail
- **MUST** provide fallback behavior for optional features
- **SHOULD** warn users about degraded functionality

```python
try:
    # Extract entities using LLM
    entity_response = await llm.ainvoke(entity_prompt)
    # ... process entities
except Exception as e:
    print(f"Warning: Could not extract entities from chunk {chunk.index}: {e}")
    continue  # Continue with next chunk
```

---

## 5. Testing Standards

### Unit Test Naming Conventions
```python
def test_create_neo4j_driver_success():
    """Test successful Neo4j driver creation"""

def test_create_neo4j_driver_auth_failure():
    """Test Neo4j driver creation with authentication failure"""

def test_process_pdf_functional_with_valid_file():
    """Test PDF processing with valid input file"""
```

### Test Coverage Requirements
- **MUST** achieve minimum 80% code coverage
- **MUST** test all public function interfaces
- **MUST** test error conditions and edge cases
- **SHOULD** test integration points with external services

### Testing Patterns for Async Functions
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_entity_extraction():
    """Test asynchronous entity extraction"""
    # Arrange
    mock_llm = MockLLM()
    chunk = create_test_chunk()
    
    # Act
    result = await extract_entities_async(mock_llm, chunk)
    
    # Assert
    assert len(result.entities) > 0
```

---

## 6. Security Guidelines

### Environment Variable Usage
- **MUST** store all sensitive information in environment variables
- **MUST** use `.env` files for local development
- **MUST** validate required environment variables at startup

```python
@classmethod
def from_env(cls) -> 'GraphRAGConfig':
    """Create configuration from environment variables"""
    load_dotenv()
    
    # Validate required variables
    required_vars = ["NEO4J_URI", "NEO4J_PASSWORD", "ANTHROPIC_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Required environment variable {var} not set")
    
    return cls(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
```

### API Key Management
- **MUST** never hardcode API keys in source code
- **MUST** use environment variables for API keys
- **SHOULD** implement API key rotation mechanisms
- **SHOULD** validate API keys before use

### Input Validation and Data Sanitization
- **MUST** validate all user inputs
- **MUST** sanitize text before storing in database
- **MUST** use parameterized queries to prevent SQL injection
- **SHOULD** implement input length limits

```python
def sanitize_text_for_storage(text: str, max_length: int = 1000) -> str:
    """Sanitize and truncate text for safe database storage"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove potentially harmful characters
    sanitized = text.replace('\x00', '').strip()
    
    # Truncate to prevent oversized storage
    return sanitized[:max_length] if len(sanitized) > max_length else sanitized
```

---

## 7. Performance Considerations

### Memory Usage Guidelines
- **MUST** limit chunk size to prevent memory overflow (max 1000 chars)
- **MUST** process documents in batches for large files
- **SHOULD** use generators for large data processing
- **SHOULD** monitor memory usage in production

### Database Connection Management
- **MUST** use connection pooling for production deployments
- **MUST** close database connections in finally blocks
- **SHOULD** implement connection retry logic
- **SHOULD** use transactions for batch operations

```python
def batch_store_embeddings(driver: neo4j.Driver, embeddings: List[Tuple[int, List[float]]]) -> None:
    """Store embeddings in batches for better performance"""
    batch_size = 100
    
    with driver.session() as session:
        with session.begin_transaction() as tx:
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                for chunk_index, embedding in batch:
                    tx.run("""
                        MATCH (c:Chunk {index: $index})
                        SET c.embedding = $embedding
                    """, index=chunk_index, embedding=embedding)
```

### Optimization Practices
- **MUST** use vector indices for similarity search
- **MUST** limit LLM context length to prevent timeouts
- **SHOULD** cache frequently accessed data
- **SHOULD** use async operations for I/O bound tasks
- **SHOULD** implement pagination for large result sets

---

*This document is living and should be updated as the project evolves.*
