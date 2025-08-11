# Import required libraries and modules
import neo4j
import os
import asyncio
from dotenv import load_dotenv
from typing import Any
from dataclasses import dataclass

from neo4j_graphrag.llm import AnthropicLLM as LLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings as Embeddings
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG

@dataclass(frozen=True) 
class GraphRAGConfig:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    anthropic_api_key: str
    index_name: str = "text_embeddings"
    pdf_file_path: str = "data/sample_IT_compliance_document.pdf"

    @classmethod
    def from_env(cls) -> 'GraphRAGConfig':
        """Create configuration from environment variables"""
        load_dotenv()
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

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
    config = GraphRAGConfig.from_env()
    driver = create_neo4j_driver(config)
    llm = create_llm(config.anthropic_api_key)
    embedder = create_embedder()
    create_vector_index(driver, config.index_name)
    retriever = create_retriever(driver, config.index_name, embedder)
    try:
        split_result = await process_pdf(config)
        embedded_chunks = await create_embeddings(embedder, split_result)
        store_chunks_and_entities(driver, embedded_chunks, llm, config.pdf_file_path)
        store_embeddings(driver, embedded_chunks, embedder)
        rag = GraphRAG(llm=llm, retriever=retriever)

        queries = [
            "What are the main topics in this document?",
            "How many levels is Company data classified?"
        ]

        for query in queries:
            print(f"\n** Query: ** {query}")
            answer = query_graph(rag, driver, query)
            print(f"** Answer: ** {answer}")

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        try:
            if driver:
                driver.close()
                print("✅ Neo4j driver closed")
        except Exception as e:
            print(f"Warning: Error closing driver: {e}")

if __name__ == "__main__":
    asyncio.run(main())
