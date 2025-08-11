# Import required libraries and modules
import neo4j
import os
from pathlib import Path
from dotenv import load_dotenv

from neo4j_graphrag.llm import AnthropicLLM as LLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings as Embeddings
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG

# Load environment variables from .env file and initialise Neo4j driver and models. This is a one-time setup step. You can run this script once to set up the database and models.
load_dotenv()

# Initialise config and models

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
INDEX_NAME = "text_embeddings"  # Name of the Neo4j vector index to use

print(f"Checking Neo4j connection to {uri}...")
    
try:
    # Create the Neo4j driver (don't use 'with' here as we need it later)
    neo4j_driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
    neo4j_driver.verify_connectivity()
    print("✅ Neo4j connection successful")
except neo4j.exceptions.ServiceUnavailable as e:
    print(f"❌ Neo4j service unavailable: {e}")
    exit(1)
except neo4j.exceptions.AuthError as e:
    print(f"❌ Authentication failed. Check username/password in .env: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Connection error: {e}")
    exit(1)

try:
    # Use Anthropic Extraction LLM for knowledge graph extraction
    ex_llm = LLM(
        model_name="claude-3-5-sonnet-20241022",
        model_params={
            "temperature": 0.1,
            "max_tokens": 1024,  # Increased from 512
        },
        api_key=ANTHROPIC_API_KEY,
    )
except Exception as e:
    print(f"Error initializing Anthropic Extraction LLM model: {e}")
    exit(1)
print ("✅ Initialized Anthropic Extraction LLM")
try:
    # Use SentenceTransformer embeddings LLM
    embedder = Embeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
except Exception as e:
    print(f"Error initializing SentenceTransformer embeddings LLM: {e}")
    exit(1)
print ("✅ Initialized SentenceTransformer embeddings LLM")

# Create vector index if it doesn't exist
try:
    if neo4j_driver:
        with neo4j_driver.session() as session:
            # Check if index exists
            result = session.run("SHOW INDEXES YIELD name WHERE name = $index_name", index_name=INDEX_NAME)
            if not result.single():
                # Create vector index
                session.run(
                    f"CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS "
                    f"FOR (n:Chunk) ON (n.embedding) "
                    f"OPTIONS {{ "
                    f"indexConfig: {{ "
                    f"`vector.dimensions`: 384, "
                    f"`vector.similarity_function`: 'cosine' "
                    f"}} "
                    f"}}"
                )
                print(f"✅ Created vector index: {INDEX_NAME}")
            else:
                print(f"✅ Vector index {INDEX_NAME} already exists")
    else:
        print("⚠️ Neo4j driver is not available, skipping vector index creation")
except Exception as e:
    print(f"Error creating vector index: {e}")
    exit(1)

# Initialize the retriever
try:
    retriever = VectorRetriever(neo4j_driver, INDEX_NAME, embedder)
except Exception as e:
    print(f"Error initializing VectorRetriever: {e}")
    exit(1)
print ("✅ Initialized VectorRetriever")
# Initialize the RAG pipeline
try:
    rag = GraphRAG(retriever=retriever, llm=ex_llm)
except Exception as e:
    print(f"Error initializing GraphRAG: {e}")
    exit(1)
print ("✅ Initialized GraphRAG")

async def main():
    try:
        # Use the sample PDF created by create_sample_pdf.py
        pdf_file_path = "data/sample_IT_compliance_document.pdf"
        
        # Check if PDF exists
        if not os.path.exists(pdf_file_path):
            print(f"❌ PDF file not found at {pdf_file_path}")
            print("Please run: python src/create_sample_pdf.py first")
            exit(1)
        print (f"✅ Found PDF file in {pdf_file_path}")
        # Load and split the PDF text by sentences
        try: 
            loader = PdfLoader()
            pdf_result = await loader.run(filepath=Path(pdf_file_path))
            
            # Split the loaded PDF text by sentences
            splitter = FixedSizeSplitter(chunk_size=512, chunk_overlap=50)
            split_result = await splitter.run(text=pdf_result.text)
            
        except Exception as e:  
            print(f"Error loading/splitting pdf: {e}")    
            exit(1)
        print (f"✅ Loaded and split PDF file from {pdf_file_path}")
        
        # Embed the chunks using the embedder model
        try:
            text_chunk_embedder = TextChunkEmbedder(embedder=embedder)
            embedded_chunks = await text_chunk_embedder.run(text_chunks=split_result)
        except Exception as e:  
            print(f"Error embedding: {e}")    
            exit(1)
        print (f"✅ Embeddings created")
        
        # Build knowledge graph manually
        try:
            with neo4j_driver.session() as session:
                # Store chunks and embeddings in Neo4j
                for chunk in embedded_chunks.chunks:
                    session.run("""
                        CREATE (c:Chunk {
                            text: $text,
                            index: $index,
                            document_path: $document_path
                        })
                    """,
                    text=chunk.text,
                    index=chunk.index,
                    document_path=pdf_file_path
                    )
                
                # Extract entities and relationships using LLM
                for chunk in embedded_chunks.chunks:
                    # Create basic chunk structure first
                    session.run("""
                        MERGE (d:Document {path: $document_path})
                        MERGE (c:Chunk {index: $index, text: $text})
                        MERGE (d)-[:CONTAINS]->(c)
                    """,
                    document_path=pdf_file_path,
                    index=chunk.index,
                    text=chunk.text[:1000]  # Truncate for storage
                    )

                    # Extract entities using LLM (simplified for now)
                    try:
                        entity_prompt = f"Extract 3-5 main entities from this text. List them as: PERSON, ORG, LOCATION, or CONCEPT. Text: {chunk.text[:500]}"
                        entity_response = await ex_llm.ainvoke(entity_prompt)

                        # Parse response and create entity nodes (basic implementation)
                        if hasattr(entity_response, 'content'):
                            entities_text = entity_response.content
                        else:
                            entities_text = str(entity_response)

                        # Simple entity extraction from response
                        lines = entities_text.split('\n')
                        for line in lines[:5]:  # Limit to 5 entities per chunk
                            if line.strip():
                                entity_name = line.strip()
                                session.run("""
                                    MERGE (e:Entity {name: $name})
                                    WITH e
                                    MATCH (c:Chunk {index: $index})
                                    MERGE (e)-[:MENTIONED_IN]->(c)
                                """,
                                name=entity_name,
                                index=chunk.index
                                )
                    except Exception as e:
                        print(f"Warning: Could not extract entities from chunk {chunk.index}: {e}")
                        continue
        except Exception as e:  
            print(f"Error building knowledge graph: {e}")    
            exit(1)
        print(f"✅ Knowledge graph built manually")

        # Create embeddings manually and store in Neo4j chunks
        try:
            with neo4j_driver.session() as session:
                print("🔄 Creating embeddings manually")

                stored_count = 0
                for chunk in embedded_chunks.chunks:
                    # Create embedding for this chunk using the embedder
                    chunk_embedding = embedder.embed_query(chunk.text)

                    # Convert to list if it's a numpy array
                    if hasattr(chunk_embedding, 'tolist'):
                        embedding_list = chunk_embedding.tolist()
                    else:
                        embedding_list = chunk_embedding

                    # Store the embedding in Neo4j
                    session.run("""
                        MATCH (c:Chunk {index: $index})
                        SET c.embedding = $embedding
                    """,
                    index=chunk.index,
                    embedding=embedding_list
                    )
                    stored_count += 1

                print(f"✅ Created and stored {stored_count} embeddings in Neo4j chunks")
        except Exception as e:
            print(f"Error creating/storing embeddings: {e}")
            print("Continuing without embeddings...")
            # Continue execution instead of failing

        # Create vector index for similarity search
        try:
            with neo4j_driver.session() as session:
                # Create vector index on chunk embeddings
                session.run(f"""
                    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                print("✅ Created vector index for embeddings")
        except Exception as e:
            print(f"Error creating vector index: {e}")

        # Initialize vector retriever
        vector_retriever = None
        rag = None

        try:
            vector_retriever = VectorRetriever(
                driver=neo4j_driver,
                index_name="text_embeddings",  # Use the existing index name
                embedder=embedder
            )
            print("✅ Initialized vector retriever")
        except Exception as e:
            print(f"Error initializing retriever: {e}")
            return  # Exit the function if retriever fails

        # Create GraphRAG for querying
        try:
            rag = GraphRAG(
                llm=ex_llm,
                retriever=vector_retriever
            )
            print("✅ GraphRAG system ready for queries")
        except Exception as e:
            print(f"Error creating GraphRAG: {e}")
            return  # Exit the function if GraphRAG fails

        # Test query with debugging
        try:
            # First, check if we have chunks in the database
            if neo4j_driver:
                with neo4j_driver.session() as session:
                    chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as count").single()["count"]
                    print(f"📊 Found {chunk_count} chunks in database")

                    # Check if embeddings are stored
                    embedding_count = session.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) as count").single()["count"]
                    print(f"📊 Found {embedding_count} chunks with embeddings")

                    if chunk_count > 0:
                        # Get a sample chunk to verify content
                        sample = session.run("MATCH (c:Chunk) RETURN c.text LIMIT 1").single()
                        if sample:
                            print(f"📄 Sample chunk: {sample['c.text'][:100]}...")
            else:
                print("⚠️ Neo4j driver is not available, skipping database checks")

            # Now test the query
            response = rag.search("What are the main topics in this document?")
            print(f"✅ Test query results: {response.answer}")
        except Exception as e:
            print(f"Error running test query: {e}")

        # Query the RAG system
        try:
            query = "How many levels is Company data classified?"
            print(f"** Query: ** {query}")
            response = rag.search(query)
            print(f"** Answer: ** {response.answer}")
        except Exception as e:
            print(f"Error in final query: {e}")

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Close driver at the very end
        try:
            if neo4j_driver:
                neo4j_driver.close()
                print("✅ Neo4j driver closed")
        except Exception as e:
            print(f"Warning: Error closing driver: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
