# Neo4j GraphRAG Tutorial

This project demonstrates how to create a Graph-based Retrieval Augmented Generation (GraphRAG) system using Neo4j and open source language models.

The use case is querying an IT compliance pdf document.

This project uses Augment code as the code assistant. 

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a sample pdf using a custom script

   ```
   python .\src\create_sample_pdf.py
   ```

3. **Set up Neo4j**:
   - Install [Neo4j Desktop](https://neo4j.com/download/) or use Docker
   - Create a new database and start it
   - Note your connection details (URI, username, password)

4. **Configure environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your actual configuration
   ```

7. **Create the knowledge graph**:
   ```bash
   python src/graph.py
   ```

## Output

```
> python .\src\graph.py
Checking Neo4j connection to neo4j+s://e40a0a8b.databases.neo4j.io...
✅ Neo4j connection successful
✅ Initialized Anthropic Extraction LLM claude-3-5-sonnet-20241022
✅ Initialized SentenceTransformer embeddings LLM sentence-transformers/all-MiniLM-L6-v2
✅ Vector index text_embeddings already exists
✅ Initialized VectorRetriever
✅ Loaded PDF file from data/sample_IT_compliance_document.pdf
✅ Split PDF into 2 chunks
✅ Stored chunks and entities
✅ Created and stored 2 embeddings

** Query: ** What are the main topics in this document?
📊 Found 50 chunks, 50 with embeddings
** Answer: ** Based on the provided context, there are two main topics in this IT Compliance Agreement for using AI:

1. Data Classification Policy - This section outlines how company data should be classified and handled, including:
   - Three classification levels (Public, Internal, and Confidential)
   - Rules for each classification level
   - Timeline requirements for classifying new data

2. LLM Usage Compliance - This section covers the requirements for using Large Language Models, specifically:     
   - Approval process by the IT Security team
   - Timeline for approval requests (48 hours)

The document appears to be focused on establishing guidelines for both data handling and AI/LLM usage within the company.

** Query: ** How many levels is Company data classified?
📊 Found 50 chunks, 50 with embeddings
** Answer: ** According to the Data Classification Policy (section 1.1), Company data is classified into three levels: Public, Internal, and Confidential.
✅ Neo4j driver closed
```


## 📁 Project Structure

```
graphrag-tutorial/
├── data/                     # Your source documents (PDF/TXT files)
├── src/                      # Python source code
│   ├── create_graph.py   # Create knowledge graph with HuggingFace models
│   └── create_sample_pdf.py # Create sample PDF for testing
├── models/                   # Local HuggingFace model cache
├── requirements.txt          # Python dependencies
├── .env.template            # Environment variables template
├── .env                     # Your actual environment variables (create from template)
├── ARCHITECTURE.md          # System architecture documentation
└── README.md               # This file
```

## 🎯 Features

- **Knowledge Graph Creation**: Automatically extract entities and relationships
- **Vector Search**: Semantic similarity search using embeddings
- **Hybrid Retrieval**: Combine graph traversal with vector search
- **Open Source Models**: Use local or free cloud models
- **Neo4j Integration**: Leverage powerful graph database capabilities


## 📚 Additional Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [HuggingFace Models Hub](https://huggingface.co/models)
- [Sentence Transformers](https://www.sbert.net/)
- [spaCy NLP Library](https://spacy.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [GraphRAG Research Paper](https://arxiv.org/abs/2404.16130)

