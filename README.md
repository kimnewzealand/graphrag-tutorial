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


### Static pipeline (PDF → Chunks → Embeddings → Query → Answer)

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

### Agentic: Dynamic, multi-step reasoning :

```
> python .\src\agentic_graphrag.py
Checking Neo4j connection to neo4j+s://e40a0a8b.databases.neo4j.io...
✅ Neo4j connection successful
✅ Initialized Anthropic Extraction LLM claude-3-5-sonnet-20241022
✅ Initialized SentenceTransformer embeddings LLM sentence-transformers/all-MiniLM-L6-v2
✅ Vector index text_embeddings already exists
✅ Initialized VectorRetriever
📚 Setting up knowledge base...
✅ Loaded PDF file from data/sample_IT_compliance_document.pdf
✅ Split PDF into 2 chunks
✅ Initialized SentenceTransformer embeddings LLM sentence-transformers/all-MiniLM-L6-v2
Checking Neo4j connection to neo4j+s://e40a0a8b.databases.neo4j.io...
✅ Neo4j connection successful
✅ Initialized Anthropic Extraction LLM claude-3-5-sonnet-20241022
✅ Stored chunks and entities
✅ Created and stored 2 embeddings
✅ Knowledge base setup complete

============================================================
Query: What are the main data classification levels and their rules?
🤖 Processing agentic query with existing functions: What are the main data classification levels and their rules?
📋 Plan: simple_fact with 1 steps
🔍 Step 1: vector_search
📊 Found 50 chunks, 50 with embeddings
🧠 Synthesizing...

🎯 Confidence: 0.60
📚 Sources: Chunk_0, Chunk_1, Chunk_2, Chunk_3, Chunk_4
🔗 Reasoning: Analyzed multiple sources → Synthesized comprehensive answer
Answer: Based on the provided information, I can provide a comprehensive answer about the main data classification levels and their rules:

The three main data classification levels are:

1. Public Data
- This is the least restricted level
- Can be freely shared with external parties
- No special handling requirements mentioned

2. Internal Data
- Restricted to company use only
- Cannot be shared externally
- Meant for internal business operations

3. Confidential Data
- Highest level of security
- Requires special handling procedures
- Must be encrypted
- Most restricted access level

Universal Rule:
- All new data must be classified within 24 hours of creation, regardless of its classification level

My reasoning:
The information clearly outlines a three-tiered classification system that progresses from least restricted (Public) to most restricted (Confidential). Each level has specific handling requirements that become more stringent as the sensitivity increases. The 24-hour classification rule appears to be an overarching policy that ensures all data is properly categorized and protected according to its sensitivity level in a timely manner. This structure is typical of data classification systems designed to protect organizational information assets while allowing appropriate access and sharing.

============================================================
Query: How do LLM usage policies relate to data classification requirements?
🤖 Processing agentic query with existing functions: How do LLM usage policies relate to data classification requirements?
📋 Plan: simple_fact with 1 steps
🔍 Step 1: vector_search
📊 Found 50 chunks, 50 with embeddings
🧠 Synthesizing...

🎯 Confidence: 0.60
📚 Sources: Chunk_0, Chunk_1, Chunk_2, Chunk_3, Chunk_4
🔗 Reasoning: Analyzed multiple sources → Synthesized comprehensive answer
Answer: Based on the provided information, I can explain how LLM usage policies relate to data classification requirements:

Answer:
LLM usage policies and data classification requirements appear to be interconnected components of an organization's IT Compliance Agreement for AI use. While they operate as separate policies, they work together to ensure proper data handling when using AI systems:

Key Relationships:
1. Approval Process Integration
- LLM usage requires IT Security approval within 48 hours
- Data must be classified within 24 hours of creation
- This suggests that data classification must be completed before or during the LLM approval process

2. Data Handling Constraints
- Different classification levels (Public, Internal, Confidential) have specific handling requirements
- These classifications would naturally restrict how data can be used with LLMs
- Particularly sensitive for Confidential data requiring encryption

Reasoning:
My analysis is based on these policies being part of the same IT Compliance Agreement, suggesting they're meant to work together. The timing requirements (24 hours for classification, 48 hours for LLM approval) and the presence of specific data handling requirements indicate that data classification status must be considered when seeking approval for LLM usage. This creates a natural dependency where data classification influences whether and how data can be used with LLMs.

While the relationship isn't explicitly stated in the provided information, the framework suggests that data classification acts as a prerequisite consideration for LLM usage approval.

============================================================
Query: What timeline requirements exist across all policies?
🤖 Processing agentic query with existing functions: What timeline requirements exist across all policies?        
📋 Plan: simple_fact with 1 steps
🔍 Step 1: vector_search
📊 Found 50 chunks, 50 with embeddings
🧠 Synthesizing...

🎯 Confidence: 0.60
📚 Sources: Chunk_0, Chunk_1, Chunk_2, Chunk_3, Chunk_4
🔗 Reasoning: Analyzed multiple sources → Synthesized comprehensive answer
Answer: Based on the provided information, I can organize the timeline requirements into three main categories with specific deadlines:

1. AI and LLM Requirements:
- 24-hour deadlines for:
  * LLM content review before use
  * AI tool usage documentation
- 48 hours for AI content factual accuracy verification
- 72 hours for sensitive data processing approval

2. Access Control Requirements:
- Immediate action for terminated employee access revocation
- 3 business days for access request approvals
- Quarterly reviews (March 30, June 30, September 30, December 30)

3. Compliance Requirements:
- 30 days to address non-compliance issues
- Quarterly reports due by the 15th of each quarter
- Annual training completion by December 31st

The reasoning behind these timelines appears to follow a logical pattern:
- Shorter windows (24-72 hours) for immediate security and AI-related concerns
- Medium-term windows (3-30 days) for administrative and access-related tasks
- Longer periodic requirements (quarterly and annual) for ongoing compliance and review processes

This structured approach helps ensure both immediate security needs and long-term compliance requirements are met while maintaining practical implementation timeframes.

============================================================
Query: Compare approval processes for different types of data and AI usage
🤖 Processing agentic query with existing functions: Compare approval processes for different types of data and AI usage
📋 Plan: comparison with 5 steps
🔍 Step 1: vector_search
📊 Found 50 chunks, 50 with embeddings
🔍 Step 2: vector_search
📊 Found 50 chunks, 50 with embeddings
🔍 Step 3: graph_traversal
🔍 Step 4: graph_traversal
🔍 Step 5: entity_analysis
🧠 Synthesizing...

🎯 Confidence: 0.80
📚 Sources: Chunk_0, Chunk_1, Chunk_2, Chunk_3, Chunk_4, Chunk_0, Chunk_1, Chunk_2, Chunk_3, Chunk_4, Graph_Traversal, Graph_Traversal, Entity_Analysis
🔗 Reasoning: Analyzed multiple sources → Synthesized comprehensive answer
Answer: Based on the provided information, I'll compare the approval processes for different types of data and AI usage:

Data Classification Approval Processes:
1. Public Data:
- Least restrictive approval process
- Can be shared externally
- Must still be classified within 24 hours of creation

2. Internal Data:
- Moderate restrictions
- Limited to company use only
- Must be classified within 24 hours of creation

3. Confidential Data:
- Most restrictive approval process
- Requires special handling and encryption
- Cannot be input into public LLM services
- Must be classified within 24 hours of creation

AI Usage Approval Processes:
1. LLM (Large Language Model) Usage:
- Requires IT Security team approval within 48 hours of request
- Approved LLM tools must be logged and monitored
- LLM-generated content must be reviewed for accuracy within 24 hours before use

2. AI Content Generation:
- Must be clearly labeled as AI-generated
- Requires factual accuracy verification within 48 hours
- Cannot process sensitive customer or business data

Key Differences and Reasoning:
1. Timing Requirements:
- Data classification: 24-hour classification requirement
- LLM approval: 48-hour approval window
- AI content verification: 48-hour accuracy check

2. Approval Authority:
- Data classification appears to be a general requirement without specific approval authority mentioned
- LLM usage specifically requires IT Security team approval
- AI content generation focuses on self-verification and labeling requirements

3. Usage Restrictions:
- More stringent restrictions exist for confidential data and sensitive information
- Public data has the most flexible usage guidelines
- AI tools have specific monitoring and logging requirements

This comparison shows a layered approach to approval processes, with stricter requirements for more sensitive data and AI applications. The policies emphasize security, accuracy, and proper documentation across all categories. 

```

## 📁 Project Structure

```
graphrag-tutorial/
├── data/                     # Your source documents (PDF/TXT files)
├── src/                      # Python source code
│   ├── agentic_graphrag.py   # Agentic GraphRAG System using existing functions from graph.py
│   ├── graph.py          # Create knowledge graph from PDF
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

