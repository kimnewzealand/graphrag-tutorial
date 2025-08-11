import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import partial, reduce
from itertools import chain

# Import existing functions from graph.py
from graph import (
    GraphRAGConfig, create_neo4j_driver, create_llm, create_embedder,
    create_vector_index, create_retriever, process_pdf, create_embeddings,
    store_chunks_and_entities, store_embeddings, query_graph
)

from neo4j_graphrag.llm import AnthropicLLM as LLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings as Embeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG

class QueryType(Enum):
    SIMPLE_FACT = "simple_fact"
    COMPLEX_REASONING = "complex_reasoning"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"

class ToolType(Enum):
    VECTOR_SEARCH = "vector_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    ENTITY_ANALYSIS = "entity_analysis"

@dataclass(frozen=True)
class QueryStep:
    step_id: str
    query: str
    tool_type: ToolType
    context_needed: Tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    query_type: QueryType
    steps: Tuple[QueryStep, ...]
    expected_complexity: int

@dataclass(frozen=True)
class RetrievalResult:
    content: str
    sources: Tuple[str, ...]
    confidence: float
    entities: Tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class AgenticResponse:
    answer: str
    reasoning_chain: Tuple[str, ...]
    confidence: float
    sources: Tuple[str, ...]
    query_plan: QueryPlan
    execution_steps: Tuple[Dict[str, Any], ...]

def parse_llm_response(response_content: str) -> Dict[str, Any]:
    """Pure function to parse LLM response"""
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        return {"query_type": "simple_fact", "complexity": 1, "steps": []}

def create_query_step(step_data: Dict[str, Any]) -> QueryStep:
    """Pure function to create QueryStep from data"""
    return QueryStep(
        step_id=step_data.get("step_id", "step_1"),
        query=step_data.get("query", ""),
        tool_type=ToolType(step_data.get("tool_type", "vector_search")),
        context_needed=tuple(step_data.get("context_needed", []))
    )

def create_query_plan(query: str, analysis: Dict[str, Any]) -> QueryPlan:
    """Pure function to create QueryPlan from analysis"""
    steps = tuple(map(create_query_step, analysis.get("steps", [])))
    
    return QueryPlan(
        original_query=query,
        query_type=QueryType(analysis.get("query_type", "simple_fact")),
        steps=steps,
        expected_complexity=analysis.get("complexity", 1)
    )

def create_fallback_plan(query: str) -> QueryPlan:
    """Pure function to create fallback plan"""
    step = QueryStep(
        step_id="step_1",
        query=query,
        tool_type=ToolType.VECTOR_SEARCH
    )
    return QueryPlan(
        original_query=query,
        query_type=QueryType.SIMPLE_FACT,
        steps=(step,),
        expected_complexity=1
    )

def create_query_analyzer(llm: LLM) -> Callable[[str], QueryPlan]:
    """Create a query analyzer function using existing LLM"""
    
    async def analyze_query(query: str) -> QueryPlan:
        analysis_prompt = f"""
        Analyze this query and determine the approach needed:
        
        Query: {query}
        
        Return JSON format:
        {{
            "query_type": "simple_fact|complex_reasoning|multi_hop|comparison|temporal",
            "complexity": 1-5,
            "steps": [
                {{
                    "step_id": "step_1",
                    "query": "What are the data classification levels?",
                    "tool_type": "vector_search|graph_traversal|entity_analysis"
                }}
            ]
        }}
        """
        
        response = llm.invoke(analysis_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        analysis = parse_llm_response(content)
        
        return create_query_plan(query, analysis) if analysis.get("steps") else create_fallback_plan(query)
    
    return analyze_query

def create_vector_searcher(rag: GraphRAG, driver) -> Callable[[str], RetrievalResult]:
    """Create vector search function using existing query_graph"""
    
    def vector_search(query: str) -> RetrievalResult:
        try:
            # Use existing query_graph function
            answer = query_graph(rag, driver, query)
            
            # Extract basic statistics for sources
            with driver.session() as session:
                chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as count").single()["count"]
                sources = tuple(f"Chunk_{i}" for i in range(min(5, chunk_count)))
            
            return RetrievalResult(
                content=answer,
                sources=sources,
                confidence=0.8,
                entities=tuple()
            )
        except Exception as e:
            return RetrievalResult(
                content=f"Error in vector search: {e}",
                sources=tuple(),
                confidence=0.0
            )
    
    return vector_search

def create_graph_traverser(driver, llm: LLM) -> Callable[[str, Dict], RetrievalResult]:
    """Create graph traversal function using existing Neo4j driver"""
    
    def graph_traversal(query: str, context: Dict[str, Any] = None) -> RetrievalResult:
        try:
            # Extract entities using existing LLM
            entity_prompt = f"Extract 2-3 main entities from: {query}"
            entity_response = llm.invoke(entity_prompt)
            entity_content = entity_response.content if hasattr(entity_response, 'content') else str(entity_response)
            entities = tuple(entity_content.split('\n')[:3])
            
            # Use existing driver for graph queries
            with driver.session() as session:
                results = []
                for entity in entities:
                    if entity.strip():
                        cypher_query = """
                        MATCH (e1:Entity)-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2:Entity)
                        WHERE e1.name CONTAINS $entity OR e2.name CONTAINS $entity
                        RETURN e1.name, e2.name, c.text[0..200] as context
                        LIMIT 5
                        """
                        result = session.run(cypher_query, entity=entity.strip())
                        results.extend(list(result))
                
                if results:
                    content = "\n".join([
                        f"{r['e1.name']} <-> {r['e2.name']}: {r['context']}"
                        for r in results
                    ])
                else:
                    content = "No graph connections found"
                
                return RetrievalResult(
                    content=content,
                    sources=("Graph_Traversal",),
                    confidence=0.7 if results else 0.3,
                    entities=entities
                )
        except Exception as e:
            return RetrievalResult(
                content=f"Error in graph traversal: {e}",
                sources=tuple(),
                confidence=0.0
            )
    
    return graph_traversal

def create_entity_analyzer(driver) -> Callable[[str, Dict], RetrievalResult]:
    """Create entity analysis function using existing Neo4j driver"""
    
    def entity_analysis(query: str, context: Dict[str, Any] = None) -> RetrievalResult:
        try:
            with driver.session() as session:
                # Use existing graph structure
                cypher_query = """
                MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
                RETURN e.name, count(c) as mentions, collect(c.text[0..100])[0..2] as contexts
                ORDER BY mentions DESC
                LIMIT 10
                """
                
                result = session.run(cypher_query)
                records = list(result)
                
                if records:
                    content = "\n".join([
                        f"Entity: {r['e.name']} (mentioned {r['mentions']} times)\nContexts: {r['contexts']}"
                        for r in records
                    ])
                    entities = tuple(r['e.name'] for r in records)
                else:
                    content = "No entities found"
                    entities = tuple()
                
                return RetrievalResult(
                    content=content,
                    sources=("Entity_Analysis",),
                    confidence=0.6 if records else 0.2,
                    entities=entities
                )
        except Exception as e:
            return RetrievalResult(
                content=f"Error in entity analysis: {e}",
                sources=tuple(),
                confidence=0.0
            )
    
    return entity_analysis

def create_tool_dispatcher(vector_searcher: Callable, graph_traverser: Callable, 
                         entity_analyzer: Callable) -> Callable:
    """Create tool dispatcher using existing retrieval functions"""
    
    tool_map = {
        ToolType.VECTOR_SEARCH: vector_searcher,
        ToolType.GRAPH_TRAVERSAL: graph_traverser,
        ToolType.ENTITY_ANALYSIS: entity_analyzer
    }
    
    def execute_step(step: QueryStep, context: Dict[str, Any] = None) -> RetrievalResult:
        tool_func = tool_map.get(step.tool_type, vector_searcher)
        
        # Handle different function signatures
        if step.tool_type == ToolType.VECTOR_SEARCH:
            return tool_func(step.query)
        else:
            return tool_func(step.query, context)
    
    return execute_step

def create_synthesizer(llm: LLM) -> Callable:
    """Create synthesizer using existing LLM"""
    
    def synthesize_results(query: str, retrieval_results: List[RetrievalResult], 
                         query_plan: QueryPlan) -> Tuple[str, Tuple[str, ...], float]:
        
        # Combine results
        combined_content = "\n\n".join([
            f"Step {i+1} ({', '.join(result.sources)}):\n{result.content}"
            for i, result in enumerate(retrieval_results)
        ])
        
        synthesis_prompt = f"""
        Based on the retrieved information, provide a comprehensive answer.
        
        Query: {query}
        Information: {combined_content}
        
        Provide a clear answer and explain your reasoning.
        """
        
        response = llm.invoke(synthesis_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Simple confidence calculation
        confidence = 0.8 if len(retrieval_results) > 1 else 0.6
        reasoning = ("Analyzed multiple sources", "Synthesized comprehensive answer")
        
        return content, reasoning, confidence
    
    return synthesize_results

def create_agentic_pipeline(config: GraphRAGConfig) -> Callable[[str], AgenticResponse]:
    """Create agentic pipeline using existing graph.py functions"""
    
    # Initialize using existing functions
    driver = create_neo4j_driver(config)
    llm = create_llm(config.anthropic_api_key)
    embedder = create_embedder()
    create_vector_index(driver, config.index_name)
    retriever = create_retriever(driver, config.index_name, embedder)
    rag = GraphRAG(llm=llm, retriever=retriever)
    
    # Create functional components
    query_analyzer = create_query_analyzer(llm)
    vector_searcher = create_vector_searcher(rag, driver)
    graph_traverser = create_graph_traverser(driver, llm)
    entity_analyzer = create_entity_analyzer(driver)
    step_executor = create_tool_dispatcher(vector_searcher, graph_traverser, entity_analyzer)
    synthesizer = create_synthesizer(llm)
    
    async def process_query(query: str) -> AgenticResponse:
        """Process query using functional composition of existing functions"""
        
        print(f"🤖 Processing agentic query with existing functions: {query}")
        
        # Step 1: Query analysis
        query_plan = await query_analyzer(query)
        print(f"📋 Plan: {query_plan.query_type.value} with {len(query_plan.steps)} steps")
        
        # Step 2: Execute steps
        results = []
        context = {}
        execution_steps = []
        
        for i, step in enumerate(query_plan.steps):
            print(f"🔍 Step {i+1}: {step.tool_type.value}")
            result = step_executor(step, context)
            results.append(result)
            context[step.step_id] = result
            
            execution_steps.append({
                "step_id": step.step_id,
                "tool_type": step.tool_type.value,
                "confidence": result.confidence,
                "sources": result.sources
            })
        
        # Step 3: Synthesis
        print("🧠 Synthesizing...")
        answer, reasoning_chain, confidence = synthesizer(query, results, query_plan)
        
        # Collect all sources
        all_sources = tuple(chain.from_iterable(result.sources for result in results))
        
        return AgenticResponse(
            answer=answer,
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            sources=all_sources,
            query_plan=query_plan,
            execution_steps=tuple(execution_steps)
        )
    
    return process_query

class FunctionalAgenticGraphRAG:
    """Agentic GraphRAG built on existing graph.py functions"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.process_query = create_agentic_pipeline(config)
    
    async def chat(self, query: str) -> str:
        """Chat interface using existing functions"""
        response = await self.process_query(query)
        
        print(f"\n🎯 Confidence: {response.confidence:.2f}")
        print(f"📚 Sources: {', '.join(response.sources)}")
        print(f"🔗 Reasoning: {' → '.join(response.reasoning_chain)}")
        
        return response.answer
    
    async def setup_knowledge_base(self):
        """Setup knowledge base using existing functions"""
        print("📚 Setting up knowledge base...")
        
        # Use existing PDF processing pipeline
        split_result = await process_pdf(self.config)
        embedder = create_embedder()
        embedded_chunks = await create_embeddings(embedder, split_result)
        
        # Use existing storage functions
        driver = create_neo4j_driver(self.config)
        llm = create_llm(self.config.anthropic_api_key)
        
        store_chunks_and_entities(driver, embedded_chunks, llm, self.config.pdf_file_path)
        store_embeddings(driver, embedded_chunks, embedder)
        
        driver.close()
        print("✅ Knowledge base setup complete")

async def demo_functional_agentic_with_existing():
    """Demo using existing graph.py infrastructure"""
    
    # Use existing configuration
    config = GraphRAGConfig.from_env()
    
    # Create functional agentic system
    agentic_rag = FunctionalAgenticGraphRAG(config)
    
    # Setup knowledge base using existing functions
    await agentic_rag.setup_knowledge_base()
    
    # Test with complex queries
    queries = [
        "What are the main data classification levels and their rules?",
        "How do LLM usage policies relate to data classification requirements?",
        "What timeline requirements exist across all policies?",
        "Compare approval processes for different types of data and AI usage"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        answer = await agentic_rag.chat(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(demo_functional_agentic_with_existing())