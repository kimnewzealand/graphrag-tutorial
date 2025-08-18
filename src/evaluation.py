# GraphRAG Evaluation Framework
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever

# Evaluation Models with Pydantic V2

class EvaluationQuery(BaseModel):
    """Single evaluation query with expected answer"""
    query: str = Field(..., min_length=1, description="Test query")
    expected_answer: str = Field(..., min_length=1, description="Expected answer or key points")
    category: str = Field(default="general", description="Query category (factual, relational, etc.)")
    difficulty: str = Field(default="medium", pattern=r"^(easy|medium|hard)$")
    
    @field_validator('query', 'expected_answer')
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        return v.strip()

class RetrievalMetrics(BaseModel):
    """Metrics for retrieval evaluation"""
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    chunks_retrieved: int = Field(..., ge=0, description="Number of chunks retrieved")
    relevant_chunks: int = Field(..., ge=0, description="Number of relevant chunks")

class GenerationMetrics(BaseModel):
    """Metrics for generation evaluation"""
    factual_accuracy: float = Field(..., ge=0.0, le=1.0, description="Factual accuracy score")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Answer completeness")
    coherence: float = Field(..., ge=0.0, le=1.0, description="Answer coherence")
    groundedness: float = Field(..., ge=0.0, le=1.0, description="How well grounded in context")

class EvaluationResult(BaseModel):
    """Complete evaluation result for a single query"""
    query: str = Field(..., description="Original query")
    generated_answer: str = Field(..., description="Generated answer")
    expected_answer: str = Field(..., description="Expected answer")
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    response_time: float = Field(..., ge=0.0, description="Response time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)

class EvaluationSuite(BaseModel):
    """Complete evaluation suite results"""
    total_queries: int = Field(..., ge=0)
    avg_retrieval_precision: float = Field(..., ge=0.0, le=1.0)
    avg_retrieval_recall: float = Field(..., ge=0.0, le=1.0)
    avg_generation_accuracy: float = Field(..., ge=0.0, le=1.0)
    avg_response_time: float = Field(..., ge=0.0)
    results: List[EvaluationResult]
    evaluation_date: datetime = Field(default_factory=datetime.now)

# Evaluation Functions

def load_evaluation_dataset(file_path: Path) -> List[EvaluationQuery]:
    """Load evaluation queries from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = []
        for item in data:
            query = EvaluationQuery(**item)
            queries.append(query)
        
        print(f"✅ Loaded {len(queries)} evaluation queries")
        return queries
    
    except Exception as e:
        print(f"❌ Error loading evaluation dataset: {e}")
        return []

def evaluate_retrieval(retrieved_chunks: List[str], expected_chunks: List[str]) -> RetrievalMetrics:
    """Evaluate retrieval quality using precision, recall, F1"""
    if not retrieved_chunks:
        return RetrievalMetrics(
            precision=0.0, recall=0.0, f1_score=0.0,
            chunks_retrieved=0, relevant_chunks=0
        )
    
    # Simple keyword-based relevance (can be enhanced with semantic similarity)
    relevant_retrieved = 0
    for chunk in retrieved_chunks:
        for expected in expected_chunks:
            if any(keyword.lower() in chunk.lower() for keyword in expected.split()):
                relevant_retrieved += 1
                break
    
    precision = relevant_retrieved / len(retrieved_chunks) if retrieved_chunks else 0.0
    recall = relevant_retrieved / len(expected_chunks) if expected_chunks else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        chunks_retrieved=len(retrieved_chunks),
        relevant_chunks=relevant_retrieved
    )

def evaluate_generation_simple(generated: str, expected: str) -> GenerationMetrics:
    """Simple generation evaluation (can be enhanced with LLM-based evaluation)"""
    generated_lower = generated.lower()
    expected_lower = expected.lower()
    
    # Simple keyword overlap for factual accuracy
    expected_keywords = set(expected_lower.split())
    generated_keywords = set(generated_lower.split())
    
    keyword_overlap = len(expected_keywords.intersection(generated_keywords))
    factual_accuracy = keyword_overlap / len(expected_keywords) if expected_keywords else 0.0
    
    # Simple heuristics for other metrics
    completeness = min(len(generated) / len(expected), 1.0) if expected else 0.0
    coherence = 1.0 if len(generated.split('.')) >= 2 else 0.7  # Multi-sentence = more coherent
    groundedness = 0.8 if len(generated) > 50 else 0.5  # Longer answers assumed more grounded
    
    return GenerationMetrics(
        factual_accuracy=factual_accuracy,
        completeness=completeness,
        coherence=coherence,
        groundedness=groundedness
    )

async def evaluate_single_query(
    rag: GraphRAG, 
    query: EvaluationQuery,
    expected_chunks: Optional[List[str]] = None
) -> EvaluationResult:
    """Evaluate a single query end-to-end"""
    start_time = time.time()
    
    try:
        # Generate answer
        response = rag.search(query.query)
        generated_answer = response.answer
        
        # For this example, we'll use simple evaluation
        # In practice, you'd extract actual retrieved chunks from the response
        retrieved_chunks = [generated_answer]  # Simplified
        expected_chunks = expected_chunks or [query.expected_answer]
        
        # Evaluate retrieval
        retrieval_metrics = evaluate_retrieval(retrieved_chunks, expected_chunks)
        
        # Evaluate generation
        generation_metrics = evaluate_generation_simple(generated_answer, query.expected_answer)
        
        response_time = time.time() - start_time
        
        return EvaluationResult(
            query=query.query,
            generated_answer=generated_answer,
            expected_answer=query.expected_answer,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            response_time=response_time
        )
    
    except Exception as e:
        print(f"❌ Error evaluating query '{query.query}': {e}")
        # Return default metrics on error
        return EvaluationResult(
            query=query.query,
            generated_answer="Error occurred",
            expected_answer=query.expected_answer,
            retrieval_metrics=RetrievalMetrics(
                precision=0.0, recall=0.0, f1_score=0.0,
                chunks_retrieved=0, relevant_chunks=0
            ),
            generation_metrics=GenerationMetrics(
                factual_accuracy=0.0, completeness=0.0,
                coherence=0.0, groundedness=0.0
            ),
            response_time=time.time() - start_time
        )

async def run_evaluation_suite(
    rag: GraphRAG, 
    queries: List[EvaluationQuery]
) -> EvaluationSuite:
    """Run complete evaluation suite"""
    print(f"🔄 Running evaluation on {len(queries)} queries...")
    
    results = []
    for i, query in enumerate(queries):
        print(f"📝 Evaluating query {i+1}/{len(queries)}: {query.query[:50]}...")
        result = await evaluate_single_query(rag, query)
        results.append(result)
    
    # Calculate averages
    avg_retrieval_precision = sum(r.retrieval_metrics.precision for r in results) / len(results)
    avg_retrieval_recall = sum(r.retrieval_metrics.recall for r in results) / len(results)
    avg_generation_accuracy = sum(r.generation_metrics.factual_accuracy for r in results) / len(results)
    avg_response_time = sum(r.response_time for r in results) / len(results)
    
    suite = EvaluationSuite(
        total_queries=len(queries),
        avg_retrieval_precision=avg_retrieval_precision,
        avg_retrieval_recall=avg_retrieval_recall,
        avg_generation_accuracy=avg_generation_accuracy,
        avg_response_time=avg_response_time,
        results=results
    )
    
    print("✅ Evaluation complete!")
    print(f"📊 Average Precision: {avg_retrieval_precision:.3f}")
    print(f"📊 Average Recall: {avg_retrieval_recall:.3f}")
    print(f"📊 Average Accuracy: {avg_generation_accuracy:.3f}")
    print(f"⏱️ Average Response Time: {avg_response_time:.3f}s")
    
    return suite

def save_evaluation_results(suite: EvaluationSuite, output_path: Path) -> None:
    """Save evaluation results to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(suite.model_dump(), f, indent=2, default=str)
        print(f"✅ Evaluation results saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

# Example usage function
async def example_evaluation():
    """Example of how to run evaluation"""
    # This would be called from your main script
    # with your actual GraphRAG instance
    pass
