# GraphRAG Evaluation Guide

## 🎯 Overview

This guide explains how to evaluate your GraphRAG system using the integrated evaluation framework. Evaluation helps you measure and improve the quality of your knowledge graph retrieval and answer generation.

## 📊 What Gets Evaluated

### **1. Retrieval Quality**
- **Precision**: How many retrieved chunks are actually relevant?
- **Recall**: How many relevant chunks were successfully retrieved?
- **F1 Score**: Harmonic mean of precision and recall

### **2. Generation Quality**
- **Factual Accuracy**: How factually correct are the generated answers?
- **Completeness**: Do answers address all aspects of the question?
- **Coherence**: Are answers well-structured and logical?
- **Groundedness**: Are answers supported by the retrieved context?

### **3. Performance Metrics**
- **Response Time**: How fast does the system respond?
- **Throughput**: How many queries can be processed per minute?

## 🚀 Quick Start

### **1. Run Evaluation**
```bash
# Run evaluation mode
python src/graph.py --eval

# Run normal mode (default)
python src/graph.py
```

### **2. View Results**
Evaluation results are saved to `data/evaluation_results_YYYYMMDD_HHMMSS.json`

## 📝 Evaluation Dataset

### **Current Dataset** (`data/evaluation_queries.json`)
Contains 8 test queries covering:
- **Factual questions**: "How many levels is Company data classified?"
- **Compliance questions**: "What are the approval requirements for LLMs?"
- **Policy questions**: "What is the purpose of the Data Classification Policy?"
- **Relational questions**: "What is the relationship between data classification and LLM usage?"

### **Adding New Test Cases**
```json
{
  "query": "Your test question here",
  "expected_answer": "Expected answer or key points",
  "category": "factual|compliance|policy|relational",
  "difficulty": "easy|medium|hard"
}
```

## 📈 Understanding Results

### **Sample Output**
```
📊 EVALUATION SUMMARY
==================================================
Total Queries: 8
Avg Retrieval Precision: 0.750
Avg Retrieval Recall: 0.625
Avg Generation Accuracy: 0.825
Avg Response Time: 1.234s
==================================================
```

### **Interpreting Scores**
- **0.8-1.0**: Excellent performance
- **0.6-0.8**: Good performance, room for improvement
- **0.4-0.6**: Fair performance, needs optimization
- **0.0-0.4**: Poor performance, requires significant improvement

## 🔧 Customizing Evaluation

### **1. Enhanced Retrieval Evaluation**
```python
def evaluate_retrieval_semantic(retrieved_chunks: List[str], expected_chunks: List[str]) -> RetrievalMetrics:
    """Use semantic similarity instead of keyword matching"""
    # Use embedding similarity for more accurate evaluation
    # Implementation would use sentence transformers
    pass
```

### **2. LLM-Based Generation Evaluation**
```python
async def evaluate_generation_llm(generated: str, expected: str, llm: LLM) -> GenerationMetrics:
    """Use LLM to evaluate answer quality"""
    evaluation_prompt = f"""
    Evaluate this answer on a scale of 0-1:
    Question: {query}
    Generated Answer: {generated}
    Expected Answer: {expected}
    
    Rate factual accuracy, completeness, and coherence.
    """
    # Use LLM to score the answer
    pass
```

### **3. Custom Metrics**
```python
class CustomMetrics(BaseModel):
    domain_relevance: float = Field(..., ge=0.0, le=1.0)
    citation_accuracy: float = Field(..., ge=0.0, le=1.0)
    bias_score: float = Field(..., ge=0.0, le=1.0)
```

## 🎯 Evaluation Best Practices

### **1. Diverse Test Cases**
- Cover different query types (factual, analytical, comparative)
- Include edge cases and difficult questions
- Test both simple and complex multi-hop reasoning

### **2. Regular Evaluation**
```bash
# Run evaluation after changes
python src/graph.py --eval

# Compare results over time
ls data/evaluation_results_*.json
```

### **3. Baseline Comparison**
- Establish baseline performance metrics
- Track improvements after system changes
- A/B test different configurations

### **4. Domain-Specific Evaluation**
- Create evaluation sets specific to your domain
- Include domain experts in creating expected answers
- Validate evaluation metrics with human judgment

## 📊 Advanced Evaluation Techniques

### **1. Human Evaluation**
```python
class HumanEvaluationResult(BaseModel):
    query: str
    generated_answer: str
    human_rating: int = Field(..., ge=1, le=5)
    human_feedback: str
    evaluator_id: str
```

### **2. Comparative Evaluation**
```python
async def compare_systems(rag_v1: GraphRAG, rag_v2: GraphRAG, queries: List[str]):
    """Compare two different RAG configurations"""
    results_v1 = await run_evaluation_suite(rag_v1, queries)
    results_v2 = await run_evaluation_suite(rag_v2, queries)
    
    # Compare metrics
    improvement = results_v2.avg_generation_accuracy - results_v1.avg_generation_accuracy
    print(f"Accuracy improvement: {improvement:.3f}")
```

### **3. Error Analysis**
```python
def analyze_failures(results: List[EvaluationResult]):
    """Analyze queries with low scores"""
    failures = [r for r in results if r.generation_metrics.factual_accuracy < 0.5]
    
    for failure in failures:
        print(f"Failed Query: {failure.query}")
        print(f"Generated: {failure.generated_answer}")
        print(f"Expected: {failure.expected_answer}")
        print("---")
```

## 🔄 Continuous Improvement Workflow

### **1. Baseline Establishment**
```bash
# Initial evaluation
python src/graph.py --eval
cp data/evaluation_results_*.json data/baseline_results.json
```

### **2. Iterative Improvement**
1. **Identify weak areas** from evaluation results
2. **Make targeted improvements** (better chunking, improved prompts, etc.)
3. **Re-evaluate** to measure improvement
4. **Compare** with baseline

### **3. Automated Evaluation Pipeline**
```bash
#!/bin/bash
# evaluation_pipeline.sh

echo "Running evaluation pipeline..."

# Run evaluation
python src/graph.py --eval

# Extract key metrics
python scripts/extract_metrics.py data/evaluation_results_*.json

# Compare with baseline
python scripts/compare_results.py data/baseline_results.json data/evaluation_results_*.json

echo "Evaluation complete!"
```

## 📋 Evaluation Checklist

Before deploying changes:

- [ ] Run full evaluation suite
- [ ] Check all metrics are within acceptable ranges
- [ ] Compare with previous baseline
- [ ] Analyze any significant performance drops
- [ ] Update baseline if improvements are confirmed
- [ ] Document changes and their impact

## 🎯 Recommended Evaluation Schedule

- **Daily**: Quick smoke tests on core queries
- **Weekly**: Full evaluation suite
- **Monthly**: Human evaluation and baseline updates
- **Before releases**: Comprehensive evaluation with comparison

## 📚 Further Reading

- [RAG Evaluation Best Practices](https://docs.llamaindex.ai/en/stable/optimizing/evaluation/)
- [LLM Evaluation Frameworks](https://github.com/microsoft/promptflow)
- [Semantic Similarity Metrics](https://www.sbert.net/)

---

*This evaluation framework helps ensure your GraphRAG system maintains high quality and continues to improve over time.*
