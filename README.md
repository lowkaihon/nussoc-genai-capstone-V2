# Multi-Agent Conversational AI with RAG, Image Generation, and Event Recommendations

## Architecture Overview
The developed system is a multi-agent conversational AI framework that integrates document-based reasoning, multimodal image generation, and real-time external data retrieval through modular subagents. It is designed for extensibility, reliability, and interpretability, using LangChain’s agentic framework with structured memory and tool orchestration.

At the core is a controller agent, built using ~~create_tool_calling_agent and AgentExecutor~~, which interprets user intent and delegates execution to the relevant tools or subagents. These components communicate via structured prompts, unified under a ChatPromptTemplate, allowing the system to dynamically compose context-aware, multi-step reasoning pipelines.

![System Architecture Diagram](./System%20Architecture%20Diagram.png)

## ✨ Upgraded to LangChain/LangGraph 1.0

**Migration Date:** 2025-11-02

### What Changed

This notebook has been upgraded from legacy LangChain to **LangChain/LangGraph 1.0** with the following improvements:

#### 🎯 Architecture Updates
- **Agent Framework**: Migrated from deprecated `create_tool_calling_agent` + `AgentExecutor` to LangChain 1.0's `create_agent`
- **Memory**: Replaced `ConversationBufferMemory` with LangGraph's `MemorySaver` checkpointer for durable state persistence
- **Subagents**: Simplified approach:
  - Weather & Event subagents → Flattened to direct API/database tools (faster, simpler)
  - Recommendation subagent → Simple LLM chain (no unnecessary overhead)

#### 🔒 Human-in-the-Loop (HITL)
- **Image Generation Approval**: Added two-step approval process to prevent unexpected Replicate API costs
  1. `request_image_generation` - Creates approval request
  2. `approve_image_generation` - Executes after user confirms

#### 💾 Conversation Persistence
- Chat history now persists across messages within the same session
- Uses LangGraph checkpointer with thread-based conversation management
- New `reset_conversation()` function to start fresh when needed

#### 🚀 Performance Improvements
- Direct API calls eliminate nested agent overhead
- Simple LLM chain for recommendations (no ReAct loop overhead)
- Better error handling and recovery
- Cleaner tool definitions with proper type hints

### New Tool Structure

| Tool Name | Type | Purpose |
|-----------|------|---------|
| `retrieve_venue_policies` | RAG | 3-stage retrieval (BM25 → Semantic → Jina Rerank) for venue policies |
| `request_image_generation` | HITL | Request image generation (requires approval) |
| `approve_image_generation` | HITL | Execute approved image generation |
| `get_current_date` | Utility | Get today's date for event queries |
| `get_weather` | API | Direct WeatherAPI call (was subagent) |
| `get_events` | Database | Direct SQLite query (was subagent) |
| `recommend_events` | LLM Chain | Simple prompt + LLM for recommendations (was subagent) |

### Usage Notes

1. **Conversation Persistence**: History is automatically saved. Use `reset_conversation()` to clear.
2. **Image Generation**: Agent will now ask for approval before generating images. Confirm by responding "yes" or approve the request.
3. **Event Recommendations**: The multi-step workflow (date → weather → events → recommendation) now executes more efficiently with direct tool calls.

For more details on LangChain/LangGraph 1.0, see: `new_langchain_langgraph.md`

## 🎯 3-Stage Retrieval Pipeline (Hybrid RAG + Reranking)

**Implementation Date:** 2025-11-03

### Overview

The venue policy retrieval system uses a **3-stage retrieval pipeline** combining keyword search, semantic search, and cross-encoder reranking for optimal relevance:

1. **Stage 1: BM25 Keyword Search** - Excels at proper nouns, postal codes, MRT codes
2. **Stage 2: Semantic Vector Search** - Understands concepts like "accessibility" or "restrictions"
3. **Stage 3: Jina AI Cross-Encoder Reranking** - Fine-grained relevance scoring of top candidates

This architecture was implemented to achieve "Tier S" performance for an ML/AI Engineer portfolio, demonstrating advanced RAG techniques beyond basic semantic search.

### Technical Implementation

```python
# Hybrid retriever (Stage 1 + 2)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]  # Equal weighting with RRF fusion
)

# Add Jina reranker (Stage 3)
compressor = JinaRerank(
    model="jina-reranker-v2-base-multilingual",
    top_n=3,
    jina_api_key=JINA_API_KEY
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)
```

### 📊 Evaluation Results

A/B testing with 20 test queries across exact-match, semantic, and hybrid query types:

| Metric | Hybrid RAG Only | Hybrid + Jina Rerank | Improvement |
|--------|-----------------|---------------------|-------------|
| **Precision@3** | 80.0% (16/20) | 90.0% (18/20) | **+12.5%** |
| **MRR** | 0.775 | 0.817 | **+5.4%** |
| **Avg Latency** | 805ms | 2022ms | +1216ms (2.5x) |
| **Cost per query** | $0 | ~$0.002 | Free tier: 1M tokens/month |

#### Performance by Query Type

| Query Type | Hybrid Only | Hybrid + Rerank | Improvement |
|------------|-------------|-----------------|-------------|
| **Exact Match** (5 queries) | 40% (2/5) | 80% (4/5) | **+100%** |
| **Semantic** (5 queries) | 80% (4/5) | 80% (4/5) | +0% |
| **Hybrid** (10 queries) | 100% (10/10) | 100% (10/10) | +0% |

### Key Insights

**Where reranking adds value:**
- ✅ **Exact-match queries** with proper nouns (venue names, codes) - doubled accuracy
- ✅ **Multi-document relevance** requiring fine-grained scoring
- ✅ **Query-document interactions** that embeddings alone miss

**Trade-offs:**
- ⚠️ 2.5x latency increase (acceptable for accuracy-critical use cases)
- ✅ Minimal cost impact with generous free tier
- ✅ No local model downloads (cloud API)

### Why Jina Reranker?

Selected Jina Reranker v2 over alternatives (Cohere, local models) for:
- **Compatibility**: Works with LangChain 1.0 (Cohere had version conflicts)
- **Cost**: 1M tokens/month free tier vs. more limited alternatives
- **Performance**: State-of-the-art cross-encoder without local GPU requirements
- **Simplicity**: Available in `langchain_community` (no extra packages)