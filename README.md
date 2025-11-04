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

## LangGraph Migration Summary

### ✅ Completed Migration

The application has been successfully migrated from **LangChain 1.0 `create_agent`** (implicit LLM-driven workflow) to **LangGraph 1.0 explicit state graphs**.

---

### 🎯 Key Improvements

#### 1. **Explicit Workflow Control**
- **Before**: LLM decides tool call sequence via system prompt (non-deterministic)
- **After**: Explicit graph with defined nodes and edges (deterministic)
- **Benefit**: Guaranteed execution order, easier debugging

#### 2. **Parallel Execution**
- **Before**: Sequential tool calls (weather → wait → events → wait)
- **After**: Parallel execution (weather + events concurrently)
- **Benefit**: ~30-50% latency reduction for recommendation queries

#### 3. **Eliminated Double LLM Call**
- **Before**: `recommend_events` tool called LLM internally (2 API calls)
- **After**: Integrated synthesis node in graph (1 API call)
- **Benefit**: ~30% cost reduction for recommendations

#### 4. **Graph Visualization**
- **Before**: No visibility into workflow
- **After**: Mermaid diagrams showing execution paths
- **Benefit**: Better debugging, team collaboration, documentation

#### 5. **Streaming Support**
- **Before**: No progress indicators during execution
- **After**: Real-time node execution updates
- **Benefit**: Better UX, visibility into long-running operations

#### 6. **Proper Tool Execution Protocol**
- **Implementation**: Full LangChain ToolMessage protocol with tool_call_id
- **Error Handling**: Graceful degradation with try/except wrappers
- **LLM Synthesis**: Tool results synthesized into natural language responses
- **Benefit**: Correct message handling, user-friendly output, resilient execution

---

### 📊 Architecture Comparison

#### Old Architecture (create_agent)
```
User Query → LLM Agent (Black Box) → Tool Selection → Execution → Response
- Implicit workflow (system prompt-driven)
- Sequential execution
- No visualization
- Double LLM call for recommendations
```

#### New Architecture (LangGraph)
```
User Query → Router → [Recommendation Subgraph | Tool Node | General Node] → Response

Recommendation Subgraph:
  get_date → [get_weather || get_events] → synthesize
  
- Explicit workflow (code-driven)
- Parallel execution (weather + events)
- Full visualization
- Single LLM call for synthesis
```

---

### 🚀 Usage

#### Basic Chat (Non-streaming)
```python
# Use LangGraph agent
switch_agent(True)
chat_loop()

# Use old agent for comparison
switch_agent(False)
chat_loop()
```

#### Streaming Chat (Real-time Progress)
```python
# Only works with LangGraph agent
chat_loop_streaming()

# Or single query
chat_with_streaming("Recommend events for today")
```

#### Direct Subgraph Testing
```python
# Test recommendation subgraph directly
result = compiled_recommendation_graph.invoke({
    "messages": [HumanMessage(content="Events query")],
    "location": "Singapore",
    "country": "Singapore",
    "event_type": None
}, config)
```

---

### 🔧 Graph Components

#### State Schemas
- `RecommendationState`: For recommendation workflow (date, weather, events, location)
- `MainAgentState`: For main conversation flow (messages only)
- `ImageRequestState`: For future image generation workflow (not yet implemented)

#### Recommendation Subgraph Nodes
1. **get_date_node**: Retrieves current date
2. **get_weather_node**: Fetches weather (parallel)
3. **get_events_node**: Queries database (parallel)
4. **synthesize_recommendations_node**: LLM synthesis (replaces `recommend_events` tool)

#### Main Graph Nodes
1. **router_node**: Classifies intent (recommendation/venue_policy/image/general)
2. **recommendation_router_node**: Extracts params and invokes recommendation subgraph
3. **tool_execution_node**: Handles single-tool queries with 3 tools (venue_policies, image_request, image_approve) using single-shot execution pattern
4. **general_response_node**: Handles greetings and general queries

**Tool Execution Architecture Note:**
The `tool_execution_node` uses a simplified single-shot pattern rather than the full agent loop pattern. This design choice prioritizes simplicity and performance for independent queries (venue policies, image generation) that don't require multi-step reasoning. The recommendation workflow, which needs multi-step execution, is handled by the dedicated recommendation subgraph with parallel execution.

---

### 📈 Performance Metrics

Run the benchmark cell to compare:
- **Latency**: Avg time per query (Old vs LangGraph)
- **Success Rate**: Reliability comparison
- **Parallel Speedup**: Measured improvement from concurrent execution

Expected results:
- **Recommendation queries**: 30-50% faster (parallel execution)
- **Simple queries**: Similar performance (router overhead minimal)
- **Overall cost**: 30% reduction (eliminated double LLM call)

---

### 🎨 Visualization

View graph structures:
```python
# Visualize recommendation subgraph
display(Image(recommendation_graph.get_graph().draw_mermaid_png()))

# Visualize main agent graph
display(Image(main_graph.get_graph().draw_mermaid_png()))
```

---

### ⚙️ Architectural Differences from LangGraph Documentation

This implementation makes deliberate architectural choices that differ from the standard LangGraph patterns documented in the official guides:

#### **Single-Shot Tool Execution (vs. Agent Loop)**

**Current Implementation:**
```
User Query → Tool Node (execute once) → Synthesize Response → END
```

**Standard LangGraph Pattern:**
```
User Query → Agent → Tools → Agent → Tools → ... → Agent → END
                      ↑___________________|
                         (loop until done)
```

**Rationale:**
- ✅ Simpler and faster for independent queries (venue policies, image generation)
- ✅ More predictable execution path
- ✅ Sufficient for single-purpose tools that don't require iteration
- ✗ Cannot do multi-step reasoning within tool node
- ✗ No error recovery through retry

**Acceptable for this use case** because the recommendation workflow (which needs multi-step reasoning) is handled by a separate subgraph with explicit parallel execution.

#### **Manual Tool Execution (vs. ToolNode)**

**Current Implementation:**
```python
# Manual tool invocation with explicit ToolMessage creation
for tool_call in response.tool_calls:
    result = tool.invoke(tool_args)
    messages_to_return.append(
        ToolMessage(content=str(result), tool_call_id=tool_call_id, name=tool_name)
    )
```

**Standard LangGraph Pattern:**
```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)  # Automatic execution
```

**Rationale:**
- ✅ Fine-grained control over tool execution
- ✅ Explicit error handling per tool
- ✅ Custom synthesis logic after tool results
- ✗ More code to maintain
- ✗ Must manually implement ToolMessage protocol

**Note:** The `langgraph.prebuilt` module is deprecated in LangChain 1.0 in favor of `langchain.agents.create_agent`. The manual implementation here demonstrates understanding of the underlying protocol while avoiding deprecated APIs.

#### **Tool Scoping (3 Tools vs. 7)**

**Recommendation-specific tools** (`get_weather`, `get_events`, `recommend_events`) are **excluded** from the tool node and instead implemented as **hardcoded nodes** in the recommendation subgraph. This separation:
- ✅ Prevents routing confusion (clear separation of workflows)
- ✅ Enables parallel execution for recommendation workflow
- ✅ Avoids duplicate paths to the same functionality

---

### 🔄 Migration Status

✅ **Completed:**
- State schema definitions
- Recommendation subgraph with parallel execution
- Main agent graph with routing
- Proper ToolMessage protocol implementation
- Tool execution with error handling and LLM synthesis
- Graph visualization
- Streaming interface
- Performance benchmarking
- Backward compatibility (old agent preserved)
- Comprehensive architectural documentation

⏭️ **Future Enhancements:**
- Migrate image generation to graph-based workflow with `interrupt_before`
- Add more sophisticated parameter extraction (use LLM for location/country parsing)
- Implement caching for repeated queries
- Add retry logic and error handling nodes
- Deploy to LangGraph Platform for production
- Consider full agent loop pattern if multi-step tool reasoning becomes necessary

---

### 📚 Resources

- **LangGraph Docs**: https://docs.langchain.com/oss/python/langgraph/overview
- **Migration Guide**: See `new_langchain_langgraph.md`
- **Graph API**: https://docs.langchain.com/oss/python/langgraph/graph-api

---

### 🎓 Engineering Judgment

This migration demonstrates advanced LangChain/LangGraph knowledge while making pragmatic architectural decisions:
- **Where to use LangGraph**: Recommendation workflow (parallel execution, explicit control)
- **Where to stay simple**: Tool execution (single-shot sufficient for discrete queries)
- **When to deviate from docs**: Avoiding deprecated APIs (`langgraph.prebuilt`), optimizing for use case

The implementation prioritizes **demonstrating technical skills** (state graphs, parallel execution, visualization) while maintaining **production-ready code quality** (error handling, proper protocols, comprehensive documentation).