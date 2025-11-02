# LangChain & LangGraph 1.0 Updates

**Last Updated:** 2025-11-02
**Status:** Both frameworks now at 1.0 (production-ready, stability commitment)

## Executive Summary

LangChain and LangGraph have reached 1.0, marking a major shift from high-level abstractions to low-level control with production-ready features. Key changes include:

- **LangChain 1.0**: Simplified agent creation, powerful middleware system, provider-agnostic content blocks
- **LangGraph 1.0**: Durable state persistence, true human-in-the-loop, deterministic concurrent execution
- **Breaking Changes**: Python 3.10+ required, legacy features moved to `langchain-classic`
- **Philosophy**: Composable approach - start with LangChain, drop to LangGraph for fine-grained control

---

## =� Essential Links

### Official Documentation
- **Main Docs Hub**: https://docs.langchain.com/
- **LangChain Python Docs**: https://docs.langchain.com/oss/python/
- **LangGraph Overview**: https://docs.langchain.com/oss/python/langgraph/overview
- **LangGraph Graph API**: https://docs.langchain.com/oss/python/langgraph/graph-api
- **LangChain Agents Guide**: https://docs.langchain.com/oss/python/langchain/agents

### Release Information
- **1.0 Announcement Blog**: https://blog.langchain.com/langchain-langgraph-1dot0/
- **LangChain v1 Release Notes**: https://docs.langchain.com/oss/python/releases/langchain-v1

### Migration Guides
- **Python Migration Guide**: https://docs.langchain.com/oss/python/migrate/langchain-v1
- **JavaScript Migration Guide**: https://docs.langchain.com/oss/javascript/migrate/langchain-v1

---

## =� LangChain 1.0 - Major Features

### 1. New `create_agent` Function

**Status**: Replaces deprecated `langgraph.prebuilt.create_react_agent`

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer for conversation persistence
checkpointer = MemorySaver()

# Agent creation with memory
agent = create_agent(
    model=model,
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant",
    checkpointer=checkpointer  # Add this for conversation memory
)
```

**How to Invoke the Agent:**

```python
from langchain_core.messages import HumanMessage, AIMessage

# Invoke with thread_id for conversation persistence
result = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    config={"configurable": {"thread_id": "user_123"}}
)

# Extract the response
ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
response = ai_messages[-1].content
print(response)
```

**Key Benefits**:
- Simplified API for building agents with any model provider
- Built on LangGraph's runtime for enhanced reliability
- Straightforward loop: model invocation � tool selection � execution � completion
- Accepts model specification, tool definitions, system prompts, and checkpointer
- Conversation history persists across invocations when using checkpointer with thread_id

### 2. Middleware System - The Defining Feature

**Concept**: Composable hooks at different execution stages for maximum customization

**Middleware Types**:
- `before_agent` / `after_agent` - Wraps entire execution
- `before_model` / `after_model` - Surrounds LLM calls
- `wrap_model_call` / `wrap_tool_call` - Intercepts requests/responses

**Built-in Middleware**:

#### PIIMiddleware
- Automatic data redaction for compliance
- Protects sensitive information (SSN, credit cards, etc.)

#### SummarizationMiddleware
- Conversation compression to manage context limits
- Keeps conversations within token budgets

#### HumanInTheLoopMiddleware
- Approval gates for high-stakes decisions
- Pause execution for human review

**Use Cases**:
- Dynamic prompt engineering
- Context management
- Guardrails implementation
- Compliance and security
- Custom logging and monitoring

### 3. Standard Content Blocks - Provider-Agnostic

**Problem Solved**: Different providers expose advanced features differently

```python
# Works uniformly across OpenAI, Anthropic, Google, AWS, Ollama
response = model.invoke(messages)
content_blocks = response.content_blocks

# Access reasoning traces, citations, tool calls consistently
for block in content_blocks:
    if block.type == "reasoning":
        print(block.reasoning)
    elif block.type == "citation":
        print(block.source)
    elif block.type == "tool_call":
        print(block.name, block.arguments)
```

**Benefits**:
- Unified `content_blocks` property across all providers
- Full type safety and backward compatibility
- No provider-specific code needed
- Future-proof against provider API changes

### 4. Structured Output Integration

- Generated **within the main agent loop** (not separate LLM calls)
- Reduces costs and improves efficiency
- Native integration with agent workflows

### 5. Package Reorganization

**New Simplified `langchain` Namespace**:
```python
langchain.agents              # Core agent creation
langchain_core.messages       # Message types, content blocks
langchain_core.tools          # Tool decorators and base classes
langchain_core.prompts        # Prompt templates
langchain_openai              # OpenAI models (ChatOpenAI, etc.)
```

**Legacy `langchain-classic` Package**:
Moved features:
- Chains
- Retrievers
- Indexing APIs
- Hub modules

If you need legacy features:
```bash
pip install langchain-classic
```

Update imports:
```python
# Old
from langchain.chains import LLMChain
from langchain.retrievers import ...

# New
from langchain_classic.chains import LLMChain
from langchain_classic.retrievers import ...
```

---

## =' LangGraph 1.0 - Production-Ready Infrastructure

### Philosophy

> "LangGraph is a low-level orchestration framework and runtime for building, managing, and deploying long-running, stateful agents."

**Organizations Using LangGraph**: Klarna, Replit, Elastic, Uber, LinkedIn

### 1. Durable State Persistence

**Key Feature**: State persists across server restarts

**Capabilities**:
- Built-in workflow resumption without custom database logic
- Agents can suspend for days or months and resume from exact checkpoint
- Checkpoint serialization to MsgPack (optionally encrypted)
- No history replay needed - latest checkpoint aggregates all computation
- O(1) execution cost regardless of history length

**Use Cases**:
- Long-running workflows
- Fault-tolerant systems
- Multi-day agent processes
- Reliable production deployments

### 2. True Human-in-the-Loop Patterns

**Not just waiting processes - true interruptions**:
- Agents can interrupt themselves at designated points
- Complete execution state saved to durable storage
- Resume after user interaction (seconds or months later)
- Scales to thousands of simultaneously interrupted agents

**Implementation**:
- Use `breakpoints` during graph compilation
- Agents save state and pause execution
- External system provides input
- Agent resumes from exact state

### 3. Pregel Algorithm - Deterministic Concurrent Execution

**Architecture**: Bulk Synchronous Parallel (BSP) algorithm

**Execution Cycle**:
1. Compare channel versions against node subscriptions
2. Execute subscribed nodes in **parallel** with isolated state copies
3. Apply updates **deterministically** (prevents data race variability)
4. Increment channel versions and continue

**Core Components**:
- **Channels**: Named data containers with versioning
- **Nodes**: Functions subscribed to channel changes
- **Edges**: Connections between nodes (fixed or conditional)

**Performance Characteristics**:
- Execution cost constant regardless of invocation history length
- **O(1) complexity** for most operations during planning and running
- Critical for long-running, complex agents
- Safe concurrent execution without data races

### 4. Six Essential Production Features

1. **Parallelization**: Run independent steps concurrently without data races
2. **Streaming**: Display intermediate results for perceived responsiveness
3. **Checkpointing**: Save computation snapshots for fault tolerance
4. **Human-in-the-loop**: Interrupt execution without restarting
5. **Tracing**: Observability into agent behavior (via LangSmith)
6. **Task queuing**: Decouple execution from requests (via LangGraph Platform)

### 5. Core Components Deep Dive

#### State Management
- Shared data structure representing application's current snapshot
- Support for `TypedDict`, `dataclass`, or `Pydantic BaseModel`
- Reducer functions for custom update logic
- Separate input/output schemas possible
- Built-in `MessagesState` for common message patterns
- `add_messages` function for message ID tracking and deserialization

**Best Practices**:
- Use `TypedDict` for best performance
- Use `dataclass` for default values
- Use `Pydantic BaseModel` only when you need recursive validation
- Implement custom reducers with `Annotated` for complex state updates

#### Nodes
- Python functions encoding agent logic
- Receive state, perform computation, return updated state
- Converted to `RunnableLambda` objects (batch/async support)
- Special nodes: `START` (entry point) and `END` (terminal marker)
- Node caching with cache policy and optional TTL

**Best Practices**:
- Keep nodes focused on single responsibilities
- Implement proper error handling in nodes
- Return partial state updates, not full state

#### Edges

**Types**:
- **Normal Edges**: Direct node-to-node transitions
- **Conditional Edges**: Route based on function output
- **Entry Points**: Initial nodes
- **Conditional Entry Points**: Dynamic starting nodes

**Advanced Features**:
- **Send**: Map-reduce patterns with dynamic edge counts
- **Command**: Combine state updates and routing for multi-agent handoffs
- Multiple outgoing edges execute in parallel during next superstep

### 6. Graph Compilation and Advanced Features

**Compilation**:
```python
# Graphs must be compiled before use
graph = builder.compile(
    checkpointer=checkpointer,  # For persistence
    interrupt_before=["node_name"],  # For HITL
)
```

**Validation**: Performs structural validation
**Runtime Arguments**: Specify checkpointers, breakpoints

**Additional Features**:
- **Graph Migrations**: Support topology changes while preserving checkpoint state
- **Runtime Context**: Pass non-state information via `context_schema`
- **Recursion Limit**: Maximum supersteps (default: 25) to prevent infinite loops
- **Visualization**: Built-in graph rendering capabilities

---

## � Breaking Changes & Migration

### Python Version Requirements

**Dropped**: Python 3.9
**Required**: Python 3.10+

### Deprecated APIs

| Old | New | Status |
|-----|-----|--------|
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` | Deprecated |
| `langgraph.prebuilt` module | `langchain.agents` | Deprecated |
| `langchain.chains.*` | `langchain_classic.chains.*` | Moved |
| `langchain.retrievers.*` | `langchain_classic.retrievers.*` | Moved |

### Migration Steps

1. **Upgrade Python**
   ```bash
   # Ensure Python 3.10+ is installed
   python --version
   ```

2. **Upgrade Packages**
   ```bash
   pip install -U langchain
   # or
   uv add langchain
   ```

3. **Update Agent Creation Code**
   ```python
   # Old
   from langgraph.prebuilt import create_react_agent
   agent = create_react_agent(model, tools)

   # New
   from langchain.agents import create_agent
   agent = create_agent(model, tools)
   ```

4. **Handle Legacy Features**
   ```bash
   # If using chains, retrievers, indexing APIs
   pip install langchain-classic
   ```

   ```python
   # Update imports
   from langchain_classic.chains import LLMChain
   from langchain_classic.retrievers import VectorStoreRetriever
   ```

5. **Leverage New Features**
   - Implement middleware for context management
   - Use standard content blocks for provider-agnostic code
   - Integrate structured outputs within agent loop

### Stability Commitment

**Both frameworks at 1.0 = stability commitment**:
- Full backward compatibility maintained
- No breaking changes until 2.0
- Safe to use in production

---

## <� When to Use What

### Agent vs Simple Chain: Quick Decision Guide

**Use `create_agent` when you need:**
- Dynamic selection between multiple tools
- Multi-step reasoning with tool calling
- Conversation memory with checkpointing
- Example: Chat assistant that routes between RAG, weather API, database queries

**Use simple LCEL chains (`prompt | llm`) when you need:**
- Single-purpose transformations (summarization, classification, recommendations)
- No tool calling required
- Simple input → LLM → output pattern
- Example: Synthesizing data into recommendations, formatting responses

**Example Simple Chain:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# For simple tasks like recommendations
recommendation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful event recommender..."),
    ("user", "{weather_and_event_data}")
])

# Simple chain: prompt → LLM
recommendation_chain = recommendation_prompt | ChatOpenAI(model="gpt-4o")

# Invoke
result = recommendation_chain.invoke({"weather_and_event_data": data})
response = result.content
```

---

### Use LangChain 1.0 For:
-  Rapid development with standard patterns
-  Simple to moderate complexity agents
-  Quick prototyping
-  Standard tool-calling workflows
-  When you want high-level abstractions
-  Getting started quickly

### Use LangGraph 1.0 For:
-  Complex, long-running workflows
-  Fine-grained control requirements
-  Production deployments requiring durability
-  Multi-agent systems
-  Custom agent architectures
-  Human-in-the-loop at specific points
-  Agents that need to pause and resume
-  Parallel execution with state management

### Best Practice: Composable Approach

**Start with LangChain � Drop to LangGraph when needed**

- LangChain agents are built on LangGraph runtime
- Seamlessly integrate custom LangGraph components
- Start high-level, drop down for control
- No need to choose one or the other
- **Most "multi-agent" patterns can be simplified:** Controller agent + tools/chains is often better than nested agents

---

## =� Best Practices & Patterns

### 1. State Design (LangGraph)

**Performance Hierarchy**:
```python
# Best performance - Use TypedDict
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_info: dict

# When you need defaults - Use dataclass
from dataclasses import dataclass

@dataclass
class AgentState:
    messages: list
    counter: int = 0

# Only when you need validation - Use Pydantic
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    messages: list
    count: int = Field(ge=0)
```

**Custom Reducers**:
```python
from typing import Annotated

def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}

class State(TypedDict):
    data: Annotated[dict, merge_dicts]
```

### 2. Middleware Strategy (LangChain)

**Compliance & Security**:
```python
from langchain.agents import create_agent
from langchain.middleware import PIIMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[PIIMiddleware()]
)
```

**Context Management**:
```python
from langchain.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        SummarizationMiddleware(max_tokens=4000)
    ]
)
```

**High-Stakes Decisions**:
```python
from langchain.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        HumanInTheLoopMiddleware(
            require_approval_for=["delete", "purchase"]
        )
    ]
)
```

**Custom Middleware**:
```python
def custom_middleware(state, next_fn):
    # Before execution
    print(f"Calling with: {state}")

    # Execute
    result = next_fn(state)

    # After execution
    print(f"Result: {result}")
    return result

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[custom_middleware]
)
```

### 3. Testing Strategy

**Essential Testing Approach**:
1. Test agents on **multiple examples** with known desired behavior
2. **Measure** agent performance with metrics
3. **Adjust** architecture and prompts based on metrics
4. Use **LangSmith** for tracing and observability

**Example Test Structure**:
```python
test_cases = [
    {"input": "...", "expected_output": "..."},
    {"input": "...", "expected_output": "..."},
]

for case in test_cases:
    result = agent.invoke(case["input"])
    assert evaluate(result, case["expected_output"])
```

### 4. Production Deployment (LangGraph)

**Always Use Checkpointers**:
```python
from langgraph.checkpoint.memory import MemorySaver
# Or use database-backed checkpointers for production
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(connection_string)

graph = builder.compile(
    checkpointer=checkpointer
)
```

**Implement Streaming**:
```python
# Stream for better UX during long operations
for event in graph.stream(inputs, config):
    print(event)
```

**Set Recursion Limits**:
```python
graph = builder.compile(
    checkpointer=checkpointer,
    recursion_limit=50  # Prevent infinite loops
)
```

**Use Breakpoints for Debugging**:
```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tool_node", "decision_node"]
)
```

### 5. Graph Design Principles (LangGraph)

**Node Design**:
-  Keep nodes focused on single responsibilities
-  Implement proper error handling in nodes
-  Return partial state updates, not full state
- L Don't put too much logic in a single node

**Edge Strategy**:
-  Use conditional edges for dynamic routing
-  Leverage parallelization for independent operations
-  Use `Send` for map-reduce patterns
-  Use `Command` for multi-agent handoffs

**State Management**:
-  Implement clear state schema with explicit types
-  Use custom reducers for complex update logic
-  Keep state minimal - only what's necessary
- L Don't store large data in state (use external storage)

### 6. Standard Content Blocks Pattern (LangChain)

**Provider-Agnostic Access**:
```python
# Initialize any model (OpenAI, Anthropic, Google, etc.)
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4", model_provider="openai")
# or
model = init_chat_model("claude-3-5-sonnet", model_provider="anthropic")

# Access features uniformly
response = model.invoke(messages)
content_blocks = response.content_blocks

for block in content_blocks:
    if block.type == "reasoning":
        # Available on models with reasoning traces
        print(f"Reasoning: {block.reasoning}")
    elif block.type == "citation":
        # Available on models with citation support
        print(f"Source: {block.source}")
    elif block.type == "tool_call":
        # Standard across all models
        print(f"Tool: {block.name}")
        print(f"Args: {block.arguments}")
```

**Benefits**:
- Write code once, works across all providers
- Easy to switch models without code changes
- Future-proof against provider API changes
- Full type safety

---

## =� Code Examples

### Basic LangChain Agent

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# Define tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72�F"

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

# Initialize model
model = init_chat_model("gpt-4", model_provider="openai")

# Create agent
agent = create_agent(
    model=model,
    tools=[get_weather, search],
    system_prompt="You are a helpful assistant."
)

# Use agent
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in NYC?"}]})
print(result)
```

### LangGraph Agent with HITL

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    approval_needed: bool

# Define nodes
def agent_node(state: State):
    # Agent logic here
    return {"messages": [{"role": "assistant", "content": "I recommend deleting file X"}],
            "approval_needed": True}

def approval_node(state: State):
    # Wait for human approval
    return {"approval_needed": False}

def action_node(state: State):
    # Execute action
    return {"messages": [{"role": "assistant", "content": "Action completed"}]}

# Build graph
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("approval", approval_node)
builder.add_node("action", action_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    lambda s: "approval" if s["approval_needed"] else "action"
)
builder.add_edge("approval", "action")
builder.add_edge("action", END)

# Compile with checkpointer and interruption
checkpointer = MemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval"]  # Pause here for human input
)

# Use graph
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"messages": [{"role": "user", "content": "Delete old files"}]}, config)

# Graph pauses at approval node
# Human provides approval, then resume:
result = graph.invoke(None, config)  # Resumes from checkpoint
```

### LangGraph with Parallel Execution

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    web_results: list
    db_results: list
    combined: list

def search_web(state: State):
    # Simulated web search
    return {"web_results": ["result1", "result2"]}

def search_db(state: State):
    # Simulated database search
    return {"db_results": ["data1", "data2"]}

def combine_results(state: State):
    combined = state["web_results"] + state["db_results"]
    return {"combined": combined}

builder = StateGraph(State)
builder.add_node("search_web", search_web)
builder.add_node("search_db", search_db)
builder.add_node("combine", combine_results)

# Parallel execution
builder.add_edge(START, "search_web")
builder.add_edge(START, "search_db")

# Both must complete before combine
builder.add_edge("search_web", "combine")
builder.add_edge("search_db", "combine")
builder.add_edge("combine", END)

graph = builder.compile()

# Execute - search_web and search_db run in parallel
result = graph.invoke({"query": "test"})
print(result["combined"])  # ['result1', 'result2', 'data1', 'data2']
```

### Using Standard Content Blocks

```python
from langchain.chat_models import init_chat_model

# Works with any provider
model = init_chat_model("claude-3-5-sonnet", model_provider="anthropic")

messages = [
    {"role": "user", "content": "Explain quantum computing and cite sources"}
]

response = model.invoke(messages)

# Access content blocks uniformly
for block in response.content_blocks:
    if block.type == "text":
        print(f"Text: {block.text}")
    elif block.type == "reasoning":
        print(f"Reasoning: {block.reasoning}")
    elif block.type == "citation":
        print(f"Citation: {block.source}")
    elif block.type == "tool_call":
        print(f"Tool: {block.name}, Args: {block.arguments}")
```

---

## = Key Takeaways

1. **LangChain 1.0** focuses on:
   - Simplified agent creation with `create_agent`
   - Powerful middleware for customization
   - Provider-agnostic content blocks
   - Production-ready simplicity

2. **LangGraph 1.0** delivers:
   - Durable state persistence
   - True human-in-the-loop patterns
   - Deterministic concurrent execution via Pregel
   - Enterprise-grade reliability

3. **Both are composable**:
   - Start with LangChain's high-level APIs
   - Drop to LangGraph for fine-grained control
   - No need to choose one exclusively

4. **Migration is straightforward**:
   - Clear guides available
   - Python 3.10+ required
   - Legacy features in `langchain-classic`

5. **Design philosophy**:
   - Minimal abstraction, maximum control
   - Production-readiness over ease-of-use
   - Feels like writing regular Python code

---

## =� Next Steps

### For New Projects
1. Install latest versions: `pip install -U langchain langgraph`
2. Start with `create_agent` for quick wins
3. Add middleware for cross-cutting concerns
4. Use LangGraph when you need durability or HITL

### For Existing Projects
1. Review migration guide: https://docs.langchain.com/oss/python/migrate/langchain-v1
2. Update Python to 3.10+
3. Update agent creation to use `create_agent`
4. Install `langchain-classic` if using deprecated features
5. Test thoroughly before deploying

### Learning Resources
- Explore examples in documentation
- Check out LangSmith for observability
- Join LangChain Discord community
- Review production patterns in docs

---

**Document Version**: 1.0
**Created**: 2025-11-02
**Framework Versions**: LangChain 1.0, LangGraph 1.0
