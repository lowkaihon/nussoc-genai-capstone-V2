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
| `retrieve_documents` | RAG | Query uploaded PDF documents |
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