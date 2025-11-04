# Event Planning Assistant for Singapore

An intelligent conversational AI system that helps users discover events, understand venue policies, and make informed planning decisions through real-time data integration, advanced RAG retrieval, and multimodal analysis.

## Overview

This project demonstrates production-grade AI application development using **LangChain/LangGraph 1.0** with:
- **Advanced RAG**: 3-stage retrieval pipeline (BM25 + Semantic + Jina Reranking) achieving 90% Precision@3
- **Multimodal Capabilities**: GPT-4o vision for venue photo analysis (accessibility, capacity, condition assessment)
- **Real-time Integration**: Live weather data and event database queries
- **Human-in-the-Loop**: Approval-based image generation preventing unexpected API costs
- **Structured Output**: GPT-4o best practices with silent reasoning and consistent Markdown formatting

**Target Users:** Event planners, venue coordinators, attendees seeking accessibility information, anyone organizing or attending events in Singapore.

**Data Sources:**
- Venue policy documents compiled from **79 online sources** via Perplexity.ai (Singapore government sources, venue websites, industry standards)
- Synthetic event data for demonstration purposes (21 sample events across Singapore and Southeast Asia)

---

## ✨ Features

### Core Capabilities

🎯 **Weather-Aware Event Recommendations**
- Synthesizes current weather, event data, and user preferences
- Multi-step tool chaining (date → weather → events → recommendation)
- Context-aware suggestions for indoor/outdoor events

📚 **Venue Policy Retrieval with Advanced RAG**
- Query Singapore venue regulations with source attribution
- Covers: Photography restrictions, sound limits, accessibility, insurance, technical specs
- 3-stage retrieval: BM25 (keyword) + Semantic (vector) + Jina Reranker (cross-encoder)
- 90% accuracy on policy queries with [Venue, p.X] citations

📸 **Multimodal Venue Photo Analysis**
- GPT-4o vision assessment of venue images
- **Accessibility**: ADA compliance, wheelchair access, ramps, facilities (1-5 rating)
- **Capacity**: Room dimensions, seating estimates for different configurations
- **Condition**: Cleanliness, maintenance, safety, readiness evaluation
- **General**: Comprehensive analysis covering all aspects

🎨 **AI Image Generation (with Approval)**
- Stable Diffusion 3.5 for conceptual visuals and mood boards
- Two-step approval process prevents unexpected costs
- Useful for visualizing event themes and decorative concepts

### Key Highlights

- ✅ **Production-Ready Architecture**: LangChain/LangGraph 1.0 with durable checkpointing
- ✅ **GPT-4o Best Practices**: Silent reasoning, declarative prompts, structured output, prompt injection defense
- ✅ **Conversation Persistence**: Thread-based memory with checkpointer
- ✅ **Error Handling**: Comprehensive validation and graceful degradation
- ✅ **Evaluation-Driven**: A/B tested retrieval pipeline with documented metrics

---

## 🏗️ Architecture Overview

The system uses a **controller agent** pattern built with LangChain 1.0's `create_agent`, which orchestrates 8 specialized tools through intelligent tool selection and chaining.

![System Architecture Diagram](./System%20Architecture%20Diagram.png)

### Core Components

- **Controller Agent**: LangChain 1.0 `create_agent` with GPT-4o
- **Hybrid RAG System**: 3-stage retrieval for venue policies (BM25 → Semantic → Jina Rerank)
- **Direct Tools**: Weather API, SQLite event database (synthetic demo data), date utilities
- **LLM Chains**: Simple chains for recommendations (no nested agent overhead)
- **Multimodal Tools**: GPT-4o vision (analysis), Stable Diffusion 3.5 (generation)
- **State Management**: LangGraph `MemorySaver` checkpointer for conversation persistence

### Design Decisions

**Why Flattened Architecture?**
- Eliminated nested subagents for weather/events (were causing overhead)
- Direct API/database calls are faster and more reliable
- Simple LLM chain for recommendations (no unnecessary ReAct loops)
- Controller agent + tools/chains pattern (LangChain 1.0 best practice)

---

## 🚀 Getting Started

### Prerequisites

**Python Environment:**
- Python 3.10 or higher
- Jupyter Notebook or JupyterLab

**Required API Keys:**
- [OpenAI API Key](https://platform.openai.com/api-keys) - For GPT-4o and embeddings
- [Replicate API Token](https://replicate.com/account/api-tokens) - For Stable Diffusion image generation
- [WeatherAPI Key](https://www.weatherapi.com/signup.aspx) - For real-time weather data (free tier available)
- [Jina AI API Key](https://jina.ai/reranker/) - For reranking (1M tokens/month free)

**Venue Policy PDFs:**
These files contain venue policies compiled from **79 online sources** (Singapore government, venue websites, industry standards) using Perplexity.ai:
- `MBS-Event-Policy.pdf` - Marina Bay Sands event policies
- `GBTB-Venue-Guide.pdf` - Gardens by the Bay venue guidelines
- `Esplanade-Manual.pdf` - Esplanade performing arts manual
- `SG-Event-Regulations.pdf` - Singapore event regulations

*Note: Ensure these files are in the project directory.*

### Installation

1. **Clone or download the repository**

2. **Install dependencies:**

   Using pip:
   ```bash
   pip install langchain langchain-openai langchain-chroma langchain-community langchain-core
   pip install langgraph replicate requests pillow
   pip install pymupdf  # For PDF loading
   pip install rank_bm25  # For BM25 retriever
   ```

   Or using uv (faster):
   ```bash
   uv pip install langchain langchain-openai langchain-chroma langchain-community langchain-core
   uv pip install langgraph replicate requests pillow pymupdf rank_bm25
   ```

3. **Set up API keys as environment variables:**

   **Windows (Command Prompt):**
   ```cmd
   set OPENAI_API_KEY=your_openai_key_here
   set REPLICATE_API_TOKEN=your_replicate_token_here
   set WEATHER_API_KEY=your_weatherapi_key_here
   set JINA_API_KEY=your_jina_key_here
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:OPENAI_API_KEY="your_openai_key_here"
   $env:REPLICATE_API_TOKEN="your_replicate_token_here"
   $env:WEATHER_API_KEY="your_weatherapi_key_here"
   $env:JINA_API_KEY="your_jina_key_here"
   ```

   **macOS/Linux:**
   ```bash
   export OPENAI_API_KEY=your_openai_key_here
   export REPLICATE_API_TOKEN=your_replicate_token_here
   export WEATHER_API_KEY=your_weatherapi_key_here
   export JINA_API_KEY=your_jina_key_here
   ```

   **Or create a `.env` file** in the project directory:
   ```
   OPENAI_API_KEY=your_openai_key_here
   REPLICATE_API_TOKEN=your_replicate_token_here
   WEATHER_API_KEY=your_weatherapi_key_here
   JINA_API_KEY=your_jina_key_here
   ```

### Running the Notebook

1. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

2. **Open `nus_genai_capstone.ipynb`**

3. **Run all cells in order:**
   - **Setup** cells: Import libraries, load API keys
   - **RAG Setup** cells: Load PDFs, create vector store, initialize retrievers
   - **Database Setup** cell: Create and populate SQLite event database
   - **Implementation** cells: Define tools and agent
   - **Testing** cell: Start the interactive chat loop

4. **Interact with the assistant:**
   ```
   You: What events would you recommend for today?
   AI: [Response with event recommendations based on weather]

   You: Can I bring a tripod to Marina Bay Sands?
   AI: [Policy information with source citation]

   You: Analyze esplanade-concert-hall.jpg for accessibility
   AI: [Detailed accessibility assessment]
   ```

5. **Exit the chat:** Type `quit`, `exit`, or `q`

---

## 💡 Usage

### Starting the Chat Interface

Execute the final cell in the notebook:
```python
chat_loop()
```

The assistant will greet you with:
```
Chat started! Type 'quit' to exit.
```

### Example Queries

**Event Recommendations:**
*(Note: Uses synthetic event data for demonstration)*
```
What events would you recommend for today?
Suggest indoor events happening today
What events are in Thailand tomorrow?
```

**Venue Policy Questions:**
```
Can I bring a tripod to Marina Bay Sands?
What are the sound restrictions at Gardens by the Bay?
Tell me about wheelchair access at Esplanade
What insurance requirements do I need for Singapore events?
```

**Venue Photo Analysis:**
```
Analyze esplanade-concert-hall.jpg for wheelchair accessibility
How many people can fit in esplanade-concert-hall.jpg?
Check the condition of esplanade-concert-hall.jpg
Provide a comprehensive analysis of esplanade-concert-hall.jpg
```

**AI Image Generation:**
```
Generate an image of a futuristic tech conference venue
Create a visual of a tropical outdoor music festival at sunset
```
*(Note: The assistant will request approval before generating images due to API costs)*

**Complex Multi-Step Queries:**
```
I'm planning an outdoor concert at Gardens by the Bay today. What should I know about weather and sound restrictions?
```

### Conversation Management

**Reset conversation history:**
```python
reset_conversation()
```

**Check conversation state:**
The assistant automatically maintains conversation context across messages within the same session.

---

## 🔧 Tools & Capabilities

The agent has access to 8 specialized tools:

| Tool Name | Purpose | Key Features |
|-----------|---------|--------------|
| `retrieve_venue_policies` | Venue policy RAG retrieval | 3-stage pipeline, [Source, p.X] citations |
| `analyze_venue_photo` | Multimodal venue assessment | Accessibility, capacity, condition analysis |
| `request_image_generation` | AI image generation (approval) | Stable Diffusion 3.5, cost protection |
| `approve_image_generation` | Execute approved generation | Two-step HITL pattern |
| `get_current_date` | Current date utility | Auto-called for date-relative queries |
| `get_weather` | Real-time weather API | WeatherAPI integration |
| `get_events` | Event database query | SQLite with date/type/country filters (synthetic demo data) |
| `recommend_events` | LLM recommendation chain | Weather-aware synthesis |

### Tool Workflow Example

**User Query:** "Recommend events for today"

**Silent Tool Chain:**
1. `get_current_date()` → "2025-11-04"
2. `get_weather("Singapore")` → "28°C, Sunny"
3. `get_events("2025-11-04", country="Singapore")` → [event list]
4. `recommend_events(weather + events)` → Final synthesis

**User Sees:** Only the final recommendation (no intermediate steps shown)

---

## 🎯 Technical Highlights

### 3-Stage Retrieval Pipeline (Hybrid RAG + Reranking)

**Architecture:**
```
User Query
    ↓
┌────────────────────────┐
│  Stage 1: BM25         │  ← Keyword search (proper nouns, codes)
│  Retrieves top 5       │
└────────────────────────┘
    ↓
┌────────────────────────┐
│  Stage 2: Semantic     │  ← Vector search (concepts, meanings)
│  Retrieves top 5       │
└────────────────────────┘
    ↓
┌────────────────────────┐
│  RRF Fusion (40/60)    │  ← Combine results with weights
│  Merged candidate pool │
└────────────────────────┘
    ↓
┌────────────────────────┐
│  Stage 3: Jina Rerank  │  ← Cross-encoder scoring
│  Returns top 3         │
└────────────────────────┘
    ↓
Final Results with Citations
```

**Implementation:**
```python
# Stage 1 + 2: Hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.4, 0.6]  # 40% keyword, 60% semantic
)

# Stage 3: Jina reranker
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

### Evaluation Results

A/B testing with 20 test queries across exact-match, semantic, and hybrid query types:

| Metric | Hybrid RAG Only | Hybrid + Jina Rerank | Improvement |
|--------|-----------------|---------------------|-------------|
| **Precision@3** | 80.0% (16/20) | 90.0% (18/20) | **+12.5%** |
| **MRR** | 0.775 | 0.817 | **+5.4%** |
| **Avg Latency** | 805ms | 2022ms | +1216ms (2.5x) |
| **Cost per query** | $0 | ~$0.002 | Free tier: 1M tokens/month |

**Performance by Query Type:**

| Query Type | Hybrid Only | Hybrid + Rerank | Improvement |
|------------|-------------|-----------------|-------------|
| **Exact Match** (5 queries) | 40% (2/5) | 80% (4/5) | **+100%** |
| **Semantic** (5 queries) | 80% (4/5) | 80% (4/5) | +0% |
| **Hybrid** (10 queries) | 100% (10/10) | 100% (10/10) | +0% |

**Key Insights:**
- ✅ Reranking doubles accuracy on exact-match queries (venue names, postal codes)
- ✅ 2.5x latency increase acceptable for accuracy-critical use case
- ✅ Minimal cost with generous free tier (1M tokens/month)

**Why Jina Reranker?**
- **Compatibility**: Works with LangChain 1.0 (Cohere had version conflicts)
- **Cost**: Best free tier among reranking options
- **Performance**: State-of-the-art cross-encoder, no local GPU needed
- **Simplicity**: Available in `langchain_community` package

---

### LangChain/LangGraph 1.0 Architecture

**Migration Date:** 2025-11-02

**Major Changes:**

1. **Agent Framework**
   - **Before:** `create_tool_calling_agent` + `AgentExecutor` (deprecated)
   - **After:** `create_agent` (LangChain 1.0 native)

2. **Memory/State Management**
   - **Before:** `ConversationBufferMemory` (legacy)
   - **After:** LangGraph's `MemorySaver` checkpointer (durable, thread-based)

3. **Subagent Simplification**
   - **Before:** Nested weather/event subagents (overhead)
   - **After:** Direct API/database tools (faster, cleaner)
   - **Before:** Recommendation subagent with full ReAct loop
   - **After:** Simple LLM chain (prompt | llm)

4. **Human-in-the-Loop**
   - **New:** Two-step approval for image generation (`request` → `approve`)
   - **Benefit:** Prevents unexpected Replicate API costs

**Benefits:**
- 🚀 **Performance**: Eliminated nested agent overhead
- 🔒 **Reliability**: Better error handling and recovery
- 💾 **Persistence**: Conversation history persists across messages
- 🎯 **Simplicity**: Cleaner tool definitions with type hints

For detailed LangChain/LangGraph 1.0 documentation, see: `new_langchain_langgraph.md`

---

### GPT-4o System Prompt Engineering

The system prompt follows GPT-4o best practices from 2025:

**Structure (Hierarchical):**
1. Identity & Mission
2. Core Capabilities
3. Behavioral Invariants (ALWAYS/NEVER)
4. Tool Usage Rules (compressed decision table)
5. Output Formatting (Markdown requirements)
6. Communication Style (professional consultant)
7. **Reasoning Visibility: MINIMAL** (silent execution)
8. Safety Boundaries
9. Knowledge Context
10. Prompt Injection Defense

**Key Features:**
- ✅ **Declarative phrasing**: "Always cite sources" not "try to cite sources"
- ✅ **Silent reasoning**: Suppresses "Let me check...", "I will now..." narration
- ✅ **Structured output**: Mandates Markdown with ## headers and bullets
- ✅ **Prompt injection defense**: Refuses instruction override attempts
- ✅ **Safety boundaries**: NEVER share PII from images, ALWAYS validate inputs

For detailed best practices, see: `gpt-4o-best-practices.md`

---

## 📁 Project Structure

```
nussoc-genai-capstone-V2/
├── nus_genai_capstone.ipynb          # Main notebook
├── README.md                          # This file
├── new_langchain_langgraph.md         # LangChain/LangGraph 1.0 documentation
├── gpt-4o-best-practices.md           # GPT-4o prompt engineering guide
├── System Architecture Diagram.png    # Architecture visualization
├── esplanade-concert-hall.jpg         # Example image for venue analysis
├── MBS-Event-Policy.pdf               # Marina Bay Sands policies (79 sources via Perplexity.ai)
├── GBTB-Venue-Guide.pdf               # Gardens by the Bay guide (79 sources via Perplexity.ai)
├── Esplanade-Manual.pdf               # Esplanade manual (79 sources via Perplexity.ai)
├── SG-Event-Regulations.pdf           # SG regulations (79 sources via Perplexity.ai)
├── events.db                          # SQLite event database (synthetic data, auto-generated)
└── venue_policies_chroma_db/          # Chroma vector store (auto-generated)
```

---

## 🔮 Future Enhancements

**Potential Improvements:**

1. **Middleware System** (LangChain 1.0 feature)
   - Add `SummarizationMiddleware` for context window management
   - Add logging/monitoring middleware
   - Add PII redaction middleware

2. **Production Deployment**
   - Upgrade to persistent checkpointer (`SqliteSaver` or `PostgresSaver`)
   - Implement streaming for better UX during RAG queries
   - Add LangSmith tracing for observability

3. **Enhanced RAG**
   - Query expansion and rewriting
   - Semantic caching for frequent queries
   - Document-type-specific indexing

4. **Additional Modalities**
   - Event flyer OCR extraction (extract_flyer_data tool)
   - Multi-image comparison for venue selection
   - Video frame extraction for venue tours

5. **Extended Coverage**
   - More Southeast Asian cities
   - Real-time event scraping APIs
   - Integration with ticketing platforms

---

## 📝 License

This project is for educational and portfolio demonstration purposes.

---

## 🙏 Acknowledgments

- **LangChain/LangGraph 1.0** - Agent framework and state management
- **OpenAI GPT-4o** - Multimodal AI capabilities
- **Jina AI** - State-of-the-art reranking models
- **Chroma** - Vector database for semantic search
- **Replicate** - Stable Diffusion image generation API
- **WeatherAPI** - Real-time weather data

---

**Built with LangChain 1.0 | GPT-4o | Jina Reranker v2**
