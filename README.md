# JTCG CRM AI Agent Chatbot

A sophisticated AI-powered customer service agent for JTCG Shop, built with OpenAI Agents SDK and ChromaDB vector database. This agent provides intelligent customer support across FAQ, product discovery, order tracking, and human handover services.

## 🚀 Features

### Core Capabilities
- **🔍 FAQ Search**: Semantic search through knowledge base with source attribution
- **🛒 Product Discovery**: AI-powered product recommendations with compatibility checking
- **📦 Order Tracking**: Real-time order status lookup and user order management
- **👤 Human Handover**: Seamless transfer to human customer service with conversation context

### Technical Highlights
- **Vector Search**: ChromaDB with `all-MiniLM-L6-v2` embeddings for semantic understanding
- **Multilingual Support**: Native Chinese/English cross-language search capabilities
- **Intent Detection**: Smart routing based on keyword analysis and context
- **Function Calling**: OpenAI function tools for specialized agent capabilities
- **Performance Monitoring**: Comprehensive evaluation framework with LLM judging

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   JTCG Agent    │───▶│  Agent Functions │───▶│   Vector DB     │
│   (Main)        │    │                  │    │   (ChromaDB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Data Processor  │    │ Intent Detection │    │ Embedding Model │
│                 │    │                  │    │ all-MiniLM-L6   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **jtcg_agent.py** | Main orchestrator using OpenAI Agents SDK | Function calling, conversation management |
| **agent_functions.py** | Specialized tools for each service area | Search, order lookup, handover logic |
| **vector_db.py** | ChromaDB vector database interface | Semantic search, embedding management |
| **data_processor.py** | CSV/JSON data loading and management | Knowledge base, products, orders |

## 📊 Data Sources

- **Knowledge Base**: `ref_data/ai-eng-test-sample-knowledges.csv` (42 FAQ items)
- **Product Catalog**: `ref_data/ai-eng-test-sample-products.csv` (6 products)
- **Test Conversations**: `ref_data/ai-eng-test-sample-conversations.json` (323 test cases)
- **Order Database**: `ref_data/orders.json` with multiple user scenarios

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd jtcg-assignment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_api_key_here
   ```

## 🚀 Usage

### Running the Agent
```bash
# Interactive chat mode
python jtcg_agent.py

# Or using the virtual environment directly
./venv/Scripts/python.exe jtcg_agent.py
```

### Testing Framework

#### Run All Tests
```bash
# Data processor tests
./venv/Scripts/python.exe test_data_processor.py

# Vector database tests
./venv/Scripts/python.exe test_vector_db.py

# Conversation tests
./venv/Scripts/python.exe test_conversations.py

# Full evaluation with LLM judge (323 conversations)
./venv/Scripts/python.exe run_full_evaluation.py
```

#### Test Results
The evaluation framework provides:
- **Technical Performance**: Success rates, response times, source link coverage
- **LLM Judge Evaluation**: Content accuracy and service scope validation
- **CSV Reports**: Detailed conversation-by-conversation analysis

Recent test results:
- ✅ **100% Success Rate** across 323 conversations
- ✅ **78% Content Accuracy** from LLM judge evaluation
- ✅ **54.2% Source Link Coverage** with proper attribution

## 🎯 Brand Voice & Guidelines

### JTCG Shop Brand Values
- **Mission**: "Better Desk, Better Focus"
- **Core Features**: Clear compatibility, reliable installation, excellent customer service
- **Voice**: Professional, trustworthy, helpful

### Response Principles
- ✅ Natural conversation (avoid AI assistant language)
- ✅ Direct answers first, then supplementary information
- ✅ Follow user's language preference (Traditional/Simplified Chinese)
- ✅ Always provide source links when available
- ❌ Never use: "簡短回答", "補充說明", "讓我為您..."

## 📈 Performance Metrics

### Evaluation Results (Latest)
```
Total Conversations: 323
Technical Success Rate: 100.0%
Content Accuracy Rate: 78.0%
Service Scope Accuracy: 100.0%
Average Response Time: 14.56s
Source Link Coverage: 54.2%
```

### Vector Database Performance
- **Knowledge Search**: Cross-language semantic search
- **Product Matching**: VESA compatibility and specification filtering
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Search Quality**: Distance-based relevance scoring

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4  # Optional, defaults to gpt-4
VECTOR_DB_PATH=chroma_db  # Optional, ChromaDB storage path
```

### Customization Options
- **Embedding Model**: Change in `vector_db.py` collection creation
- **Agent Instructions**: Modify system prompt in `_get_system_instructions()`
- **Function Tools**: Add new capabilities in `agent_functions.py`

## 🧪 Development

### Project Structure
```
jtcg-assignment/
├── jtcg_agent.py              # Main agent orchestrator
├── agent_functions.py         # Specialized function tools
├── vector_db.py              # ChromaDB vector database
├── data_processor.py         # Data loading and management
├── test_conversations.py     # Conversation testing framework
├── test_data_processor.py    # Data processor tests
├── test_vector_db.py         # Vector database tests
├── run_full_evaluation.py    # Comprehensive evaluation with LLM judge
├── handover_simple_mock.py   # Human handover mock service
├── ref_data/                 # Data sources
│   ├── ai-eng-test-sample-knowledges.csv
│   ├── ai-eng-test-sample-products.csv
│   ├── ai-eng-test-sample-conversations.json
│   └── orders.json
├── requirements.txt          # Python dependencies
├── CLAUDE.md                # Development guidelines
└── README.md                # This file
```

### Adding New Features
1. **New Function Tools**: Add to `agent_functions.py` and register in `jtcg_agent.py`
2. **Data Sources**: Update `data_processor.py` loading methods
3. **Search Capabilities**: Extend `vector_db.py` collection methods
4. **Evaluation**: Add test cases to conversation JSON files

## 📋 Dependencies

### Core Framework
- `openai-agents` - OpenAI Agents SDK for function calling
- `chromadb==1.1.0` - Vector database for semantic search
- `openai` - OpenAI API client

### Data Processing
- `pandas` - CSV/JSON data manipulation
- `pydantic` - Data validation and models

### Development
- `python-dotenv` - Environment variable management

## 🤝 Contributing

1. Follow the brand voice guidelines in `CLAUDE.md`
2. Use `./venv/Scripts/python.exe` for all development
3. Run tests before submitting changes
4. Update evaluation metrics for new features

## 📄 License

This project is developed for JTCG Shop customer service automation.

---

**Built with ❤️ for JTCG Shop - Better Desk, Better Focus**