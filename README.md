# GenAI-Powered Ticket Analysis for Telecom

A Generative AI-based ticket analysis assistant for telecom service providers that analyzes incoming ticket descriptions and suggests solutions based on similar resolved tickets, ranking the top 3 solutions by suitability percentage.

## 🎯 Objective

This system helps telecom support teams by:
- Analyzing customer ticket descriptions
- Finding similar resolved tickets from historical data
- Generating and ranking top 3 solution options with suitability percentages
- Providing reasoning for each recommended solution

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │
│  (User Input)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │
│  (FAISS Index)  │
│  + Embeddings   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Similar Ticket  │
│   Retrieval     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generator  │
│(Groq/OpenAI/    │
│    Gemini)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ranked Solutions│
│  (Top 3 + %)    │
└─────────────────┘
```

### Components

1. **Data Processor** (`data_processor.py`)
   - Loads and preprocesses telecom ticket data
   - Combines ticket fields (issue, root cause, resolution) for embedding
   - Prepares metadata for retrieval context

2. **Vector Store** (`vector_store.py`)
   - Uses Sentence Transformers for embeddings (`all-MiniLM-L6-v2`)
   - Stores embeddings in FAISS for efficient similarity search
   - Supports index persistence and loading

3. **Solution Generator** (`solution_generator.py`)
   - Supports OpenAI GPT, Groq LLM, and Google Gemini models
   - Generates ranked solutions with suitability percentages
   - Provides reasoning for each solution
   - Groq offers faster inference, OpenAI offers higher quality, Gemini offers Google's AI

4. **Streamlit App** (`app.py`)
   - User-friendly web interface
   - Ticket input and solution display
   - Similar tickets reference view

## 📋 Prerequisites

- Python 3.8+
- API key for LLM provider (choose one):
  - **Groq API key** (recommended for speed) - Get one at https://console.groq.com
  - **OpenAI API key** (for higher quality) - Get one at https://platform.openai.com/api-keys
  - **Gemini API key** (Google AI) - Get one at https://aistudio.google.com/apikey
- 2GB+ RAM (for embedding model and FAISS index)

##  Setup Instructions

### 1. Clone/Download the Repository

```bash
cd HCL
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The first run will download the sentence transformer model (~80MB), which may take a few minutes.

### 3. Set API Key (Required - Environment Variable)

⚠️ **Important:** API keys are read from environment variables only (not shown in UI for security).

Choose one provider and set the environment variable:

```bash
# For Groq (recommended - fast & free)
export GROQ_API_KEY="your-groq-api-key-here"

# OR for OpenAI
export OPENAI_API_KEY="your-openai-api-key-here"

# OR for Gemini (Google AI)
export GEMINI_API_KEY="your-gemini-api-key-here"
```

**Get API Keys:**
| Provider | Free Tier | Get API Key |
|----------|-----------|-------------|
| **Groq** | ✅ Yes | https://console.groq.com |
| **Gemini** | ✅ Yes | https://aistudio.google.com/apikey |
| **OpenAI** | ❌ Paid | https://platform.openai.com/api-keys |

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### 5. Initialize the System

Click **"🔄 Initialize System"** in the sidebar. The first time, it will:
1. Load the ticket data from `telecom_tickets_10000_12cats.json`
2. Generate embeddings for all tickets
3. Build the FAISS index
4. Save the index for future use

This process takes approximately 5-10 minutes for 10,000 tickets.

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage

1. **Set API Key**: Set your API key as environment variable before running (see Setup Instructions)
2. **Run App**: Run `streamlit run app.py`
3. **Initialize System**: Click "🔄 Initialize System" in the sidebar (first time only)
4. **Select Provider**: Choose Groq, OpenAI, or Gemini from the dropdown
5. **Enter Ticket**: Type or paste the customer issue description
6. **Analyze**: Click "🔍 Analyze Ticket"
7. **Review Results**: 
   - View similar resolved tickets (reference)
   - Review top 3 ranked solutions with suitability percentages
   - Read reasoning for each solution

### Example Ticket Description

```
Customer reports intermittent internet connectivity on mobile data in the late afternoon. Connection drops frequently and websites fail to load.
```

## 📊 Dataset

The system uses a dataset of 10,000 telecom support tickets across 12 categories:
- Network issues
- Billing
- Service activation
- Router problems
- Fiber connectivity
- Roaming
- SIM/KYC
- And more...

Each ticket contains:
- `ticket_id`: Unique identifier
- `category`: Issue category
- `customer_issue_description`: Customer's reported issue
- `root_cause`: Identified root cause
- `final_resolution`: How the issue was resolved

## ☁️ Cloud Deployment

### Deploying on Replit

1. **Create a new Replit project**
   - Choose Python template
   - Upload all project files

2. **Install dependencies**
   - Replit will auto-install from `requirements.txt`
   - Or run: `pip install -r requirements.txt`

3. **Set environment variables**
   - Go to Secrets tab
   - Add one of: `GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`

4. **Run the app**
   ```bash
   streamlit run app.py --server.port 8501
   ```

5. **Make it public**
   - Use Replit's webview or deploy as a web app

### Alternative: Deploy on Streamlit Cloud

1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set your API key (`GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`) in secrets
4. Deploy!

### Alternative: Deploy on AWS/Azure/GCP

1. Create a VM instance (t2.medium or larger)
2. Install Python and dependencies
3. Run Streamlit with a reverse proxy (nginx)
4. Set up SSL certificate
5. Configure firewall rules

## ⚙️ Configuration

### Changing the Embedding Model

Edit `vector_store.py`:
```python
vector_store = TicketVectorStore(model_name='your-model-name')
```

Popular alternatives:
- `all-mpnet-base-v2` (better quality, slower)
- `paraphrase-MiniLM-L6-v2` (faster, good quality)

### Changing the LLM Model

Edit `app.py` or `solution_generator.py`:
```python
# For Groq (default: llama-3.3-70b-versatile)
generator = SolutionGenerator(provider="groq", model="llama-3.3-70b-versatile")

# For OpenAI (default: gpt-3.5-turbo)
generator = SolutionGenerator(provider="openai", model="gpt-4")

# For Gemini (default: gemini-2.0-flash)
generator = SolutionGenerator(provider="gemini", model="gemini-2.0-flash")
```

**Available Groq Models:**
- `llama-3.3-70b-versatile` (default, best quality)
- `llama-3.1-8b-instant` (faster, good quality)
- `mixtral-8x7b-32768` (long context)

**Available OpenAI Models:**
- `gpt-3.5-turbo` (default, cost-effective)
- `gpt-4` (higher quality, more expensive)
- `gpt-4-turbo` (best quality)

**Available Gemini Models:**
- `gemini-2.0-flash` (default, fast and recommended)
- `gemini-1.5-pro-latest` (higher quality)
- `gemini-1.5-flash-latest` (fast alternative)

### Adjusting Similarity Search

Edit `app.py`:
```python
similar_tickets = vector_store.search(ticket_description, k=10)  # Get more results
```

## 🎓 Learnings & Challenges

### Challenges Faced

1. **Embedding Model Selection**
   - **Challenge**: Balancing quality vs. speed
   - **Solution**: Chose `all-MiniLM-L6-v2` for good balance
   - **Learning**: Smaller models work well for domain-specific tasks

2. **FAISS Index Building**
   - **Challenge**: Initial index building was slow
   - **Solution**: Implemented index persistence to avoid rebuilding
   - **Learning**: Cache embeddings and indices for production use

3. **LLM Response Parsing**
   - **Challenge**: LLM responses sometimes not in expected JSON format
   - **Solution**: Implemented fallback parser with regex
   - **Learning**: Always have error handling for LLM outputs

4. **Memory Management**
   - **Challenge**: Loading all embeddings in memory
   - **Solution**: Used FAISS for efficient storage and retrieval
   - **Learning**: Vector databases are essential for production systems

5. **Solution Ranking**
   - **Challenge**: Getting consistent suitability percentages
   - **Solution**: Structured prompts with explicit format requirements
   - **Learning**: Clear instructions improve LLM output quality

### Key Learnings

1. **Embedding Quality Matters**: Better embeddings = better similarity search = better solutions

2. **Context is Critical**: Including root cause and resolution in embeddings improves matching

3. **Hybrid Approach Works Best**: Combining vector search (for retrieval) with LLM (for generation) provides best results

4. **User Experience**: Clear UI with progress indicators and error handling improves adoption

5. **Scalability**: FAISS handles thousands of tickets efficiently; for millions, consider cloud vector DBs (Pinecone, Weaviate)

## 🔧 Troubleshooting

### ImportError: cannot import name 'PreTrainedModel' from 'transformers'

This error occurs due to version incompatibility between `transformers` and `sentence-transformers`.

### ImportError: cannot import name 'is_quanto_available' from 'transformers.utils'

This error occurs when `transformers` version is too old and doesn't have the `is_quanto_available` function required by `sentence-transformers 2.7.0`.

### ImportError: cannot import name 'translate_to_torch_parallel_style' from 'transformers.pytorch_utils'

This error occurs when `transformers` version is too old and doesn't have the `translate_to_torch_parallel_style` function required by `sentence-transformers 2.7.0`.

### ImportError: cannot import name 'ImageNetInfo' from 'timm.data'

This error occurs when `timm` (PyTorch Image Models) version is too old. Upgrading `timm` resolves this issue.

**Quick Fix (fixes all import errors):**
```bash
# Uninstall conflicting packages
pip uninstall -y transformers sentence-transformers tokenizers timm

# Reinstall compatible versions (tested and working)
pip install transformers==4.48.0
pip install tokenizers==0.21.4
pip install sentence-transformers==2.7.0
pip install timm>=1.0.0
pip install "pillow>=7.1.0,<11"
```

Or use the provided fix script:
```bash
chmod +x fix_import_error.sh
./fix_import_error.sh
```

**Note:** These versions are tested and work together:
- `transformers==4.48.0` (has all required functions and compatible with sentence-transformers)
- `tokenizers==0.21.4` (compatible with transformers 4.48.0)
- `sentence-transformers==2.7.0` (stable version)
- `timm>=1.0.0` (required for transformers 4.48.0)
- `pillow>=7.1.0,<11` (compatible with Streamlit)

### Other Common Issues

**Issue**: "Module not found" errors
- **Solution**: Run `pip install -r requirements.txt` to ensure all dependencies are installed

**Issue**: "Index not found"
- **Solution**: Click "Initialize System" in the Streamlit sidebar to build the index

**Issue**: "API error" or "Rate limit exceeded"
- **Solution**: Check your API key, verify credits/access, and check rate limits in your provider dashboard

**Issue**: App is slow on first run
- **Solution**: First run builds the vector index (5-10 minutes). Subsequent runs are fast as the index is cached.

## 🚀 Future Enhancements

- [ ] Support for more LLM providers (Anthropic, Cohere, etc.)
- [ ] Fine-tuned embedding model on telecom domain data
- [ ] Multi-language support
- [ ] Solution confidence scoring
- [ ] Integration with ticketing systems (Jira, ServiceNow)
- [ ] Analytics dashboard for solution effectiveness
- [ ] Feedback loop to improve recommendations

## 📄 License

This project is for educational/demonstration purposes.

## 👥 Contributors

Built as part of GenAI Evaluation Project.

## 📚 Documentation

- **[Technical Documentation](TECHNICAL_DOCUMENTATION.md)**: In-depth technical details, architecture, algorithms, and step-by-step workflow
- **[Quick Start Guide](QUICKSTART.md)**: Get started in 5 minutes
- **[Deployment Guide](REPLIT_DEPLOY.md)**: Deploy on Replit or other platforms

## 📞 Support

For issues or questions, please check:
- OpenAI API documentation: https://platform.openai.com/docs
- Streamlit documentation: https://docs.streamlit.io
- FAISS documentation: https://github.com/facebookresearch/faiss

---

**Note**: This system requires an API key (Groq, OpenAI, or Gemini) for solution generation. 
- **Groq**: Offers free tier with generous limits, very fast inference
- **Gemini**: Offers free tier, Google's AI models
- **OpenAI**: Pay-per-use, monitor usage to avoid unexpected costs

