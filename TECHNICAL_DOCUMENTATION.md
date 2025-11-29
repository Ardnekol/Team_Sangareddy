# Technical Documentation: GenAI-Powered Ticket Analysis System

## 📋 Table of Contents
1. [System Architecture](#system-architecture)
2. [Tech Stack](#tech-stack)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Component Deep Dive](#component-deep-dive)
5. [Data Flow](#data-flow)
6. [Algorithm Details](#algorithm-details)

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│                    (Streamlit Web App)                        │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Data         │  │ Vector      │  │ Solution     │      │
│  │ Processor    │  │ Store       │  │ Generator    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA & MODEL LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ JSON Ticket  │  │ FAISS Vector │  │ LLM API      │      │
│  │ Data         │  │ Index        │  │ (Groq/OpenAI)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Frontend
- **Streamlit** (v1.28.0+)
  - Purpose: Web UI for ticket input and solution display
  - Why: Rapid prototyping, Python-native, no frontend code needed

### Backend Core
- **Python 3.8+**
  - Main programming language
  - Handles all business logic

### Embedding & Vector Search
- **Sentence Transformers** (v2.7.0)
  - Model: `all-MiniLM-L6-v2`
  - Purpose: Converts text to dense vector embeddings
  - Dimensions: 384
  - Why: Fast, efficient, good quality for semantic search

- **FAISS** (Facebook AI Similarity Search) (v1.7.4+)
  - Purpose: Efficient similarity search in high-dimensional space
  - Index Type: `IndexFlatIP` (Inner Product for cosine similarity)
  - Why: Fast nearest neighbor search, handles millions of vectors

- **Transformers** (v4.49.0)
  - Purpose: Underlying library for sentence transformers
  - Provides: Pre-trained models and tokenization

### LLM Integration
- **Groq API** (v0.4.0+)
  - Models: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`
  - Purpose: Generate ranked solutions
  - Why: Fast inference, free tier available

- **OpenAI API** (v1.3.0+)
  - Models: `gpt-3.5-turbo`, `gpt-4`
  - Purpose: Alternative LLM provider
  - Why: High quality, reliable

### Data Processing
- **NumPy** (v1.24.0+)
  - Purpose: Numerical operations on embeddings
  - Used for: Array operations, normalization

- **PyTorch** (v2.0.0+)
  - Purpose: Deep learning backend for sentence transformers
  - Used for: Model inference

### Additional Dependencies
- **Tokenizers** (v0.21.4): Text tokenization
- **Timm** (v1.0.22+): Model utilities
- **Pillow** (v10.4.0): Image processing (dependency)

---

## 🔄 Step-by-Step Workflow

### Phase 1: System Initialization

#### Step 1.1: Data Loading
```
User clicks "Initialize System"
    ↓
app.py → initialize_system()
    ↓
TicketDataProcessor.load_tickets()
    ↓
Reads: telecom_tickets_10000_12cats.json
    ↓
Returns: List of 10,000 ticket dictionaries
```

**Technical Details:**
- File format: JSON array
- Each ticket contains:
  - `ticket_id`: Unique identifier
  - `category`: Issue category (12 categories)
  - `customer_issue_description`: Problem description
  - `root_cause`: Identified cause
  - `final_resolution`: How it was resolved

#### Step 1.2: Text Preparation
```
TicketDataProcessor.get_all_ticket_texts()
    ↓
For each ticket:
    Combine: category + issue + root_cause + resolution
    Format: "Category: X\nIssue: Y\nRoot Cause: Z\nResolution: W"
    ↓
Returns: List of 10,000 combined text strings
```

**Example:**
```python
Input ticket:
{
  "category": "billing",
  "customer_issue_description": "Duplicate charge",
  "root_cause": "Pro-rated calculation error",
  "final_resolution": "Refunded duplicate amount"
}

Output text:
"Category: billing
Issue: Duplicate charge
Root Cause: Pro-rated calculation error
Resolution: Refunded duplicate amount"
```

#### Step 1.3: Embedding Generation
```
TicketDataProcessor.get_all_ticket_texts()
    ↓
VectorStore.build_index(ticket_texts)
    ↓
SentenceTransformer.encode(ticket_texts)
    ↓
For each text:
    Tokenize → Model forward pass → 384-dim vector
    ↓
Returns: NumPy array (10000, 384)
```

**Technical Process:**
1. **Tokenization**: Text → Token IDs
   - Uses WordPiece/BPE tokenization
   - Max sequence length: 256 tokens
   - Padding/truncation as needed

2. **Model Forward Pass**:
   ```
   Input tokens → Embedding layer → Transformer layers → Pooling → Output vector
   ```
   - 6 transformer layers
   - Mean pooling of all token embeddings
   - Output: 384-dimensional dense vector

3. **Batch Processing**:
   - Batch size: 32
   - Progress bar shows encoding progress
   - ~5-10 minutes for 10,000 tickets

#### Step 1.4: Vector Normalization
```
Embeddings (10000, 384)
    ↓
faiss.normalize_L2(embeddings)
    ↓
Each vector: v / ||v||
    ↓
Normalized embeddings (unit vectors)
```

**Why Normalize?**
- Enables cosine similarity via inner product
- `cosine_similarity(a, b) = dot_product(normalized_a, normalized_b)`
- More stable for similarity search

#### Step 1.5: FAISS Index Building
```
Normalized embeddings
    ↓
faiss.IndexFlatIP(384)  # Inner Product index
    ↓
index.add(embeddings)
    ↓
Index contains 10,000 vectors
    ↓
faiss.write_index(index, 'ticket_index.faiss')
```

**Index Type: IndexFlatIP**
- **Flat**: No compression, exact search
- **IP**: Inner Product (for normalized vectors = cosine similarity)
- **Search Time**: O(n) where n = number of vectors
- **Memory**: ~15MB for 10,000 vectors (384 dims × 4 bytes × 10,000)

#### Step 1.6: Metadata Storage
```
Ticket metadata (IDs, categories, etc.)
    ↓
Pickle serialization
    ↓
Save: ticket_index_metadata.pkl
```

**Stored Metadata:**
- Ticket ID
- Category
- Original issue description
- Root cause
- Resolution

---

### Phase 2: Query Processing

#### Step 2.1: User Input
```
User enters ticket description in Streamlit UI
    ↓
Example: "Customer reports duplicate billing charge"
    ↓
Stored in: ticket_description variable
```

#### Step 2.2: Query Embedding
```
ticket_description (string)
    ↓
VectorStore.search(query, k=15)
    ↓
SentenceTransformer.encode([query])
    ↓
Query embedding (1, 384)
    ↓
faiss.normalize_L2(query_embedding)
```

**Process:**
- Same embedding model as training
- Same normalization
- Ensures compatibility

#### Step 2.3: Similarity Search
```
Normalized query embedding
    ↓
index.search(query_embedding, k=15)
    ↓
For each vector in index:
    similarity = dot_product(query, vector)
    ↓
Returns: Top 15 most similar tickets
    (indices, similarity_scores)
```

**Algorithm:**
```python
# Pseudo-code
def search(query_vector, k=15):
    similarities = []
    for i, vector in enumerate(index):
        similarity = dot_product(query_vector, vector)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return similarities[:k]
```

**Time Complexity:** O(n) where n = 10,000
**Space Complexity:** O(k) for results

#### Step 2.4: Diversity Enhancement
```
Top 15 similar tickets
    ↓
_prepare_context(similar_tickets)
    ↓
Group by category
    ↓
Select diverse tickets (different categories)
    ↓
Returns: 12 diverse tickets for context
```

**Diversity Logic:**
```python
seen_categories = set()
diverse_tickets = []

for ticket, score in similar_tickets:
    category = ticket.get('category')
    if category not in seen_categories:
        diverse_tickets.append((ticket, score))
        seen_categories.add(category)
    if len(diverse_tickets) >= 12:
        break
```

**Why?**
- Ensures solutions consider different problem types
- Prevents bias toward single category
- Improves solution diversity

---

### Phase 3: Solution Generation

#### Step 3.1: Context Preparation
```
12 diverse similar tickets
    ↓
Format each ticket:
    "Similar Ticket 1 (Similarity: 85%):
     Category: billing
     Issue: Duplicate charge
     Root Cause: Pro-rated error
     Resolution: Refunded amount"
    ↓
Combined context string
```

#### Step 3.2: Prompt Construction
```
Customer issue + Similar tickets context
    ↓
_create_prompt(query, context)
    ↓
Structured prompt with:
    - Customer issue
    - Similar tickets (12 examples)
    - Instructions for 3 diverse solution types
    - JSON format specification
```

**Prompt Structure:**
```
Analyze the following customer issue and provide the top 3 DISTINCT solution options...

CRITICAL REQUIREMENTS FOR DIVERSITY:
- Solution 1: IMMEDIATE ACTION (refund, credit, quick fix)
- Solution 2: INVESTIGATION/DIAGNOSTIC (verify, check, analyze)
- Solution 3: ALTERNATIVE STRATEGY (escalate, upgrade, workaround)

Customer Issue: [user input]

Similar Resolved Tickets: [12 diverse examples]

[JSON format specification]
```

#### Step 3.3: LLM API Call
```
Constructed prompt
    ↓
Groq/OpenAI API call
    ↓
Parameters:
    - model: llama-3.3-70b-versatile / gpt-3.5-turbo
    - temperature: 0.8 (creativity)
    - max_tokens: 1500
    - messages: [system, user]
    ↓
API Response (JSON string)
```

**API Request Format:**
```python
{
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {
            "role": "system",
            "content": "You are a telecom support expert..."
        },
        {
            "role": "user",
            "content": "[constructed prompt]"
        }
    ],
    "temperature": 0.8,
    "max_tokens": 1500
}
```

**LLM Processing (Internal):**
1. Tokenize prompt
2. Forward pass through transformer layers
3. Generate tokens autoregressively
4. Stop at max_tokens or end token
5. Return generated text

#### Step 3.4: Response Parsing
```
LLM response (JSON string)
    ↓
_parse_solutions(response_text)
    ↓
Extract JSON using regex
    ↓
json.loads() → Python dict
    ↓
Extract solutions array
    ↓
Returns: List of 3 solution dictionaries
```

**Parsed Structure:**
```python
[
    {
        "rank": 1,
        "solution": "Process refund for duplicate charge...",
        "suitability_percentage": 95,
        "reasoning": "Directly addresses the issue..."
    },
    {
        "rank": 2,
        "solution": "Verify account and transaction history...",
        "suitability_percentage": 80,
        "reasoning": "Thorough investigation approach..."
    },
    {
        "rank": 3,
        "solution": "Escalate to billing specialist...",
        "suitability_percentage": 65,
        "reasoning": "Alternative strategy if standard solutions fail..."
    }
]
```

#### Step 3.5: Diversity Post-Processing
```
Parsed solutions
    ↓
_ensure_diversity(solutions)
    ↓
Check solution types:
    - Immediate action keywords
    - Investigation keywords
    - Alternative keywords
    ↓
If all same type → Modify solutions
    ↓
Returns: Diverse solutions
```

**Diversity Check:**
```python
immediate_keywords = ['refund', 'credit', 'process', 'apply']
investigation_keywords = ['verify', 'check', 'analyze', 'review']
alternative_keywords = ['escalate', 'upgrade', 'alternative', 'workaround']

# Classify each solution
# If all same type → Force diversity by adding type prefixes
```

---

### Phase 4: Result Display

#### Step 4.1: UI Rendering
```
3 ranked solutions
    ↓
Streamlit display loop
    ↓
For each solution:
    - Display rank and suitability %
    - Show progress bar
    - Display solution text
    - Display reasoning
    ↓
User sees formatted results
```

**UI Components:**
- Progress bars (visual suitability indicator)
- Color coding (green/yellow/orange by suitability)
- Expandable sections (similar tickets reference)
- Responsive layout

---

## 🔍 Component Deep Dive

### 1. Data Processor (`data_processor.py`)

**Class: `TicketDataProcessor`**

**Methods:**
- `load_tickets()`: Loads JSON file, returns list of tickets
- `prepare_ticket_text()`: Combines ticket fields into searchable text
- `get_all_ticket_texts()`: Returns all ticket texts for embedding
- `get_ticket_metadata()`: Returns metadata for retrieval

**Key Design Decisions:**
- Combines all fields (category, issue, root cause, resolution) for richer embeddings
- Preserves metadata separately for display
- Handles file I/O errors gracefully

### 2. Vector Store (`vector_store.py`)

**Class: `TicketVectorStore`**

**Methods:**
- `build_index()`: Creates FAISS index from embeddings
- `save_index()`: Persists index to disk
- `load_index()`: Loads existing index
- `search()`: Finds similar tickets

**Embedding Model: `all-MiniLM-L6-v2`**
- Architecture: 6-layer transformer
- Parameters: ~22M
- Input: Text (max 256 tokens)
- Output: 384-dim vector
- Training: Trained on 1B+ sentence pairs

**FAISS Index:**
- Type: `IndexFlatIP` (Flat Inner Product)
- Advantages: Exact search, no approximation
- Disadvantages: O(n) search time
- Suitable for: < 1M vectors

### 3. Solution Generator (`solution_generator.py`)

**Class: `SolutionGenerator`**

**Methods:**
- `generate_solutions()`: Main method, orchestrates solution generation
- `_prepare_context()`: Formats similar tickets with diversity
- `_create_prompt()`: Constructs LLM prompt
- `_parse_solutions()`: Parses LLM JSON response
- `_ensure_diversity()`: Post-processes for solution diversity

**LLM Models:**

**Groq - llama-3.3-70b-versatile:**
- Architecture: Llama 3.3, 70B parameters
- Context: 128K tokens
- Speed: Very fast (GPU-accelerated)
- Cost: Free tier available

**OpenAI - gpt-3.5-turbo:**
- Architecture: GPT-3.5, ~175B parameters
- Context: 16K tokens
- Speed: Fast
- Cost: Pay-per-use

**Prompt Engineering:**
- System message: Sets role and diversity requirements
- User prompt: Structured with examples and format
- Temperature: 0.8 for creativity
- Max tokens: 1500 for detailed solutions

### 4. Streamlit App (`app.py`)

**Functions:**
- `initialize_system()`: Cached resource initialization
- `main()`: Main UI loop

**Session State:**
- `vector_store`: Cached vector store instance
- `initialized`: System status flag
- `provider`: Selected LLM provider
- `llm_model`: Selected model name

**UI Components:**
- Sidebar: Configuration and status
- Main area: Ticket input and results
- Expanders: Collapsible sections for details

---

## 📊 Data Flow

### Complete Flow Diagram

```
┌─────────────┐
│ User Input  │
│ (Ticket)    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Embed Query     │
│ (384-dim vector)│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ FAISS Search    │
│ (Top 15)        │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Diversity Filter│
│ (12 diverse)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Format Context  │
│ (Prompt)        │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ LLM API Call    │
│ (Groq/OpenAI)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Parse Response  │
│ (3 solutions)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Diversity Check │
│ (Post-process)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Display Results │
│ (Streamlit UI)  │
└─────────────────┘
```

---

## 🧮 Algorithm Details

### 1. Embedding Generation Algorithm

```python
def generate_embedding(text):
    # Step 1: Tokenization
    tokens = tokenizer(text, 
                      max_length=256,
                      padding='max_length',
                      truncation=True,
                      return_tensors='pt')
    
    # Step 2: Model forward pass
    with torch.no_grad():
        outputs = model(**tokens)
        # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
    
    # Step 3: Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # embeddings shape: (batch, hidden_dim) = (1, 384)
    
    return embeddings.numpy()
```

### 2. Similarity Search Algorithm

```python
def cosine_similarity_search(query_vector, index, k=15):
    # Normalize query vector
    query_norm = query_vector / np.linalg.norm(query_vector)
    
    # Compute similarities (inner product for normalized vectors)
    similarities = index.search(query_norm.reshape(1, -1), k)
    # Returns: (distances, indices)
    # distances: cosine similarities (higher = more similar)
    # indices: ticket IDs
    
    return similarities
```

**Mathematical Foundation:**
- Cosine Similarity: `cos(θ) = (A · B) / (||A|| × ||B||)`
- For normalized vectors: `cos(θ) = A · B` (dot product)
- Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite

### 3. Diversity Selection Algorithm

```python
def select_diverse_tickets(similar_tickets, max_tickets=12):
    seen_categories = set()
    diverse_tickets = []
    
    # First pass: Prioritize different categories
    for ticket, score in similar_tickets:
        category = ticket.get('category')
        if category not in seen_categories:
            diverse_tickets.append((ticket, score))
            seen_categories.add(category)
        if len(diverse_tickets) >= max_tickets:
            break
    
    # Second pass: Fill remaining slots
    if len(diverse_tickets) < max_tickets:
        for ticket, score in similar_tickets:
            if (ticket, score) not in diverse_tickets:
                diverse_tickets.append((ticket, score))
            if len(diverse_tickets) >= max_tickets:
                break
    
    return diverse_tickets
```

### 4. Solution Diversity Check Algorithm

```python
def check_solution_diversity(solutions):
    immediate_keywords = ['refund', 'credit', 'process', 'apply']
    investigation_keywords = ['verify', 'check', 'analyze', 'review']
    alternative_keywords = ['escalate', 'upgrade', 'alternative']
    
    solution_types = []
    for sol in solutions:
        text = sol['solution'].lower()
        
        # Count keyword matches
        immediate_score = sum(1 for kw in immediate_keywords if kw in text)
        investigation_score = sum(1 for kw in investigation_keywords if kw in text)
        alternative_score = sum(1 for kw in alternative_keywords if kw in text)
        
        # Classify solution type
        if immediate_score > max(investigation_score, alternative_score):
            solution_types.append('immediate')
        elif investigation_score > alternative_score:
            solution_types.append('investigation')
        else:
            solution_types.append('alternative')
    
    # Check if all same type
    if len(set(solution_types)) == 1:
        # Force diversity by modifying solutions
        modify_solutions_for_diversity(solutions, solution_types[0])
    
    return solutions
```

---

## 🔧 Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Embedding generation | O(n × m) | n = tickets, m = tokens per ticket |
| Index building | O(n) | Linear with number of vectors |
| Similarity search | O(n) | Flat index, no approximation |
| Solution generation | O(1) | API call, constant time |
| Total query time | O(n) | Dominated by similarity search |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Embeddings | O(n × d) | n = tickets, d = dimensions (384) |
| FAISS index | O(n × d) | Stores all vectors |
| Metadata | O(n) | Ticket information |
| Total | O(n × d) | ~15MB for 10,000 tickets |

### Performance Metrics

- **Index Building**: 20-30 minutes for 10,000 tickets
- **Query Processing**: < 1 second (embedding + search)
- **Solution Generation**: 2-5 seconds (LLM API call)
- **Total Response Time**: 3-6 seconds per query

---

## 🎯 Key Design Decisions

### 1. Why Sentence Transformers?
- **Pre-trained**: No need for domain-specific training
- **Efficient**: Fast inference, small model size
- **Effective**: Good semantic understanding

### 2. Why FAISS?
- **Fast**: Optimized C++ implementation
- **Scalable**: Handles millions of vectors
- **Flexible**: Multiple index types available

### 3. Why RAG (Retrieval-Augmented Generation)?
- **Accuracy**: Grounded in real examples
- **Transparency**: Shows similar tickets
- **No Training**: Uses pre-trained models
- **Updatable**: Easy to add new tickets

### 4. Why Multiple Solution Types?
- **Comprehensive**: Covers different approaches
- **Flexible**: Users can choose best fit
- **Realistic**: Mirrors real support workflows

---

## 📈 Scalability Considerations

### Current Limits
- **Tickets**: 10,000 (tested)
- **Query Time**: < 1 second
- **Index Size**: ~15MB

### Scaling Options

**For 100K+ Tickets:**
1. Use FAISS `IndexIVFFlat` (approximate search)
2. Reduce embedding dimensions (PCA)
3. Use GPU acceleration

**For Production:**
1. Deploy vector store separately (Pinecone, Weaviate)
2. Cache frequent queries
3. Use async API calls
4. Load balancing for multiple users

---

## 🔐 Security & Privacy

### Data Handling
- **Local Processing**: Embeddings generated locally
- **API Keys**: Stored in environment variables
- **No Data Storage**: LLM responses not persisted
- **User Privacy**: No tracking or logging

### Best Practices
- Never commit API keys
- Use environment variables
- Validate user inputs
- Rate limiting for API calls

---

## 🐛 Troubleshooting

### Common Issues

1. **Slow Index Building**
   - Solution: Use smaller batch size or GPU

2. **Memory Errors**
   - Solution: Process in batches, reduce index size

3. **Low Similarity Scores**
   - Solution: Check embedding model, verify normalization

4. **Repetitive Solutions**
   - Solution: Increase temperature, improve prompt diversity

---

## 📚 References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Groq API Documentation](https://console.groq.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: GenAI Ticket Analysis System

