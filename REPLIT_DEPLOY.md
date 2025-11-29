# Deploying on Replit

This guide will help you deploy the GenAI Ticket Analysis system on Replit.

## Step-by-Step Deployment

### 1. Create a New Replit Project

1. Go to [Replit](https://replit.com) and sign in
2. Click "Create Repl"
3. Choose "Python" template
4. Name your project (e.g., "genai-ticket-analysis")

### 2. Upload Project Files

Upload all project files to your Replit:
- `app.py`
- `data_processor.py`
- `vector_store.py`
- `solution_generator.py`
- `telecom_tickets_10000_12cats.json`
- `requirements.txt`
- `setup.py` (optional)
- `README.md` (optional)

### 3. Install Dependencies

In the Replit shell, run:
```bash
pip install -r requirements.txt
```

**Note**: This may take a few minutes as it downloads:
- Sentence Transformers model (~80MB)
- PyTorch and dependencies
- Other required packages

### 4. Set Environment Variables (Optional)

**Note**: Groq API key is hardcoded, so you don't need to set it. However, if you want to use OpenAI:

1. Click on the "Secrets" tab (lock icon) in the left sidebar
2. Add a new secret:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key

**Groq is pre-configured** - no API key setup needed!

### 5. Build the Vector Index (First Time Only)

Run the setup script in the shell:
```bash
python setup.py
```

This will:
- Load all tickets
- Generate embeddings
- Build the FAISS index
- Save it for future use

**Note**: This takes 20-30 minutes for 10,000 tickets.

### 6. Run the Application

In the Replit shell, run:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 7. Make it Public

1. Click the "Webview" tab in Replit
2. Or use Replit's "Deploy" feature to create a public URL

## Replit-Specific Considerations

### Memory Limits
- Free Repls have limited RAM
- If you encounter memory issues, consider:
  - Using a smaller embedding model
  - Reducing the number of tickets
  - Upgrading to a paid Replit plan

### Persistence
- FAISS index files are saved in the Replit filesystem
- They persist between sessions
- Consider backing up `*.faiss` and `*_metadata.pkl` files

### Performance
- First run (index building) is slow
- Subsequent runs are fast (index is cached)
- Consider using Replit's "Always On" feature for production

## Troubleshooting

### Issue: "Module not found"
**Solution**: Run `pip install -r requirements.txt` again

### Issue: "Out of memory"
**Solution**: 
- Restart the Repl
- Use a smaller model: Edit `vector_store.py` and change model to `'paraphrase-MiniLM-L3-v2'`

### Issue: "Index not found"
**Solution**: Run `python setup.py` to build the index

### Issue: "API error"
**Solution**: 
- Check your API key (Groq or OpenAI) in Secrets
- Verify you have API credits/access
- Check rate limits
- For Groq: Ensure you're using a valid API key from console.groq.com

## Alternative: Use Replit's Deploy Feature

1. After setting up, click "Deploy" button
2. Configure deployment settings
3. Replit will create a public URL
4. The app will auto-restart on code changes

## Cost Considerations

- Replit free tier: Limited resources, may be slow
- **Groq API**: Free tier available with generous limits, very fast inference (recommended)
- **OpenAI API**: Pay-per-use, monitor costs
- Consider setting usage limits in provider dashboard

