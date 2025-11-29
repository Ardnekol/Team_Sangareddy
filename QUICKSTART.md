# Quick Start Guide

Get up and running with the GenAI Ticket Analysis system in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] API key (choose one):
  - [ ] **Groq API key** (recommended - fast & free tier available) at https://console.groq.com
  - [ ] **OpenAI API key** (higher quality) at https://platform.openai.com/api-keys
- [ ] 2GB+ free RAM
- [ ] Internet connection (for downloading models)

## Installation (3 steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set API Key

**Option A: Groq (Recommended - Fast & Free)**
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

**Option B: OpenAI**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Or select provider and enter it in the Streamlit UI (see Step 3).

### Step 3: Run the App
```bash
streamlit run app.py
```

The app will:
- Automatically open in your browser
- Build the vector index on first run (takes 5-10 minutes)
- Save the index for future use

## First Use

1. **Initialize System** (first time only):
   - Click "🔄 Initialize System" in the sidebar
   - Wait for "✅ System Ready" message

2. **Select Provider & Enter API Key** (if not set as env variable):
   - Choose "Groq" or "OpenAI" from the dropdown
   - Enter your API key in the sidebar text field

3. **Analyze a Ticket**:
   - Type a ticket description in the main text area
   - Click "🔍 Analyze Ticket"
   - View the top 3 ranked solutions!

## Example Queries

Try these sample ticket descriptions:

```
Customer reports intermittent internet connectivity on mobile data in the late afternoon.
```

```
Router lights show internet active, but no websites open on any connected device.
```

```
Activation charges for a value-added service appear on the bill without consent.
```

## Troubleshooting

**Problem**: "Module not found"
- **Solution**: Run `pip install -r requirements.txt`

**Problem**: "Index not found"
- **Solution**: Click "Initialize System" in the sidebar

**Problem**: "API error"
- **Solution**: Check your API key (Groq or OpenAI) and ensure you have credits/access

**Problem**: App is slow
- **Solution**: First run builds the index (5-10 min). Subsequent runs are fast.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [REPLIT_DEPLOY.md](REPLIT_DEPLOY.md) for cloud deployment
- Run `python test_system.py` to verify everything works

## Need Help?

- Check the README for architecture details
- Review error messages in the Streamlit app
- Ensure all dependencies are installed correctly

Happy analyzing! 🎫✨

