# Quick Replit Deployment Guide

## 🚀 Fast Deployment (5 Steps)

### Step 1: Create Repl
1. Go to [Replit](https://replit.com)
2. Click "Create Repl"
3. Choose "Python" template
4. Name: `genai-ticket-analysis`

### Step 2: Upload Files
Upload these files to your Repl:
- ✅ `app.py`
- ✅ `data_processor.py`
- ✅ `vector_store.py`
- ✅ `solution_generator.py`
- ✅ `telecom_tickets_10000_12cats.json`
- ✅ `requirements.txt`
- ✅ `.replit` (optional, auto-configures)
- ✅ `setup.py` (optional, for manual setup)

### Step 3: Install Dependencies
In Replit shell, run:
```bash
pip install -r requirements.txt
```
⏱️ Takes 5-10 minutes (downloads models)

### Step 4: Build Index (One-Time)
```bash
python setup.py
```
⏱️ Takes 20-30 minutes for 10,000 tickets

**OR** use the quick setup script:
```bash
chmod +x replit_setup.sh
./replit_setup.sh
```

### Step 5: Run App
Click the **"Run"** button in Replit, or run:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## ✅ That's It!

- **Groq API**: Pre-configured (hardcoded key)
- **OpenAI API**: Optional (set in Secrets if needed)
- **Webview**: Click "Webview" tab to see your app
- **Public URL**: Use Replit's "Deploy" feature

## 🐛 Troubleshooting

**Out of Memory?**
- Use smaller dataset: Switch to `telecom_tickets_2000_12cats.json` in `app.py`
- Or upgrade Replit plan

**Index Not Found?**
- Run: `python setup.py`

**Module Errors?**
- Run: `pip install -r requirements.txt --upgrade`

## 📝 Notes

- First run builds index (20-30 min) - be patient!
- Index persists between sessions
- Groq key is hardcoded - ready to use immediately
- Free tier works but may be slow

---

**Need help?** Check `REPLIT_DEPLOY.md` for detailed instructions.

