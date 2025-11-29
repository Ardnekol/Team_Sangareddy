# ✅ Replit Deployment Checklist

Follow these steps to deploy your GenAI Ticket Analysis system on Replit.

## 📋 Pre-Deployment Checklist

- [ ] All project files ready
- [ ] Groq API key is hardcoded (already done ✅)
- [ ] Requirements.txt is up to date
- [ ] .replit file is configured

## 🚀 Deployment Steps

### 1. Create Replit Project
- [ ] Go to https://replit.com and sign in
- [ ] Click "Create Repl"
- [ ] Select "Python" template
- [ ] Name it: `genai-ticket-analysis` (or your preferred name)
- [ ] Click "Create Repl"

### 2. Upload Project Files
Upload these files to your Repl (drag & drop or use file upload):

**Required Files:**
- [ ] `app.py`
- [ ] `data_processor.py`
- [ ] `vector_store.py`
- [ ] `solution_generator.py`
- [ ] `telecom_tickets_10000_12cats.json` (or 2000 version if memory is limited)
- [ ] `requirements.txt`
- [ ] `.replit` (optional - auto-configures Replit)

**Optional Files:**
- [ ] `setup.py`
- [ ] `replit_setup.sh`
- [ ] `README.md`

### 3. Install Dependencies
In the Replit shell (bottom panel), run:
```bash
pip install -r requirements.txt
```
- [ ] Wait for installation to complete (5-10 minutes)
- [ ] Check for any errors

**Expected output:**
- Downloads sentence-transformers model (~80MB)
- Installs PyTorch, FAISS, and other dependencies
- May take several minutes

### 4. Build Vector Index (One-Time Setup)
Run in Replit shell:
```bash
python setup.py
```
- [ ] Wait for index building (20-30 minutes for 10,000 tickets)
- [ ] You'll see progress: "Encoding X tickets..."
- [ ] Wait for "✅ Setup complete!" message

**OR** use the quick setup script:
```bash
chmod +x replit_setup.sh
./replit_setup.sh
```

**Note:** This is a one-time process. The index will be saved and reused.

### 5. Run the Application

**Option A: Use Replit's Run Button**
- [ ] Click the green "Run" button at the top
- [ ] Replit will automatically run: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`

**Option B: Manual Run**
In the shell, run:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

- [ ] Wait for Streamlit to start
- [ ] Look for: "You can now view your Streamlit app in your browser"
- [ ] URL will be shown in the output

### 6. Access Your App
- [ ] Click the "Webview" tab in Replit
- [ ] Or use the URL shown in the shell output
- [ ] Your app should open in a new tab

### 7. Test the Application
- [ ] Click "🔄 Initialize System" in the sidebar
- [ ] Wait for "✅ System Ready" message
- [ ] Select "groq" as provider (API key is pre-configured)
- [ ] Enter a test ticket description
- [ ] Click "🔍 Analyze Ticket"
- [ ] Verify solutions are generated

### 8. Make it Public (Optional)
- [ ] Click "Deploy" button in Replit (if available)
- [ ] Configure deployment settings
- [ ] Get a public URL
- [ ] Share with others!

## 🔧 Configuration

### Groq (Pre-configured)
- ✅ API key is hardcoded
- ✅ No setup needed
- ✅ Ready to use immediately

### OpenAI (Optional)
If you want to use OpenAI instead:
- [ ] Go to "Secrets" tab (lock icon)
- [ ] Add: `OPENAI_API_KEY` = `your-key-here`
- [ ] Select "openai" in the app

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "Out of memory"
**Solutions:**
1. Use smaller dataset: Edit `app.py` line 29, change to:
   ```python
   data_path = 'telecom_tickets_2000_12cats.json'
   ```
2. Restart Repl
3. Upgrade Replit plan

### Issue: "Index not found"
**Solution:**
```bash
python setup.py
```

### Issue: "Port already in use"
**Solution:**
- Stop the current process
- Restart Repl
- Run again

### Issue: App is slow
**Solutions:**
- First run builds index (20-30 min) - be patient
- Subsequent runs are fast
- Consider using "Always On" feature

## 📊 Performance Tips

1. **First Run**: 20-30 minutes (index building)
2. **Subsequent Runs**: < 1 minute (index cached)
3. **Query Time**: 3-6 seconds per analysis
4. **Memory Usage**: ~2-3GB for 10,000 tickets

## ✅ Success Indicators

You'll know it's working when:
- ✅ "System Ready" message appears
- ✅ You can enter ticket descriptions
- ✅ Solutions are generated with percentages
- ✅ Similar tickets are shown in expandable section

## 🎉 You're Done!

Your GenAI Ticket Analysis system is now live on Replit!

**Next Steps:**
- Share the public URL
- Test with different ticket types
- Monitor API usage (Groq free tier is generous)
- Consider upgrading Replit plan for better performance

---

**Need Help?**
- Check `REPLIT_DEPLOY.md` for detailed guide
- Check `REPLIT_QUICKSTART.md` for quick reference
- Review error messages in Replit shell

