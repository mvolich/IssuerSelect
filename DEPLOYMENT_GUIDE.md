# Quick Deployment Guide - Issuer Screening App

## 🚀 5-Minute Deployment to Streamlit Cloud

### Step 1: Prepare Your Files (1 minute)

You should have these files ready:
- ✅ `issuer_screening_app.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules
- ✅ `secrets.toml.template` - Secrets template

### Step 2: Push to GitHub (2 minutes)

**Option A: Using GitHub Web Interface**
1. Go to github.com and create a new repository
2. Name it: `issuer-screening-app` (or your choice)
3. Keep it private
4. Upload all files using "Add file" → "Upload files"
5. Commit the files

**Option B: Using Git Command Line**
```bash
# Navigate to your folder with the files
cd path/to/your/files

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Issuer Screening App"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/issuer-screening-app.git

# Push
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud (2 minutes)

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Click "Sign in with GitHub"

2. **Create New App**
   - Click "New app" button
   - Select your repository: `issuer-screening-app`
   - Branch: `main`
   - Main file path: `issuer_screening_app.py`

3. **Add Secrets**
   - Click "Advanced settings" before deploying
   - In the "Secrets" section, paste:
   ```toml
   api_key = "sk-your-actual-openai-api-key-here"
   ```
   - Replace with your actual OpenAI API key

4. **Deploy**
   - Click "Deploy!"
   - Wait 2-3 minutes for deployment

5. **Done!**
   - Your app will be live at: `https://[app-name].streamlit.app`

---

## 📱 Using the App

1. **Access your app** at the Streamlit Cloud URL

2. **Upload data**:
   - Use the sidebar file uploader
   - Upload your S&P Excel file
   - Wait for processing (~10 seconds for 2,000 issuers)

3. **Explore the tabs**:
   - **Overview**: See IG/HY positioning maps
   - **Top Rankings**: View top performers
   - **Rating Groups**: Analyze by rating tier
   - **Detailed Data**: Filter and export data
   - **AI Analysis**: Get GPT-4 insights (requires API key)

---

## 🔧 Updating Your App

After initial deployment, to update:

```bash
# Make changes to issuer_screening_app.py
# Then commit and push:
git add issuer_screening_app.py
git commit -m "Updated feature X"
git push

# Streamlit Cloud will auto-redeploy in ~1 minute
```

---

## ⚙️ Configuration Options

### Change App Settings

In Streamlit Cloud:
1. Go to your app dashboard
2. Click the ⚙️ settings icon
3. Options available:
   - Change app URL
   - Update secrets
   - View logs
   - Reboot app

### Update OpenAI API Key

1. Go to app settings
2. Click "Secrets"
3. Update the `api_key` value
4. Click "Save"
5. Reboot app

---

## 🐛 Troubleshooting

### App won't start
- Check logs in Streamlit Cloud dashboard
- Verify all files are in repository
- Ensure requirements.txt has correct versions

### "Please upload file" message
- This is normal - app needs the Excel data file
- Upload via sidebar on each session

### AI Analysis not working
- Verify API key is in secrets
- Check you have GPT-4 access on your OpenAI account
- Check API key format: `sk-...`

### Data processing errors
- Verify Excel file format matches expected structure
- Check column names match exactly
- Ensure S&P ratings are valid

---

## 💰 Cost Considerations

**Streamlit Cloud: FREE**
- Free tier includes:
  - 1 private app
  - Unlimited public apps
  - Community support

**OpenAI API:**
- ~$0.15-0.25 per full AI analysis
- GPT-4 pricing: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- To reduce costs: switch to `gpt-3.5-turbo` in code

---

## 🔒 Security Best Practices

✅ **DO:**
- Keep repository private if containing proprietary data
- Use Streamlit secrets for API keys
- Rotate API keys regularly
- Monitor API usage in OpenAI dashboard

❌ **DON'T:**
- Never commit API keys to git
- Don't share your app URL publicly (unless intended)
- Don't commit actual data files

---

## 📊 App Performance

**Expected Performance:**
- Data processing: ~10 seconds for 2,000 issuers
- Visualization rendering: <2 seconds
- AI analysis: ~30-45 seconds (4 GPT-4 calls)
- Total time to results: ~1 minute

**Optimization tips:**
- Data is cached after first load
- Refresh only when uploading new file
- AI analysis is not cached (runs each time)

---

## 🆘 Getting Help

**Streamlit Issues:**
- Docs: https://docs.streamlit.io
- Forum: https://discuss.streamlit.io
- Status: https://streamlit.statuspage.io

**App-Specific Issues:**
- Check README.md in repository
- Review code comments in issuer_screening_app.py
- Test locally before deploying updates

**OpenAI Issues:**
- Docs: https://platform.openai.com/docs
- API status: https://status.openai.com
- Usage dashboard: https://platform.openai.com/usage

---

## ✨ Next Steps After Deployment

1. **Test thoroughly**
   - Upload sample data
   - Test all tabs
   - Run AI analysis
   - Export data

2. **Share with team**
   - Send app URL to authorized users
   - Provide data upload instructions
   - Document any custom settings

3. **Monitor usage**
   - Check Streamlit Cloud analytics
   - Monitor OpenAI API usage
   - Track costs

4. **Customize**
   - Adjust scoring weights if needed
   - Modify category thresholds
   - Update branding/colors

---

## 🎉 You're Done!

Your issuer screening app is now live and ready to use. The entire deployment should take less than 5 minutes.

**App URL format:** `https://issuer-screening-app-[random-id].streamlit.app`

Remember to bookmark your app URL and share it with your team!

---

*Deployment completed: Ready to screen issuers!*
