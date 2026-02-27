# Deploy to Your Existing GitHub Repository

## Your Repository
https://github.com/mckayje3/eagle-sports.git

## Files to Upload

You need to upload these 4 files to your repository:

1. **`streamlit_app_standalone.py`** - The standalone app
2. **`requirements.txt`** - Rename from `requirements_streamlit.txt`
3. **`cfb_games.db`** - Your games database
4. **`users.db`** - Your users/predictions database

## Option 1: Upload via GitHub Website (Easiest)

### Step 1: Go to Your Repository
Visit: https://github.com/mckayje3/eagle-sports

### Step 2: Upload Files
1. Click "Add file" → "Upload files"
2. Drag and drop or select these files:
   - `streamlit_app_standalone.py`
   - `requirements_streamlit.txt` (but rename it to `requirements.txt` before uploading)
   - `cfb_games.db`
   - `users.db`
3. Add commit message: "Add Streamlit app with authentication"
4. Click "Commit changes"

### Step 3: Check Files Are There
After upload, your repository should show:
```
eagle-sports/
├── streamlit_app_standalone.py
├── requirements.txt
├── cfb_games.db
└── users.db
```

## Option 2: Push via Git Command Line (If you have Git installed)

```bash
# Navigate to your sports directory
cd C:\Users\jbeast\documents\coding\sports

# Initialize git if not already done
git init

# Add your GitHub repository as remote
git remote add origin https://github.com/mckayje3/eagle-sports.git

# Or if remote already exists, set the URL
git remote set-url origin https://github.com/mckayje3/eagle-sports.git

# Rename requirements file
copy requirements_streamlit.txt requirements.txt

# Add the files
git add streamlit_app_standalone.py
git add requirements.txt
git add cfb_games.db
git add users.db

# Commit
git commit -m "Add Streamlit app with authentication and databases"

# Push to GitHub
git push -u origin main
# Or if your branch is called master:
# git push -u origin master
```

## Deploy to Streamlit Cloud

### Step 1: Go to Streamlit Cloud
Visit: https://share.streamlit.io/

### Step 2: Sign In
- Click "Sign in"
- Use your GitHub account (mckayje3)

### Step 3: Create New App
1. Click "New app" (or "Deploy an app")
2. **Repository:** `mckayje3/eagle-sports`
3. **Branch:** `main` (or `master` if that's your default)
4. **Main file path:** `streamlit_app_standalone.py`
5. Click "Deploy!"

### Step 4: Wait for Deployment
- First deployment takes 2-5 minutes
- Watch the logs to see progress
- You'll get a URL like: `mckayje3-eagle-sports.streamlit.app`

### Step 5: Test Your App
1. Visit your new URL
2. Try logging in with: user1 / password123
3. Check that predictions load

### Step 6: Share with Beta Testers
Send your 5 friends/family:

**Your Sports Prediction App**
- URL: https://mckayje3-eagle-sports.streamlit.app
- Login credentials:
  - Username: user1, user2, user3, user4, or user5
  - Password: password123

## Troubleshooting

### "No such file or directory"
- Make sure all 4 files are in the repository root (not in a subfolder)
- Check file names match exactly (case-sensitive)

### "No predictions available"
- Verify `users.db` has data in the `prediction_cache` table
- If needed, run `populate_real_predictions.py` locally first

### Database files too large
If you get an error about file size:
```bash
# Check database sizes
dir cfb_games.db users.db

# If cfb_games.db is > 100MB, you may need Git LFS
# Or filter to just 2024 season data
```

### Authentication failed on git push
If you get an authentication error when pushing:
1. GitHub now requires Personal Access Token instead of password
2. Go to: https://github.com/settings/tokens
3. Generate new token (classic)
4. Select: repo (full control)
5. Use token as password when prompted

## After Deployment

### Update Your App
Any changes pushed to GitHub will automatically deploy:
1. Edit files locally
2. Commit and push to GitHub
3. Streamlit Cloud auto-deploys in 1-2 minutes

### Monitor Your App
In Streamlit Cloud dashboard you can:
- View app logs
- See resource usage
- Check visitor stats
- Restart app if needed

### Add More Predictions
1. Run `populate_real_predictions.py` with new weeks locally
2. Upload the updated `users.db` to GitHub
3. App will auto-update with new predictions

## Your URLs

- **GitHub Repository:** https://github.com/mckayje3/eagle-sports
- **Streamlit App:** (will be) https://mckayje3-eagle-sports.streamlit.app
- **Local Test:** http://localhost:8504 (currently running)

## Need Help?

Common issues:
- Files not uploading? Try smaller batches
- Database too large? Let me know, I can help filter it
- Authentication issues? Use Personal Access Token

---

**Estimated time:** 5-10 minutes
**Result:** Public URL for your beta testers! 🎉
