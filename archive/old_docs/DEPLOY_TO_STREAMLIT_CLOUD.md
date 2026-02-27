# Deploy to Streamlit Cloud - Step by Step Guide

## What I've Created

**`streamlit_app_standalone.py`** - A standalone version that works on Streamlit Cloud:
- ✅ No separate API needed
- ✅ Authentication built-in (6 users)
- ✅ All features included
- ✅ Database files uploaded with app

## Prerequisites

1. **GitHub Account** (free) - https://github.com/signup
2. **Streamlit Cloud Account** (free) - https://share.streamlit.io/

## Step-by-Step Deployment

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `sports-predictions` (or your choice)
3. Set to **Private** (recommended for beta)
4. Click "Create repository"

### Step 2: Upload Your Files

You need to upload these files to GitHub:

**Required Files:**
```
sports-predictions/
├── streamlit_app_standalone.py  (rename to app.py or streamlit_app.py)
├── requirements_streamlit.txt   (rename to requirements.txt)
├── cfb_games.db                 (your game database)
└── users.db                     (your users/predictions database)
```

**Option A: Upload via GitHub Website** (Easiest)
1. On your new repository page, click "uploading an existing file"
2. Drag and drop all 4 files
3. Commit the files

**Option B: Use GitHub Desktop** (Recommended if you know Git)
1. Install GitHub Desktop
2. Clone your repository
3. Copy the files
4. Commit and push

### Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `sports-predictions`
5. **Main file path:** `streamlit_app_standalone.py` (or whatever you named it)
6. Click "Deploy!"

### Step 4: Wait for Deployment

- First deployment takes 2-5 minutes
- Streamlit Cloud will install dependencies
- You'll get a URL like: `yourname-sports-predictions.streamlit.app`

### Step 5: Share with Beta Testers

Send your 5 friends/family:
- **URL:** `https://your-app-name.streamlit.app`
- **Credentials:**
  - Username: user1, user2, user3, user4, or user5
  - Password: password123

## Customizing User Credentials

Before deploying, you can edit the credentials in `streamlit_app_standalone.py`:

```python
USERS = {
    "john": {
        "password_hash": hashlib.sha256("johns_password".encode()).hexdigest(),
        "full_name": "John Smith",
        "email": "john@example.com"
    },
    # ... add your 5 friends
}
```

To generate a password hash:
```python
import hashlib
print(hashlib.sha256("your_password".encode()).hexdigest())
```

## Updating Your App

After deployment, any changes you push to GitHub will automatically update your app:

1. Edit files on your computer
2. Commit and push to GitHub
3. Streamlit Cloud auto-deploys (takes 1-2 minutes)

## Troubleshooting

### "File not found" errors
- Make sure `cfb_games.db` and `users.db` are in the repository root
- Check the file names match exactly

### "No predictions available"
- Run `populate_real_predictions.py` locally first
- Upload the updated `users.db` file
- The database needs to have predictions in the `prediction_cache` table

### Database is too large
- GitHub has a 100MB file limit
- If `cfb_games.db` is too large, you can:
  - Use Git LFS (Large File Storage)
  - Or filter to only 2024 season data

### App is slow
- Streamlit Cloud free tier has resource limits
- Use `@st.cache_data` decorators (already included)
- Consider upgrading to paid tier if needed

## Alternative: Quick Test Without GitHub

If you just want to test the standalone version locally first:

```bash
py -m streamlit run streamlit_app_standalone.py
```

This will run at http://localhost:8501 (different port from the API version)

## Updating Predictions

To add predictions for new weeks:

1. Run locally: `py populate_real_predictions.py` (modify for different weeks)
2. The updated `users.db` file will have new predictions
3. Upload the new `users.db` to GitHub
4. Streamlit Cloud will auto-update

## Monitoring Usage

Streamlit Cloud dashboard shows:
- Number of visitors
- App health status
- Resource usage
- Logs for debugging

## Cost

- **Streamlit Cloud:** FREE forever
- **Limitations on free tier:**
  - 1GB resources
  - Public apps (private requires Teams plan)
  - Sleeps after 7 days of inactivity

For private beta with 5 users, the free tier is perfect!

## Security Note

The standalone version uses SHA-256 password hashing, which is better than plain text but not as secure as the full API with bcrypt. For a short beta test, it's fine. For production, use the full API deployment.

## Next Steps After Beta

If the beta goes well, consider:
1. Deploy the full API + Streamlit setup (more secure)
2. Use Railway or Heroku for the API
3. Add a custom domain
4. Implement proper password reset
5. Add analytics

---

**Need Help?**
- Streamlit docs: https://docs.streamlit.io/streamlit-community-cloud
- My contact: [your contact info]

**Your app is ready to deploy!** Just follow the steps above and you'll have a public URL in minutes.
