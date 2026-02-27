# Streamlit Cloud Deployment - Quick Checklist

## ✅ Files Ready to Deploy

I've prepared everything you need:

- [x] `streamlit_app_standalone.py` - Standalone app (no API needed)
- [x] `requirements_streamlit.txt` - Dependencies list
- [x] `cfb_games.db` - Your game database (already exists)
- [x] `users.db` - Your users/predictions database (already exists)
- [x] `DEPLOY_TO_STREAMLIT_CLOUD.md` - Full deployment guide

## 📋 Quick Start (10 minutes)

### 1. Create GitHub Account
- [ ] Go to https://github.com/signup
- [ ] Verify your email

### 2. Create Repository
- [ ] Go to https://github.com/new
- [ ] Name: `sports-predictions` (or your choice)
- [ ] Set to **Private**
- [ ] Click "Create repository"

### 3. Upload Files
- [ ] Click "uploading an existing file"
- [ ] Upload these 4 files:
  - `streamlit_app_standalone.py`
  - `requirements_streamlit.txt` (rename to `requirements.txt`)
  - `cfb_games.db`
  - `users.db`
- [ ] Commit the upload

### 4. Deploy to Streamlit Cloud
- [ ] Go to https://share.streamlit.io/
- [ ] Sign in with GitHub
- [ ] Click "New app"
- [ ] Select your repository
- [ ] Main file: `streamlit_app_standalone.py`
- [ ] Click "Deploy!"

### 5. Wait & Test
- [ ] Wait 2-5 minutes for deployment
- [ ] You'll get a URL like: `yourname-sports-predictions.streamlit.app`
- [ ] Test login with: user1 / password123

### 6. Share with Beta Users
- [ ] Send URL to your 5 friends/family
- [ ] Send credentials:
  - user1 / password123
  - user2 / password123
  - user3 / password123
  - user4 / password123
  - user5 / password123

## 🎯 Your Beta Credentials

**For testing, you have 6 accounts:**
- user1 through user5 (Password: password123)
- admin (Password: admin123)

## 💡 Optional: Customize Before Deploying

If you want to use real names/passwords, edit `streamlit_app_standalone.py` line 33:

```python
USERS = {
    "john": {
        "password_hash": hashlib.sha256("johns_secure_password".encode()).hexdigest(),
        "full_name": "John Smith",
        "email": "john@email.com"
    },
    # ... add your 5 friends
}
```

## ⚠️ Important Notes

1. **Database files must be uploaded** - Without `cfb_games.db` and `users.db`, the app won't work
2. **First deployment takes 2-5 minutes** - Be patient
3. **Automatic updates** - Push to GitHub = auto-deploy
4. **It's free!** - Streamlit Cloud is free for public apps

## 🆘 If Something Goes Wrong

**"File not found" error:**
- Make sure `cfb_games.db` and `users.db` are in the repository

**"No predictions available":**
- Check that `users.db` has predictions in the `prediction_cache` table
- Run `populate_real_predictions.py` locally first if needed

**Database too large:**
- If `cfb_games.db` > 100MB, you may need to filter the data
- Or use Git LFS (I can help with this)

## 📞 Need Help?

Read the full guide: `DEPLOY_TO_STREAMLIT_CLOUD.md`

---

**Estimated Time:** 10-15 minutes total

**Result:** A public URL your beta testers can access from anywhere! 🎉
