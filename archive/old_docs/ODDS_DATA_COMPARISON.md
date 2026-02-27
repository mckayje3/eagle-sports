# Odds Data Sources Comparison

## Current Setup

You're currently using **TWO** different odds sources:

### 1. **VegasInsider**
- **Used for:** CFB games
- **Records in DB:** ~1,506 CFB odds
- **Method:** Web scraping from VegasInsider website
- **Free:** Yes
- **Data provided:** Spread, moneyline, totals

### 2. **The Odds API**
- **Used for:** NFL games (and CFB backup)
- **Records in DB:** 419 NFL odds
- **Method:** REST API
- **Free tier:** 500 requests/month
- **Cost:** Limited free, then paid
- **Data provided:** Multiple sportsbooks, spreads, moneylines, totals

### 3. **ESPN** ❓
- **Available:** Sometimes in their API
- **Reliable:** No - not consistently provided
- **Our finding:** Odds NOT included in most responses

---

## Detailed Comparison

### ESPN Odds (NOT RECOMMENDED)

**Pros:**
- Already using ESPN for game data
- No extra API calls needed
- No rate limits

**Cons:**
- ❌ **NOT CONSISTENTLY AVAILABLE** - missing from most games
- ❌ No control over which sportsbook
- ❌ No historical odds/line movements
- ❌ Unknown update frequency
- ❌ Not reliable for pregame predictions

**Verdict:** ESPN does not provide consistent, reliable odds data. While the endpoint exists, it's frequently empty.

---

### The Odds API (CURRENTLY USING)

**Official site:** https://the-odds-api.com

**Pros:**
- ✅ Multiple sportsbooks (consensus lines available)
- ✅ Real-time odds updates
- ✅ Historical data available (from June 2020)
- ✅ Clean REST API
- ✅ Well-documented
- ✅ Reliable and consistent
- ✅ Opening lines + line movements
- ✅ Both pre-game and live odds

**Cons:**
- ❌ Free tier limited to 500 requests/month
- ❌ Costs money beyond free tier
- ❌ Need to manage API key
- ❌ Rate limiting

**Current pricing (as of 2024):**
- Free: 500 requests/month
- Starter: $50/month = 10,000 requests
- Pro: $200/month = 50,000 requests
- Each request can fetch multiple games

**Your usage:**
- NFL: ~15 games/week × 18 weeks = 270 requests/season
- CFB: ~60 games/week × 15 weeks = 900 requests/season
- **Total:** ~1,170 requests/season (would exceed free tier)

---

### VegasInsider (CURRENTLY USING)

**Website:** https://www.vegasinsider.com

**Pros:**
- ✅ Free (web scraping)
- ✅ No API limits
- ✅ Opening/current/closing lines
- ✅ Consensus lines
- ✅ Historical odds
- ✅ Line movement history

**Cons:**
- ❌ Web scraping (can break if site changes)
- ❌ Slower than API
- ❌ May violate ToS
- ❌ No official support
- ❌ Requires maintenance

**Your current scraper:**
- Appears to be working well
- Already have 1,506 CFB odds records
- Scraping HTML/parsing odds data

---

## 📊 Recommendation

### Keep Your Current Hybrid Approach ✅

**For CFB: Use VegasInsider**
- You already have it working
- Free and unlimited
- Sufficient for CFB where you need many games
- 1,506 records show it's working well

**For NFL: Use The Odds API**
- Only ~270 requests/season (fits free tier)
- More reliable than scraping
- NFL is smaller dataset (14-15 games/week)
- Official API support

**DO NOT use ESPN for odds:**
- Too unreliable
- Missing from most games
- Not designed as primary odds source

---

## 💡 Optimization Strategy

### Option 1: Stay Free (Recommended)
- **CFB:** VegasInsider scraping (free, unlimited)
- **NFL:** The Odds API (270 requests < 500 free limit)
- **Total cost:** $0/month
- **Downside:** Need to maintain scraper

### Option 2: All-In on The Odds API
- **CFB + NFL:** The Odds API
- **Requests needed:** ~1,170/season
- **Cost:** $50/month (10,000 requests)
- **Benefit:** No scraping maintenance, multiple sportsbooks
- **Downside:** $600/year

### Option 3: Optimize Request Usage
- Fetch odds once per day (not per game)
- One request can get all games at once
- **CFB:** ~15 weeks × 1 request/day = 15 requests
- **NFL:** ~18 weeks × 1 request/day = 18 requests
- **Total:** ~33 requests/season (well under free tier!)

---

## 🔧 Implementation: Optimize The Odds API Usage

**Current approach (inefficient):**
```python
for game in games:
    fetch_odds(game_id)  # 1 request per game
```

**Optimized approach:**
```python
# One request gets ALL games
all_odds = fetch_current_odds()  # Returns ~60 games in 1 request
for game in all_odds:
    save_to_db(game)
```

**Result:**
- Instead of 1,170 requests/season
- You'd only need ~33 requests/season
- **Stays well under free tier!**

---

## ✅ Final Recommendation

1. **Keep VegasInsider for CFB** - it's working and free
2. **Keep The Odds API for NFL** - you're under the free limit
3. **Optimize The Odds API calls** - fetch all games at once, not per-game
4. **Don't use ESPN for odds** - too unreliable

### Additional Benefits of The Odds API:

- **Multiple sportsbooks:** See consensus vs individual books
- **Sharp vs public betting:** Identify where sharp money is
- **Line shopping:** Find best lines across books
- **Opening lines:** Compare opening to current lines

### The Odds API Provides:

```json
{
  "bookmakers": [
    {
      "key": "fanduel",
      "markets": [
        {
          "key": "h2h",  // moneyline
          "outcomes": [...]
        },
        {
          "key": "spreads",
          "outcomes": [...]
        },
        {
          "key": "totals",
          "outcomes": [...]
        }
      ]
    },
    {
      "key": "draftkings",
      // ... same structure
    }
    // ... more sportsbooks
  ]
}
```

You get **multiple sportsbooks** in one request, which is much more valuable than ESPN's single/unreliable line.

---

## 🎯 Action Items

1. ✅ Continue using VegasInsider for CFB
2. ✅ Continue using The Odds API for NFL
3. 🔨 Optimize The Odds API scraper to batch requests
4. ❌ Don't rely on ESPN for odds data

Would you like me to optimize your The Odds API scraper to reduce request usage?
