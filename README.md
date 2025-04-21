# Mexico City Activities Reddit Scraper

This script scrapes Reddit for activities and things to do in Mexico City using the PRAW (Python Reddit API Wrapper) library.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Create a Reddit application:

   - Go to https://www.reddit.com/prefs/apps
   - Click "create another app..."
   - Choose "script"
   - Fill in the required information
   - Note down your client ID and client secret
3. Create a `.env` file in the project root with the following variables:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

## Usage

Run the script:

```bash
python mexico_city_activities.py
```

The script will:

- Search through multiple Mexico City-related subreddits
- Look for posts about activities and things to do
- Save the results to `mexico_city_activities.json`

## Output

The script generates a JSON file containing:

- Post titles
- URLs
- Scores (upvotes)
- Creation dates
- Subreddit sources
- Post content (first 500 characters)
- Number of comments

The results are sorted by score and duplicates are removed.



To see it  work locally, run 
```bash
python app.py
```
