import praw
import os
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment variables
load_dotenv()

def setup_reddit():
    """Initialize Reddit API connection"""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            read_only=True  # Explicitly set read-only mode
        )
        # Test the connection
        reddit.user.me()  # This will raise an error if authentication fails
        return reddit
    except Exception as e:
        print(f"Error during Reddit setup: {str(e)}")
        raise

def get_comments(submission, limit=20):
    """Extract top comments from a submission"""
    submission.comments.replace_more(limit=0)  # Replace MoreComments objects with actual comments
    comments = []
    
    # Get top comments sorted by score
    top_comments = sorted(submission.comments.list(), key=lambda comment: comment.score, reverse=True)[:limit]
    
    for comment in top_comments:
        try:
            # Skip deleted or removed comments
            if comment.author is None or comment.body in ['[deleted]', '[removed]']:
                continue
                
            comment_data = {
                'author': str(comment.author),
                'body': comment.body,
                'score': comment.score,
                'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat()
            }
            comments.append(comment_data)
        except Exception as e:
            print(f"Error processing comment: {str(e)}")
            continue
            
    return comments

def get_activities(reddit):
    """Scrape activities from Mexico City subreddits"""
    subreddits = [
        'MexicoCity',
        'CDMX',
        'MexicoCityTravel',
        'MexicoCityFood'
    ]
    
    activities = []
    
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            print(f"Searching in r/{subreddit_name}...")
            
            # Search for posts about things to do
            search_terms = [
                'tourist attractions',
                'American food',
                'Traditional Mexican food',
                'Hole in the wall',
                'best food',  # Added for food-specific searches
                'restaurant recommendations',  # Added for food-specific searches
                'best tacos',  # Added for food-specific searches
                'food recommendations'  # Added for food-specific searches
            ]
            
            for term in search_terms:
                try:
                    for submission in subreddit.search(term, limit=10):
                        if submission.score > 5:  # Only include posts with some engagement
                            # Get comments for this submission
                            print(f"Fetching comments for: {submission.title}")
                            comments = get_comments(submission)
                            
                            activity = {
                                'title': submission.title,
                                'url': submission.url,
                                'score': submission.score,
                                'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                                'subreddit': subreddit_name,
                                'text': submission.selftext[:500] if submission.selftext else '',  # First 500 chars
                                'num_comments': submission.num_comments,
                                'comments': comments  # Add the comments to the data
                            }
                            activities.append(activity)
                except Exception as e:
                    print(f"Error searching term '{term}' in r/{subreddit_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error accessing subreddit r/{subreddit_name}: {str(e)}")
            continue
    
    return activities

def save_activities(activities):
    """Save activities to a JSON file"""
    with open('mexico_city_activities_with_comments.json', 'w', encoding='utf-8') as f:
        json.dump(activities, f, ensure_ascii=False, indent=2)

def main():
    print("Starting Reddit scraper for Mexico City activities and food recommendations...")
    
    try:
        reddit = setup_reddit()
        print("Successfully connected to Reddit API")
        
        activities = get_activities(reddit)
        
        # Sort activities by score
        activities.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_activities = []
        for activity in activities:
            if activity['url'] not in seen_urls:
                seen_urls.add(activity['url'])
                unique_activities.append(activity)
        
        save_activities(unique_activities)
        print(f"Successfully scraped {len(unique_activities)} unique activities with comments!")
        print("Results saved to mexico_city_activities_with_comments.json")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your Reddit API credentials and try again.")

if __name__ == "__main__":
    main()
