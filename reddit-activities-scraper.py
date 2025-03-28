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

def get_travel_info(reddit):
    """Scrape travel information and attractions from Mexico City subreddits"""
    subreddits = [
        'MexicoCity',
        'CDMX',
        'MexicoCityTravel',
        'travel',
        'TravelHacks'
    ]
    
    travel_posts = []
    
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            print(f"Searching in r/{subreddit_name}...")
            
            # Search for posts about travel and attractions
            search_terms = [
                'places to visit Mexico City',
                'Mexico City attractions',
                'things to do CDMX',
                'Mexico City travel',
                'must see Mexico City',
                'Mexico City museums',
                'Mexico City day trips',
                'Mexico City neighborhoods',
                'Mexico City walking tour',
                'Mexico City landmarks',
                'Mexico City markets',
                'Mexico City parks',
                'Mexico City cultural sites',
                'Mexico City hidden gems'
            ]
            
            for term in search_terms:
                try:
                    print(f"Searching for term: '{term}'")
                    for submission in subreddit.search(term, limit=15):
                        if submission.score > 5:  # Only include posts with some engagement
                            # Get comments for this submission
                            print(f"Fetching comments for: {submission.title}")
                            comments = get_comments(submission)
                            
                            post = {
                                'title': submission.title,
                                'url': submission.url,
                                'score': submission.score,
                                'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                                'subreddit': subreddit_name,
                                'text': submission.selftext[:800] if submission.selftext else '',  # First 800 chars
                                'num_comments': submission.num_comments,
                                'search_term': term,  # Include the search term that found this post
                                'comments': comments  # Add the comments to the data
                            }
                            travel_posts.append(post)
                except Exception as e:
                    print(f"Error searching term '{term}' in r/{subreddit_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error accessing subreddit r/{subreddit_name}: {str(e)}")
            continue
    
    return travel_posts

def extract_attractions(posts):
    """Extract mentioned attractions from posts and comments"""
    # This function can be expanded to use NLP techniques to extract specific locations
    # For now, we're just keeping track of where attractions are mentioned
    
    for post in posts:
        # Add a flag for key terms in the post title
        post['contains_attraction_names'] = any(term in post['title'].lower() for term in [
            'museum', 'park', 'castle', 'pyramid', 'temple', 'market', 'plaza', 'palace',
            'cathedral', 'teotihuacan', 'chapultepec', 'zocalo', 'anthropology', 'frida',
            'reforma', 'xochimilco', 'coyoacan', 'polanco', 'condesa', 'roma'
        ])
    
    return posts

def save_travel_info(travel_posts):
    """Save travel information to a JSON file"""
    with open('mexico_city_travel_attractions.json', 'w', encoding='utf-8') as f:
        json.dump(travel_posts, f, ensure_ascii=False, indent=2)

def main():
    print("Starting Reddit scraper for Mexico City travel and attractions...")
    
    try:
        reddit = setup_reddit()
        print("Successfully connected to Reddit API")
        
        travel_posts = get_travel_info(reddit)
        print(f"Collected {len(travel_posts)} posts about Mexico City travel and attractions")
        
        # Process and enrich the data
        processed_posts = extract_attractions(travel_posts)
        
        # Sort posts by score
        processed_posts.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_posts = []
        for post in processed_posts:
            if post['url'] not in seen_urls:
                seen_urls.add(post['url'])
                unique_posts.append(post)
        
        save_travel_info(unique_posts)
        print(f"Successfully scraped {len(unique_posts)} unique posts about Mexico City travel!")
        print("Results saved to mexico_city_travel_attractions.json")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your Reddit API credentials and try again.")

if __name__ == "__main__":
    main()
