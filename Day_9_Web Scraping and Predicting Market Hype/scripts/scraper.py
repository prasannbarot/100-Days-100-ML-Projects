import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()]
)

class RedditScraper:
    def __init__(self, subreddit="wallstreetbets", user_agent="StockSentimentBot/2.0"):
        self.base_url = "https://www.reddit.com"
        self.subreddit = subreddit
        self.session = self._init_session(user_agent)
        self.raw_data_path = Path("data/raw")
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def _init_session(self, user_agent):
        """Creates a resilient session with automatic retries."""
        session = requests.Session()
        session.headers.update({'User-Agent': user_agent})
        
        # Retry strategy: Wait 1s, 2s, 4s... on 429 (Too Many Requests) or 5xx errors
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch_threads(self, days=7):
        """Finds relevant discussion threads from the last N days."""
        # Expanded search queries to get more data
        queries = [
            f"title:\"Daily Discussion\" subreddit:{self.subreddit}",
            f"title:\"What Are Your Moves Tomorrow\" subreddit:{self.subreddit}",
            f"title:\"Weekend Discussion\" subreddit:{self.subreddit}"
        ]
        
        threads = []
        seen_ids = set()

        logging.info(f" Searching threads for the last {days} days...")
        
        for query in queries:
            try:
                # 'q' is query, 'sort' is new to get latest, 'limit' increased to 50
                url = f"{self.base_url}/r/{self.subreddit}/search.json?q={query}&restrict_sr=on&sort=new&limit=50&raw_json=1"
                response = self.session.get(url)
                response.raise_for_status()
                
                posts = response.json().get('data', {}).get('children', [])
                
                for post in posts:
                    data = post['data']
                    pid = data['id']
                    created = datetime.fromtimestamp(data['created_utc'])
                    
                    # Filter by date and uniqueness
                    if (datetime.now() - created).days <= days and pid not in seen_ids:
                        threads.append({
                            'post_id': pid,
                            'title': data['title'],
                            'created': created,
                            'url': f"{self.base_url}{data['permalink']}"
                        })
                        seen_ids.add(pid)
            except Exception as e:
                logging.error(f"Failed to fetch threads for query '{query}': {e}")

        logging.info(f" Found {len(threads)} relevant threads.")
        return threads

    def fetch_comments_for_thread(self, thread):
        """Fetches comments for a single thread (worker function)."""
        url = f"{self.base_url}/comments/{thread['post_id']}.json?raw_json=1&limit=500" # Limit increased
        extracted = []
        
        try:
            response = self.session.get(url)
            # No manual raise_for_status here; let the retry adapter handle soft fails, 
            # but if it fails hard, we catch it.
            if response.status_code != 200:
                return []

            data = response.json()
            # Reddit JSON is [post_info, comments_info]
            if len(data) < 2: 
                return []
                
            comments = data[1]['data']['children']
            
            for c in comments:
                if c['kind'] == 't1' and 'body' in c['data']:
                    body = c['data']['body']
                    # Filter out AutoModerator and deleted comments
                    if c['data']['author'] != 'AutoModerator' and body != '[deleted]':
                        extracted.append({
                            'date': thread['created'].date(),
                            'post_title': thread['title'],
                            'comment': body,
                            'score': c['data'].get('score', 0), # Added score (useful feature)
                            'post_url': thread['url']
                        })
        except Exception as e:
            logging.warning(f" Error scraping thread {thread['post_id']}: {e}")
            
        return extracted

    def run(self, days=7):
        threads = self.fetch_threads(days)
        if not threads:
            logging.warning("No threads found. Exiting.")
            return

        all_comments = []
        
        # Parallel Execution: Scrape multiple threads at once
        # Max_workers=5 is safe for Reddit (avoids aggressive rate limiting)
        logging.info(f" Scraping {len(threads)} threads using parallel workers...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_thread = {executor.submit(self.fetch_comments_for_thread, t): t for t in threads}
            
            for future in as_completed(future_to_thread):
                result = future.result()
                if result:
                    all_comments.extend(result)
                    
        # Save to CSV
        if all_comments:
            df = pd.DataFrame(all_comments)
            # Remove duplicates (sometimes stickied comments appear twice)
            df.drop_duplicates(subset=['comment', 'date'], inplace=True)
            
            output_path = self.raw_data_path / "scraped_posts.csv"
            
            # If file exists, append to it instead of overwriting (optional, good for scaling)
            # For this logic, we will overwrite to keep it simple but scalable
            df.to_csv(output_path, index=False)
            logging.info(f" Success! Saved {len(df)} comments to {output_path}")
        else:
            logging.warning("No comments extracted.")

if __name__ == "__main__":
    # How to run:
    scraper = RedditScraper()
    scraper.run(days=28) # Scrape last 4 weeks for more data