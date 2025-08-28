import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from transformers import pipeline

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("YouTube API key not found. Please set it in the .env file.")

# YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

# -----------------------------
# Helper Functions
# -----------------------------
def get_video_id(url: str) -> str:
    """Extracts video ID from different YouTube URL formats."""
    parsed_url = urlparse(url)

    # Case 1: Standard watch?v= format
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/")[2]

    # Case 2: youtu.be short links
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    raise ValueError("Invalid YouTube URL format.")

def fetch_youtube_comments(video_url: str, max_results: int = 20):
    """Fetch top-level comments from a YouTube video."""
    video_id = get_video_id(video_url)
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment_text = snippet["textDisplay"]
            comments.append(comment_text)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# -----------------------------
# Toxicity Classifier
# -----------------------------
classifier = pipeline("text-classification", model="unitary/toxic-bert")

def analyze_comments(comments):
    """Run toxicity analysis on a list of comments."""
    results = []
    for text in comments:
        prediction = classifier(text)[0]  # returns [{'label': 'toxic', 'score': 0.95}]
        results.append({
            "comment": text,
            "label": prediction["label"],
            "score": round(prediction["score"], 4)
        })
    return results

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    video_link = input("Enter YouTube video URL: ").strip()
    try:
        comments = fetch_youtube_comments(video_link, max_results=20)
        print(f"\nRetrieved {len(comments)} comments. Running toxicity analysis...\n" + "-"*60)
        
        analyzed = analyze_comments(comments)
        for i, res in enumerate(analyzed, start=1):
            print(f"{i}. Comment: {res['comment']}")
            print(f"   → Prediction: {res['label']} (score: {res['score']})\n")
    except Exception as e:
        print("Error:", e)
