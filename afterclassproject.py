import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time
import sys

init(autoreset=True)

def load_data(file_path="imdb_top_1000.csv"):
    try:
        df = pd.read_csv(file_path)
        df['combined_features'] = df['Genre'].fillna('') + " " + df['Overview'].fillna('')
        return df
    except FileNotFoundError:
        print(Fore.RED + "Error: The file 'imdb_top_1000.csv' was not found.")
        return None

movies_df = load_data()
if movies_df is None or movies_df.empty:
    print(Fore.RED + "No movie data available. Exiting.")
    sys.exit(1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def list_genres(df):
    return sorted(set(genre.strip() for sublist in df['Genre'].dropna().str.split(',') for genre in sublist))

genres = list_genres(movies_df)

def get_movie_recommendations(genre=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df.copy()

    if genre:
        if not any(genre.lower() == g.lower() for g in genres):
            print(Fore.RED + f"Error: Genre '{genre}' is not recognized.")
            return []
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre, case=False, na=False)]

    if rating:
        filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= rating]

    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

    recommendations = []
    for _, row in filtered_df.iterrows():
        overview = row['Overview']
        polarity = TextBlob(overview).sentiment.polarity
        if (mood == "positive" and polarity > 0) or (mood == "negative" and polarity < 0) or (mood == "neutral") or (mood is None):
            recommendations.append((row['Series_Title'], polarity))
        if len(recommendations) >= top_n:
            break

    return recommendations if recommendations else "No recommendations found."

def display_recommendations(recs, name):
    print(Fore.CYAN + f"\n✨ AI-analyzed Movie Recommendations for {name}: ✨")
    if isinstance(recs, list):
        for i, (title, polarity) in enumerate(recs, 1):
            row = movies_df[movies_df['Series_Title'] == title].iloc[0]
            sentiment = "😊 Positive" if polarity > 0 else "😞 Negative" if polarity < 0 else "😐 Neutral"
            print(f"{i}. {title} | Genre: {row['Genre']} | IMDB: {row['IMDB_Rating']} (Polarity: {polarity:.2f}, Sentiment: {sentiment})")
    else:
        print(Fore.RED + recs)

def processing_animation():
    for i in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)
    print("")

def handle_ai(name):
    print(Fore.BLUE + f"\n🤖 Let's find the perfect movie for you!\n")
    print(Fore.GREEN + "\nAvailable Genres:", end="")
    for i, genre in enumerate(genres, 1):
        print(f" {i}. {genre}", end="")
    print("\n" + "-" * 50)

    while True:
        genre_input = input(Fore.YELLOW + "Enter genre number or name: ").strip()
        try:
            genre_id = int(genre_input)
            if 1 <= genre_id <= len(genres):
                genre = genres[genre_id - 1]
                break
        except ValueError:
            matches = [g for g in genres if genre_input.lower() in g.lower()]
            if matches:
                genre = matches[0]
                break
        print(Fore.RED + "Invalid input. Try again.\n")

    mood = input(Fore.YELLOW + "How do you feel today? (Describe your mood): ").strip()
    print(Fore.BLUE + "\nAnalyzing mood", end="", flush=True)
    processing_animation()
    polarity = TextBlob(mood).sentiment.polarity
    mood = 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
    print(f"Mood detected: {mood.capitalize()} (Polarity: {polarity:.2f})\n")

    while True:
        rating_input = input(Fore.YELLOW + "Enter minimum IMDB rating (7.0–9.3) or 'skip': ").strip()
        if rating_input.lower() == 'skip':
            rating = None
            break
        try:
            rating = float(rating_input)
            if 7.0 <= rating <= 9.3:
                break
            else:
                print(Fore.RED + "Rating out of range. Try again.\n")
        except ValueError:
            print(Fore.RED + "Invalid input. Try again.\n")

    print(Fore.BLUE + f"\nFinding movies for {name}...", end="", flush=True)
    processing_animation()
    recs = get_movie_recommendations(genre=genre, mood=mood, rating=rating, top_n=5)
    display_recommendations(recs, name)

    while True:
        action = input(Fore.YELLOW + "\nWould you like more recommendations? (yes/no): ").strip().lower()
        if action == 'no':
            print(Fore.GREEN + "🎬 Enjoy your movie picks, " + name + "! 🍿")
            break
        elif action == 'yes':
            recs = get_movie_recommendations(genre=genre, mood=mood, rating=rating, top_n=5)
            display_recommendations(recs, name)
        else:
            print(Fore.RED + "Invalid choice. Try again.\n")

def main():
    print(Fore.BLUE + "\nWelcome to your Personal Movie Recommendation Assistant! 🎥\n")
    name = input(Fore.YELLOW + "What's your name? ").strip()
    print(Fore.GREEN + f"Great to meet you, {name}!\n")
    handle_ai(name)

if __name__ == "__main__":
    main()