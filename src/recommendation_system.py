"""
Main Recommendation System
Unified interface for all recommendation methods.
"""

import pandas as pd
from .data_loader import DataLoader
from .popularity_recommender import PopularityRecommender
from .collaborative_recommender import CollaborativeRecommender
from .content_recommender import ContentRecommender
from .hybrid_recommender import HybridRecommender


class BookRecommendationSystem:
    """Main recommendation system class."""
    
    def __init__(self, data_path="data", min_book_ratings=50, min_user_ratings=5):
        """
        Initialize the recommendation system.
        
        Args:
            data_path: Path to the data directory
            min_book_ratings: Minimum number of ratings per book for collaborative filtering
            min_user_ratings: Minimum number of ratings per user for collaborative filtering
        """
        self.data_path = data_path
        self.min_book_ratings = min_book_ratings
        self.min_user_ratings = min_user_ratings
        
        # Initialize components
        self.data_loader = DataLoader(data_path)
        self.popularity_recommender = None
        self.collaborative_recommender = None
        self.content_recommender = None
        self.hybrid_recommender = None
        
        self.books_with_rating = None
        self.filtered_ratings = None
    
    def initialize(self):
        """Load and preprocess data, build all models."""
        print("=" * 60)
        print("Initializing Book Recommendation System")
        print("=" * 60)
        
        # Load and preprocess data
        if not self.data_loader.process_all():
            raise RuntimeError("Failed to load or preprocess data")
        
        self.books_with_rating = self.data_loader.books_with_rating
        self.filtered_ratings = self.data_loader.get_filtered_ratings(
            min_book_ratings=self.min_book_ratings,
            min_user_ratings=self.min_user_ratings
        )
        
        # Initialize recommenders
        print("\n" + "=" * 60)
        print("Building Recommendation Models")
        print("=" * 60)
        
        self.popularity_recommender = PopularityRecommender(
            self.books_with_rating,
            self.data_loader.book_rating_merged
        )
        
        self.collaborative_recommender = CollaborativeRecommender(
            self.filtered_ratings,
            self.books_with_rating
        )
        
        self.content_recommender = ContentRecommender(
            self.books_with_rating
        )
        
        self.hybrid_recommender = HybridRecommender(
            self.collaborative_recommender,
            self.content_recommender
        )
        
        print("\n" + "=" * 60)
        print("✓ System Initialized Successfully")
        print("=" * 60)
    
    def get_popular_books(self, n=10, min_ratings=50):
        """Get popular book recommendations."""
        return self.popularity_recommender.recommend(n=n, min_ratings=min_ratings)
    
    def recommend_by_isbn(self, isbn, n=10, method='hybrid'):
        """
        Get recommendations for a specific book by ISBN.
        
        Args:
            isbn: ISBN of the book
            n: Number of recommendations
            method: 'popularity', 'collaborative', 'content', or 'hybrid'
        """
        if method == 'popularity':
            return self.popularity_recommender.recommend(n=n)
        elif method == 'collaborative':
            return self.collaborative_recommender.recommend(isbn, n=n)
        elif method == 'content':
            return self.content_recommender.recommend(isbn, n=n)
        elif method == 'hybrid':
            return self.hybrid_recommender.recommend(isbn, n=n)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def recommend_by_title(self, book_title, n=10, method='hybrid'):
        """
        Get recommendations based on book title.
        
        Args:
            book_title: Title of the book (can be partial)
            n: Number of recommendations
            method: 'popularity', 'collaborative', 'content', or 'hybrid'
        """
        # Find books matching the title
        matching_books = self.books_with_rating[
            self.books_with_rating['Book-Title'].str.contains(
                book_title, case=False, na=False
            )
        ]
        
        if matching_books.empty:
            print(f"No books found matching '{book_title}'")
            return pd.DataFrame()
        
        # Use the first matching book
        isbn = matching_books.iloc[0]['ISBN']
        book_title_full = matching_books.iloc[0]['Book-Title']
        
        print(f"Found book: {book_title_full}")
        print(f"Getting recommendations using {method} method...\n")
        
        return self.recommend_by_isbn(isbn, n=n, method=method)
    
    def recommend_for_user(self, user_id, n=10, method='hybrid'):
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            method: 'popularity', 'collaborative', 'content', or 'hybrid'
        """
        # Get books the user has already rated
        user_rated_books = self.filtered_ratings[
            self.filtered_ratings['User-ID'] == user_id
        ]['ISBN'].tolist()
        
        if not user_rated_books:
            # If user has no ratings, return popular books
            return self.popularity_recommender.recommend(n=n)
        
        # Get recommendations based on user's highest rated books
        user_ratings = self.filtered_ratings[
            self.filtered_ratings['User-ID'] == user_id
        ]
        user_ratings = user_ratings.sort_values('Book-Rating', ascending=False)
        
        # Get top 3 books the user rated highly
        top_user_books = user_ratings.head(3)['ISBN'].tolist()
        
        if method == 'popularity':
            return self.popularity_recommender.recommend(n=n)
        
        # Get recommendations for each top book and combine
        all_recommendations = []
        for isbn in top_user_books:
            if method == 'collaborative':
                recs = self.collaborative_recommender.recommend(isbn, n=n)
            elif method == 'content':
                recs = self.content_recommender.recommend(isbn, n=n)
            else:  # hybrid
                recs = self.hybrid_recommender.recommend(isbn, n=n)
            
            if not recs.empty:
                all_recommendations.append(recs)
        
        if all_recommendations:
            combined = pd.concat(all_recommendations, ignore_index=True)
            # Remove books user already rated
            combined = combined[~combined['ISBN'].isin(user_rated_books)]
            # Remove duplicates
            combined = combined.drop_duplicates(subset='ISBN')
            return combined.head(n)
        
        return pd.DataFrame()

