"""
Popularity-Based Recommendation System
Recommends books based on average ratings and number of ratings.
"""

import pandas as pd


class PopularityRecommender:
    """Popularity-based recommendation system."""
    
    def __init__(self, books_with_rating, book_rating_merged):
        """
        Initialize PopularityRecommender.
        
        Args:
            books_with_rating: DataFrame with books and their average ratings
            book_rating_merged: DataFrame with merged book and rating data
        """
        self.books_with_rating = books_with_rating
        self.book_rating_merged = book_rating_merged
        
    def recommend(self, n=10, min_ratings=50):
        """
        Get the most popular books.
        
        Args:
            n: Number of recommendations to return
            min_ratings: Minimum number of ratings required
            
        Returns:
            DataFrame with top N popular books
        """
        # Calculate number of ratings per book
        rating_counts = self.book_rating_merged.groupby('ISBN')['Book-Rating'].count().reset_index()
        rating_counts.rename(columns={'Book-Rating': 'Rating-Count'}, inplace=True)
        
        # Merge with books_with_rating
        popular_books = pd.merge(self.books_with_rating, rating_counts, on='ISBN')
        
        # Filter books with minimum ratings
        popular_books = popular_books[popular_books['Rating-Count'] >= min_ratings]
        
        # Sort by average rating and rating count
        popular_books = popular_books.sort_values(
            by=['Average-Rating', 'Rating-Count'], 
            ascending=[False, False]
        )
        
        return popular_books[[
            'ISBN', 'Book-Title', 'Book-Author', 
            'Average-Rating', 'Rating-Count', 'Publisher'
        ]].head(n)

