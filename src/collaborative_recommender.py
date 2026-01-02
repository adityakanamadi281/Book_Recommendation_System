"""
Collaborative Filtering Recommendation System
Item-based collaborative filtering using cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


class CollaborativeRecommender:
    """Item-based collaborative filtering recommendation system."""
    
    def __init__(self, filtered_ratings, books_with_rating):
        """
        Initialize CollaborativeRecommender.
        
        Args:
            filtered_ratings: Filtered ratings dataframe
            books_with_rating: DataFrame with books and their average ratings
        """
        self.filtered_ratings = filtered_ratings
        self.books_with_rating = books_with_rating
        self.user_item_matrix = None
        self.item_similarity_df = None
        self._build_model()
    
    def _build_model(self):
        """Build the user-item matrix and similarity matrix."""
        print("Building collaborative filtering model...")
        
        # Create user-item matrix
        self.user_item_matrix = self.filtered_ratings.pivot_table(
            index='ISBN', 
            columns='User-ID', 
            values='Book-Rating',
            fill_value=0
        )
        
        print(f"  User-item matrix shape: {self.user_item_matrix.shape}")
        sparsity = (1 - (self.user_item_matrix != 0).sum().sum() / 
                   (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100
        print(f"  Sparsity: {sparsity:.2f}%")
        
        # Calculate item-item similarity
        print("  Calculating item-item similarity...")
        item_similarity = cosine_similarity(self.user_item_matrix)
        self.item_similarity_df = pd.DataFrame(
            item_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print("✓ Collaborative filtering model built")
    
    def recommend(self, isbn, n=10):
        """
        Get book recommendations using collaborative filtering.
        
        Args:
            isbn: ISBN of the book to find similar books for
            n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books
        """
        if isbn not in self.item_similarity_df.index:
            print(f"ISBN {isbn} not found in the model")
            return pd.DataFrame()
        
        # Get similarity scores
        similar_books = self.item_similarity_df[isbn].sort_values(ascending=False)
        
        # Remove the book itself
        similar_books = similar_books[similar_books.index != isbn]
        
        # Get top N similar books
        top_similar_isbns = similar_books.head(n).index.tolist()
        
        # Get book details
        recommendations = self.books_with_rating[
            self.books_with_rating['ISBN'].isin(top_similar_isbns)
        ].copy()
        
        # Add similarity scores
        similarity_scores = similar_books[top_similar_isbns].values
        recommendations['Similarity-Score'] = similarity_scores
        
        # Sort by similarity score
        recommendations = recommendations.sort_values('Similarity-Score', ascending=False)
        
        return recommendations[[
            'ISBN', 'Book-Title', 'Book-Author', 
            'Average-Rating', 'Similarity-Score', 'Publisher'
        ]]

