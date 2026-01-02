"""
Content-Based Recommendation System
Recommends books based on content features (author, publisher, title).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


class ContentRecommender:
    """Content-based recommendation system."""
    
    def __init__(self, books_with_rating):
        """
        Initialize ContentRecommender.
        
        Args:
            books_with_rating: DataFrame with books and their average ratings
        """
        self.books_with_rating = books_with_rating
        self.tfidf_matrix = None
        self.content_similarity_df = None
        self._build_model()
    
    def _build_model(self):
        """Build the TF-IDF matrix and similarity matrix."""
        print("Building content-based model...")
        
        # Combine book features into a single text feature
        self.books_with_rating['Content'] = (
            self.books_with_rating['Book-Author'].fillna('') + ' ' +
            self.books_with_rating['Publisher'].fillna('') + ' ' +
            self.books_with_rating['Book-Title'].fillna('')
        )
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.books_with_rating['Content'])
        
        print(f"  TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Calculate cosine similarity
        print("  Calculating content similarity...")
        content_similarity = cosine_similarity(self.tfidf_matrix)
        self.content_similarity_df = pd.DataFrame(
            content_similarity,
            index=self.books_with_rating['ISBN'],
            columns=self.books_with_rating['ISBN']
        )
        
        print("✓ Content-based model built")
    
    def recommend(self, isbn, n=10):
        """
        Get book recommendations using content-based filtering.
        
        Args:
            isbn: ISBN of the book to find similar books for
            n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books
        """
        if isbn not in self.content_similarity_df.index:
            print(f"ISBN {isbn} not found in the model")
            return pd.DataFrame()
        
        # Get similarity scores
        similar_books = self.content_similarity_df[isbn].sort_values(ascending=False)
        
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

