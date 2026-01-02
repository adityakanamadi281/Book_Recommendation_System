"""
Hybrid Recommendation System
Combines collaborative and content-based filtering.
"""

import pandas as pd
from .collaborative_recommender import CollaborativeRecommender
from .content_recommender import ContentRecommender


class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches."""
    
    def __init__(self, collaborative_recommender, content_recommender):
        """
        Initialize HybridRecommender.
        
        Args:
            collaborative_recommender: CollaborativeRecommender instance
            content_recommender: ContentRecommender instance
        """
        self.collab_recommender = collaborative_recommender
        self.content_recommender = content_recommender
    
    def recommend(self, isbn, n=10, collab_weight=0.5, content_weight=0.5):
        """
        Get book recommendations using hybrid approach.
        
        Args:
            isbn: ISBN of the book to find similar books for
            n: Number of recommendations to return
            collab_weight: Weight for collaborative filtering scores
            content_weight: Weight for content-based scores
            
        Returns:
            DataFrame with recommended books
        """
        # Get recommendations from both methods
        collab_recs = self.collab_recommender.recommend(isbn, n=n*2)
        content_recs = self.content_recommender.recommend(isbn, n=n*2)
        
        # If either is empty, return the non-empty one
        if collab_recs.empty:
            return content_recs.head(n)
        if content_recs.empty:
            return collab_recs.head(n)
        
        # Merge recommendations
        all_isbns = set(collab_recs['ISBN'].tolist()) | set(content_recs['ISBN'].tolist())
        
        hybrid_scores = {}
        for rec_isbn in all_isbns:
            collab_score = 0
            content_score = 0
            
            if rec_isbn in collab_recs['ISBN'].values:
                collab_score = collab_recs[
                    collab_recs['ISBN'] == rec_isbn
                ]['Similarity-Score'].values[0]
            
            if rec_isbn in content_recs['ISBN'].values:
                content_score = content_recs[
                    content_recs['ISBN'] == rec_isbn
                ]['Similarity-Score'].values[0]
            
            # Weighted combination
            hybrid_score = (collab_weight * collab_score) + (content_weight * content_score)
            hybrid_scores[rec_isbn] = hybrid_score
        
        # Sort by hybrid score
        sorted_isbns = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_isbns = [isbn for isbn, score in sorted_isbns[:n]]
        
        # Get book details
        recommendations = self.collab_recommender.books_with_rating[
            self.collab_recommender.books_with_rating['ISBN'].isin(top_isbns)
        ].copy()
        
        recommendations['Hybrid-Score'] = [hybrid_scores[isbn] for isbn in recommendations['ISBN']]
        recommendations = recommendations.sort_values('Hybrid-Score', ascending=False)
        
        return recommendations[[
            'ISBN', 'Book-Title', 'Book-Author', 
            'Average-Rating', 'Hybrid-Score', 'Publisher'
        ]]

