"""
Simple test script to verify the recommendation system works
"""

import sys
from src.recommendation_system import BookRecommendationSystem

def test_system():
    """Test the recommendation system."""
    print("Testing Book Recommendation System...")
    print("="*60)
    
    try:
        # Initialize
        system = BookRecommendationSystem(data_path="data")
        system.initialize()
        print("\n✓ System initialized successfully")
        
        # Test popular books
        print("\nTesting popularity-based recommendations...")
        popular = system.get_popular_books(n=5)
        assert not popular.empty, "Popular books should not be empty"
        print(f"✓ Got {len(popular)} popular book recommendations")
        
        # Test by ISBN
        print("\nTesting ISBN-based recommendations...")
        sample_isbn = system.books_with_rating['ISBN'].iloc[0]
        recs = system.recommend_by_isbn(sample_isbn, n=5, method='hybrid')
        assert not recs.empty, "Recommendations should not be empty"
        print(f"✓ Got {len(recs)} recommendations for ISBN {sample_isbn}")
        
        # Test by title
        print("\nTesting title-based recommendations...")
        sample_title = system.books_with_rating['Book-Title'].iloc[0]
        recs = system.recommend_by_title(sample_title[:20], n=5, method='hybrid')
        assert not recs.empty, "Title recommendations should not be empty"
        print(f"✓ Got {len(recs)} recommendations for title search")
        
        # Test user recommendations
        print("\nTesting user-based recommendations...")
        sample_user = system.filtered_ratings['User-ID'].iloc[0]
        recs = system.recommend_for_user(sample_user, n=5, method='hybrid')
        assert not recs.empty, "User recommendations should not be empty"
        print(f"✓ Got {len(recs)} recommendations for user {sample_user}")
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)

