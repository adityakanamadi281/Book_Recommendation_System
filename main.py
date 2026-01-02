"""
Main script to run the Book Recommendation System
"""

from src.recommendation_system import BookRecommendationSystem
import pandas as pd


def main():
    """Main function to demonstrate the recommendation system."""
    
    # Initialize the system
    print("\n" + "="*80)
    print("BOOK RECOMMENDATION SYSTEM")
    print("="*80 + "\n")
    
    system = BookRecommendationSystem(
        data_path="data",
        min_book_ratings=50,
        min_user_ratings=5
    )
    
    # Initialize (load data and build models)
    system.initialize()
    
    # Example 1: Get popular books
    print("\n" + "="*80)
    print("EXAMPLE 1: Popular Books")
    print("="*80)
    popular_books = system.get_popular_books(n=10, min_ratings=50)
    print(popular_books.to_string(index=False))
    
    # Example 2: Get recommendations by book title
    print("\n" + "="*80)
    print("EXAMPLE 2: Recommendations by Book Title")
    print("="*80)
    
    # Get a sample book title
    sample_title = system.books_with_rating['Book-Title'].iloc[0]
    print(f"\nSearching for: '{sample_title[:50]}...'")
    
    recommendations = system.recommend_by_title(
        sample_title[:30], 
        n=10, 
        method='hybrid'
    )
    
    if not recommendations.empty:
        print(f"\nTop 10 Recommendations:")
        print(recommendations.to_string(index=False))
    
    # Example 3: Get recommendations by ISBN
    print("\n" + "="*80)
    print("EXAMPLE 3: Recommendations by ISBN")
    print("="*80)
    
    sample_isbn = system.books_with_rating['ISBN'].iloc[0]
    sample_book = system.books_with_rating[
        system.books_with_rating['ISBN'] == sample_isbn
    ]
    
    if not sample_book.empty:
        print(f"\nBook: {sample_book['Book-Title'].values[0]}")
        print(f"Author: {sample_book['Book-Author'].values[0]}")
        print(f"ISBN: {sample_isbn}")
        
        # Get recommendations using different methods
        methods = ['collaborative', 'content', 'hybrid']
        for method in methods:
            print(f"\n{method.upper()} Recommendations:")
            recs = system.recommend_by_isbn(sample_isbn, n=5, method=method)
            if not recs.empty:
                print(recs[['Book-Title', 'Book-Author', 'Average-Rating']].to_string(index=False))
    
    # Example 4: Get recommendations for a user
    print("\n" + "="*80)
    print("EXAMPLE 4: User-Based Recommendations")
    print("="*80)
    
    sample_user_id = system.filtered_ratings['User-ID'].iloc[0]
    print(f"\nUser ID: {sample_user_id}")
    
    # Show user's rated books
    user_books = system.filtered_ratings[
        system.filtered_ratings['User-ID'] == sample_user_id
    ].merge(
        system.books_with_rating[['ISBN', 'Book-Title', 'Book-Author']], 
        on='ISBN'
    )[['Book-Title', 'Book-Author', 'Book-Rating']].head(5)
    
    print("\nUser's Rated Books:")
    print(user_books.to_string(index=False))
    
    # Get recommendations
    user_recs = system.recommend_for_user(sample_user_id, n=10, method='hybrid')
    if not user_recs.empty:
        print("\nRecommended Books for User:")
        print(user_recs[['Book-Title', 'Book-Author', 'Average-Rating']].to_string(index=False))
    
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

