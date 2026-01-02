"""
Interactive script to get book recommendations
"""

from src.recommendation_system import BookRecommendationSystem
import sys


def interactive_mode():
    """Run the recommendation system in interactive mode."""
    
    print("\n" + "="*80)
    print("BOOK RECOMMENDATION SYSTEM - Interactive Mode")
    print("="*80 + "\n")
    
    # Initialize system
    print("Initializing system...")
    system = BookRecommendationSystem(
        data_path="data",
        min_book_ratings=50,
        min_user_ratings=5
    )
    
    try:
        system.initialize()
        print("\n✓ System ready!\n")
    except Exception as e:
        print(f"\n✗ Error initializing system: {e}")
        return
    
    while True:
        print("\n" + "-"*80)
        print("OPTIONS:")
        print("1. Get popular books")
        print("2. Get recommendations by book title")
        print("3. Get recommendations by ISBN")
        print("4. Get recommendations for a user")
        print("5. Exit")
        print("-"*80)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            try:
                n = int(input("Number of recommendations (default 10): ") or "10")
                min_ratings = int(input("Minimum ratings (default 50): ") or "50")
                recs = system.get_popular_books(n=n, min_ratings=min_ratings)
                print("\n" + "="*80)
                print("POPULAR BOOKS")
                print("="*80)
                print(recs.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            try:
                title = input("Enter book title (or partial title): ").strip()
                if not title:
                    print("Please enter a book title")
                    continue
                
                method = input("Method (popularity/collaborative/content/hybrid, default hybrid): ").strip() or "hybrid"
                n = int(input("Number of recommendations (default 10): ") or "10")
                
                recs = system.recommend_by_title(title, n=n, method=method)
                if not recs.empty:
                    print("\n" + "="*80)
                    print("RECOMMENDATIONS")
                    print("="*80)
                    print(recs.to_string(index=False))
                else:
                    print("No recommendations found")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            try:
                isbn = input("Enter ISBN: ").strip()
                if not isbn:
                    print("Please enter an ISBN")
                    continue
                
                method = input("Method (popularity/collaborative/content/hybrid, default hybrid): ").strip() or "hybrid"
                n = int(input("Number of recommendations (default 10): ") or "10")
                
                recs = system.recommend_by_isbn(isbn, n=n, method=method)
                if not recs.empty:
                    print("\n" + "="*80)
                    print("RECOMMENDATIONS")
                    print("="*80)
                    print(recs.to_string(index=False))
                else:
                    print("No recommendations found")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            try:
                user_id = input("Enter User ID: ").strip()
                if not user_id:
                    print("Please enter a User ID")
                    continue
                
                try:
                    user_id = int(user_id)
                except ValueError:
                    print("User ID must be a number")
                    continue
                
                method = input("Method (popularity/collaborative/content/hybrid, default hybrid): ").strip() or "hybrid"
                n = int(input("Number of recommendations (default 10): ") or "10")
                
                recs = system.recommend_for_user(user_id, n=n, method=method)
                if not recs.empty:
                    print("\n" + "="*80)
                    print("USER RECOMMENDATIONS")
                    print("="*80)
                    print(recs.to_string(index=False))
                else:
                    print("No recommendations found")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            print("\nThank you for using the Book Recommendation System!")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)

