"""
Data Loading and Preprocessing Module
Handles loading and cleaning of book, user, and rating data.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class DataLoader:
    """Class to load and preprocess book recommendation data."""
    
    def __init__(self, data_path="data"):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.books = None
        self.users = None
        self.ratings = None
        self.books_with_rating = None
        self.book_rating_merged = None
        
    def load_data(self):
        """Load all datasets (books, users, ratings)."""
        try:
            self.books = pd.read_csv(f"{self.data_path}/Books (1).csv")
            self.users = pd.read_csv(f"{self.data_path}/Users.csv", encoding="latin1")
            self.ratings = pd.read_csv(f"{self.data_path}/Ratings (1).csv", encoding="latin1")
            print("✓ All datasets loaded successfully")
            return True
        except FileNotFoundError as e:
            print(f"✗ File Not Found: {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def preprocess_books(self):
        """Clean and preprocess books data."""
        if self.books is None:
            raise ValueError("Books data not loaded. Call load_data() first.")
        
        # Drop missing values in critical columns
        self.books = self.books.dropna(subset=['Book-Author', 'Publisher', 'Image-URL-L'])
        
        # Convert Year-Of-Publication to numeric
        self.books["Year-Of-Publication"] = pd.to_numeric(
            self.books["Year-Of-Publication"], errors="coerce"
        )
        
        # Remove invalid years
        self.books = self.books[
            (self.books["Year-Of-Publication"] >= 1900) &
            (self.books["Year-Of-Publication"] <= 2025) &
            (self.books["Year-Of-Publication"] != 0)
        ]
        
        print(f"✓ Books preprocessed: {self.books.shape[0]} books remaining")
    
    def preprocess_users(self):
        """Clean and preprocess users data."""
        if self.users is None:
            raise ValueError("Users data not loaded. Call load_data() first.")
        
        # Drop missing age values
        self.users = self.users.dropna(subset=['Age'])
        
        # Filter valid age range
        self.users = self.users[
            (self.users["Age"] >= 15) & 
            (self.users["Age"] <= 100) &
            (self.users["Age"] != 0)
        ]
        
        print(f"✓ Users preprocessed: {self.users.shape[0]} users remaining")
    
    def preprocess_ratings(self):
        """Clean and preprocess ratings data."""
        if self.ratings is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
        
        # Ratings are already clean, just merge with books
        self.book_rating_merged = pd.merge(self.ratings, self.books, on="ISBN")
        
        print(f"✓ Ratings preprocessed: {self.book_rating_merged.shape[0]} ratings")
    
    def create_books_with_rating(self):
        """Create books_with_rating dataframe with average ratings."""
        if self.book_rating_merged is None:
            raise ValueError("Ratings not preprocessed. Call preprocess_ratings() first.")
        
        # Calculate average ratings
        average_rating = pd.DataFrame(
            self.book_rating_merged.groupby('ISBN')['Book-Rating'].mean().round(1)
        )
        average_rating.reset_index(inplace=True)
        average_rating.rename(columns={'Book-Rating': 'Average-Rating'}, inplace=True)
        
        # Get unique average ratings
        average_rating_unique = average_rating.drop_duplicates(subset=['ISBN'])
        
        # Merge with books
        self.books_with_rating = pd.merge(
            self.books, 
            average_rating_unique, 
            on='ISBN', 
            how='inner'
        )
        
        # Select relevant columns
        self.books_with_rating = self.books_with_rating[[
            'ISBN', 'Book-Title', 'Book-Author', 'Average-Rating',
            'Year-Of-Publication', 'Publisher', 'Image-URL-S', 
            'Image-URL-M', 'Image-URL-L'
        ]]
        
        print(f"✓ Books with ratings created: {self.books_with_rating.shape[0]} books")
    
    def get_filtered_ratings(self, min_book_ratings=50, min_user_ratings=5):
        """
        Get filtered ratings for collaborative filtering.
        
        Args:
            min_book_ratings: Minimum number of ratings per book
            min_user_ratings: Minimum number of ratings per user
            
        Returns:
            Filtered ratings dataframe
        """
        if self.book_rating_merged is None:
            raise ValueError("Ratings not preprocessed. Call preprocess_ratings() first.")
        
        # Filter books with sufficient ratings
        book_counts = self.book_rating_merged['ISBN'].value_counts()
        valid_books = book_counts[book_counts >= min_book_ratings].index
        
        # Filter users with sufficient ratings
        user_counts = self.book_rating_merged['User-ID'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        
        # Filter the data
        filtered_ratings = self.book_rating_merged[
            (self.book_rating_merged['ISBN'].isin(valid_books)) & 
            (self.book_rating_merged['User-ID'].isin(valid_users))
        ]
        
        print(f"✓ Filtered ratings: {filtered_ratings.shape[0]} ratings "
              f"({filtered_ratings['User-ID'].nunique()} users, "
              f"{filtered_ratings['ISBN'].nunique()} books)")
        
        return filtered_ratings
    
    def process_all(self):
        """Run all preprocessing steps in order."""
        if not self.load_data():
            return False
        
        self.preprocess_books()
        self.preprocess_users()
        self.preprocess_ratings()
        self.create_books_with_rating()
        
        return True

