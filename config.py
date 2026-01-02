"""
Configuration file for the Book Recommendation System
"""

# Data paths
DATA_PATH = "data"
BOOKS_FILE = "Books (1).csv"
USERS_FILE = "Users.csv"
RATINGS_FILE = "Ratings (1).csv"

# Data preprocessing parameters
MIN_BOOK_RATINGS = 50  # Minimum ratings per book for collaborative filtering
MIN_USER_RATINGS = 5   # Minimum ratings per user for collaborative filtering

# Book filtering parameters
MIN_YEAR = 1900
MAX_YEAR = 2025
MIN_AGE = 15
MAX_AGE = 100

# Recommendation parameters
DEFAULT_N_RECOMMENDATIONS = 10
DEFAULT_MIN_RATINGS = 50

# Hybrid recommendation weights
COLLAB_WEIGHT = 0.5
CONTENT_WEIGHT = 0.5

# TF-IDF parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = 'english'

