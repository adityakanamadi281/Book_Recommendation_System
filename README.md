# Book Recommendation System

A comprehensive machine learning-based book recommendation system with multiple recommendation algorithms.

## Features

- **Popularity-Based Recommendations**: Recommends books with high average ratings and many ratings
- **Collaborative Filtering**: Item-based collaborative filtering using cosine similarity
- **Content-Based Filtering**: Recommends books based on content features (author, publisher, title)
- **Hybrid Approach**: Combines collaborative and content-based methods for better recommendations
- **User-Based Recommendations**: Personalized recommendations based on user rating history
- **Title-Based Search**: Search books by title and get recommendations

## Project Structure

```
Book_Recommendation_System/
├── data/
│   ├── Books (1).csv
│   ├── Users.csv
│   └── Ratings (1).csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── popularity_recommender.py
│   ├── collaborative_recommender.py
│   ├── content_recommender.py
│   ├── hybrid_recommender.py
│   └── recommendation_system.py # Main system class
├── notebook/
│   └── Book_Recommendation_System.ipynb
├── main.py                      # Demo script
├── run_recommendations.py       # Interactive script
├── config.py                    # Configuration file
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adityakanamadi281/Book_Recommendation_System.git
cd Book_Recommendation_System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Demo Script

Run the demo script to see examples of all recommendation methods:

```bash
python main.py
```

### Option 2: Interactive Mode

Run the interactive script for a user-friendly interface:

```bash
python run_recommendations.py
```

### Option 3: Use as a Library

```python
from src.recommendation_system import BookRecommendationSystem

# Initialize the system
system = BookRecommendationSystem(
    data_path="data",
    min_book_ratings=50,
    min_user_ratings=5
)

# Initialize (loads data and builds models)
system.initialize()

# Get popular books
popular_books = system.get_popular_books(n=10, min_ratings=50)

# Get recommendations by book title
recommendations = system.recommend_by_title(
    "Harry Potter", 
    n=10, 
    method='hybrid'
)

# Get recommendations by ISBN
recommendations = system.recommend_by_isbn(
    "034545104X", 
    n=10, 
    method='hybrid'
)

# Get recommendations for a user
user_recommendations = system.recommend_for_user(
    user_id=12345, 
    n=10, 
    method='hybrid'
)
```

## Recommendation Methods

### 1. Popularity-Based
Recommends books with the highest average ratings and most ratings.

```python
popular_books = system.get_popular_books(n=10, min_ratings=50)
```

### 2. Collaborative Filtering
Finds books similar to a given book based on user rating patterns.

```python
recommendations = system.recommend_by_isbn(isbn, n=10, method='collaborative')
```

### 3. Content-Based
Recommends books with similar features (author, publisher, title).

```python
recommendations = system.recommend_by_isbn(isbn, n=10, method='content')
```

### 4. Hybrid
Combines collaborative and content-based approaches.

```python
recommendations = system.recommend_by_isbn(isbn, n=10, method='hybrid')
```

## Configuration

Edit `config.py` to customize:
- Data paths
- Minimum ratings thresholds
- Recommendation parameters
- Hybrid weights
- TF-IDF parameters

## Data Requirements

The system expects three CSV files in the `data/` directory:
- **Books (1).csv**: Book information (ISBN, Title, Author, Publisher, etc.)
- **Users.csv**: User information (User-ID, Location, Age)
- **Ratings (1).csv**: Rating data (User-ID, ISBN, Book-Rating)

## Algorithm Details

### Collaborative Filtering
- Creates a user-item matrix from ratings
- Calculates item-item similarity using cosine similarity
- Recommends books similar to books the user has rated highly

### Content-Based Filtering
- Combines book features (author, publisher, title) into text
- Uses TF-IDF vectorization to create feature vectors
- Calculates cosine similarity between books
- Recommends books with similar content features

### Hybrid Approach
- Combines scores from both collaborative and content-based methods
- Uses weighted combination (default: 50% each)
- Provides more robust recommendations

## Dependencies

- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn

## Performance Notes

- Initial model building may take a few minutes for large datasets
- Collaborative filtering requires books with at least 50 ratings
- Content-based filtering works with all books in the dataset
- Hybrid approach provides the best balance of accuracy and coverage

## License

See LICENSE file for details.
