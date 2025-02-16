from knn import find_nearest_users
from cosinesim_clean import rank_users_by_similarity
from openai import OpenAI
from typing import List, Tuple

def get_best_matches(
    query_features: List[float],
    query_text: str,
    user_features: List[List[float]],
    user_texts: List[str],
    user_ids: List[str],
    n_neighbors: int = 50,
    client: OpenAI = None
) -> List[Tuple[str, float]]:
    """
    Find best matches using a two-step process:
    1. Use KNN to find nearest neighbors based on feature vectors
    2. Rank those neighbors using cosine similarity of text embeddings
    
    Args:
        query_features: feature vector of the query user
        query_text: text description of the query user
        user_features: list of user feature vectors
        user_texts: list of user text descriptions
        user_ids: list of user IDs
        n_neighbors: number of neighbors to consider in KNN step
        client: OpenAI client instance
    
    Returns:
        list of (user_id, similarity_score) tuples, sorted by final similarity
    """
    # Step 1: Get nearest neighbors using KNN
    _, neighbor_ids = find_nearest_users(user_features, user_ids, query_features, n_neighbors)
    
    # Get corresponding texts for the nearest neighbors
    neighbor_texts = [user_texts[user_ids.index(uid)] for uid in neighbor_ids]
    
    # Step 2: Rank nearest neighbors by text similarity
    if client is None:
        client = OpenAI()
        
    ranked_matches = rank_users_by_similarity(query_text, neighbor_texts, neighbor_ids, client)
    
    return ranked_matches

# Example usage:
if __name__ == "__main__":
    import numpy as np
    
    # Example data
    client = OpenAI()  # Make sure OPENAI_API_KEY is set in environment
    
    # Generate sample data
    np.random.seed(42)
    n_users = 100
    user_ids = [f"USER_{str(i+1).zfill(3)}" for i in range(n_users)]
    user_features = np.random.random((n_users, 7))
    user_texts = [
        f"User with interests in area {np.random.randint(1, 5)}" for _ in range(n_users)
    ]
    
    # Query example
    query_features = np.random.random(7)
    query_text = "User interested in area 3"
    
    # Get best matches
    matches = get_best_matches(
        query_features,
        query_text,
        user_features,
        user_texts,
        user_ids,
        n_neighbors=10,
        client=client
    )
    
    print("\nBest matches (KNN + Embedding similarity):")
    for user_id, similarity in matches:
        print(f"User ID: {user_id}, Final Similarity: {similarity:.4f}")
