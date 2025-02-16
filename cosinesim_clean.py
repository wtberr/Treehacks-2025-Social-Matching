import numpy as np
from openai import OpenAI
from typing import List, Tuple

def get_embedding(text: str, client: OpenAI) -> np.ndarray:
    """
    Get embedding for a text string using OpenAI's API.
    
    Args:
        text: string to get embedding for
        client: initialized OpenAI client
    
    Returns:
        numpy array of the embedding
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def rank_users_by_similarity(
    query_text: str,
    user_texts: List[str],
    user_ids: List[str],
    client: OpenAI
) -> List[Tuple[str, float]]:
    """
    Rank users based on cosine similarity of their text embeddings.
    
    Args:
        query_text: text to compare against
        user_texts: list of user text descriptions
        user_ids: list of corresponding user IDs
        client: initialized OpenAI client
    
    Returns:
        list of (user_id, similarity_score) tuples, sorted by similarity
    """
    # Get query embedding
    query_embedding = get_embedding(query_text, client)
    
    # Get embeddings for all users
    similarities = []
    for idx, user_text in enumerate(user_texts):
        user_embedding = get_embedding(user_text, client)
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, user_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(user_embedding)
        )
        
        similarities.append((user_ids[idx], similarity))
    
    # Sort by similarity score in descending order
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Example usage:
if __name__ == "__main__":
    client = OpenAI()  # Make sure to set OPENAI_API_KEY in your environment
    
    # Example data
    query = "Software engineer interested in AI and machine learning"
    users = [
        "Data scientist with focus on NLP",
        "Frontend developer working with React",
        "Machine learning engineer specializing in computer vision"
    ]
    user_ids = ["USER_001", "USER_002", "USER_003"]
    
    ranked_users = rank_users_by_similarity(query, users, user_ids, client)
    
    print("Query:", query)
    print("\nRanked users by embedding similarity:")
    for user_id, similarity in ranked_users:
        print(f"User ID: {user_id}, Similarity: {similarity:.4f}")
