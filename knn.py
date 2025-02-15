#generall knn application

import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_nearest_points(points, query_point, n_neighbors):
    """
    Basic KNN application, finding nearest neighbor(s) to a query point using euclidian distance.
    Arguments: 
        points- numpy array of shape (n_samples, n_features) containing the users feature vectors
        query_point- numpy array of shape (n_features,), the point to find neighbors for
        n_neighbors- number of nearest neighbor(s) to find 
    Returns:
        distances: distances to the nearest neighbors
        indices: indices of the nearest neighbors in the original points array
    """
    # Reshape query_point if it's 1D
    if query_point.ndim == 1:
        query_point = query_point.reshape(1, -1)
    
    # Initialize and fit the nearest neighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(points)
    
    # Find nearest neighbors
    distances, indices = nn.kneighbors(query_point)
    
    return distances[0], indices[0]  # Return 1D arrays

def find_nearest_users(user_features, user_ids, query_features, n_neighbors=10):
    """
    Find the nearest users based on their feature vectors.
    
    Args:
        user_features: numpy array of shape (n_users, n_features) containing all users' features
        user_ids: array-like of user IDs corresponding to the features
        query_features: numpy array of shape (n_features,) or (1, n_features) - features to find neighbors for
        n_neighbors: number of nearest neighbors to find (default=10)
    
    Returns:
        distances: distances to the nearest neighbors
        neighbor_ids: user IDs of the nearest neighbors
    """
    distances, indices = find_nearest_points(user_features, query_features, n_neighbors)
    neighbor_ids = [user_ids[idx] for idx in indices]
    return distances, neighbor_ids

# Example usage:
if __name__ == "__main__":
    # Generate 100 random users with 7D features
    np.random.seed(42)  # For reproducibility
    n_users = 100
    
    # Generate random user IDs (e.g., "USER_001" to "USER_100")
    user_ids = [f"USER_{str(i+1).zfill(3)}" for i in range(n_users)]
    
    # Generate random feature vectors for each user
    # Each feature is normalized between 0 and 1
    user_features = np.random.random((n_users, 7))
    
    # Example query user (could be a new user or existing user)
    query_features = np.random.random(7)
    
    # Find 10 nearest neighbors
    distances, neighbor_ids = find_nearest_users(user_features, user_ids, query_features)
    
    print("Query features:", query_features)
    print("\nNearest neighbors:")
    for dist, user_id in zip(distances, neighbor_ids):
        print(f"User ID: {user_id}, Distance: {dist:.4f}")
