import networkx as nx
import numpy as np
import itertools
"""
Finding Similarity between two graphs with respect to the betweenness centrality
"""
def compare_betweenness_centrality(original_graph, reduced_graph, removed_nodes):
    # Calculate betweenness centrality for the first graph
    betweenness_centrality_original_graph = nx.betweenness_centrality(original_graph)
    # Calculate betweenness centrality for the second graph
    betweenness_centrality_reduced_graph = nx.betweenness_centrality(reduced_graph)
    # Calculate the average absolute difference between centrality values
    differences = []
    for node in original_graph.nodes():
        centrality1 = betweenness_centrality_original_graph[node]
        if node not in removed_nodes:
            centrality2 = betweenness_centrality_reduced_graph[node]
        else:
            centrality2 = 0
        difference = abs(centrality1 - centrality2)
        differences.append(difference)
    average_difference = sum(differences) / len(differences)
    # Calculate the similarity percentage
    similarity_percentage = (1 - average_difference) * 100
    return round(similarity_percentage, 2)


"""
Calculating Similarity between the original graph and the reduced graph with respect to the degree distribution
"""
def calculate_degree_sequence(original_graph, reduced_graph):
    # Find the degree sequence
    """Finding the degree sequence of the Original Graph"""
    degree_sequence_original = [degree for node, degree in original_graph.degree()]
    # Find the degree sequence
    """Finding the degree sequence of the Reduced Graph"""
    degree_sequence_reduced = [degree for node, degree in reduced_graph.degree()]
    return degree_sequence_original, degree_sequence_reduced


"""
Calculate Degree Distribution from a given degree sequence and number of nodes
"""
def calculate_degree_distribution(degree_sequence, size):
    # Initialize an array to store the degree distribution
    degree_distribution = np.zeros(size)
    # Calculate the degree distribution
    for degree in degree_sequence:
        if degree < size:
            degree_distribution[degree] += 1

    # Normalize the degree distribution to get probabilities
    degree_distribution = degree_distribution / len(degree_sequence)

    # Print the degree distribution array
    output = ", ".join(str(value) for value in degree_distribution)
    return degree_distribution


"""
Compare the degree distributions of an original graph and a reduced graph and return the similarity score.
"""
def compare_degree_distributions(original_graph, reduced_graph):
    degree_sequence_original, degree_sequence_reduced = calculate_degree_sequence(
        original_graph, reduced_graph)
    size = len(original_graph)
    """Get the degree distribution for the degree sequence of the origianl graph"""
    degree_distribution_original = calculate_degree_distribution(degree_sequence_original, size)
    """Get the degree distribution for the degree sequence of the reduced graph"""
    degree_distribution_reduced = calculate_degree_distribution(degree_sequence_reduced, size)
    # Calculate Euclidean distance
    euclidean_distance = np.linalg.norm(degree_distribution_original - degree_distribution_reduced)
    # Calculate maximum possible distance
    max_distance = np.linalg.norm(np.ones_like(degree_distribution_original) - np.zeros_like(degree_distribution_reduced))
    # Normalize Euclidean distance to obtain similarity score
    similarity_score = 1 - (euclidean_distance / max_distance)
    return round(similarity_score*100, 2)

"""
Finding similarity between two graphs with respect to the Eigenvector centralities
"""
def compare_eigen_vector_centrality(original_graph, reduced_graph):
    eigenvector_centrality_original = nx.eigenvector_centrality(original_graph)
    eigenvector_centrality_reduced = nx.eigenvector_centrality(reduced_graph)
    # Get the common nodes present in both graphs
    common_nodes = set(eigenvector_centrality_original.keys()).intersection(
        eigenvector_centrality_reduced.keys())
    # Extract the centralities for the common nodes
    eigenvector_centrality_original = np.array([eigenvector_centrality_original[node] for node in common_nodes])
    eigenvector_centrality_reduced = np.array([eigenvector_centrality_reduced[node] for node in common_nodes])
    # Normalize the centralities by dividing by the maximum centrality value
    max_centrality = max(np.max(eigenvector_centrality_original),np.max(eigenvector_centrality_reduced))
    normalized_centrality_original = eigenvector_centrality_original / max_centrality
    normalized_centrality_reduced = eigenvector_centrality_reduced / max_centrality
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((normalized_centrality_original -normalized_centrality_reduced) ** 2)
    # Convert the MSE to a percentage scale
    similarity_percentage = 100 * (1 - mse)
    return round(similarity_percentage, 2)

"""
Calculate the average path length of a graph
"""
def average_path_length(graph):
    total_length = 0
    num_pairs = 0
    # Iterate over all nodes as source and target pairs
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                try:
                    # Calculate the shortest path length between the source and target nodes
                    length = nx.shortest_path_length(graph, source, target)
                    total_length += length
                    num_pairs += 1
                except nx.NetworkXNoPath:
                    continue
    # Calculate the average path length if there are pairs of nodes
    if num_pairs > 0:
        return total_length / num_pairs
    else:
        return 0

"""
Finding Similarity between two graphs with respect to the Average Path Length
"""
def compare_average_path_length(original_graph, reduced_graph):
    # Calculate the average path length
    average_path_length_original = average_path_length(original_graph)
    average_path_length_reduced = average_path_length(reduced_graph)

    # Print the average path length
    similarity_avg_path_length = round(1-abs(average_path_length_reduced-average_path_length_original)/max(average_path_length_original, average_path_length_reduced), 2)*100
    return similarity_avg_path_length

"""
Finding Similarity between two graphs with respect to the Jaccard Similarity
"""
def calculate_jaccard_similarity(graph1, graph2):
    jaccard_scores = []
    for u, v in itertools.combinations(graph1.nodes(), 2):
        if not graph2.has_edge(u, v):
            intersection = len(set(graph1.neighbors(
                u)).intersection(graph1.neighbors(v)))
            union = len(set(graph1.neighbors(u)).union(graph1.neighbors(v)))
            jaccard_scores.append((u, v, intersection / union))
    # Calculate the similarity as a percentage
    num_pairs = len(jaccard_scores)
    similarity_percentage = (
        num_pairs / (graph1.number_of_nodes() * (graph1.number_of_nodes() - 1) / 2)) * 100
    return round(similarity_percentage, 2)
