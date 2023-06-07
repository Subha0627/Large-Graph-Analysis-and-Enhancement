import networkx as nx
import numpy as np
import itertools
"""Finding Similarity between two graphs with respect to the betweenness centrality"""

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
Calculate Degree Distribution from a given degree and number of nodes
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
Two degree distributions 'array1' and 'array2' are compared and the similarity score is returned
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
The clustering coefficient is a measure of how interconnected the nodes in a graph are. A higher clustering coefficient 
indicates a higher level of clustering or connectivity between the neighbors of a node.In our case, the original graph has a
clustering coefficient of 0.57, while the reduced graph has a clustering coefficient of 0.17. This implies that the original
graph has a higher level of local clustering compared to the reduced graph.The reduction in the clustering coefficient could be due
to the removal of certain edges or nodes in the reduced graph. When edges or nodes are removed, it can disrupt the local 
connectivity and decrease the clustering coefficient. A lower clustering coefficient in the reduced graph suggests that the nodes 
in the graph are less interconnected, and there may be fewer clusters or communities present compared to the original graph. This 
reduction in clustering could indicate a loss of local structure or a more dispersed network configuration. Overall, the difference in
clustering coefficients between the original and reduced graphs can provide insights into the structural changes and the level of 
clustering in the network. It can be used to study the impact of network modifications or analyze the resilience and robustness of
the graph's connectivity. The characterization of the reduced graph as "bad" or not depends on the specific context and goals of your 
analysis. The reduced graph may not necessarily be considered "bad," but rather different or simplified compared to the original graph.
Reducing a graph can be done for various reasons, such as improving computational efficiency, removing noise or outliers, or focusing
on specific aspects of the network. It may help in identifying the core structure or essential relationships within the graph.However, it's 
important to consider the implications of the reduction. Removing nodes or edges can result in a loss of information and potentially 
alter the characteristics of the graph. If the reduced graph still captures the essential properties or patterns you are interested in,
it can be considered a useful simplification. It's crucial to evaluate the reduced graph in relation to your specific analysis objectives. 
If the reduction significantly impacts the interpretation or understanding of the network, you may need to reassess the approach or consider
alternative methods to retain more of the original graph's structure. Ultimately, whether the reduced graph is considered "bad" or 
not depends on its utility and relevance to the particular analysis or problem at hand.
"""
def compare_clustering_coefficient(original_graph, reduced_graph):
    original_clustering_coefficient = nx.average_clustering(original_graph)
    reduced_clustering_coefficient = nx.average_clustering(reduced_graph)
    similarity_clustering_coefficient = 1-abs(original_clustering_coefficient-reduced_clustering_coefficient)/max(original_clustering_coefficient, reduced_clustering_coefficient)
    return round(similarity_clustering_coefficient*100, 2)

# Calculate eigenvector centralities
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

def average_path_length(graph):
    total_length = 0
    num_pairs = 0
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                try:
                    length = nx.shortest_path_length(graph, source, target)
                    total_length += length
                    num_pairs += 1
                except nx.NetworkXNoPath:
                    continue
    if num_pairs > 0:
        return total_length / num_pairs
    else:
        return 0

def compare_average_path_length(original_graph, reduced_graph):
    # Calculate the average path length
    average_path_length_original = average_path_length(original_graph)
    average_path_length_reduced = average_path_length(reduced_graph)

    # Print the average path length
    similarity_avg_path_length = round(1-abs(average_path_length_reduced-average_path_length_original)/max(average_path_length_original, average_path_length_reduced), 2)*100
    return similarity_avg_path_length

"""
The formula `(graph1.number_of_nodes() * (graph1.number_of_nodes() - 1) / 2)` calculates the total number of possible pairs of nodes in `graph1`
excluding self-loops. When computing the Jaccard similarity, we are interested in comparing pairs of nodes that do not have an edge between
them in `graph2`. To calculate the similarity as a percentage, we need to divide the number of such pairs with non-zero Jaccard similarity by 
the total number of possible pairs.The total number of possible pairs of nodes in `graph1` excluding self-loops can be calculated as
`(graph1.number_of_nodes() * (graph1.number_of_nodes() - 1) / 2)`. We divide `num_pairs` (the number of pairs with non-zero Jaccard similarity)
by this total number of possible pairs, and then multiply by 100 to express the result as a percentage. This gives us the similarity percentage.
By using this formula, we can obtain a normalized measure of similarity between the graphs that accounts for the total possible pairs 
of nodes in `graph1`.
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
