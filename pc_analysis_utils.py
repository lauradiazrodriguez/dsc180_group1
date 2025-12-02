import os
import glob
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Tuple

# Causal-Learn Imports
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.SHD import SHD 
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph

def plot_comparison_graphs(true_graph_obj, learned_graph_obj, labels, title_prefix=""):
    """
    helper func to plot the ground truth and learned graphs side-by-side
    
    Args:
        true_graph_obj (GeneralGraph): The graph object for the ground truth.
        learned_graph_obj (GeneralGraph): The graph object returned by the PC algorithm.
        labels (list): A list of variable names.
        title_prefix (str): A title for the entire plot.
    """
    
    # 1-row, 2-column plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title_prefix, fontsize=24, y=1.05) # Add a main title
    
    
    # pydot object from the graph
    pyd_true = GraphUtils.to_pydot(true_graph_obj, labels=labels)
    
    # convert pydot to PNG image
    tmp_png_true = pyd_true.create_png(f="png")
    fp_true = io.BytesIO(tmp_png_true)
    img_true = mpimg.imread(fp_true, format='png')
    
    # display image on first plot
    axes[0].imshow(img_true)
    axes[0].set_title("Ground Truth DAG", fontsize=20)
    axes[0].axis('off') # Hide axes
    
    
    # create pydot object from learned graph
    pyd_learned = GraphUtils.to_pydot(learned_graph_obj, labels=labels)
    
    # convert pydot to PNG image
    tmp_png_learned = pyd_learned.create_png(f="png")
    fp_learned = io.BytesIO(tmp_png_learned)
    img_learned = mpimg.imread(fp_learned, format='png')
    
    # display image on second plot
    axes[1].imshow(img_learned)
    axes[1].set_title("Learned Graph (CPDAG) from PC", fontsize=20)
    axes[1].axis('off')
    
    plt.show()


def calculate_and_print_metrics(true_graph_obj: GeneralGraph, learned_graph_obj: GeneralGraph) -> Tuple[float, float, float]:
    """
    Calculates and prints Structural Precision, Recall, and F1-score 
    (based on edge adjacency only) by manually comparing the adjacency matrices.
    
    The comparison is done on UNDIRECTED edges (existence of an edge, 
    A-B vs. A->B or B->A).
    
    Args:
        true_graph_obj (GeneralGraph): The ground truth graph object.
        learned_graph_obj (GeneralGraph): The graph object learned by the PC algorithm.
        
    Returns:
        Tuple[float, float, float]: (Precision, Recall, F1-score)
    """
    
    # 1. Extract Adjacency Matrices using the public 'graph' attribute.
    # The 'graph' attribute is listed in dir(true_graph_obj) and typically holds 
    # the underlying NumPy matrix for the GeneralGraph object in this library.
    
    true_adj = true_graph_obj.graph
    learned_adj = learned_graph_obj.graph
    
    # 2. Symmetrize Matrices for Adjacency-Only Comparison
    
    def symmetrize_matrix(A):
        """Returns a binary matrix where A[i,j]=1 if A[i,j] or A[j,i] is non-zero."""
        # Convert non-zero entries (1, 2, 3, 4, etc. for different edge types) to 1 (edge exists)
        A_binary = (A != 0).astype(int)
        # Symmetrize: an edge exists if A->B OR B->A exists.
        return ((A_binary + A_binary.T) != 0).astype(int)

    True_Edges_Sym = symmetrize_matrix(true_adj)
    Learned_Edges_Sym = symmetrize_matrix(learned_adj)
    
    # 3. Calculate Confusion Matrix components (TP, FP, FN)
    
    # Mask for unique edges (upper triangle, excluding diagonal)
    N = True_Edges_Sym.shape[0]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Flatten the matrices to compare only unique potential edges
    True_Edges_Flat = True_Edges_Sym[mask]
    Learned_Edges_Flat = Learned_Edges_Sym[mask]
    
    # True Positives (TP): Edges present in BOTH true and learned graphs
    TP = np.sum(True_Edges_Flat * Learned_Edges_Flat)
    
    # False Positives (FP): Edges in Learned but NOT in True
    FP = np.sum(Learned_Edges_Flat * (1 - True_Edges_Flat))
    
    # False Negatives (FN): Edges in True but NOT in Learned
    FN = np.sum(True_Edges_Flat * (1 - Learned_Edges_Flat))
    
    # Total True Edges (P = Positives): TP + FN
    num_edges_in_true = np.sum(True_Edges_Flat)
    
    # Total Learned Edges (P' = Predicted Positives): TP + FP
    num_edges_in_learned = np.sum(Learned_Edges_Flat)
    
    # 4. Calculate Precision, Recall, and F1-Score

    # Precision: TP / (TP + FP)
    if num_edges_in_learned == 0:
        precision = 1.0  # Convention: 1.0 if no edges are predicted
    else:
        precision = TP / num_edges_in_learned
        
    # Recall: TP / (TP + FN)
    if num_edges_in_true == 0:
        recall = 1.0  # Convention: 1.0 if the true graph has no edges
    else:
        recall = TP / num_edges_in_true

    # F1-Score
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"|--- Structural Metrics (Edges Only) ---")
    print(f"| Total True Edges (TP + FN): {num_edges_in_true}")
    print(f"| Total Learned Edges (TP + FP): {num_edges_in_learned}")
    print(f"| TP (Correctly Found Edges): {TP}")
    print(f"| FP (Spurious Edges): {FP}")
    print(f"| FN (Missed Edges): {FN}")
    print(f"| Precision: {precision:.4f}")
    print(f"| Recall:    {recall:.4f}")
    print(f"| F1-Score:  {f1_score:.4f}")
    print(f"|---------------------------------------")
    
    return precision, recall, f1_score


def run_pc_analysis_on_dataset(data_csv_path, truth_graph_npy_path, alpha=0.05, test_choice = 'fisherz'):
    """
    runs complete PC analysis on a single dataset and compares it to the ground truth
    
    1. Load data and true graph
    2. Run PC algorithm
    3. Score the result using Structural Hamming Distance (SHD)
    4. Plot the true graph vs. the learned graph.
    
    Args:
        data_csv_path (str): Filepath to the simulation's _data.csv file.
        truth_graph_npy_path (str): Filepath to the simulation's _graph.npy file.
        alpha (float): Significance level for the PC algorithm's independence tests.
                       A common value is 0.05.
        
    Returns:
        dict: A dictionary containing the learned graph object and the SHD score.
    """
    
    print(f"Processing: {os.path.basename(data_csv_path)}")
    
    # LOAD AND PREPARE DATA
    df = pd.read_csv(data_csv_path)
    if 'domain_index' in df.columns:
        df = df.drop('domain_index', axis=1)
    labels = df.columns.tolist()
    data = df.to_numpy()
    
    # LOAD & CREATE GROUND TRUTH DAG OBJECT
    # LOAD SAVED ADJACENCY MATRIX
    true_adj_transposed = np.load(truth_graph_npy_path)
    true_adj_matrix = true_adj_transposed.T
    
    # Create the Ground Truth Graph Object 
    # must convert NumPy matrix into GeneralGraph object
    nodes = [GraphNode(name) for name in labels]
    true_graph_obj = GeneralGraph(nodes)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if true_adj_matrix[i, j] != 0:
                true_graph_obj.add_directed_edge(nodes[i], nodes[j])
    
    # PC alg
    print("Running PC alg")
    cg = pc(data, alpha=alpha, indep_test=test_choice)
    
    print("PC alg finished.")
    
    # get learned graph object
    learned_graph_obj = cg.G
    
    #check accuracy with SHD
    #pass graph objects to SHD
    # SHD receives true_graph_obj and learned_graph_obj
    shd_score = SHD(true_graph_obj, learned_graph_obj).get_shd()
    testing = SHD(true_graph_obj, learned_graph_obj)


    calculate_and_print_metrics(true_graph_obj, learned_graph_obj)


    
    print("\n analysis done")
    print(f"Structural Hamming Distance (SHD): {shd_score}")
    if shd_score == 0:
        print("PC perfectly recovered the true causal graph")
    else:
        print(f"Learned graph is {shd_score} edge(s) different from the truth.")
    print("-------------------------\n")
    
    ### 5. VISUALIZE RESULTS ###
    plot_title = os.path.basename(data_csv_path).replace('_data.csv', '')
    
    # Pass the two graph objects to the plotting function
    plot_comparison_graphs(true_graph_obj, learned_graph_obj, labels, title_prefix=plot_title)
    return {
        'learned_graph_obj': learned_graph_obj,
        'shd_score': shd_score,
        'test': testing
    }

