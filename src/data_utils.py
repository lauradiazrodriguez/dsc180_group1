import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import pandas as pd
from typing import Dict, List, Tuple, Callable, Union
import os
import json
from datetime import datetime
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor


# helper functions
def set_random_seed(seed):
    #for reproducibility
    random.seed(seed)
    np.random.seed(seed)

def is_dag(W):
    #checks with weight adjancency matrix is a DAG
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def simulate_dag(d, s0, graph_type):
    """simulates random DAG with some expected number of edges
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def random_permutation(M):
        #permutes rows and columns of a matrix
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def random_acyclic_orientation(B_und):
        #takes undirected graph matrix and returns a DAG
        return np.tril(random_permutation(B_und), k=-1)

    def graph_adjmat(G):
        #converts igraph graph object to numpy adjacency matrix
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi: set probability for an edge to be inserted 
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0) # undirected graph with d nodes and s0 edges
        B_und = graph_adjmat(G_und)
        B = random_acyclic_orientation(B_und) #randomly set orientation for acyclic
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert: certain nodes have more edges (hubs)
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = graph_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite: two sets of nodes with edges only between the two sets
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = graph_adjmat(G)
    else:
        raise ValueError('unknown graph type')

    #another permuation to randomize order
    B_perm = random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """simulate Structural Equation Model (SEM) parameters for DAG
    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges
    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    #randomly select which range of weights each edge is
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        #generate uniform random weights
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_sem(W, n, sem_type, noise_scale=None, discrete_ratio=0.0, max_categories=10):
    """
    simulate samples from linear SEM with specified type of noise and mixed continuous/discrete variables
    Args:
        W (np.ndarray): weighted adjacency matrix
        n (int): # of samples to generate
        sem_type (str): Type of noise/non-linearity ('gaussian', 'uniform', 'poisson', etc.)
        noise_scale (float/np.ndarray): Scale parameter for the noise
        discrete_ratio (float): Proportion of nodes that should be discrete
        max_categories (int): Max number of categories for discrete nodes
    
    """
    
    def simulate_single_equation(X, w, scale, is_discrete=False, n_cats=None):
        """
        X: [n, num of parents]
        w: [num of parents], x: [n]
        """
        if is_discrete: #discrete case: uses a multinomial logistic model
            if X.shape[1] == 0: #no parents: noise directly defines logits
                logits = np.random.normal(scale=scale, size=(n, n_cats))
            else:
                #calculate logits based on a linear combination of parent values
                logits = np.zeros((n, n_cats))
                for k in range(n_cats):
                    #linear componenent for each category k
                    if sem_type == 'gaussian':
                        logits[:, k] = X @ w
                    elif sem_type == 'exponential':
                        logits[:, k] = X @ w
                    elif sem_type == 'gumbel':
                        logits[:, k] = X @ w
                    elif sem_type == 'uniform':
                        logits[:, k] = X @ w
                    else:
                        raise ValueError('unsupported sem type for discrete variables')
            # logit -> probability
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            #sample categories based on probabilities
            return np.array([np.random.choice(n_cats, p=p) for p in probs]).astype(float)
        else:
            if sem_type == 'gaussian':
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'exponential':
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            elif sem_type == 'logistic':
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            elif sem_type == 'poisson':
                x = np.random.poisson(np.exp(X @ w)) * 1.0
            else:
                raise ValueError('unknown sem type')
            return x

    d = W.shape[0]

    #noise scale
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):
        if sem_type == 'gauss':
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')


    #setting up which columns will be discrete variables
    n_discrete = int(d * discrete_ratio)
    discrete_cols = np.random.choice(d, size=n_discrete, replace=False)

    #topological order for simulation
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    
    assert len(ordered_vertices) == d
    X = np.zeros([n, d]) #data matrix
    for j in ordered_vertices:
        #loop through nodes in causal order
        parents = G.neighbors(j, mode=ig.IN)
        is_discrete = j in discrete_cols
        
        #determining how many categories for discrete variables
        if max_categories > 2:
            n_categories = np.random.randint(2, max_categories)
        else:
            n_categories = 2

        #getting X[:, j] from X[:, parents]
        X[:, j] = simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j], is_discrete, n_categories)
    return X

def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None, discrete_ratio=0.3, max_categories=10):
    """simulate samples from nonlinear SEM with mixed continuous and discrete variables.
    Args:
        B (np.ndarray): Binary adjacency matrix
        n (int): # of samples to generate
        sem_type (str): Type of non-linear function ('mlp', 'mim', 'gp', 'gp-add')
        noise_scale (float): Scale parameter for the noise (assumed Gaussian)
        discrete_ratio (float): Proportion of nodes that should be discrete
        max_categories (int): Max # of categories for discrete nodes
    """
    def simulate_single_equation(X, scale, is_discrete=False, n_cats=None):
        #simulates value of a single variable using nonlinear function
        pa_size = X.shape[1]
        
        if is_discrete:
            if pa_size == 0:
                logits = np.random.normal(scale=scale, size=(n, n_cats))
            else:
                if sem_type == 'mlp': #Multi-Layer Perceptron
                    hidden = 100
                    logits = np.zeros((n, n_cats))
                    for k in range(n_cats): #randomly assign nonlinear weights to each category
                        W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                        W1[np.random.rand(*W1.shape) < 0.5] *= -1
                        W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                        W2[np.random.rand(hidden) < 0.5] *= -1
                        logits[:, k] = sigmoid(X @ W1) @ W2
                elif sem_type == 'mim': #combo of sin, cos, tan
                    logits = np.zeros((n, n_cats))
                    #randomly assign linear combination of weights
                    for k in range(n_cats):
                        w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w1[np.random.rand(pa_size) < 0.5] *= -1
                        w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w2[np.random.rand(pa_size) < 0.5] *= -1
                        w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                        w3[np.random.rand(pa_size) < 0.5] *= -1
                        logits[:, k] = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3)
                elif sem_type == 'gp' or sem_type == 'gp-add':
                    logits = np.zeros((n, n_cats))
                    for k in range(n_cats):
                        gp = GaussianProcessRegressor()
                        if sem_type == 'gp':
                            logits[:, k] = gp.sample_y(X, random_state=None).flatten()
                        else:  # gp-add
                            logits[:, k] = sum([gp.sample_y(X[:, i, None], random_state=None).flatten() 
                                                for i in range(X.shape[1])])
                else:
                    raise ValueError('unknown sem type')
            
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            return np.array([np.random.choice(n_cats, p=p) for p in probs]).astype(float)
            
        else:
            z = np.random.normal(scale=scale, size=n)
            if pa_size == 0:
                return z
                
            if sem_type == 'mlp':
                hidden = 100
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'mim':
                w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w1[np.random.rand(pa_size) < 0.5] *= -1
                w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w2[np.random.rand(pa_size) < 0.5] *= -1
                w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w3[np.random.rand(pa_size) < 0.5] *= -1
                x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            elif sem_type == 'gp':
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gp-add':
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                         for i in range(X.shape[1])]) + z
            else:
                raise ValueError('unknown sem type')
            return x

    d = B.shape[0]
    scale_vec = noise_scale * np.ones(d) if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    
    n_discrete = int(d * discrete_ratio)
    discrete_columns = np.random.choice(d, size=n_discrete, replace=False)
    
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        is_discrete = j in discrete_columns
        
        # CHANGE: checker
        # would fail if max_categories was 2 (since low >= high)
        if max_categories > 2:
            n_categories = np.random.randint(2, max_categories)
        else:
            n_categories = 2

        X[:, j] = simulate_single_equation(X[:, parents], scale_vec[j], is_discrete, n_categories)
    return X






#-----------------



# DataSimulator Class from repo (modified) 

class DataSimulator:
    def __init__(self):
        self.data = None
        self.graph = None
        self.ground_truth = {}
        self.variable_names = None

    def generate_graph(self, n_nodes: int, edge_probability: float = 0.3, variable_names: List[str] = None, graph_type: str = 'ER') -> None:
        """Generate a random directed acyclic graph (DAG) using specified method."""
        self.graph = simulate_dag(n_nodes, int(edge_probability * n_nodes * (n_nodes - 1) / 2), graph_type)
        if variable_names and len(variable_names) == n_nodes:
            self.variable_names = variable_names
            self.graph_dict = {i: name for i, name in enumerate(variable_names)}
        else:
            self.variable_names = [f'X{i+1}' for i in range(n_nodes)]
            self.graph_dict = {i: f'X{i+1}' for i in range(n_nodes)}
        self.ground_truth['graph'] = self.graph_dict
        self.ground_truth['edge_probability'] = edge_probability

    def generate_single_domain_data(self, n_samples: int, noise_scale: float, noise_type: str, 
                                      function_type: Union[str, List[str], Dict[str, str]], 
                                      discrete_ratio: float = 0.0, max_categories: int = 5) -> pd.DataFrame:
        """Generate data for a single domain based on the graph structure."""
        # CHANGED: Replaced 'logger.debug' with print
        print(f"Data generation: {function_type}, {noise_type}, scale={noise_scale}")
        assert isinstance(function_type, str) and function_type in ['linear', 'mlp', 'mim', 'gp', 'gp-add']
        assert noise_type in ['gaussian', 'exponential', 'gumbel', 'uniform', 'logistic', 'poisson']
        if function_type != 'linear':
            # CHANGED: replaced logger.debug with print
            print("Non-linear function requires gaussian noise")
        assert isinstance(noise_scale, float) and noise_scale > 0
        if function_type == 'linear':
            W = simulate_parameter(self.graph)
            data = simulate_linear_sem(W, n_samples, noise_type, noise_scale, discrete_ratio, max_categories)
        else:
            data = simulate_nonlinear_sem(self.graph, n_samples, function_type, noise_scale, discrete_ratio, max_categories)
        
        data_df = pd.DataFrame(data, columns=self.variable_names)
        return data_df
    
    def generate_multi_domain_data(self, n_samples: int, noise_scale: float, noise_type: str, 
                                     function_type: Union[str, List[str], Dict[str, str]], 
                                     discrete_ratio: float = 0.0, max_categories: int = 5, edge_probability: float = 0.3) -> pd.DataFrame:
        """Generate data for multiple domains based on the graph structure."""
        # CHANGED: Replaced 'logger.debug' with print
        print(f"Data generation: {function_type}, {noise_type}, scale={noise_scale}")
        assert isinstance(function_type, str) and function_type in ['linear', 'mlp', 'mim', 'gp', 'gp-add']
        assert noise_type in ['gaussian', 'exponential', 'gumbel', 'uniform', 'logistic', 'poisson']
        if function_type != 'linear':
            # CHANGED: replaced logger.debug with print
            print("Non-linear function requires gaussian noise")
        assert isinstance(noise_scale, float) and noise_scale > 0

        if function_type == 'linear':
            W = simulate_parameter(self.graph)
            base_data = []
            for i in range(self.n_domains):
                base_data.append(simulate_linear_sem(W, n_samples, noise_type, noise_scale, discrete_ratio, max_categories))
        else:
            base_data = []
            for i in range(self.n_domains):
                base_data.append(simulate_nonlinear_sem(self.graph, n_samples, function_type, noise_scale, discrete_ratio, max_categories))
        
        n_nodes = self.graph.shape[0]
        node_connections = np.sum(self.graph, axis=1) + np.sum(self.graph, axis=0)
        less_connected_nodes = np.argsort(node_connections)
        n_affected = max(3, int(edge_probability * n_nodes))
        affected_nodes = less_connected_nodes[:n_affected]
        
        # CHANGED: replaced logger.debug with print
        print(f"Domain affects {len(affected_nodes)} variables")
        
        data = []
        step = 10 / self.n_domains
        
        base_data_array = np.vstack([d for d in base_data])
        base_correlation_matrix = np.corrcoef(base_data_array.T)
        # CHANGED: replaced logger.debug with print
        print(f"Base correlation matrix: {base_correlation_matrix.shape}")
        
        n_vars = base_correlation_matrix.shape[0]
        high_corr_threshold = 0.7
        high_corr_pairs_base = []
        
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                corr = base_correlation_matrix[i, j]
                if abs(corr) > high_corr_threshold:
                    high_corr_pairs_base.append((self.variable_names[i], self.variable_names[j], corr))
        
        if high_corr_pairs_base:
            # CHANGED: replaced logger.debug with print
            print(f"Found {len(high_corr_pairs_base)} highly correlated pairs (base)")
        else:
            # CHANGED: replaced logger.debug with print
            print(f"No high correlations found (base)")
        
        domain_effect_multiplier = 0.3
        max_attempts = 5
        attempts = 0
        new_high_corr_pairs_found = False
        
        temp_data = [] # Initialize here
        
        while attempts < max_attempts and not new_high_corr_pairs_found:
            temp_data = []
            for i in range(self.n_domains):
                domain_data = base_data[i].copy()
                
                if function_type == 'linear':
                    domain_effect = (i + 1) * step * domain_effect_multiplier
                    for node_idx in affected_nodes:
                        domain_data[:, node_idx] += domain_effect
                else:
                    domain_effect = (i + 1) * step * domain_effect_multiplier
                    for node_idx in affected_nodes:
                        base_values = domain_data[:, node_idx]
                        domain_data[:, node_idx] += domain_effect * np.sign(base_values) * (base_values ** 2)
                
                temp_data.extend(domain_data)
            
            temp_data_array = np.vstack(temp_data)
            temp_correlation_matrix = np.corrcoef(temp_data_array.T)
            
            high_corr_pairs_temp = []
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    corr = temp_correlation_matrix[i, j]
                    if abs(corr) > high_corr_threshold:
                        high_corr_pairs_temp.append((self.variable_names[i], self.variable_names[j], corr))
            
            if len(high_corr_pairs_temp) > len(high_corr_pairs_base):
                new_high_corr_pairs_found = True
                data = temp_data
                correlation_matrix = temp_correlation_matrix
                high_corr_pairs = high_corr_pairs_temp
            else:
                domain_effect_multiplier *= 1.5
                attempts += 1
        
        if not new_high_corr_pairs_found:
            data = temp_data
            correlation_matrix = temp_correlation_matrix
            high_corr_pairs = high_corr_pairs_temp
        
        #  CHANGED: replaced logger.debug with print
        print(f"Final correlation matrix: {correlation_matrix.shape}")
        
        if high_corr_pairs:
            #  CHANGED: replaced logger.debug with print
            print(f"Found {len(high_corr_pairs)} highly correlated pairs (final)")
        else:
            #  CHANGED: replaced logger.debug with print
            print(f"No high correlations found (final)")
            
        data_df = pd.DataFrame(data, columns=self.variable_names)
        data_df['domain_index'] = np.repeat(range(1, 1 + self.n_domains), n_samples)
        return data_df

    def generate_data(self, n_samples: int, noise_scale: float = 1.0, 
                        noise_type: str = 'gaussian', 
                        function_type: Union[str, List[str], Dict[str, str]] = 'linear',
                        n_domains: int = 1, variable_names: List[str] = None,
                        discrete_ratio: float = 0.0, max_categories: int = 5, edge_probability: float = 0.3) -> None:
        """Generate heterogeneous data from multiple domains."""
        if self.graph is None:
            raise ValueError("Generate graph first")

        domain_size = n_samples // n_domains
        self.n_domains = n_domains

        if n_domains == 1:
            domain_df = self.generate_single_domain_data(domain_size, noise_scale, noise_type, function_type, 
                                                         discrete_ratio, max_categories)
        else:
            domain_df = self.generate_multi_domain_data(domain_size, noise_scale, noise_type, function_type, 
                                                        discrete_ratio, max_categories, edge_probability)
        
        self.data = domain_df
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        if variable_names is not None:
            if n_domains == 1:
                self.data.columns = variable_names
            else:
                self.data.columns = variable_names + ['domain_index']
                
        self.ground_truth['n_domains'] = n_domains
        self.ground_truth['noise_type'] = noise_type
        self.ground_truth['function_type'] = function_type
        self.ground_truth['discrete_ratio'] = discrete_ratio
        self.ground_truth['max_categories'] = max_categories

    def add_measurement_error(self, error_std: float = 0.3, error_rate: float = 0.5) -> None:
        """Randomly sample a subset of columns to add gaussian measurement error."""
        if self.data is None:
            raise ValueError("Generate data first")

        available_cols = [col for col in self.data.columns if col != 'domain_index']
        n_cols = int(error_rate * len(available_cols))
        columns = np.random.choice(available_cols, size=n_cols, replace=False)

        for col in columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] += np.random.randn(len(self.data)) * error_std
        
        self.ground_truth['measurement_error'] = {col: error_std for col in columns}
        self.ground_truth['measurement_error_value'] = error_rate
        self.ground_truth['measurement_error_desc'] = f"The measurement error is Gaussian with a standard deviation of {error_std} on {error_rate} of the columns (some are selected, some are not, except domain_index), and the measurement error is added to the original data."

    def add_missing_values(self, missing_rate: float = 0.1) -> None:
        """Introduce missing values to the whole dataframe with a specified missing rate."""
        if self.data is None:
            raise ValueError("Generate data first")

        columns = [col for col in self.data.columns if col != 'domain_index']
        
        mask = np.random.random(size=self.data[columns].shape) < missing_rate
        
        self.data[columns] = np.where(mask, np.nan, self.data[columns])
        
        affected_columns = [col for col in columns if mask[:, columns.index(col)].any()]
        self.ground_truth['missing_rate'] = {col: missing_rate for col in affected_columns}
        self.ground_truth['missing_rate_value'] = missing_rate
        self.ground_truth['missing_data_desc'] = f"The missing values are randomly sampled with a missing rate of {missing_rate} on all column data (except domain_index), and the missing values are replaced with NaN."

    def generate_dataset(self, n_nodes: int, n_samples: int, edge_probability: float = 0.3,
                         noise_scale: float = 1.0, noise_type: str = 'gaussian',
                         function_type: Union[str, List[str], Dict[str, str]] = 'linear', discrete_ratio: float = 0.0, max_categories: int = 5,
                         add_measurement_error: bool = False, add_missing_values: bool = False, n_domains: int = 1, 
                         error_std: float = 0.3, error_rate: float = 0.5, missing_rate: float = 0.1,
                         variable_names: List[str] = None, graph_type: str = 'ER') -> Tuple[Dict[int, str], pd.DataFrame]:
        """
        Generate a complete heterogeneous dataset with various characteristics.
        """
        self.generate_graph(n_nodes, edge_probability, variable_names, graph_type)
        self.generate_data(n_samples, noise_scale, noise_type, function_type, n_domains, variable_names, discrete_ratio, max_categories, edge_probability)
                
        if add_measurement_error:
            self.add_measurement_error(error_std=error_std, error_rate=error_rate)
        else:
            self.ground_truth['measurement_error'] = None
            self.ground_truth['measurement_error_value'] = None
            self.ground_truth['measurement_error_desc'] = None
        
        if add_missing_values:
            self.add_missing_values(missing_rate=missing_rate)
        else:
            self.ground_truth['missing_rate'] = None
            self.ground_truth['missing_rate_value'] = None
            self.ground_truth['missing_data_desc'] = None
        
        return self.graph.T, self.data

    def save_simulation(self, output_dir: str = 'simulated_data', prefix: str = 'base') -> None:
        """
        Save the simulated data, graph structure, and simulation settings.
        """
        if self.data is None or self.graph is None:
            raise ValueError("No data or graph to save. Generate dataset first.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_nodes = len(self.graph)
        n_samples = len(self.data)
        folder_name = f"{timestamp}_{prefix}_nodes{n_nodes}_samples{n_samples}"
        save_dir = os.path.join(output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        data_filename = os.path.join(save_dir, f'{prefix}_data.csv')
        data_to_save = self.data.copy()
        if self.ground_truth.get('n_domains', 1) == 1 and 'domain_index' in data_to_save.columns:
            data_to_save = data_to_save.drop('domain_index', axis=1)
        data_to_save.to_csv(data_filename, index=False)
        # <<< CHANGE: Replaced 'logger.success' with 'print', keeping your notebook's success format.
        print(f"\u001b[92m✓ SUCCESS\u001b[0m Data saved: {data_filename}")
        
        graph_filename = os.path.join(save_dir, f'{prefix}_graph.npy')
        np.save(graph_filename, self.graph.T)
        # <<< CHANGE: Replaced 'logger.success' with 'print'.
        print(f"\u001b[92m✓ SUCCESS\u001b[0m Graph saved: {graph_filename}")
        
        config_filename = os.path.join(save_dir, f'{prefix}_config.json')
        config = {
            'n_nodes': n_nodes,
            'n_samples': n_samples,
            'n_domains': self.ground_truth.get('n_domains'),
            'noise_type': self.ground_truth.get('noise_type'),
            'function_type': self.ground_truth.get('function_type'),
            'node_functions': self.ground_truth.get('node_functions'),
            'categorical': self.ground_truth.get('categorical'),
            'measurement_error': self.ground_truth.get('measurement_error'),
            'selection_bias': self.ground_truth.get('selection_bias'),
            'confounding': self.ground_truth.get('confounding'),
            'missing_rate': self.ground_truth.get('missing_rate'),
            'edge_probability': self.ground_truth.get('edge_probability'),
            'discrete_ratio': self.ground_truth.get('discrete_ratio'),
            'max_categories': self.ground_truth.get('max_categories'),
            'missing_rate_value': self.ground_truth.get('missing_rate_value'),
            'measurement_error_value': self.ground_truth.get('measurement_error_value'),
            'missing_data_desc': self.ground_truth.get('missing_data_desc'),
            'measurement_error_desc': self.ground_truth.get('measurement_error_desc'),
        }
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        # CHANGED: Replaced 'logger.success' with print
        print(f"\u001b[92m✓ SUCCESS\u001b[0m Config saved: {config_filename}")

    def generate_and_save_dataset(self, n_nodes: int, n_samples: int, output_dir: str = 'simulated_data', prefix: str = 'base', **kwargs) -> None:
        """
        Generate a dataset and save the results.
        """
        self.generate_dataset(n_nodes, n_samples, **kwargs)
        self.save_simulation(output_dir, prefix)

