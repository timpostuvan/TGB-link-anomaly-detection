from typing import Optional, Dict, Any, Tuple
import numpy as np
import networkx as nx
from tqdm import tqdm


def calculate_remaining_hyperparameters(
    N: int,
    K: int,
    M: int,
    mean_edge_reoccurrence: int,
    ratio_in_out: int,
    t_max: int,
    p_mean_edge_timespan: float,
    p_std_edge_timespan: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate remaining hyperparameters that are needed for graph generation.

    Args:
        N (int): Number of nodes.
        K (int): Number of planted communities.
        M (int): Approximate number of temporal edges.
        mean_edge_reoccurrence (int): Average number of occurrences of edges.
        ratio_in_out (float): Ratio between number of edges within communities and
            number of edges between communities.
        t_max (int): Maximal timestamp (i.e., total timespan of the temporal graph).
        p_mean_edge_timespan (float): Parameter determining average of log-normal distribution
            for sampling edge timespans.
        p_std_edge_timespan (float): Parameter determining standard deviation of log-normal
            distribution for sampling edge timespans.

    Returns:
        Tuple[float, float, float, float]:
            Occurrence probability of an edge within a community,
            occurrence probability of an edge between communities,
            parameters of log-normal distribution (mu and sigma).
    """
    # STRUCTURAL HYPERPARAMETERS

    # Number of unique edges (without timestamps)
    M_unique = M / mean_edge_reoccurrence
    # Number of unique edges (without timestamps) within communities
    M_in = ratio_in_out * M_unique / (ratio_in_out + 1)
    # Number of unique edges (without timestamps) between communities
    M_out = 1 * M_unique / (ratio_in_out + 1)

    # Number of nodes per community
    C_size = N / K
    # Number of all possible edges within communities
    M_all_in = K * C_size * (C_size - 1)
    # Number of all possible edges between communities
    M_all_out = K * C_size * (N - C_size)

    # Occurrence probability of an edge within a community
    p_in = M_in / M_all_in
    # Occurrence probability of an edge between communities
    p_out = M_out / M_all_out

    # --------------------------------------------------------------------------------------------
    # TEMPORAL HYPERPARAMETERS

    # Expected value and standard deviation of log-normal distribution for sampling edge timespans
    mean_edge_timespan = p_mean_edge_timespan * t_max
    std_edge_timespan = p_std_edge_timespan * t_max

    # Parameters of log-normal distribution (mu and sigma)
    mean_lognormal_edge_timespan = np.log(
        mean_edge_timespan
        / np.sqrt((std_edge_timespan**2 / mean_edge_timespan**2) + 1)
    )
    std_lognormal_edge_timespan = np.sqrt(
        np.log((std_edge_timespan**2 / mean_edge_timespan**2) + 1)
    )

    return (
        p_in,
        p_out,
        mean_lognormal_edge_timespan,
        std_lognormal_edge_timespan,
    )


def calculate_mean_messages(T: int, d: int) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Calculate mean messages between different types of nodes.
    Messages are orthogonal and have the following structure:
    (0, ..., 0, 1, ..., 1, 0, ..., 0).

    Args:
        T (int): Number of latent node types. Defaults to 3.
        d (int): Dimensionality of messages. Defaults to 100.

    Returns:
        Dict[Tuple[int, int], np.ndarray]: Dictonary with mean messages for each pair of node types.
    """
    num_ones = np.floor(d / T**2).astype(int)
    assert (
        num_ones > 0
    ), "There are too many types for the current message dimensionality!"

    mean_messages = dict()
    for i in range(T):
        for j in range(T):
            start_ind = (i * T + j) * num_ones
            end_ind = start_ind + num_ones

            message = np.zeros(shape=d)
            message[start_ind:end_ind] = 1

            mean_messages[(i, j)] = message

    return mean_messages


def generate_synthetic_graph(
    N: Optional[int] = 10000,
    K: Optional[int] = 10,
    M: Optional[int] = 10**6,
    mean_edge_reoccurrence: Optional[int] = 50,
    ratio_in_out: Optional[float] = 6,
    t_max: Optional[int] = 10**8,
    p_mean_edge_timespan: Optional[float] = 0.01,
    p_std_edge_timespan: Optional[float] = 0.005,
    p_std_noise_timestamps: Optional[float] = 0.05,
    T: Optional[int] = 5,
    d: Optional[int] = 100,
    std_noise_messages: Optional[float] = 0.05,
    seed: Optional[int] = 44,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic graph.

    Args:
        N (Optional[int], optional): Number of nodes. Defaults to 10000.
        K (Optional[int], optional): Number of planted communities. Defaults to 10.
        M (Optional[int], optional): Approximate number of temporal edges. Defaults to 10**6.
        mean_edge_reoccurrence (Optional[int], optional): Average number of occurrences of edges. Defaults to 50.
        ratio_in_out (Optional[float], optional): Ratio between number of edges within communities and
            number of edges between communities. Defaults to 6.
        t_max (Optional[int], optional): Maximal timestamp (i.e., total timespan of the temporal graph).
            Defaults to 10**8.
        p_mean_edge_timespan (Optional[float], optional): Parameter determining average of log-normal distribution
            for sampling edge timespans. Defaults to 0.01.
        p_std_edge_timespan (Optional[float], optional): Parameter determining standard deviation of log-normal
            distribution for sampling edge timespans. Defaults to 0.005.
        p_std_noise_timestamps (Optional[float], optional): Parameter determining standard deviation for
            perturbations of timestamps. Defaults to 0.05.
        T (Optional[int], optional): Number of latent node types. Defaults to 5.
        d (Optional[int], optional): Dimensionality of messages. Defaults to 100.
        std_noise_messages (Optional[float], optional): Standard deviation for perturbation of messages. Defaults to 0.05.
        seed (Optional[int], optional): Randomness seed. Defaults to 44.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Sources, destinations, messages, timestamps, labels and edge indices.
    """
    assert N % K == 0, "Number of nodes has to be divisible with number of communities!"

    # Set seed for reproducibility.
    np.random.seed(seed)

    (
        p_in,
        p_out,
        mean_lognormal_edge_timespan,
        std_lognormal_edge_timespan,
    ) = calculate_remaining_hyperparameters(
        N=N,
        K=K,
        M=M,
        mean_edge_reoccurrence=mean_edge_reoccurrence,
        ratio_in_out=ratio_in_out,
        t_max=t_max,
        p_mean_edge_timespan=p_mean_edge_timespan,
        p_std_edge_timespan=p_std_edge_timespan,
    )

    # Underlying static structure is generated by a stochastic block model (SBM).
    community_sizes = [(N // K) for _ in range(K)]
    p = []
    for i in range(K):
        p.append([(p_in if i == j else p_out) for j in range(K)])

    G = nx.stochastic_block_model(
        sizes=community_sizes,
        p=p,
        directed=True,
        seed=seed,
    )

    # Calculate the mean message for each type pair.
    mean_messages = calculate_mean_messages(T=T, d=d)
    # Assign nodes random latent type.
    node_types = {node: np.random.randint(low=0, high=T, size=1)[0] for node in G.nodes}

    src, dst, t, msg = [], [], [], []
    for u, v in tqdm(G.edges, desc="Generating synthetic graph"):
        # Sample number of occurrrences of the edge.
        edge_reoccurrences = np.random.poisson(lam=mean_edge_reoccurrence, size=1)[0]
        # Require that at each edge appears at least twice.
        edge_reoccurrences = max(edge_reoccurrences, 2)

        # Sample timespan of the edge.
        edge_timespan = np.random.lognormal(
            mean=mean_lognormal_edge_timespan, sigma=std_lognormal_edge_timespan, size=1
        )[0]
        edge_timespan_start = np.random.uniform(
            low=0, high=(t_max - edge_timespan), size=1
        )[0]
        edge_timespan_end = edge_timespan_start + edge_timespan

        # Equidistant timestamps
        edge_timestamps = np.linspace(
            start=edge_timespan_start,
            stop=edge_timespan_end,
            num=edge_reoccurrences,
        )

        # Interevent time for the edge
        interevent_time = edge_timespan / edge_reoccurrences
        # Standard deviation for perturbations of timestamps of the edge
        std_noise_timestamps = p_std_noise_timestamps * interevent_time

        # Add perturbation to all timestamps, except for the first one and the last one.
        noise_timestamps = np.random.normal(
            loc=0, scale=std_noise_timestamps, size=edge_reoccurrences - 2
        )
        noise_timestamps = np.concatenate(
            [np.array([0]), noise_timestamps, np.array([0])]
        )
        edge_timestamps = edge_timestamps + noise_timestamps

        message_type = (node_types[u], node_types[v])
        messages = np.random.normal(
            loc=mean_messages[message_type],
            scale=std_noise_messages,
            size=(edge_reoccurrences, d),
        )

        for i in range(edge_reoccurrences):
            src.append(u)
            dst.append(v)
            t.append(edge_timestamps[i])
            msg.append(messages[i])

    src = np.array(src)
    dst = np.array(dst)
    t = np.array(t)
    msg = np.array(msg)

    # Sort edges according to timestamps.
    perm = np.argsort(t)
    src = src[perm]
    dst = dst[perm]
    t = t[perm]
    msg = msg[perm]

    # Convert timestamps to integers.
    t = t.astype(int)

    # Set dummy values to labels and indices for compatibility with other datasets.
    label = np.zeros(shape=src.shape)
    idx = np.arange(src.shape[0])

    return src, dst, msg, t, label, idx


if __name__ == "__main__":
    src, dst, msg, t, label, idx = generate_synthetic_graph(
        N=10, K=5, M=20, mean_edge_reoccurrence=5, t_max=1000, T=2, d=4
    )

    for i in range(src.shape[0]):
        print(
            f"idx {idx[i]}: src: {src[i]}, dst: {dst[i]}, t: {t[i]}, label: {label[i]}"
        )
        print(f"Message: {msg[i]}")
