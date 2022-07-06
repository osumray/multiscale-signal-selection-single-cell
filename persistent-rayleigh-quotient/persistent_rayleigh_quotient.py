import numpy as np
from numpy.linalg import inv as inverse
from numpy.linalg import pinv as pseudo_inverse


def schur_complement(M, I):
    """Compute the generalised Schur complement of the matrix M with respect to the index I.

    Parameters
    ----------
    M : ndarray
        Matrix on which the Schur complement is performed.
    I : list
        Indices which define the submatrix in M on which the Schur complement is performed.
    
    Returns
    -------
    complement : ndarray
        The Schur complement M/M[I] where M[I] is the submatrix of M with row and column
        indices given by I.
    """
    I_c = [i for i in range(len(M)) if i not in I]
    complement = M[I_c, :][:, I_c] - M[I_c, :][:, I] @ pseudo_inverse(M[I, :][:, I]) @ M[I, :][:, I_c]
    return complement


def weighted_transpose(M, W_1, W_2):
    """Takes the transpose of the matrix M with respect to two inner products.
    The inner product is given by W_1 and W_2 which give the inner product
    weights on the domain and codomain of M respectively.

    Parameters
    ----------
    M : ndarray of shape (n, m)
        Matrix to be transposed.
    W_1 : ndarray of shape (m, m)
        Weights for the inner product on the domain.
    W_2 : ndarray of shape (n, n)
        Weights for the inner product on the codomain.

    Returns
    -------
    M_transpose : ndarray of shape (n, m)
        Weighted transpose of M.
    """
    M_transpose = np.linalg.inv(W_1).T @ M.T @ W_2.T
    return M_transpose


def persistent_laplacian(B_K_q, B_L_qp1, W_K_qm1, W_K_q, W_L_q, W_L_qp1):
    """Compute the qth persistent Laplacian of a simplicial pair K, L.
    Uses algorithm 2 in https://arxiv.org/abs/2012.02808.
    Assumes that the simplices are ordered such that the simplices in K are first.

    Parameters
    ----------
    B_K_q : ndarray of shape (l, n)
        qth boundary matrix of K.
    B_L_qp1 : ndarray of shape (t, s)
        (q+1)th boundary matrix of L
    W_K_qm1 : ndarray of shape (l, l)
        Weights for the inner product on the (q-1)th chain group of K.
    W_K_q : ndarray of shape (n, n)
        Weights for the inner product on the qth chain group of K.
    W_L_q : ndarray of shape (s, s)
        Weights for the inner product on the qth chain group of L.
    W_L_qp1 : ndarray of shape (r, r)
        Weights for the innser product on the (q + 1)th chain group fo L.

    Returns
    -------
    persistent_laplacian : ndarray of shape (n, n)
        qth persistent Laplacian of K within L.
    """
    B_K_q_dual = weighted_transpose(B_K_q, W_K_q, W_K_qm1)
    laplacian_down_K_q = B_K_q_dual @ B_K_q
    B_L_qp1_dual = weighted_transpose(B_L_qp1, W_L_qp1, W_L_q)
    laplacian_up_L_q = B_L_qp1 @ B_L_qp1_dual
    _, n_K_q = B_K_q.shape
    n_L_q, _ = B_L_qp1.shape
    if n_K_q == n_L_q:
        persistent_laplacian = laplacian_up_L_q + laplacian_down_K_q
    else:
        persistent_laplacian_up = schur_complement(laplacian_up_L_q, [i for i in range(n_K_q, n_L_q)])
        persistent_laplacian = persistent_laplacian_up + laplacian_down_K_q
    return persistent_laplacian


class PersistentRayleighQuotient(object):
    """
    Computes the persistent Rayleigh quotient of a graph signal.

    Parameters
    ----------
    normalised : bool
        Whether to calculate the normalised persistent Rayleigh quotient.
    """

    def __init__(self, normalised=False):
        self.normalised = normalised

    def _lower_star_filtration(self, boundary, filtration):
        _, num_edges = boundary.shape
        edge_filtration = np.zeros(num_edges)
        for col in range(num_edges):
            edge_filtration[col] = filtration[np.argwhere(boundary[:, col])].max()
        return edge_filtration

    def _reorder_boundary(self, boundary, filtration):
        edge_filtration = self._lower_star_filtration(boundary, filtration)
        
        node_reorder = np.argsort(filtration)
        self.node_reorder = node_reorder
        filtration = filtration[node_reorder]
        edge_reorder = np.argsort(edge_filtration)
        edge_filtration = edge_filtration[edge_reorder]
        boundary = boundary[node_reorder, :][:, edge_reorder]

        node_sublevel_set_size = [ sum(filtration <= t) for t in filt_numbers]
        edge_sublevel_set_size = [ sum(edge_filtration <= t) for t in filt_numbers]
        boundary_corners = list(zip(node_sublevel_set_size, edge_sublevel_set_size))
         
        return boundary, boundary_corners

    def fit(self, boundary, edge_weights, filtration):
        """
        Define graph and filtration that the persistent Rayleigh quotient
        is taken with respect to.

        Parameters
        ----------
        boundary : ndarray of shape (n, m)
            The boundary matrix for the graph.
            Each edge is given an orientation:
            if e = (u, v) then e_u = -1 and e_v = 1.
            If e is the ith edge and u and v are the jth and kth nodes respectively
            then boundary[j, i] = e_u = -1 and boundary[k, i] = e_v = -1.
            All other entries are zero.
        edge_weights : ndarray of shape (m,)
            Non-negative weight for each edge.
        filtration : ndarray of shape (n,)
            Filtration value for each node.

        Side-effects
        ------------
        self.node_reorder is the ordering of indices of nodes by filtration value.
        self.laplacian[i, j] is the persistent Laplacian between the ith and jth filtration values.
        """
        boundary, boundary_corners = self._reorder_boundary(boundary, filtration)
        W_1 = np.diag(1 / edge_weights)

        num_filts = len(boundary_corners)
        
        self.laplacians = []
        for i in range(num_filts):
            row = []
            for j in range(i, num_filts):
                birth_node, birth_edge = boundary_corners[i]
                death_node, death_edge = boundary_corners[j]
                num_K = birth_node
                num_L = death_node
                W_K_m1 = np.array([]).reshape(0, 0)
                W_K_0 = np.eye(birth_node)
                W_L_0 = np.eye(death_node)
                W_L_1 = W_1[:death_edge, :death_edge]
                K_0 = np.array([]).reshape(0, birth_node)
                L_1 = boundary[:death_node, :death_edge]
                laplacian = persistent_laplacian(K_0, L_1, W_K_m1, W_K_0, W_L_0, W_L_1)
                row.append(laplacian)
            self.laplacians.append(row)

    def _rayleigh_quotient(self, laplacian, signal):
        rq_numerator = signal @ laplacian @ signal
        if self.normalised:
            diag = np.diag(np.diag(laplacian))
            rq_denominator = signal @ diag @ signal
        else:
            rq_denominator = signal @ signal
        if rq_numerator == 0:
            return 0
        elif rq_denominator == 0:
            raise ZeroDivisionError('Denominator in Rayleigh quotient is zero.')
        else:
            rq = rq_numerator / rq_denominator
            return rq
        
    def __call__(self, signal):
        """
        Calculate the persistent Rayleigh quotient of the signal.

        Parameters
        ----------
        signal : ndarray of shape (n,)

        Returns
        -------
        prq : list
            prq[i][j] is the persistent Rayleigh quotient of the signal
            between the ith and jth levels of the filtration.
        """
        signal = signal[self.node_reorder]
        rqs = []
        for birth in self.laplacians:
            rqs_birth = []
            for death in birth:
                signal_restricted = signal[:len(death)]
                rq = self._rayleigh_quotient(death, signal_restricted)
                rqs_birth.append(rq)
            rqs.append(rqs_birth)
        return rqs
       
if __name__ == '__main__':
    test = {'boundary' : np.array([[-1, -1],
                                   [ 1,  0],
                                   [ 0,  1]]),
            'edge_weights' : np.array([1, 1, 1]),
            'filtration' : np.array([1, 0, 0])}
    nprq = PersistentRayleighQuotient(normalised=True)
    nprq.fit(**test)
    print(nprq.laplacians)
    test_signals = [np.array([1, 1, 1]),
                    np.array([1, 1, 0]),
                    np.array([0, 0, 1]),
                    np.array([0, 1, 1])]
    for signal in test_signals:
        print(nprq(signal))
