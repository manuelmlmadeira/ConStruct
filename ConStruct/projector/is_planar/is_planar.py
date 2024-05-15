###############################################################################
#
# Adapted from https://github.com/satemochi/is_planar/
#
###############################################################################


from collections import defaultdict
from itertools import islice
from projector.is_planar.fringe import fringe

__all__ = ["is_planar"]


def is_planar(g):
    """
    The top of the left-right algorithm for testing planarity of graphs.

    We can refer its basic concepts and notations in [1]_.

    Parameters
    ----------
    g : networkx.Graph
        A simple undirected graph of NetworkX.

        If g has four interfaces size(), order(), nodes(), and
        g[v] or neighbors(v), then is_planar does not require NetworkX.

            1. size() : returns the number of edges as int
            2. order() : returns the number of vertices as int
            3. nodes(): returns all vertices in g as iterable
            4. g[v] or neighbors(v) : returns the neighbors of v as iterable
                    If you would like to exchange to neighbor(), then
                    just edit in the function lr_algorithm.

    Returns
    -------
    bool
        True if the graph g is planar otherwise False

    References
    ----------
    .. [1] H. de Fraysseix and P. O. de Mendez (2012).
           Tremaux trees and planarity.
           Europ. J. Combin., 33(3):279-293.
           https://core.ac.uk/download/pdf/82483715.pdf
    """

    if g.size() < 9 or g.order() < 5:
        return True
    if g.size() > 3 * g.order() - 6:
        return False
    dfs_heights = defaultdict(lambda: -1)
    for v in g.nodes():
        if dfs_heights[v] < 0:
            dfs_heights[v] = 0
            if not lr_algorithm(g, v, dfs_heights):
                return False
    return True


def lr_algorithm(g, root, dfs_heights):
    """
    The framework of depth-first search (DFS) in the left-right algorithm.

    Parameters
    ----------
    g : networkx.Graph
        A simple undirected graph of NetworkX.

    root : any hashable or immutable variable
        A vertex in g for starting DFS, or DFS root.

    dfs_heights : collections.defaultdict
        This maintains DFS height for each vertex, or the distance
        (the number of vertices on the path) from the vertex root.
        The DFS height of each root is 0.
        In addition, it is used as a checklist visiting vertices, then
        each unvisited vertex is initialized as -1.

        It is not necessary to use defaultdict, else dict such as a map
        from V to a domain [-1, 0, 1, ..., n], where V is the vertex set of g
        and n is the number of vertices in g, may be appropriative.
        However, since lr_algorithm immediately terminates as soon as
        finding a violation against the left-right criterion,
        so we have specified defaultdict.

        Noting that dfs_heights is the call by sharing, it is also
        accessed for checking all components in g are completely traversed.

    Returns
    -------
    bool
        True if the connected component reachable from root is planar,
        otherwise False.
    """

    fringes = [[]]
    dfs_stack = [(root, iter(g[root]))]
    while dfs_stack:
        x, children = dfs_stack[-1]
        try:
            y = next(children)
            if dfs_heights[y] < 0:  # tree edge
                fringes.append([])
                dfs_heights[y] = dfs_heights[x] + 1
                dfs_stack.append((y, iter([u for u in g[y] if u != x])))
            else:
                if dfs_heights[x] > dfs_heights[y]:  # back edge
                    fringes[-1].append(fringe(dfs_heights[y]))
        except StopIteration:
            dfs_stack.pop()
            if len(fringes) > 1:
                try:
                    merge_fringes(fringes, dfs_heights[dfs_stack[-1][0]])
                except Exception:
                    return False
    return True


def merge_fringes(fringes, dfs_height):
    """
    merge fringes and prune back edges

    Parameters
    ----------
    fringes : list of list of fringe
        The stack of fringes of all tree edges have been traversed. Except
        the top of stack, each list of fringes is under construction.

    dfs_height: int
        To be used as expiring condition so that back edges are caused to
        exit from the fringe of the top tree edge (x, y).
        The expired back edges are never crossing in the progress.
    """

    mf = get_merged_fringe(fringes.pop())
    if mf is not None:
        mf.prune(dfs_height)
        if mf.fops:
            fringes[-1].append(mf)


def get_merged_fringe(upper_fringes):
    """
    merge (upper) fringes

    In order to construct the fringe of the tree edge (x, y) that is
    the top of dfs_stack, this function merges all fringes of tree edges
    outgoing from y and back edges outgoing from y.

    Parameters
    ----------
    upper_fringes : list of fringe
        upper_fringes consists of all fringes of tree edges outgoing from y
        and back edges outgoing from y.

    Returns
    -------
    new_fringe : fringe
        Returns new_fringe as the merged fringe if upper_fringes does not
        contain any violation against the left-right criterion.

        new_fringe may be None if upper_fringe is empty. This is the case
        in which the tree edge (on the top of dfs_stack) is a bridge
        (whose deletion increases its number of connected components).
    """

    if len(upper_fringes) > 0:
        upper_fringes.sort()
        new_fringe = upper_fringes[0]
        for f in islice(upper_fringes, 1, len(upper_fringes)):
            new_fringe.merge(f)
        return new_fringe
