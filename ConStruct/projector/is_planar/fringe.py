###############################################################################
#
# Adapted from https://github.com/satemochi/is_planar/
#
###############################################################################

from itertools import islice
from collections import deque
from projector.is_planar.fringe_opposed_subset import fringe_opposed_subset as fop


class fringe:
    r"""The class of a fringe of a tree edge

    This class maintains fringe of a tree edge (x, y).
    The definition of the fringe can be found in [1]_.

    Roughly, from Definition 2.2 in [1]_:
        The fringe Fringe(e) of a tree edge e = (x, y) is defined by

        Fringe(e) = {f in E \ T : f >= e and low(f) < x}.

            - E : edge set of a given graph
            - T : tree edge set of a DFS tree (T is subset of E)
            - f : any back edge
            - low : destination vertex if for a back edge
            - Each binary relation compares between heights or the distance
              from the DFS root.

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        In other words, the fringe of a tree edge e = (x, y) is the set of
        all back edges linking a vertex in the subtree rooted at y and
        a vertex strictly smaller than x.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    !!!Caution!!!
        The above is excerpt from the literature, we have a little correction.
        We omit "strictly", thus our fringe contains back edges incoming to x.

    Attributes
    ----------
    fops : collections.deque of fringe_opposed_subset
        The fringe of a tree edge (x, y) mainly consists of
        the fringe-opposed subsets for each tree edges outgoing from y.
        More strictly, it also contains one-sided single fringe-opposed
        subsets for each back edges outgoing from y.

        The number of elements in fops is less than or equal to
            n1 (the number of tree edges outgoing from y) +
            n2 (the number of back edges outgoing from y).

        This is important when we evaluate time complexities.
        If for each backtracking on DFS, the total time complexity of
        merging fringes is in O(n1 + n2), then is_planr is in linear time.

    H : The static pointer to the fringe-opposed subset containing a back edge
        with the highest dfs_height in self.fops under __lt__.

    L : The static pointer to fringe-opposed subset containing a back edge
        with the lowest dfs_height in self.fops under __lt__.

    Methods
    _______
    merge(other)
        Merge other fringe into self one.

    prune(dfs_height)
        Prune back edges in which the DFS height is equal to dfs_height.

    References
    ----------
    .. [1] H. de Fraysseix and P. O. de Mendez (2012).
           Tremaux trees and planarity.
           Europ. J. Combin., 33(3):279-293.
           https://core.ac.uk/download/pdf/82483715.pdf
    """

    __slots__ = ["fops"]

    def __init__(self, dfs_h=None):
        self.fops = deque() if dfs_h is None else deque([fop(dfs_h)])

    def __lt__(self, other):
        """
        The ordering criterion for fringes

        We intend to pack as many back edges as possible into the left side
        of a fringe-opposed subset.
        Anyway, it is important to note which fringe is nested in the other.
        """

        diff = self.L.l_lo - other.L.l_lo
        if diff != 0:
            return diff < 0
        return self.H.l_hi < other.H.l_hi

    def __repr__(self):
        """print in terminal with colors
        see https://stackoverflow.com/questions/287871/
        """
        return (
            "\33[1m\33[91m{\33[0m"
            + " ".join([repr(c) for c in self.fops])
            + "\33[91m\33[1m}\33[0m"
        )

    @property
    def H(self):
        return self.fops[0]

    @property
    def L(self):
        return self.fops[-1]

    def merge(self, other):
        """
        Merge other fringe into self one

        Parameters
        ----------
        other : fringe
            Eiter
            - a fringe of a tree edge outgoing from the vertex y,
              where y is the destination of the tree edge (x, y)
              corresponding to self, or
            - a fringe consisting of a single back edge outgoing from y.

        Notes
        -----
        We would like to cofirm correctness.
        There are two facts easily checkable:

            - At each raising an exception, we can construct Kuratowski
              subgraph (a subdivision of K5 or K3,3), from the lowest
              back edge or an edge with the onion violation (see
              _make_onion_structure_example.png) and the two-sided
              fringe-opposed subset. Anyways, the induced subgraph of union of
              edges in four fundamental cycles makes a Kuratowski subgraph.
              Thereby, if given graph is planar, we never return False
              as non-planar.
                  (see _merge_t_alike_edges and _make_onion_structure)
                  (see also An_extraction_of_Kuratowski_subgraph.png)

            - The remaining is the case we are given non-planar graph,
              and incorrectly return True as planar. It's never happend.
              Since after all tree edges are processed, all back edges
              could have been colored either 1 (left) or -1 (right) when
              it's pruned, depending on which side of the fringe-opposed
              subset it belongs to.
              This coloring admits the F-coloring defined in [1]_.
                  (see _align_duplicates and prune)

        This sketch is leading to valid proofs, isn't it?
        """

        other._merge_t_alike_edges()
        self._merge_t_opposite_edges_into(other)
        if not self.H.right:
            other._align_duplicates(self.L.l_hi)
        else:
            self._make_onion_structure(other)
        if other.H.left:
            self.fops.appendleft(other.H)

    def _merge_t_alike_edges(self):
        """
        Merge back edges into the left side of self fringe-opposed subset
        as T-alike ones.

        This is called for a fringe of a tree edge nested in some fringe
        containing lower DFS height.
        So, all edges should be T-alike each other.

        Raises
        ------
        Exception
            If self is two sided, then non-planar.

        Notes
        -----
        Invariants:
            - Each left/right side of a fringe-opposed subset is
              sorted by the DFS height in decreasing order.
        """

        if self.H.right:
            raise Exception
        for f in islice(self.fops, 1, len(self.fops)):
            if f.right:
                raise Exception
            self.H.left.extend(f.left)
        self.fops = deque([self.fops[0]])

    def _merge_t_opposite_edges_into(self, other):
        """
        Move back edges in self to the right side of the fringe-opposed
        subset of other as T-opposit back edges.

        Parameters
        ----------
        other : firnge
            A fringe consists of a one-sided fringe-opposed subset, and
            nested in self two-sided fringe-opposed subset.

        Notes
        -----
            See '_merge_t_opposite_edges_into.png'.
        """

        while not self.H.right and self.H.l_hi > other.H.l_lo:
            other.H.right.extend(self.H.left)
            self.fops.popleft()

    def _align_duplicates(self, dfs_h):
        """
        Reap a duplicated (namely unnecessary) back edge and
        align a property of a fringe-opposed subset.

        Parameters
        ----------
        dfs_h : int
            The DFS height of a boundary condition of the left-right criterion;
            When the boundary condition holds, dfs_h should be coincidence
            point between self highest back edge in the left side edge with
            other lowest.
            We can assume _align_duplicates is called when self is one-sided
        """

        if self.H.l_lo == dfs_h:
            self.H.left.pop()
            self._swap_side()

    def _swap_side(self):
        """
        Swap side if the left side of the highest fop is empty

        Notes
        -----
        Invariants:
            - Not permited; in any fringe-opposed subset,
              the left side is empty while the right side is not empty.
        """

        if not self.H.left or (self.H.right and self.H.l_lo > self.H.r_lo):
            self.H.c[0], self.H.c[1] = self.H.c[1], self.H.c[0]

    def _make_onion_structure(self, other):
        """
        Catch up in which T-opposed back edges still conforms
        the left-right criterion in constant time.

            See '_make_onion_structure_example.png', for example.

        Parameters
        ----------
        other : fringe
            A fringe consists of a one-sided fringe-opposed subset, and
            nested in self two-sided fringe-opposed subset.

        Raises
        ------
        Exception
            If other cannot penetrate correctly in self fringe-opposed subset,
            thus Kuratowski subgraph can be extracted from self two-sided
            fop and other back edge (corresponding to other.H.l_lo).

        Notes
        -----
        We would have used the term of onion structures analogous to
        the k-outer planarity in Baker's technique for designing PTASs, or
        the convex/onion layers for counting crossing-free structures.
        """

        lo, hi = (0, 1) if self.H.l_hi < self.H.r_hi else (1, 0)
        if other.H.l_lo < self.H.c[lo][0]:
            raise Exception
        elif other.H.l_lo < self.H.c[hi][0]:
            self.H.c[lo].extendleft(reversed(other.H.left))
            self.H.c[hi].extendleft(reversed(other.H.right))
            other.H.left.clear()
            other.H.right.clear()

    def prune(self, dfs_height):
        """
        Prune back edges whose DFS height greater than or
        equal to dfs_height.

        Parameters
        ----------
        dfs_height : int
            To be used as a threshold whether is a back edge survived.

        Notes
        -----
        We have to prove two statements;
            1) prune completely remove all back edges of height dfs_height
            2) in O(n1) time, where n1 is the number of tree edges outgoing y.

            The second one is obvious. We note that the number of back
            edges outgoing from y are omitted, since our graph is simple.

            The first one is...
            First of all,
                - we can assume that self fringe conforms the left-right
                  criterion, and
                - 'while loop' is progressed from inner to outer in
                  the onion structure.
            If self.fops consists of one fringe-opposed subset (fop) then
            obvious, since it just check the highest in the both sides.
            If self.fops has at least two fops containing the back
            edge of dfs_height, each inner fop must be one-sided and
            must consist of just one back edge, excluding the maximal
            exterior one if exists. Q.E.D.
        """

        left_, right_ = self.__lr_condition(dfs_height)
        while self.fops and (left_ or right_):
            if left_:
                self.H.left.popleft()
            if right_:
                self.H.right.popleft()
            if not self.H.left and not self.H.right:
                self.fops.popleft()
            else:
                self._swap_side()
            if self.fops:
                left_, right_ = self.__lr_condition(dfs_height)

    def __lr_condition(self, dfs_height):
        return (
            self.H.left and self.H.l_hi >= dfs_height,
            self.H.right and self.H.r_hi >= dfs_height,
        )
