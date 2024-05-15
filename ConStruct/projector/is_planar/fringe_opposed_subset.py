###############################################################################
#
# Adapted from https://github.com/satemochi/is_planar/
#
###############################################################################

from collections import deque


class fringe_opposed_subset:
    """The class of the fringe-opposed subset

    Roughly speaking, a fringe-opposed subset maintains the property of
    the T-alike/T-opposite relations for traversed back edges.
    The T-alike (T-opposite) means that two back edges will be on same
    (different) side(s) of the DFS-tree in every planar drawing.

    The detailed definition for fringe-opposed subsets can be found in [1]_.

    Attributes
    ----------
    c : list of collections.deque of int
        The pair of left side back edges and right side ones.
        The left side back edges and right ones are T-opposed each other.

        This does not maintain a back edge (x, y) itself,
        but the DFS-height of the destination (lowpoint) y.

        The lowest height for back edges in the left is lower than the right
        ones as far as possible. This is not an invariant.

        Each side is ordered according to DFS height, or the traversal order.
        The head has the highest lowpoint height, and the tail has the lowest.

        We use deque, since
            - Back edges are expired from higher ones (popleft).
            - T-alike back edges are concatenated from lower side (extend).
            - And so on...

    References
    ----------
    .. [1] H. de Fraysseix and P. O. de Mendez (2012).
           Tremaux trees and planarity.
           Europ. J. Combin., 33(3):279-293.
           https://core.ac.uk/download/pdf/82483715.pdf
    """

    __slots__ = ["c"]

    def __init__(self, h):
        self.c = [deque([h]), deque()]

    def __repr__(self):
        """Print in terminal with colors
        see https://stackoverflow.com/questions/287871/
        """
        return (
            "\33[1m\33[90m(\33[0m"
            + str(list(self.c[0]))
            + ", "
            + str(list(self.c[1]))
            + "\33[1m\33[90m)\33[0m"
        )

    @property
    def left(self):
        return self.c[0]

    @property
    def right(self):
        return self.c[1]

    @property
    def l_lo(self):
        return self.c[0][-1]

    @property
    def l_hi(self):
        return self.c[0][0]

    @property
    def r_lo(self):
        return self.c[1][-1]

    @property
    def r_hi(self):
        return self.c[1][0]
