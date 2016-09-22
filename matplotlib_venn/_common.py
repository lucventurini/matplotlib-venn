"""
Venn diagram plotting routines.
Functionality, common to venn2 and venn3.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

import numpy as np
from functools import reduce
from itertools import combinations
from collections import Counter, OrderedDict
from math import factorial
import warnings
from matplotlib.colors import ColorConverter
from matplotlib_venn._math import *


class VennDiagram:
    """
    A container for a set of patches and patch labels and set labels, which make up the rendered venn diagram.
    This object is returned by a venn2 or venn3 function call.
    """
    id2idx = {'10': 0, '01': 1, '11': 2,
              '100': 0, '010': 1, '110': 2, '001': 3, '101': 4, '011': 5, '111': 6, 'A': 0, 'B': 1, 'C': 2}

    def __init__(self, patches, subset_labels, set_labels, centers, radii):
        self.patches = patches
        self.subset_labels = subset_labels
        self.set_labels = set_labels
        self.centers = centers
        self.radii = radii
        
    def get_patch_by_id(self, id):
        """Returns a patch by a "region id". 
           A region id is a string '10', '01' or '11' for 2-circle diagram or a 
           string like '001', '010', etc, for 3-circle diagram."""
        return self.patches[self.id2idx[id]]

    def get_label_by_id(self, id):
        """
        Returns a subset label by a "region id". 
        A region id is a string '10', '01' or '11' for 2-circle diagram or a 
        string like '001', '010', etc, for 3-circle diagram.
        Alternatively, if the string 'A', 'B'  (or 'C' for 3-circle diagram) is given, the label of the
        corresponding set is returned (or None)."""
        if len(id) == 1:
            return self.set_labels[self.id2idx[id]] if self.set_labels is not None else None
        else:
            return self.subset_labels[self.id2idx[id]]

    def get_circle_center(self, id):
        """
        Returns the coordinates of the center of a circle as a numpy array (x,y)
        id must be 0, 1 or 2 (corresponding to the first, second, or third circle). 
        This is a getter-only (i.e. changing this value does not affect the diagram)
        """
        return self.centers[id]
    
    def get_circle_radius(self, id):
        """
        Returns the radius of circle id (where id is 0, 1 or 2).
        This is a getter-only (i.e. changing this value does not affect the diagram)
        """
        return self.radii[id]

    def hide_zeroes(self):
        """
        Sometimes it makes sense to hide the labels for subsets whose size is zero.
        This utility method does this.
        """
        for v in self.subset_labels:
            if v is not None and v.get_text() == '0':
                v.set_visible(False)


def mix_colors(col1, col2, col3=None):
    """
    Mixes two colors to compute a "mixed" color (for purposes of computing
    colors of the intersection regions based on the colors of the sets.
    Note that we do not simply compute averages of given colors as those seem
    too dark for some default configurations. Thus, we lighten the combination up a bit.
    
    Inputs are (up to) three RGB triples of floats 0.0-1.0 given as numpy arrays.
    
    >>> mix_colors(np.array([1.0, 0., 0.]), np.array([1.0, 0., 0.])) # doctest: +NORMALIZE_WHITESPACE
    array([ 1.,  0.,  0.])
    >>> mix_colors(np.array([1.0, 1., 0.]), np.array([1.0, 0.9, 0.]), np.array([1.0, 0.8, 0.1])) # doctest: +NORMALIZE_WHITESPACE
    array([ 1. ,  1. , 0.04])    
    """
    if col3 is None:
        mix_color = 0.7 * (col1 + col2)
    else:
        mix_color = 0.4 * (col1 + col2 + col3)
    mix_color = np.min([mix_color, [1.0, 1.0, 1.0]], 0)    
    return mix_color


def prepare_venn_axes(ax, centers, radii):
    """
    Sets properties of the axis object to suit venn plotting. I.e. hides ticks, makes proper xlim/ylim.
    """
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    min_x = min([centers[i][0] - radii[i] for i in range(len(radii))])
    max_x = max([centers[i][0] + radii[i] for i in range(len(radii))])
    min_y = min([centers[i][1] - radii[i] for i in range(len(radii))])
    max_y = max([centers[i][1] + radii[i] for i in range(len(radii))])
    ax.set_xlim([min_x - 0.1, max_x + 0.1])
    ax.set_ylim([min_y - 0.1, max_y + 0.1])
    ax.set_axis_off()


def compute_venn_subsets(*args):
    """
    Generic function to calculate the intersection area of each possible combination of
    any number of set-like objects.

    :param args:
    :return:


    >>> compute_venn_subsets({1,2,3}, {2,3,4}, {3,4,5,6})
    {(0, 0, 1): 2, (0, 1, 0): 0, (0, 1, 1): 1, (1, 0, 1): 0, (0, 1, 1): 1, (1, 0, 1): 0, (1, 1, 0): 1, (1, 1, 1): 1}
    >>> compute_venn_subsets(Counter([1,2,3]), Counter([2,3,4]), Counter([3,4,5,6]))
    {(0, 0, 1): 2, (0, 1, 0): 0, (0, 1, 1): 1, (1, 0, 1): 0, (0, 1, 1): 1, (1, 0, 1): 0, (1, 1, 0): 1, (1, 1, 1): 1}
    >>> compute_venn_subsets(Counter([1,1,1]), Counter([1,1,1]), Counter([1,1,1,1]))

    >>> compute_venn_subsets(Counter([1,1,2,2,3,3]), Counter([2,2,3,3,4,4]), Counter([3,3,4,4,5,5,6,6]))
    (2, 0, 2, 4, 0, 2, 2)
    >>> compute_venn_subsets(Counter([1,2,3]), Counter([2,2,3,3,4,4]), Counter([3,3,4,4,4,5,5,6]))
    (1, 1, 1, 4, 0, 3, 1)
    >>> compute_venn_subsets(set([]), set([]), set([]))
    (0, 0, 0, 0, 0, 0, 0)
    >>> compute_venn_subsets(set([1]), set([]), set([]))
    (1, 0, 0, 0, 0, 0, 0)
    >>> compute_venn_subsets(set([]), set([1]), set([]))
    (0, 1, 0, 0, 0, 0, 0)
    >>> compute_venn_subsets(set([]), set([]), set([1]))
    (0, 0, 0, 1, 0, 0, 0)
    >>> compute_venn_subsets(Counter([]), Counter([]), Counter([1]))
    (0, 0, 0, 1, 0, 0, 0)
    >>> compute_venn_subsets(set([1]), set([1]), set([1]))
    (0, 0, 0, 0, 0, 0, 1)
    >>> compute_venn_subsets(set([1,3,5,7]), set([2,3,6,7]), set([4,5,6,7]))
    (1, 1, 1, 1, 1, 1, 1)
    >>> compute_venn_subsets(Counter([1,3,5,7]), Counter([2,3,6,7]), Counter([4,5,6,7]))
    (1, 1, 1, 1, 1, 1, 1)
    >>> compute_venn_subsets(Counter([1,3,5,7]), set([2,3,6,7]), set([4,5,6,7]))
    """

    if len(set(type(_) for _ in args)) > 1:
        raise ValueError("All arguments must be of the same type")
    # We cannot use len to compute the cardinality of a Counter
    set_size = len if type(args[0]) != Counter else lambda x: sum(list(x.values()))
    result = dict()
    empty = type(args[0])()

    for num in range(1, len(args) + 1):
        for combination in combinations(range(len(args)), num):
            key = "".join(["0" if _ not in combination else "1" for _ in range(len(args))])

            minuend = reduce(lambda x, y: x & y, [args[_] for _ in combination])
            # The null container ("empty") is to avoid a crash for the last combination
            subtrahend = reduce(lambda x, y: x | y,
                                [empty] + [args[_] for _ in range(len(args)) if _ not in combination])

            result[key] = set_size(minuend - subtrahend)

    return result


def compute_venn_areas(diagram_areas, num_sets=4, normalize_to=1.0, _minimal_area=1e-6):
    """
    The list of venn areas is given as 7 values, corresponding to venn diagram areas in the following order:
     (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    (i.e. last element corresponds to the size of intersection A&B&C).
    The return value is a list of areas (A_a, A_b, A_c, A_ab, A_bc, A_ac, A_abc),
    such that the total area of all circles is normalized to normalize_to.
    If the area of any circle is smaller than _minimal_area, makes it equal to _minimal_area.

    Assumes all input values are nonnegative (to be more precise, all areas are passed through and abs() function)
    >>> compute_venn_areas((1, 1, 0, 1, 0, 0, 0))
    (0.33..., 0.33..., 0.33..., 0.0, 0.0, 0.0, 0.0)
    >>> compute_venn_areas((0, 0, 0, 0, 0, 0, 0))
    (1e-06, 1e-06, 1e-06, 0.0, 0.0, 0.0, 0.0)
    >>> compute_venn_areas((1, 1, 1, 1, 1, 1, 1), normalize_to=7)
    (4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0)
    >>> compute_venn_areas((1, 2, 3, 4, 5, 6, 7), normalize_to=56/2)
    (16.0, 18.0, 22.0, 10.0, 13.0, 12.0, 7.0)
    """
    # Normalize input values to sum to 1
    total = 0
    for num in range(num_sets + 1):
        total += int(factorial(num_sets) / (factorial(num) * factorial(num_sets - num)))

    if total != len(diagram_areas):
        raise ValueError("Expected {} combinations from {} sets, received {} instead".format(
            total, num_sets, len(diagram_areas)))

    areas = np.array(np.abs(diagram_areas), float)
    total_area = np.sum(areas)
    if np.abs(total_area) < _minimal_area:
        warnings.warn("All circles have zero area")
        return tuple([1e-06] * num_sets + [0] * (len(diagram_areas) - num_sets))
    else:
        areas = areas / total_area * normalize_to
        sums = OrderedDict()
        index = -1
        for num in range(1, num_sets + 1):
            for combination in combinations(range(num_sets), num):
                index += 1
                for inum in range(1, len(combination) + 1):
                    for subcomb in combinations(combination, inum):
                        if subcomb not in sums:
                            sums[subcomb] = set()
                        sums[subcomb].add(index)

        for num in range(num_sets):
            area = np.array([areas[_] for _ in sums[(num,)]]).sum()
            if area < _minimal_area:
                warnings.warn("Circle # {} has zero area".format(num + 1))
                areas[num] = _minimal_area

        newareas = []
        for key in sums:
            newareas.append(np.array([areas[_] for _ in sums[key]]).sum())

        return tuple(newareas)


def compute_venn_colors(set_colors):
    """
    Given any amount of base colors, computes combinations of colors corresponding to all regions of the venn diagram.
    It returns a list of X elements, where X is the number of available combinations.

    >>> compute_venn_colors(['r', 'g', 'b', 'y'])
    (array([ 1.,  0.,  0.]),..., array([ 0.4,  0.2,  0.4]))
    """
    ccv = ColorConverter()
    base_colors = [np.array(ccv.to_rgb(c)) for c in set_colors]
    final = []
    for num in range(len(set_colors)):
        for combination in combinations(range(len(set_colors)), num):
            final.append(mix_colors(*[base_colors[_] for _ in combination]))

    return tuple(final)


def solve_venn_circles(venn_areas, num_sets=3):
    """
    Given the list of "venn areas" (as output from compute_venn3_areas, i.e. [A, B, C, AB, BC, AC, ABC]),
    finds the positions and radii of the three circles.
    The return value is a tuple (coords, radii), where coords is a 3x2 array of coordinates and
    radii is a 3x1 array of circle radii.

    Assumes the input values to be nonnegative and not all zero.
    In particular, the first three values must all be positive.

    The overall match is only approximate (to be precise, what is matched are the areas of the circles and the
    three pairwise intersections).

    >>> c, r = solve_venn_circles((1, 1, 1, 0, 0, 0, 0))
    >>> np.round(r, 3)
    array([ 0.564,  0.564,  0.564])
    >>> c, r = solve_venn_circles(compute_venn3_areas((1, 2, 40, 30, 4, 40, 4)))
    >>> np.round(r, 3)
    array([ 0.359,  0.476,  0.453])
    """

    venn_areas = np.array(venn_areas, float)
    # (A_a, A_b, A_c, A_ab, A_bc, A_ac, A_abc) = list(map(float, venn_areas))
    intersection_areas = [venn_areas[_] for _ in range(num_sets, len(venn_areas) - 1)]
    common_area = venn_areas[-1]
    num_nonzero = sum(np.array(intersection_areas) > tol)
    if num_sets <= 3:
        radii = np.array([np.sqrt( venn_areas[_] /np.pi) for _ in range(num_sets)])
        # Hypothetical distances between circle centers that assure
        # that their pairwise intersection areas match the requirements.
        dists = [find_distance_by_area(radii[i], radii[j], intersection_areas[i])
                 for (i, j) in [(0, 1), (1, 2), (2, 0)]]
    else:
        # TODO: implement for the 4 or 5 cases
        num_nonzero = 0

    # How many intersections have nonzero area?
    # Handle four separate cases:
    #    1. All pairwise areas nonzero
    #    2. Two pairwise areas nonzero
    #    3. One pairwise area nonzero
    #    4. All pairwise areas zero.

    # (A_a, A_b, A_ab) = list(map(float, venn_areas))
    # r_a, r_b = np.sqrt(A_a / np.pi), np.sqrt(A_b / np.pi)
    # radii = np.array([r_a, r_b])
    # if A_ab > tol:
    #     # Nonzero intersection
    #     coords = np.zeros((2, 2))
    #     coords[1][0] = find_distance_by_area(radii[0], radii[1], A_ab)
    # else:
    #     # Zero intersection
    #     coords = np.zeros((2, 2))
    #     # The max here is needed for the case r_a = r_b = 0
    #     coords[1][0] = radii[0] + radii[1] + max(np.mean(radii) * 1.1, 0.2)
    # coords = normalize_by_center_of_mass(coords, radii)
    # return (coords, radii)

    if num_nonzero == num_sets:
        # The "generic" case, simply use dists to position circles at the vertices of a triangle.
        # Before we need to ensure that resulting circles can be at all positioned on a triangle,
        # use an ad-hoc fix.
        for i in range(num_sets):
            

            i, j, k = (i, (i + 1) % num_sets, (i + 2) % num_sets)
            if dists[i] > dists[j] + dists[k]:
                a, b = (j, k) if dists[j] < dists[k] else (k, j)
                dists[i] = dists[b] + dists[a]*0.8
                warnings.warn("Bad circle positioning")
        coords = position_venn3_circles_generic(radii, dists)
    elif num_nonzero == 2:
        # One pair of circles is not intersecting.
        # In this case we can position all three circles in a line
        # The two circles that have no intersection will be on either sides.
        for i in range(3):
            if intersection_areas[i] < tol:
                (left, right, middle) = (i, (i + 1) % 3, (i + 2) % 3)
                coords = np.zeros((3, 2))
                coords[middle][0] = dists[middle]
                coords[right][0] = dists[middle] + dists[right]
                # We want to avoid the situation where left & right still intersect
                if coords[left][0] + radii[left] > coords[right][0] - radii[right]:
                    mid = (coords[left][0] + radii[left] + coords[right][0] - radii[right]) / 2.0
                    coords[left][0] = mid - radii[left] - 1e-5
                    coords[right][0] = mid + radii[right] + 1e-5
                break
    elif num_nonzero == 1:
        # Only one pair of circles is intersecting, and one circle is independent.
        # Position all on a line first two intersecting, then the free one.
        for i in range(3):
            if intersection_areas[i] > tol:
                (left, right, side) = (i, (i + 1) % 3, (i + 2) % 3)
                coords = np.zeros((3, 2))
                coords[right][0] = dists[left]
                coords[side][0] = dists[left] + radii[right] + radii[side] * 1.1  # Pad by 10%
                break
    else:
        # All circles are non-touching. Put them all in a sequence
        coords = np.zeros((3, 2))
        coords[1][0] = radii[0] + radii[1] * 1.1
        coords[2][0] = radii[0] + radii[1] * 1.1 + radii[1] + radii[2] * 1.1

    coords = normalize_by_center_of_mass(coords, radii)
    return (coords, radii)