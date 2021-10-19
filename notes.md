# Notes

These are notes and observations made while working on this project. As of now,
experiments have only been run on SE(2) but none of this project explicitly
requires any assumptions unique to SE(2).

We define the range-only SLAM (RO-SLAM) problem as Pose graph SLAM +
pure range measurements to a static beacon.

## Relaxing Rotations

When solving RO-SLAM it was observed that under certain noise regimes we could
entirely relax away the SO(d) constraint on rotations and still obtain good
solutions (after rounding the solved result back to SO(d) via the SVD).

The relaxation referred to here is effectively the well-studied
chordal-initialization approach. We use the rotation matrix representation of
rotations and noticed that dropping the rotation constraints (i.e. $R^\top @ R =
I$, $det(R) = 1$) still resulted in good solutions after rounding was performed.
It was noticed that under larger noises and with a larger number of poses the
solutions obtained were further from true rotations as observed by how far the
determinant of each solution was from unity (i.e. $\text{error} = det(R) - 1$).

## SOCP Relaxation

We have noted that the range-only SLAM problem can be relaxed to a nonconvex
QCQP, with the nonconvex quadratic (equality) constraints coming from the
rotation matrix ($R^\top @ R = I$) and the distance variables ($||t_{i} - t_{j}||_2^2
= r_{ij}$). As discussed in [Relaxing Rotations](Relaxing Rotations), we can
relax away the rotation constraint. However, the distance constraints cannot be
entirely dropped without substantially effecting the cost function. However, it
is noted that the distance constraints can be relaxed to a second-order cone
constraint ($||t_{i} - t_{j}||_2^2 \leq r_{ij}$), which is a convex constraint.

By performing these two relaxations the range-only SLAM problem can be solved as
a fully convex program.

## Scalability

There were two observed factors affecting runtime: graph sparsity, and number of
variables. Neither of these are particularly surprising.

Graph sparsity played a substantial role when allowing for loop closures between
poses.
