import trimesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_trimesh_emd(gt_points, gen_mesh, offset, scale, num_mesh_samples=500):
    """
    This function computes the earth mover distance

    gt_points: trimesh.points.PointCloud of just points, sampled from the surface
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method

    """
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    gt_points_np = gt_points.vertices

    # computes euclidian distance
    d = cdist(gen_points_sampled, gt_points_np)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(gen_points_sampled), len(gt_points_np))