import argparse
import logging
import os
import pandas as pd
import trimesh
import numpy as np
from datasets import SceneNet
import trimesh
from sklearn.neighbors import KDTree

from multiprocessing import Pool
from functools import partial

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''

    logging.debug(f"distance_p2p - {points_tgt.shape} {points_src.shape}")

    logging.debug("distance_p2p - KDTree construction")
    kdtree = KDTree(points_tgt)

    logging.debug("distance_p2p - query")
    dist, idx = kdtree.query(points_src)
    idx = idx[:,0]

    logging.debug("distance_p2p - normals")
    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


def process_meshes(source_filename, F_threshold):


    source_filename, shape_id = source_filename

    pred_filename = source_filename.split(args.gtdir)[-1]
    if pred_filename[0] == "/":
        pred_filename = pred_filename[1:]

    category, fname = pred_filename.split("/")

    pred_filename = os.path.join(args.gendir, args.meshdir, pred_filename)
    if os.path.splitext(pred_filename)[1] != ".ply":
        pred_filename = pred_filename+".ply"

    logging.info(f"{os.getppid()} - {os.getpid()} - {category} - {fname} - {F_threshold}")

    logging.debug("Sampling source")
    source_mesh = trimesh.load(source_filename)
    source_pts, face_index = trimesh.sample.sample_surface(source_mesh, args.npoints)
    source_nls = source_mesh.face_normals[face_index]

    logging.debug("Sampling prediction")
    pred_mesh = trimesh.load(pred_filename)
    pred_pts, face_index = trimesh.sample.sample_surface(pred_mesh, args.npoints)
    pred_nls = pred_mesh.face_normals[face_index]

    pred_pts = pred_pts.astype(np.float32)
    pred_nls = pred_nls.astype(np.float32)
    source_pts = source_pts.astype(np.float32)
    source_nls = source_nls.astype(np.float32)

    logging.debug("Completeness")
    # Completeness: how far are the points of gt from the prediction
    completeness, completeness_normals = distance_p2p(source_pts.copy(), source_nls.copy(), pred_pts.copy(), pred_nls.copy())
    completeness2 = completeness**2
    recall = (completeness <= F_threshold).mean()
    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    logging.debug("Accuracy")
    # Accuracy: how far are the points of the prediction from the gt
    accuracy, accuracy_normals = distance_p2p(pred_pts.copy(), pred_nls.copy(), source_pts.copy(), source_nls.copy())
    accuracy2 = accuracy**2
    precision = (accuracy <= F_threshold).mean()
    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    logging.debug("Chamfer and F-score")

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    chamferL1 = 0.5 * (completeness + accuracy)

    # Normal correctness
    normals_consistency = 0.5 * (completeness_normals + accuracy_normals)

    # F-Score
    F = 2 * recall * precision / (recall + precision)

    out_dict = {
        'idx': shape_id,
        'class': category,
        'name': fname,
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals_consistency': normals_consistency,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer-L2': chamferL2,
        'chamfer-L1': chamferL1,
        'f-score': F,
    }
    logging.debug(f"{os.getppid()} - {os.getpid()} - {category} - {fname} - done procesing")

    return out_dict


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
    parser.add_argument("--gendir", type=str, help="Path to generated data", required=True)
    parser.add_argument("--gtdir", type=str, help="Path to ground truth data", required=True)
    parser.add_argument("--meshdir", type=str, default="meshes")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--npoints", type=int, default=4000000)
    parser.add_argument("--Fthreshold", type=float, default=0.025)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--logging", type=str, default="INFO")
    parser.add_argument("--num_mesh", type=int, default=None)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.logging)

    eval_dataset = SceneNet(args.gtdir,
                split="test",
                filter_name=args.filter,
                num_non_manifold_points=1,
                )

    source_filenames =eval_dataset.filenames
    F_threshold = args.Fthreshold

    zipped_source_filenames = list(zip(source_filenames, list(range(len(source_filenames)))))
    chunked_filenames = list(chunks(zipped_source_filenames, args.threads))

    eval_dicts = []

    for ch in chunked_filenames:
        with Pool(args.threads) as p:
            chunk_eval_dicts = p.map(partial(process_meshes, F_threshold=F_threshold), ch)
            eval_dicts += chunk_eval_dicts        

    out_file = os.path.join(args.gendir, 'eval_meshes_full.pkl')
    out_file_class = os.path.join(args.gendir, 'eval_meshes.csv')

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.mean()
    eval_df_class.to_csv(out_file_class)

    # Print results
    print(eval_df_class)