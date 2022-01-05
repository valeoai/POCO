
import argparse
import logging
import os
from tqdm import tqdm
import pandas as pd
import trimesh
from lightconvpoint.datasets.dataset import get_dataset
import torch_geometric.transforms as T
import datasets
import numpy as np
from eval.src.eval import MeshEvaluator

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
    parser.add_argument("--gendir", type=str, help="Path to generated data", required=True)
    parser.add_argument("--gtdir", type=str, help="Path to ground truth data", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name", required=True)
    parser.add_argument("--meshdir", type=str, default="")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--npoints", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--logging", type=str, default="INFO")
    parser.add_argument("--num_meshes", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--remove_wall", action="store_true") # specific to SyntheticRooms
    args = parser.parse_args()

    logging.getLogger().setLevel(args.logging)


    DatasetClass = get_dataset(eval("datasets."+args.dataset))
    eval_dataset = DatasetClass(args.gtdir,
                split=args.split,
                filter_name=args.filter,
                num_non_manifold_points=1,
                dataset_size=args.num_meshes,
                )


    evaluator = MeshEvaluator(n_points=args.npoints)

    source_filenames = eval_dataset.filenames

    eval_dicts = []
    for shape_id, source_filename in enumerate(tqdm(source_filenames, ncols=50)):

        # print(source_filename)

        category = eval_dataset.get_category(shape_id)
        object_name = eval_dataset.get_object_name(shape_id)

        if args.meshdir is None:
            pred_name = os.path.join(args.gendir, category, object_name+".ply")
        else:
            pred_name = os.path.join(args.gendir, args.meshdir, category, object_name+".ply")

        ## load the ground truth
        data_shape, data_space = eval_dataset.get_data_for_evaluation(shape_id)
        pointcloud_tgt = data_shape['points'].astype(np.float32)
        normals_tgt = data_shape['normals'].astype(np.float32)
        points_tgt = data_space["points"].astype(np.float32)
        occ_tgt = np.unpackbits(data_space['occupancies']).astype(np.int64)

        # np.savetxt("/root/no_backup/test.xyz", pointcloud_tgt)

        ## load the prediction
        pred_mesh = trimesh.load(pred_name, process=False)
        out_dict = evaluator.eval_mesh(
            pred_mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt, remove_wall=args.remove_wall)

        out_dict['idx'] = shape_id
        out_dict['class'] = eval_dataset.get_class_name(shape_id)
        out_dict['category'] = category
        out_dict['name'] = object_name

        eval_dicts.append(out_dict)

    out_file = os.path.join(args.gendir, 'eval_meshes_full.pkl')
    out_file_class = os.path.join(args.gendir, 'eval_meshes.csv')

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class']).mean()
    eval_df_class.loc['mean'] = eval_df_class.mean()
    eval_df_class.to_csv(out_file_class)
    print(eval_df_class)
