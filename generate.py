
import os
import logging
import numpy as np
from tqdm import tqdm
import math

from skimage import measure
import open3d as o3d
from scipy.spatial import KDTree
import torch_geometric.transforms as T

# torch imports
import torch
import torch.nn.functional as F

# lightconvpoint imports
from lightconvpoint.datasets.dataset import get_dataset
import lightconvpoint.utils.transforms as lcp_T
from lightconvpoint.utils.logs import logs_file
from lightconvpoint.utils.misc import dict_to_device

import networks
import datasets
import utils.argparseFromFile as argparse

def export_mesh_and_refine_vertices_region_growing_v2(
    network,latent,
    resolution,
    padding=0,
    mc_value=0,
    device=None,
    num_pts=50000, 
    refine_iter=10, 
    simplification_target=None,
    input_points=None,
    refine_threshold=None,
    out_value=np.nan,
    step = None,
    dilation_size=2,
    whole_negative_component=False,
    return_volume=False
    ):

    bmin=input_points.min()
    bmax=input_points.max()

    if step is None:
        step = (bmax-bmin) / (resolution -1)
        resolutionX = resolution
        resolutionY = resolution
        resolutionZ = resolution
    else:
        bmin = input_points.min(axis=0)
        bmax = input_points.max(axis=0)
        resolutionX = math.ceil((bmax[0]-bmin[0])/step)
        resolutionY = math.ceil((bmax[1]-bmin[1])/step)
        resolutionZ = math.ceil((bmax[2]-bmin[2])/step)

    bmin_pad = bmin - padding * step
    bmax_pad = bmax + padding * step

    pts_ids = (input_points - bmin)/step + padding
    pts_ids = pts_ids.astype(np.int)

    # create the volume
    volume = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), np.nan, dtype=np.float64)
    mask_to_see = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), True, dtype=bool)
    while(pts_ids.shape[0] > 0):

        # print("Pts", pts_ids.shape)

        # creat the mask
        mask = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)
        mask[pts_ids[:,0], pts_ids[:,1], pts_ids[:,2]] = True

        # dilation
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i,0])
            yc = int(pts_ids[i,1])
            zc = int(pts_ids[i,2])
            mask[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True

        # get the valid points
        valid_points_coord = np.argwhere(mask).astype(np.float32)
        valid_points = valid_points_coord * step + bmin_pad

        # get the prediction for each valid points
        z = []
        near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
        for pnts in tqdm(torch.split(near_surface_samples_torch,num_pts,dim=0), ncols=100, disable=True):

            latent["pos_non_manifold"] = pnts.unsqueeze(0)
            occ_hat = network.from_latent(latent)

            # get class and max non class
            class_dim = 1
            occ_hat = torch.stack([occ_hat[:, class_dim] , occ_hat[:,[i for i in range(occ_hat.shape[1]) if i!=class_dim]].max(dim=1)[0]], dim=1)
            occ_hat = F.softmax(occ_hat, dim=1)
            occ_hat[:, 0] = occ_hat[:, 0] * (-1)
            if class_dim == 0:
                occ_hat = occ_hat * (-1)


            # occ_hat = -occ_hat.sum(dim=1)
            occ_hat = occ_hat.sum(dim=1)
            outputs = occ_hat.squeeze(0)
            z.append(outputs.detach().cpu().numpy())

        z = np.concatenate(z,axis=0)
        z  = z.astype(np.float64)

        # update the volume
        volume[mask] = z

        # create the masks
        mask_pos = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)
        mask_neg = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)

        
        # dilation
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i,0])
            yc = int(pts_ids[i,1])
            zc = int(pts_ids[i,2])
            mask_to_see[xc,yc,zc] = False
            if volume[xc,yc,zc] <= 0:
                mask_neg[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True
            if volume[xc,yc,zc] >= 0:
                mask_pos[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True

        # get the new points
        
        new_mask = (mask_neg & (volume>=0) & mask_to_see) | (mask_pos & (volume<=0) & mask_to_see)
        pts_ids = np.argwhere(new_mask).astype(np.int)

    volume[0:padding, :, :] = out_value
    volume[-padding:, :, :] = out_value
    volume[:, 0:padding, :] = out_value
    volume[:, -padding:, :] = out_value
    volume[:, :, 0:padding] = out_value
    volume[:, :, -padding:] = out_value

    # volume[np.isnan(volume)] = out_value
    maxi = volume[~np.isnan(volume)].max()
    mini = volume[~np.isnan(volume)].min()

    if not (maxi > mc_value and mini < mc_value):
        return None

    if return_volume:
        return volume

    # compute the marching cubes
    verts, faces, _, _ = measure.marching_cubes(
            volume=volume.copy(),
            level=mc_value,
            )

    # removing the nan values in the vertices
    values = verts.sum(axis=1)
    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
    mesh.remove_vertices_by_mask(np.isnan(values))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)


    if refine_iter > 0:

        dirs = verts - np.floor(verts)
        dirs = (dirs>0).astype(dirs.dtype)

        mask = np.logical_and(dirs.sum(axis=1)>0, dirs.sum(axis=1)<2)
        v = verts[mask]
        dirs = dirs[mask]

        # initialize the two values (the two vertices for mc grid)
        v1 = np.floor(v)
        v2 = v1 + dirs

        # get the predicted values for both set of points
        v1 = v1.astype(int)
        v2 = v2.astype(int)
        preds1 = volume[v1[:,0], v1[:,1], v1[:,2]]
        preds2 = volume[v2[:,0], v2[:,1], v2[:,2]]

        # get the coordinates in the real coordinate system
        v1 = v1.astype(np.float32)*step + bmin_pad
        v2 = v2.astype(np.float32)*step + bmin_pad

        # tmp mask
        mask_tmp = np.logical_and(
                        np.logical_not(np.isnan(preds1)),
                        np.logical_not(np.isnan(preds2))
                        )
        v = v[mask_tmp]
        dirs = dirs[mask_tmp]
        v1 = v1[mask_tmp]
        v2 = v2[mask_tmp]
        mask[mask] = mask_tmp

        # initialize the vertices
        verts = verts * step + bmin_pad
        v = v * step + bmin_pad

        # iterate for the refinement step
        for iter_id in tqdm(range(refine_iter), ncols=50, disable=True):

            # print(f"iter {iter_id}")

            preds = []
            pnts_all = torch.tensor(v, dtype=torch.float, device=device)
            for pnts in tqdm(torch.split(pnts_all,num_pts,dim=0), ncols=100, disable=True):

                
                latent["pos_non_manifold"] = pnts.unsqueeze(0)
                occ_hat = network.from_latent(latent)

                # get class and max non class
                class_dim = 1
                occ_hat = torch.stack([occ_hat[:, class_dim] , occ_hat[:,[i for i in range(occ_hat.shape[1]) if i!=class_dim]].max(dim=1)[0]], dim=1)
                occ_hat = F.softmax(occ_hat, dim=1)
                occ_hat[:, 0] = occ_hat[:, 0] * (-1)
                if class_dim == 0:
                    occ_hat = occ_hat * (-1)


                # occ_hat = -occ_hat.sum(dim=1)
                occ_hat = occ_hat.sum(dim=1)
                outputs = occ_hat.squeeze(0)


                # outputs = network.predict_from_latent(latent, pnts.unsqueeze(0), with_sigmoid=True)
                # outputs = outputs.squeeze(0)
                preds.append(outputs.detach().cpu().numpy())
            preds = np.concatenate(preds,axis=0)

            mask1 = (preds*preds1)>0
            v1[mask1] = v[mask1]
            preds1[mask1] = preds[mask1]

            mask2 = (preds*preds2)>0
            v2[mask2] = v[mask2]
            preds2[mask2] = preds[mask2]

            v = (v2 + v1)/2

            verts[mask] = v

            # keep only the points that needs to be refined
            if refine_threshold is not None:
                mask_vertices = (np.linalg.norm(v2 - v1, axis=1) > refine_threshold)
                # print("V", mask_vertices.sum() , "/", v.shape[0])
                v = v[mask_vertices]
                preds1 = preds1[mask_vertices]
                preds2 = preds2[mask_vertices]
                v1 = v1[mask_vertices]
                v2 = v2[mask_vertices]
                mask[mask] = mask_vertices

                if v.shape[0] == 0:
                    break
                # print("V", v.shape[0])

    else:
        verts = verts * step + bmin_pad


    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

    if simplification_target is not None and simplification_target > 0:
        mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, simplification_target)

    return mesh

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(config):    
    
    config = eval(str(config))  
    logging.getLogger().setLevel(config["logging"])
    disable_log = (config["log_mode"] != "interactive")
    device = torch.device(config["device"])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

    savedir_root = config["save_dir"]

    # create the network
    N_LABELS = config["network_n_labels"]
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]
    decoder = {'name':config["network_decoder"], 'k': config['network_decoder_k']}
    

    logging.info("Creating the network")
    def network_function():
        return networks.Network(3, latent_size, N_LABELS, backbone, decoder)
    net = network_function()
    checkpoint = torch.load(os.path.join(savedir_root, "checkpoint.pth"))
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    net.eval()
    logging.info(f"Network -- Number of parameters {count_parameters(net)}")

    
    logging.info("Getting the dataset")
    DatasetClass = get_dataset(eval("datasets."+config["dataset_name"]))
    test_transform = []

    # downsample 
    if config["manifold_points"] is not None and config["manifold_points"] > 0:
        test_transform.append(lcp_T.FixedPoints(config["manifold_points"], item_list=["x", "pos", "normal", "y", "y_object"]))
    test_transform.append(lcp_T.FixedPoints(1, item_list=["pos_non_manifold", "occupancies", "y_v", "y_v_object"]))

    # add noise to data
    if (config["random_noise"] is not None) and (config["random_noise"] > 0):
        logging.info("Adding random noise")
        test_transform.append(lcp_T.RandomNoiseNormal(sigma=config["random_noise"]))

    if config["normals"]:
        logging.info("Normals as features")
        test_transform.append(lcp_T.FieldAsFeatures(["normal"]))

    # operate the permutations
    test_transform = test_transform + [
                                            lcp_T.Permutation("pos", [1,0]),
                                            lcp_T.Permutation("pos_non_manifold", [1,0]),
                                            lcp_T.Permutation("normal", [1,0]),
                                            lcp_T.Permutation("x", [1,0]),
                                            lcp_T.ToDict(),]
    test_transform = T.Compose(test_transform)

    # build the dataset
    gen_dataset = DatasetClass(config["dataset_root"],
                split=config["test_split"], 
                transform=test_transform, 
                network_function=network_function, 
                filter_name=config["filter_name"], 
                num_non_manifold_points=config["non_manifold_points"],
                dataset_size=config["num_mesh"]
                )

    # build the data loaders
    gen_loader = torch.utils.data.DataLoader(
        gen_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    
    with torch.no_grad():

        gen_dir = f"gen_{config['dataset_name']}"
        gen_dir += f"_{config['test_split']}"
        if config['manifold_points'] <= 0:
            gen_dir += f"_allPts"
        else:
            gen_dir += f"_{config['manifold_points']}"

        if "gen_descriptor" in config:
            gen_dir += f"_{config['gen_descriptor']}"

        savedir_mesh_root = os.path.join(savedir_root, gen_dir)


        for data in tqdm(gen_loader, ncols=100):

            shape_id = data["shape_id"].item()
            category_name = gen_dataset.get_category(shape_id)
            object_name = gen_dataset.get_object_name(shape_id)

            print(f"{shape_id} | {category_name} - {object_name} - {data['pos'].shape}")

            # create the directories
            savedir_points = os.path.join(savedir_mesh_root, "input", category_name)
            os.makedirs(savedir_points, exist_ok=True)
            savedir_mesh = os.path.join(savedir_mesh_root, "meshes", category_name)
            os.makedirs(savedir_mesh, exist_ok=True)

            # if resume skip if the file already exists
            if config["resume"]:
                if os.path.splitext(object_name)[1] == ".ply":
                    if os.path.isfile(os.path.join(savedir_mesh, object_name)):
                        continue
                else:
                    if os.path.isfile(os.path.join(savedir_mesh, object_name+".ply")):
                        continue

            data = dict_to_device(data, device)


            # save the input
            pts = data["pos"][0].transpose(1,0).cpu().numpy()
            nls = data["x"][0].transpose(1,0).cpu().numpy()
            pts = np.concatenate([pts, nls], axis=1)
            pts = pts.astype(np.float16)
            np.savetxt(os.path.join(savedir_points, object_name+".xyz"), pts)

            # auto scale (for big scenes)
            if "gen_autoscale" in config and config["gen_autoscale"]:
                logging.info("Autoscale computation")
                autoscale_target = config["gen_autoscale_target"] # 0.01 # estimated on shapenet 3000
                pos = data["pos"][0].cpu().transpose(0,1).numpy()
                tree = KDTree(pos)
                mean_dist = tree.query(pos, 2)[0].max(axis=1).mean()
                scale = autoscale_target / mean_dist
                logging.info(f"Autoscale {scale}")
            else:
                scale = 1

            # scale the points
            data["pos"] = data["pos"] * scale


            # if too musch points and no subsample iteratively compute the latent vectors
            if data["pos"].shape[2] > 100000 and ("gen_subsample_manifold" not in config or config["gen_subsample_manifold"] is None):
                
                # create the KDTree
                pos = data["pos"][0].cpu().transpose(0,1).numpy()
                tree = KDTree(pos)

                # create the latent storage
                latent = torch.zeros((pos.shape[0], config["network_latent_size"]), dtype=torch.float)
                counts = torch.zeros((pos.shape[0],), dtype=torch.float)
                
                n_views = 3
                logging.info(f"Latent computation - {n_views} views")
                for current_value in range(0,n_views):
                    while counts.min() < current_value+1:
                        valid_ids = np.argwhere(counts.cpu().numpy()==current_value)
                        # print(valid_ids.shape)
                        pt_id = torch.randint(0, valid_ids.shape[0], (1,)).item()
                        pt = pos[valid_ids[pt_id]]
                        k = 100000
                        distances, neighbors = tree.query(pt, k=k)

                        neighbors = neighbors[0]

                        data_partial = {
                            "pos": data["pos"][0].transpose(1,0)[neighbors].transpose(1,0).unsqueeze(0),
                            "x": data["x"][0].transpose(1,0)[neighbors].transpose(1,0).unsqueeze(0)
                        }

                        partial_latent = net.get_latent(data_partial, with_correction=False)["latents"]

                        latent[neighbors] += partial_latent[0].cpu().numpy().transpose(1,0)
                        counts[neighbors] += 1

                latent = latent / counts.unsqueeze(1)
                latent = latent.transpose(1,0).unsqueeze(0).to(device)
                data["latents"] = latent
                latent = data
                logging.info("Latent done")

            elif "gen_subsample_manifold" in config and config["gen_subsample_manifold"] is not None:
                    logging.info("Submanifold sampling")

                    # create the KDTree
                    pos = data["pos"][0].cpu().transpose(0,1).numpy()

                    # create the latent storage
                    latent = torch.zeros((pos.shape[0], config["network_latent_size"]), dtype=torch.float)
                    counts = torch.zeros((pos.shape[0],), dtype=torch.float)


                    iteration = 0
                    for current_value in range(config["gen_subsample_manifold_iter"]):
                        while counts.min() < current_value+1:
                            # print("iter", iteration, current_value)
                            valid_ids = torch.tensor(np.argwhere(counts.cpu().numpy()==current_value)[:,0]).long()
                            
                            if pos.shape[0] >= config["gen_subsample_manifold"]:

                                ids = torch.randperm(valid_ids.shape[0])[:config["gen_subsample_manifold"]]
                                ids = valid_ids[ids]
                                
                                if ids.shape[0] < config["gen_subsample_manifold"]:
                                    ids = torch.cat([ids, torch.randperm(pos.shape[0])[:config["gen_subsample_manifold"] - ids.shape[0]]], dim=0)
                                assert(ids.shape[0] == config["gen_subsample_manifold"])
                            else:
                                ids = torch.arange(pos.shape[0])


                            data_partial = {
                                "pos": data["pos"][0].transpose(1,0)[ids].transpose(1,0).unsqueeze(0),
                                "x": data["x"][0].transpose(1,0)[ids].transpose(1,0).unsqueeze(0)
                            }

                            partial_latent = net.get_latent(data_partial, with_correction=False)["latents"]
                            latent[ids] += partial_latent[0].cpu().numpy().transpose(1,0)
                            counts[ids] += 1

                            iteration += 1

                    latent = latent / counts.unsqueeze(1)
                    latent = latent.transpose(1,0).unsqueeze(0).to(device)
                    data["latents"] = latent
                    latent = data
                
            else:
                # all prediction
                latent = net.get_latent(data, with_correction=False)

            
            if "gen_resolution_metric" in config and config["gen_resolution_metric"] is not None:
                step = config['gen_resolution_metric'] * scale
                resolution = None
            elif config["gen_resolution_global"] is not None:
                step = None
                resolution = config["gen_resolution_global"]
            else:
                raise ValueError("You must specify either a global resolution or a metric resolution")



            print("POS", data["pos"].shape)
            mesh = export_mesh_and_refine_vertices_region_growing_v2(
                net, latent,
                resolution=resolution,
                padding=1,
                mc_value=0,
                device=device,
                input_points=data["pos"][0].cpu().numpy().transpose(1,0),
                refine_iter=config["gen_refine_iter"],
                out_value=1,
                step=step
            )

            
            if mesh is not None:

                vertices = np.asarray(mesh.vertices)
                vertices = vertices / scale
                vertices = o3d.utility.Vector3dVector(vertices)
                mesh.vertices = vertices

                print(os.path.join(savedir_mesh, object_name))

                if os.path.splitext(object_name)[1] == ".ply":
                    o3d.io.write_triangle_mesh(os.path.join(savedir_mesh, object_name), mesh)
                else:
                    o3d.io.write_triangle_mesh(os.path.join(savedir_mesh, object_name+".ply"), mesh)

            else:
                logging.warning("mesh is None")


def replace_values_of_config(config, config_update):

    for key, value in config_update.items():
        if key not in config:
            print(f"replace warning unknown key '{key}'")
            continue
        if isinstance(value, dict):
            config[key] = replace_values_of_config(config[key], value)
        else:
            config[key] = value
    return config

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("trimesh").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParserFromFile(description='Process some integers.')
    parser.add_argument('--config_default', type=str, default="configs/config_default.yaml")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_mesh', type=int, default=None)
    parser.add_argument("--gen_refine_iter", type=int, default=10)

    parser.update_file_arg_names(["config_default", "config"])
    config = parser.parse(use_unknown=True)
    
    logging.getLogger().setLevel(config["logging"])
    if config["logging"] == "DEBUG":
        config["threads"] = 0
    
    config["save_dir"] = os.path.dirname(config["config"])

    main(config)
