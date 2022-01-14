
import os
import numpy as np
import yaml
from tqdm import tqdm
import logging
import shutil

from sklearn.metrics import confusion_matrix
import torch_geometric.transforms as T

# torch imports
import torch
import torch.nn.functional as F

# lightconvpoint imports
from lightconvpoint.datasets.dataset import get_dataset
import lightconvpoint.utils.transforms as lcp_T
from lightconvpoint.utils.logs import logs_file
from lightconvpoint.utils.misc import dict_to_device

import utils.argparseFromFile as argparse
from utils.utils import wblue, wgreen
import utils.metrics as metrics
import datasets
import networks

from torch.utils.tensorboard import SummaryWriter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_config_file(config, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def main(config):

    
    config = eval(str(config))    
    disable_log = (config["log_mode"] != "interactive")
    device = torch.device(config['device'])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

    savedir_root = os.path.join(config["save_dir"],f"{config['dataset_name']}_{config['experiment_name']}_{config['network_backbone']}_{config['network_decoder']}_{config['filter_name']}")


    logging.getLogger().setLevel(config["logging"])

    # create the network
    N_LABELS = config["network_n_labels"]
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]
    decoder = {'name':config["network_decoder"], 'k': config['network_decoder_k']}

    logging.info("Creating the network")
    def network_function():
        return networks.Network(3, latent_size, N_LABELS, backbone, decoder)
    net = network_function()
    net.to(device)
    logging.info(f"Network -- Number of parameters {count_parameters(net)}")
    
    logging.info("Getting the dataset")
    DatasetClass = get_dataset(eval("datasets."+config["dataset_name"]))
    train_transform = []
    test_transform = []

    # downsample 
    train_transform.append(lcp_T.FixedPoints(config["manifold_points"], item_list=["x", "pos", "normal", "y", "y_object"]))
    test_transform.append(lcp_T.FixedPoints(config["manifold_points"], item_list=["x", "pos", "normal", "y", "y_object"]))
    train_transform.append(lcp_T.FixedPoints(config["non_manifold_points"], item_list=["pos_non_manifold", "occupancies", "y_v", "y_v_object"]))
    test_transform.append(lcp_T.FixedPoints(config["non_manifold_points"], item_list=["pos_non_manifold", "occupancies", "y_v", "y_v_object"]))

    random_rotation_x = config["training_random_rotation_x"]
    random_rotation_y = config["training_random_rotation_y"]
    random_rotation_z = config["training_random_rotation_z"]
    if random_rotation_x is not None and random_rotation_x > 0:
        train_transform += [lcp_T.RandomRotate(random_rotation_x, axis=0, item_list=["pos", "normal", "pos_non_manifold"]),]
    if random_rotation_y is not None and random_rotation_y > 0:
        train_transform += [lcp_T.RandomRotate(random_rotation_y, axis=1, item_list=["pos", "normal", "pos_non_manifold"]),]
    if random_rotation_z is not None and random_rotation_z > 0:
        train_transform += [lcp_T.RandomRotate(random_rotation_z, axis=2, item_list=["pos", "normal", "pos_non_manifold"]),]

    # add noise to data
    if (config["random_noise"] is not None) and (config["random_noise"] > 0):
        train_transform.append(lcp_T.RandomNoiseNormal(sigma=config["random_noise"]))
        test_transform.append(lcp_T.RandomNoiseNormal(sigma=config["random_noise"]))

    if config["normals"]:
        logging.info("Normals as features")
        test_transform.append(lcp_T.FieldAsFeatures(["normal"]))

    # operate the permutations
    train_transform = train_transform + [
                                            lcp_T.Permutation("pos", [1,0]),
                                            lcp_T.Permutation("pos_non_manifold", [1,0]),
                                            lcp_T.Permutation("normal", [1,0]),
                                            lcp_T.Permutation("x", [1,0]),
                                            lcp_T.ToDict(),]
    test_transform = test_transform + [
                                            lcp_T.Permutation("pos", [1,0]),
                                            lcp_T.Permutation("pos_non_manifold", [1,0]),
                                            lcp_T.Permutation("normal", [1,0]),
                                            lcp_T.Permutation("x", [1,0]),
                                            lcp_T.ToDict(),]


    train_transform = T.Compose(train_transform)
    test_transform = T.Compose(test_transform)

    # build the dataset
    train_dataset = DatasetClass(config["dataset_root"], 
                split=config["train_split"], 
                transform=train_transform, 
                network_function=network_function, 
                filter_name=config["filter_name"], 
                num_non_manifold_points=config["non_manifold_points"]
                )
    test_dataset = DatasetClass(config["dataset_root"],
                split=config["val_split"], 
                transform=test_transform, 
                network_function=network_function, 
                filter_name=config["filter_name"], 
                num_non_manifold_points=config["non_manifold_points"],
                dataset_size=config["val_num_mesh"]
                )


    # build the data loaders
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["training_batch_size"],
            shuffle=True,
            num_workers=config["threads"],
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training_batch_size"],
        shuffle=False,
        num_workers=config["threads"],
    )

    # create the optimizer
    logging.info("Creating the optimizer")
    optimizer = torch.optim.Adam(net.parameters(),config["training_lr_start"])
    
    # save the config file in the directory to restore the configuration
    if config["resume"] and os.path.exists(savedir_root):
        checkpoint = torch.load(os.path.join(savedir_root, "checkpoint.pth"), map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"]
        train_iter_count = len(train_loader) * epoch_start
    else:
        if os.path.exists(savedir_root):
            shutil.rmtree(savedir_root)
        os.makedirs(savedir_root, exist_ok=True)
        save_config_file(eval(str(config)), os.path.join(savedir_root, "config.yaml"))
        epoch_start = 0
        train_iter_count = 0

    # create the loss layer
    loss_layer = torch.nn.CrossEntropyLoss()


    # create the summary writer
    logging.info("Creating tensorboard summary writer")
    writer = SummaryWriter(log_dir=os.path.join(savedir_root, "logs_tb"))

    epoch = epoch_start
    while True:

        # break if the number of iterations is reached
        if train_iter_count >= config["training_iter_nbr"]:
            break
        
        net.train()
        error = 0
        cm = np.zeros((N_LABELS, N_LABELS))

        t = tqdm(
            train_loader,
            desc="Epoch " + str(epoch),
            ncols=130,
            disable=disable_log,
        )
        for data in t:

            data = dict_to_device(data, device)
            optimizer.zero_grad()

            outputs = net(data, spectral_only=True)
            occupancies = data["occupancies"]

            loss = loss_layer(outputs, occupancies)
            loss.backward()
            optimizer.step()

            # compute scores
            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            target_np = occupancies.cpu().numpy()
            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(N_LABELS))
            )
            cm += cm_
            error += loss.item()

            # point wise scores on training
            train_oa = metrics.stats_overall_accuracy(cm)
            train_aa = metrics.stats_accuracy_per_class(cm)[0]
            train_iou = metrics.stats_iou_per_class(cm)[0]
            train_aloss = error / cm.sum()

            description = f"Epoch {epoch} | OA {train_oa*100:.2f} | AA {train_aa*100:.2f} | IoU {train_iou*100:.2f} | Loss {train_aloss:.4e}"
            t.set_description_str(wblue(description))

            train_iter_count += 1

            if train_iter_count >= config["training_iter_nbr"]:
                break


        # save the logs
        train_log_data = {
            "OA_train": train_oa,
            "AA_train": train_aa,
            "IoU_train": train_iou,
            "Loss_train": train_aloss,
        }

        # create the root folder
        os.makedirs(savedir_root, exist_ok=True)

        
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(savedir_root, "checkpoint.pth"),
        )

        logs_file(os.path.join(savedir_root, "logs_train.csv"), train_iter_count, train_log_data)


        # tensorboard logging
        writer.add_scalar('Loss/loss_train', train_aloss, train_iter_count)
        writer.add_scalar('Metrics/iou_train', train_iou, train_iter_count)


        # validation
        if (epoch+1)%config["val_interval"]==0:

            net.eval()
            error = 0
            cm = np.zeros((N_LABELS, N_LABELS))
            with torch.no_grad():

                t = tqdm(
                    test_loader,
                    desc="  Test " + str(epoch),
                    ncols=100,
                    disable=disable_log,
                )
                for data in t:
                
                    
                    # data = data.to(device)
                    data = dict_to_device(data, device)
                    # output_data = net(data)
                    # outputs = output_data["outputs"]

                    if config["normals"]:
                        data["x"] = data["normal"]
                    outputs = net(data, spectral_only=True)
                    occupancies = data["occupancies"]

                    loss = loss_layer(outputs, occupancies)

                    outputs = F.softmax(outputs, dim=1)
                    outputs_np = outputs.cpu().detach().numpy()
                    targets_np = occupancies.cpu().numpy()
                    pred_labels = np.argmax(outputs_np, axis=1)
                    cm_ = confusion_matrix(targets_np.ravel(), pred_labels.ravel(), labels=list(range(N_LABELS)))
                    cm += cm_
                    error += loss.item()

                    # point-wise scores on testing
                    test_oa = metrics.stats_overall_accuracy(cm)
                    test_aa = metrics.stats_accuracy_per_class(cm)[0]
                    test_iou = metrics.stats_iou_per_class(cm)[0]
                    test_aloss = error / cm.sum()

                    description = f"Val. {epoch}  | OA {test_oa*100:.2f} | AA {test_aa*100:.2f} | IoU {test_iou*100:.2f} | Loss {test_aloss:.4e}"
                    t.set_description_str(wgreen(description))
            
            # save the logs
            val_log_data = {
                "OA_val": test_oa,
                "AA_val": test_aa,
                "IoU_val": test_iou,
                "Loss_val": test_aloss,
            }
            logs_file(os.path.join(savedir_root, "logs_val.csv"), train_iter_count, val_log_data)

            # tensorboard logging
            writer.add_scalar('Loss/loss_train', test_aloss, train_iter_count)
            writer.add_scalar('Metrics/iou_train', test_iou, train_iter_count)

        epoch += 1



if __name__ == "__main__":


    parser = argparse.ArgumentParserFromFile(description='Process some integers.')
    parser.add_argument('--config_default', type=str, default="configs/config_default.yaml")
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.update_file_arg_names(["config_default", "config"])

    config = parser.parse(use_unknown=True)

    logging.getLogger().setLevel(config["logging"])
    if config["logging"] == "DEBUG":
        config["threads"] = 0
    
    main(config)
