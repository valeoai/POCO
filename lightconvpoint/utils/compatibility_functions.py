# compatibility function, to be used to load KPConvSeg networks from previous version
def rename_deprecated_network_weights(deprecated_state_dict, network_state_dict_keys):
    deprecated_state_dict = dict(deprecated_state_dict)
    renamed_state_dict = {}

    for k in network_state_dict_keys:
        if k in deprecated_state_dict:
            renamed_state_dict[k] = deprecated_state_dict[k]
            continue

        k_tmp = k.replace('network.', '')
        if k_tmp in deprecated_state_dict:
            renamed_state_dict[k] = deprecated_state_dict[k_tmp]
            continue
        
        k_tmp = k.replace('cv1.norm.', 'bn1.')
        if k_tmp in deprecated_state_dict and 'resnet' in k_tmp:
            renamed_state_dict[k] = deprecated_state_dict[k_tmp]
            continue
        
        k_tmp = k.replace('cv0.norm.', 'bn0.')
        if k_tmp in deprecated_state_dict and 'resnet' not in k_tmp:
            renamed_state_dict[k] = deprecated_state_dict[k_tmp]
            continue
        else:
            print('Error - missing key', k)

    return renamed_state_dict

## Script used to convert old models

# net.load_state_dict(
#     rename_deprecated_network_weights(
#         torch.load(os.path.join(savedir_root, "checkpoint.pth"))["state_dict"],
#                     list(net.state_dict().keys())))

# ckpt = torch.load(os.path.join(savedir_root, "checkpoint.pth"))
# ckpt["state_dict"] = net.state_dict()
# torch.save(ckpt, os.path.join(savedir_root, "checkpoint2.pth"))
# exit()