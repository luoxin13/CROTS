import os
import torch
import argparse
import collections


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--pl-weight-path", type=str, default="")
    args.add_argument("--model-weight-keyname", type=str, default="")
    args.add_argument("--save-path", type=str, default="./model.pth")
    args = args.parse_args()

    assert os.path.isfile(args.pl_weight_path), f"{args.pl_weight_path} is not valid"

    pl_weight = torch.load(args.pl_weight_path, map_location="cpu")['state_dict']
    model_weight = collections.OrderedDict()

    for key in pl_weight.keys():
        if key.startswith(f"{args.model_weight_keyname}."):
            model_weight[key.replace(f"{args.model_weight_keyname}.", "")] = pl_weight[key]

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model_weight, args.save_path)
