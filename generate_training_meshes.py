#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import numpy as np
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def code_to_mesh(experiment_directory, output_directory, checkpoint, keep_normalized=False, cluster=False):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth"),map_location=device
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.to(device)

    decoder.eval()

    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    if(cluster):
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        data_paths_polyaxon = get_data_paths()
        data_source = os.path.join(data_paths_polyaxon['data1'], "USShapeCompletion", specs["DataSource"])
    else:
        data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

    print(len(instance_filenames), " vs ", len(latent_vectors))

    for i, latent_vector in enumerate(latent_vectors):

        dataset_name, class_name, instance_name = instance_filenames[i].split("/")
        instance_name = instance_name.split(".")[0]

        print("{} {} {}".format(dataset_name, class_name, instance_name))

        mesh_dir = os.path.join(
            output_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )
        print(mesh_dir)

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)

        print(instance_filenames[i])

        offset = None
        scale = None

        if not keep_normalized:

            normalization_params = np.load(
                ws.get_normalization_params_filename(
                    data_source, dataset_name, class_name, instance_name
                )
            )
            offset = normalization_params["offset"]
            scale = normalization_params["scale"]

        latent_vector = latent_vector.to(device)
        with torch.no_grad():
            deep_sdf.mesh.create_mesh(
                decoder,
                latent_vector,
                mesh_filename,
                N=256,
                max_batch=int(2 ** 18),
                offset=offset,
                scale=scale,
            )


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--keep_normalization",
        dest="keep_normalized",
        default=False,
        action="store_true",
        help="If set, keep the meshes in the normalized scale.",
    )
    arg_parser.add_argument(
        "--cluster",
        dest="cluster",
        default=False,
        action="store_true",
        help="If set, training on cluster will be enabled",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    if (args.cluster):
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        output_directory = get_outputs_path()
    else:
        output_directory = args.experiment_directory
    code_to_mesh(args.experiment_directory, output_directory, args.checkpoint, args.keep_normalized, cluster=args.cluster)
