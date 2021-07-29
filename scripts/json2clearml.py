# Script to convert json files to a ClearML dataset.

import os
import json
from glob import glob

from allegroai import Dataset, DatasetVersion, FrameGroup, SingleFrame
from pathlib2 import Path

if __name__ == "__main__":

    import argparse

    from tqdm import tqdm

    # CLI Parameters
    parser = argparse.ArgumentParser(
        description="Upload image/mask pairs from json to ClearML Dataset."
    )
    parser.add_argument("--name", type=str, help="ClearML Dataset name")
    parser.add_argument("--version", type=str, help="ClearML Dataset version")
    parser.add_argument("--workdir", type=str, default="./", help="Data directory")
    parser.add_argument(
        "--jsondir", type=str, default="", help="Json directory, defaults to the workdir"
    )
    parser.add_argument("--uploadURI", type=str, help="Remote bucket URI (GCS or S3)")
    parser.add_argument("--batch", type=int, default=1000, help="How many frames to add at once")
    parser.add_argument("--startix", type=int, default=0, help="Starting index, for resuming")
    args = parser.parse_args()

    # Collect the various .jsons in the main dir
    jsondir = args.workdir if args.jsondir == "" else args.jsondir
    jsons = glob(jsondir + "*.json")
    print(f"Found the following json files: {jsons}")

    # Collect the frameGroups
    # In the metadata of the frameGroup:
    #   'use' is the json file that we found the image in
    frameGroups = []
    for jsonfile in jsons:
        with open(jsonfile) as trainf:
            jsonclass = Path(jsonfile).stem
            for _, item in tqdm(json.load(trainf).items()):
                frameArgs = {"metadata": {"face_landmarks": item["image"]["face_landmarks"]}}
                parent_path, sub_folder, file_name = item["image"]["file_path"].split("/")
                parent_path = "ffhq-unpacked"
                sub_folder = "000" + sub_folder[0:2]
                file_name = "img000" + file_name
                impath = os.path.join(args.workdir, parent_path, sub_folder, file_name)
                imframe = SingleFrame(source=impath)
                frameArgs["image"] = imframe
                frameG = FrameGroup(**frameArgs)
                frameGroups.append(frameG)

    # Pushing things to Allegro
    print(f"Adding {len(frameGroups)} framegroups to dataset {args.name}:{args.version}")
    dataset = DatasetVersion.create_version(dataset_name=args.name, version_name=args.version)
    # Manual batching, so that we can update the progress bar
    bar = tqdm(total=len(frameGroups), desc="Adding frames")
    ix = args.startix
    bar.update(ix)
    while ix < len(frameGroups):
        frameBatch = frameGroups[ix : ix + args.batch]
        ix += args.batch
        dataset.add_frames(
            frameBatch,
            auto_upload_destination=args.uploadURI,
            local_dataset_root_path=args.workdir,
            batch_size=args.batch,
        )
        bar.update(len(frameBatch))
    dataset.commit_version()
