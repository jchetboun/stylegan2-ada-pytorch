# Pytorch Lightning template (ClearML)

[ClearML/AllegroAI](https://app.lightricks.hosted.allegro.ai/) is a machine learning platform for managing projects, experiments and datasets. For projects, they keep track of experiments (logs, configs, artifacts, runtime stats), and the various checkpoints for the model. For datasets, their architecture is based on keeping all the metadata for the dataset on their platform, while leaving the actual data on our servers. They offer advanced querying on the metadata, for both training and exploration, and fairly good visualization tools.

This README details the extra/alternative steps to take, compared to the main `README.md`, to setup and train with a ClearML dataset.

## Cnvrg secrets/Environment variables

For ClearML to work correctly, the following *environment variables* (or *Settings>Secrets* in a Cnvrg project) need to be set in advance:
* `WANDB_API_KEY`: get it from https://wandb.ai/authorize
* `CLEARML_ACCESS_KEY`, `CLEARML_SECRET_KEY`: From your [ClearML profile page](https://app.lightricks.hosted.allegro.ai/profile), go APP CREDENTIALS>Create new credentials, to generate the access and secret key.
* `S3_KEY`, `S3_SECRET`, `S3_REGION`: The credentials for the AWS S3 bucket that you plan to use (should be in 1pass).
Will soon update to use GCS

## Prerun.sh and clearml_template.conf

The installation and configuration of ClearML is done in the following section of `prerun.sh`:
```
pip install --extra-index-url https://shared:HF6w0RbukY@packages.allegro.ai/repository/allegroai/simple allegroai==3.4.0

apt-get install gettext-base
envsubst < /cnvrg/clearml_template.conf > /cnvrg/clearml.conf
mv /cnvrg/clearml.conf /root
```

The file `clearml_template.conf` is a template configuration file, in which we plug the env variables and move it to where ClearML expects it.

## Structure of ClearML datasets

A ClearML dataset is described by its *name*, and it can contain a number of different *versions*.

For a segmentation task where the dataset is made of image/mask pairs, such as this example, a ClearML dataset is basically a list of *FrameGroup* objects, each containing two *SingleFrame* objects (one for image, one for mask). These objects hold only a pointer (such as an S3 URL) to the images, together with a bunch of metadata. We can also add our own metadata, for instance whether a FrameGroup (image/mask pair) belongs to the train/validation set.

## Uploading a dataset to ClearML

1. If necessary run the `scripts/create_json_files.py` script on your data, to generate `train.json` and `validation.json`.
2. Create a new dataset on the ClearML site.
3. Run the `scripts/json2clearml.py` script:
  ```
  python json2clearml.py --name <dataset name> --version <dataset version name> --workdir <data directory> --s3URI <AWS bucket URI/folder>
  ```
  You can put your data in our us-east-1 bucket, with a URI like `s3://ltx-research-data/datasets/allegroai/<dataset name>/`.
4. Check that you can see your data on the ClearML page of the dataset. Until you click *Publish* the data can be modified or deleted.
5. You can see data from just the train/validation sets, by *Switch to advanced filters>Add frame rule* and entering `meta.use:"train"` or `meta.use:"validation"`.

The `example_dataset.zip` dataset has already been uploaded to [template_example_dataset (us-east-1 version)](https://app.lightricks.hosted.allegro.ai/datasets/1e86d1bb014a4ab0baf23631738af0fb/info;version=cd69fd7c99f541dca5867ddd10298a70).

## Training on a ClearML dataset

Once the dataset is registered with ClearML (which also automatically uploads it to the S3 bucket), add/modify the following section in `src/config/datamodule/allegro.yaml`:
```
project: <project name>
task: <name of the experiment, shows up in the Experiments tab of the project>
dataset_name: <dataset name>
dataset_version: <dataset version>
frame_query: <can be used to filter the dataset; can be left blank>
```
Call the training script as usual, there are no extra CLI args associated with allegro. Note that since the datasets are stored in S3 buckets, performance is better on EKS machines.

## Overview of the training code

In `src/datasets/segmentation_allegro_dataset.py`, we define a pytorch `Dataset`-type (abstract) class that represents the ClearML dataset. This is the core initialisation code:
```
dataview = allegroai.DataView()
dataview.add_query(
    dataset_name=self.dataset_name,
    version_name=self.version_name,
    frame_query=self.frame_query,
)
dataview.prefetch_files()
self.data = self.dataview.to_list()
```
The dataview represents the (perhaps filtered) dataset, and calling its `prefetch_files()` method starts immediately downloading the actual data to the local FS. We are also grabbing a list with all the `FrameGroup` objects in the dataview (this is cheap, because it does not contain any actual image data, just URLs and metadata). Whenever pytorch asks for the n-th sample, we map this to the FrameGroup`frame=self.data[n]`.
we convert the FrameGroup to an actual (image,mask) image tuple, by using `get_local_source()` to get to the local cached files:
```
image = imageio.imread(frame["image"].get_local_source())
mask = imageio.imread(frame["mask"].get_local_source())
```

The two pytorch datasets that we need, train and validation, are created in `datasets/segmentation_allegro_data_module.py`, by taking advantage of the `frame_query` argument:
```
self._train_dataset = SegmentationDataset(
    dataset_name=self.config.allegro.dataset_name,
    version_name=self.config.allegro.dataset_version,
    frame_query=query + 'meta.use:"train"',
    transforms=self._get_transforms(is_train=True),
)

self._validation_dataset = SegmentationDataset(
    dataset_name=self.config.allegro.dataset_name,
    version_name=self.config.allegro.dataset_version,
    frame_query=query + 'meta.use:"validation"',
    transforms=self._get_transforms(is_train=False),
)
```
If we add a Allegro Task then we can easily see in the web what code used what dataset with what qeury etc.
to do that we add in  `train.py`, `Task` object at the very start of `main`
```
task = Task.init(project_name=config.allegro.project, task_name=config.allegro.task)
```
This makes the experiment visible to ClearML.
