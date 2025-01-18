# Points2Surf: Learning Implicit Surfaces from Point Clouds (ECCV 2020 Spotlight)

[<img src="https://api.gitsponsors.com/api/badge/img?id=277855446" height="20">](https://api.gitsponsors.com/api/badge/link?p=kkh1dwC2sS0d9de9cEN+aPSJ7YTBQLtpXBXSrzhwHxptWSiTGsSMy6VxIw4cxE7cHxfvPz/UHAcTbNtB0/+L8w==)

Please see our follow-up work [PPSurf](https://github.com/cg-tuwien/ppsurf). It's easier to use, much faster and better.

This is our implementation of [Points2Surf](https://www.cg.tuwien.ac.at/research/publications/2020/erler-2020-p2s/),
a network that estimates a signed distance function from point clouds. This SDF is turned into a mesh with Marching Cubes. For more details, please watch the [short video](https://www.cg.tuwien.ac.at/research/publications/2020/erler-2020-p2s/erler-2020-p2s-short%20video.mp4) and [long video](https://www.cg.tuwien.ac.at/research/publications/2020/erler-2020-p2s/erler-2020-p2s-long%20video.mp4).

![Points2Surf reconstructs objects from arbitrary points clouds more accurately than DeepSDF, AtlasNet and Screened Poisson Surface Reconstruction.](images/teaser.png)

The architecture is similar to [PCPNet](https://github.com/paulguerrero/pcpnet/). In contrast to other ML-based surface reconstruction methods, e.g. [DeepSDF](https://github.com/facebookresearch/DeepSDF) and [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet), Points2Surf is patch-based and therefore independent from classes. The strongly improved generalization leads to much better results, even better than [Screened Poisson Surface Reconstruction](http://hhoppe.com/proj/screenedpoisson/) in most cases.

This code was mostly written by [Philipp Erler](https://philipperler.net/) and [Paul Guerrero](https://paulguerrero.github.io).
This work was published at [ECCV 2020](https://eccv2020.eu/).


## Prerequisites
* Python >= 3.7
* PyTorch >= 1.6
* CUDA and CuDNN if using GPU
* BlenSor 1.0.18 RC 10 for dataset generation


## Quick Start

To get a minimal working example for training and reconstruction, follow these steps. We recommend using [Anaconda](https://anaconda.org/) to manage the Python environment. Otherwise, you can install the required packages with Pip as defined in the requirements.txt.

``` bash
# clone this repo
# a minimal dataset is included (2 shapes for training, 1 for evaluation)
git clone https://github.com/ErlerPhilipp/points2surf.git

# go into the cloned dir
cd points2surf

# create a conda environment with the required packages
conda env create --file p2s.yml

# activate the new conda environment
conda activate p2s

# train and evaluate the vanilla model with default settings
python full_run.py
```


## Reconstruct Surfaces from our Point Clouds

Reconstruct meshes from a point clouds to replicate our results:
``` bash
# download the test datasets
python datasets/download_datasets_abc.py
python datasets/download_datasets_famous.py
python datasets/download_datasets_thingi10k.py
python datasets/download_datasets_real_world.py

# download the pre-trained models
python models/download_models_vanilla.py
python models/download_models_ablation.py
python models/download_models_max.py

# start the evaluation for each model
# Points2Surf main model, trained for 150 epochs
bash experiments/eval_p2s_vanilla.sh

# ablation models, trained to for 50 epochs
bash experiments/eval_p2s_small_radius.sh
bash experiments/eval_p2s_medium_radius.sh
bash experiments/eval_p2s_large_radius.sh
bash experiments/eval_p2s_small_kNN.sh
bash experiments/eval_p2s_large_kNN.sh
bash experiments/eval_p2s_shared_transformer.sh
bash experiments/eval_p2s_no_qstn.sh
bash experiments/eval_p2s_uniform.sh
bash experiments/eval_p2s_vanilla_ablation.sh

# additional ablation models, trained for 50 epochs
bash experiments/eval_p2s_regression.sh
bash experiments/eval_p2s_shared_encoder.sh

# best model based on the ablation results, trained for 250 epochs
bash experiments/eval_p2s_max.sh
```

Each eval script reconstructs all test sets using the specified model. Running one evaluation takes around one day on a normal PC with e.g. a 1070 GTX and grid resolution = 256.

To get the best results, take the Max model. It's 15% smaller and produces 4% better results (mean Chamfer distance over all test sets) than the Vanilla model. It avoids the QSTN and uses uniform sub-sampling.


## Training with our Dataset

To train the P2S models from the paper with our training set:
``` bash
# download the ABC training and validation set
python datasets/download_datasets_abc_training.py

# start the evaluation for each model
# Points2Surf main model, train for 150 epochs
bash experiments/train_p2s_vanilla.sh

# ablation models, train to for 50 epochs
bash experiments/train_p2s_small_radius.sh
bash experiments/train_p2s_medium_radius.sh
bash experiments/train_p2s_large_radius.sh
bash experiments/train_p2s_small_kNN.sh
bash experiments/train_p2s_large_kNN.sh
bash experiments/train_p2s_shared_transformer.sh
bash experiments/train_p2s_no_qstn.sh
bash experiments/train_p2s_uniform.sh
bash experiments/train_p2s_vanilla_ablation.sh

# additional ablation models, train for 50 epochs
bash experiments/train_p2s_regression.sh
bash experiments/train_p2s_shared_encoder.sh

# best model based on the ablation results, train for 250 epochs
bash experiments/train_p2s_max.sh
```

With 4 RTX 2080Ti, we trained around 5 days to 150 epochs. Full convergence is at 200-250 epochs but the Chamfer distance doesn't change much. The topological noise might be reduced, though.

Logging of loss (absolute distance, sign logits and both) with Tensorboard is done by default. Additionally, we log the accuracy, recall and F1 score for the sign prediction. You can start a Tensorboard server with: 
``` bash
bash start_tensorboard.sh
```

## Make your own Datasets

The point clouds are stored as [NumPy](https://numpy.org/) arrays of type np.float32 with ending `.npy` where each line contains the 3 coordinates of a point. The point clouds need to be normalized to the (-1..+1)-range.

A dataset is given by a text file containing the file name (without extension) of one point cloud per line. The file name is given relative to the location of the text file.


### Dataset from Meshes for Training and Reconstruction

To make your own dataset from meshes, place your ground-truth meshes in `./datasets/(DATASET_NAME)/00_base_meshes/`. Meshes must be of a type that [Trimesh](https://trimsh.org/) can load, e.g. `.ply`, `.stl`, `.obj` or `.off`. Because we need to compute signed distances for the training set, these input meshes must represent solids, i.e be manifold and watertight. Triangulated CAD objects like in the [ABC-Dataset](https://archive.nyu.edu/handle/2451/43778) work in most cases. Next, create the text file `./datasets/(DATASET_NAME)/settings.ini` with the following settings:
``` ini
[general]
only_for_evaluation = 0
grid_resolution = 256
epsilon = 5
num_scans_per_mesh_min = 5
num_scans_per_mesh_max = 30
scanner_noise_sigma_min = 0.0
scanner_noise_sigma_max = 0.05
```
When you set `only_for_evaluation = 1`, the dataset preparation script skips most processing steps and produces only the text file for the test set.

For the point-cloud sampling, we used [BlenSor 1.0.18 RC 10](https://www.blensor.org/). Windows users need to fix [a bug in the BlenSor code](https://github.com/mgschwan/blensor/issues/30). Make sure that the `blensor_bin` variable in `make_dataset.py` points to your BlenSor binary, e.g. `blensor_bin = "bin/Blensor-x64.AppImage"`.

You may need to change other paths or the number of worker processes and run: 
```
python make_dataset.py
```

The ABC var-noise dataset with about 5k shapes takes around 8 hours using 15 worker processes on a Ryzen 7. Most computation time is required for the sampling and the GT signed distances.


### Dataset from Point Clouds for Reconstruction

If you only want to reconstruct your own point clouds, place them in `./datasets/(DATASET_NAME)/00_base_pc/`. The accepted file types are the same as for meshes.

You need to change some settings like the dataset name and the number of worker processes in `make_pc_dataset.py` and then run it with:
```
python make_pc_dataset.py
```


### Manually Created Dataset for Reconstruction

In case you already have your point clouds as Numpy files, you can create a dataset manually. Put the *.npy files in the `(DATASET_NAME)/04_pts/` directory. Then, you need to list the names (without extensions, one per line) in a textfile `(DATASET_NAME)/testset.txt`.


## Related Work

### [Kazhdan, Michael, and Hugues Hoppe. "Screened poisson surface reconstruction." ACM Transactions on Graphics (ToG) 32.3 (2013): 1-13.](https://dl.acm.org/doi/abs/10.1145/2487228.2487237)

This work is the most important baseline for surface reconstruction. It fits a surface into a point cloud.


### [Groueix, Thibault, et al. "A papier-mâché approach to learning 3d surface generation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.](https://openaccess.thecvf.com/content_cvpr_2018/html/Groueix_A_Papier-Mache_Approach_CVPR_2018_paper.html)

This is one of the first data-driven methods for surface reconstruction. It learns to approximate objects with 'patches', deformed and subdivided rectangles.


### [Park, Jeong Joon, et al. "Deepsdf: Learning continuous signed distance functions for shape representation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html)

This is one of the first data-driven methods for surface reconstruction. It learns to approximate a signed distance function from points.


### [Chabra, Rohan, et al. "Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction." arXiv preprint arXiv:2003.10983 (2020).](https://arxiv.org/abs/2003.10983)

This concurrent work uses a similar approach as ours. It produces smooth surfaces but requires point normals.


## Citation
If you use our work, please cite our paper:
```
@InProceedings{ErlerEtAl:Points2Surf:ECCV:2020,
  title   = {{Points2Surf}: Learning Implicit Surfaces from Point Clouds}, 
  author="Erler, Philipp
    and Guerrero, Paul
    and Ohrhallinger, Stefan
    and Mitra, Niloy J.
    and Wimmer, Michael",
  editor="Vedaldi, Andrea
    and Bischof, Horst
    and Brox, Thomas
    and Frahm, Jan-Michael",
  year    = {2020},
  booktitle="Computer Vision -- ECCV 2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="108--124",
  abstract="A key step in any scanning-based asset creation workflow is to convert unordered point clouds to a surface. Classical methods (e.g., Poisson reconstruction) start to degrade in the presence of noisy and partial scans. Hence, deep learning based methods have recently been proposed to produce complete surfaces, even from partial scans. However, such data-driven methods struggle to generalize to new shapes with large geometric and topological variations. We present Points2Surf, a novel patch-based learning framework that produces accurate surfaces directly from raw scans without normals. Learning a prior over a combination of detailed local patches and coarse global information improves generalization performance and reconstruction accuracy. Our extensive comparison on both synthetic and real data demonstrates a clear advantage of our method over state-of-the-art alternatives on previously unseen classes (on average, Points2Surf brings down reconstruction error by 30{\%} over SPR and by 270{\%}+ over deep learning based SotA methods) at the cost of longer computation times and a slight increase in small-scale topological noise in some cases. Our source code, pre-trained model, and dataset are available at: https://github.com/ErlerPhilipp/points2surf.",
  isbn="978-3-030-58558-7"
  doi = {10.1007/978-3-030-58558-7_7},
}
```
