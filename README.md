<div align="center">   
  
# EdgeFormer: Local Patch-based Edge Detection Transformer on Point Clouds
</div>

![EdgeFormer Architecture](./figures/EdgeFormer%20Architecture.png)

## Dependency
- PCL 1.8.1
- Python 3.7.16
- Pytorch 1.8.1
- CUDA 11.1
- cuDNN 8.0.5

## Dataset
We used [Chunk 0000](https://archive.nyu.edu/handle/2451/44309) of the [ABC dataset](https://deep-geometry.github.io/abc-dataset/) and [PartNet dataset](https://www.shapenet.org/download/parts).

## Usage
### Pre-processing
For the point clouds from the ABC dataset, the normal vectors are computed using the "Re-Compute Vertex Normals" method available in MeshLab. The PartNet dataset leverages the pre-computed normal vectors included with its point cloud models. 

### Local Patch Encoding
To compute local patch descriptors from point cloud, it is necessary to configure specific parameters to align with your local environment and directory structure. The following settings need to be adjusted accordingly:
- **'full_name'**: Define the complete filename of the output file, appending .ply to the filename.
- **'cloud_path'**: The directory path where the point cloud data files (*.ply) are stored.
- **'feature_path'**: The directory path where the Ground Trouth files (*.ply) are stored.
- **'txt_path'**: The path for saving output files. The `.content` files, which store feature descriptors, and `Label.txt`, which contains point classification labels, are saved in 'txt_path'.

Example Configuration:
```bash
string name = argv[1];
string full_name = name + ".ply";

//point cloud path
string cloud_path = "cloud_path/";
//Ground Truth path
string feature_path = "feature_path/";
//save_path
string txt_path = "save_path/" + name;
```

### EdgeFormer
To train the model, execute the following command in your terminal:
```bash
python train_lr_1e-6.py
```
To test the model, specify the path to the test dataset and the dataset name using the command-line arguments as follows:
```bash
python reload_model.py --test_dataset_path <path> --dataset <dataset_name>
```
For example, use the following command to test `00000009_9b3d6a97e8de4aa193b81000_trimesh_001.ply`:
```bash
python reload_model.py --test_dataset_path ./datasets/test_datasets --dataset 00000009_9b3d6a97e8de4aa193b81000_trimesh_001
```

### Visualization
The predicted labels are mapped back to the respective points in the point cloud. We can see them in `./datasets/test_datasets`.