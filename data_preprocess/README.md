## Data preprocessing

Our data preprocessing code is largely built upon scripts from [Structured3D](https://github.com/bertjiazheng/Structured3D) and scripts provided by the authors of [MonteFloor](https://openaccess.thecvf.com/content/ICCV2021/papers/Stekovic_MonteFloor_Extending_MCTS_for_Reconstructing_Accurate_Large-Scale_Floor_Plans_ICCV_2021_paper.pdf).

### Structured3D

* Step 1: download and unzip the Structured3D dataset from [here](https://github.com/bertjiazheng/Structured3D). Note only the ```Structured3D_panorama``` folder is required. 
* Step 2: convert the registered multi-view RGB-D panoramas to point clouds:
  ```shell
  python generate_point_cloud_stru3d.py --data_root=Structured3D_panorama
  ```
* Step 3: project point cloud to density map and generate coco format annotation:
  ```shell
  python generate_coco_stru3d.py --data_root=Structured3D_panorama --output=coco_stru3d
  ```

### SceneCAD
* Step 1: download 3D scans (\<scanId\>\_vh_clean_2.ply) from [ScanNet](https://github.com/ScanNet/ScanNet). 
* Step 2: download scan transformation matrics from [here](https://drive.google.com/file/d/1zq5fDeV45ar8FMAlTffa1f1XeIeXjmw8/view?usp=sharing).
* Step 3: download 3D layout annotation for ScanNet from [SceneCAD](https://github.com/skanti/SceneCAD).
* Step 4: extract 2D polygon from the 3D layout annotation:
  ```shell
  python extract_2dfloor.py --scannet_planes_path=scannet_planes --scans_transform_path=scans_transform --out_path=2Dfloor_planes
  ```
* Step 5: project point cloud to density map and generate coco format annotation:
  ```shell
  python generate_coco_scenecad.py --data_root=ScanNet --scannet_floor_path=2Dfloor_planes --scans_transform_path=scans_transform --output=coco_scenecad
  ```

*** ***Clarify details for SceneCAD*** ***: 

1. Unlike Structured3D, SceneCAD is dominated by single-room scenes. Due to the extreme data imbalance, we filtered out multi-room scenes (which are rare) in SceneCAD. Another motivation is to make evaluation on Floor-SP easier. When evaluating the sequential room-wise shortest
path module in Floor-SP, we need to downsample the density map to 64×64 pixels. However, this operation is not proper for multi-room scenes. Please note all methods are benchmarked on the filtered version for a fair comparison.

2. We use slightly different projection methods for Structured3D and SceneCAD. One important difference lies in the quantization operation (see L75-L80 [here](https://github.com/ywyue/RoomFormer/blob/f53cc2d1597836ce935cef1ec7db40b32e695750/data_preprocess/stru3d/PointCloudReaderPanorama.py#L75)) in Structured3D. We directly adopted the code from the authors of MonteFloor and only realized this operation in a late stage and didn't change this. In other words, all methods reported in Tab.1 in the paper are trained and tested on the quantized density maps. For SceneCAD, we didn't apply this quantization operation. However, when performing the cross-data generalization experiments (sec. 4.3 in the paper), we evaluate HEAT and RoomFormer on both the original density map and the quantized ones and report the best metrics.