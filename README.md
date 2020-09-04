# python_version_Grid_GCN
This is a python version Grid_GCN model


## Reference: 
This code is adapted from https://github.com/Xharlie/Grid-GCN and https://github.com/hetong007/dgl/tree/pointnet/examples/pytorch/pointcloud/pointnet

## Train:
python train_cls.py --model grid_gcn

## line profiler: 
kernprof -l train_tmp.py

## line profiler visualize: 
python -m line_profiler train_tmp.py.lprof
