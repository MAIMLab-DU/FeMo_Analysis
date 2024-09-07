### Data processing pipeline

```
[ load ] → [ preprocess ] → [ segmentation ] → [ fusion ] → 
[ extraction ] → [ build dataset ]
```


- [ ] Check for new `.dat` files (from `data_manifest.json`)
- [x] `load_data` function test passed
- [x] `preprocess_data` function test passed
- [x] `create_imu_map` function test passed
- [x] `create_fm_map` function test passed
- [x] `get_labeled_user_scheme` function test passed
- [x] `extract_detections_for_inference` function test passed
- [ ] `extract_detections_for_train` function test passed
- [ ] `extract_features` function test passed
- [ ] Save features as `.csv` or `.parquet` files
