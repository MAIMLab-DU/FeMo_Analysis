### Data processing pipeline

```
[ load ] → [ preprocess ] → [ segmentation ] → [ fusion ] → 
[ extraction ] → [ build dataset]
```


- [ ] Check for new `.dat` files (from `data_manifest.json`)
- [x] `load_data` function test passed
- [x] `preprocess_data` function test passed
- [x] `create_imu_map` function test passed
- [x] `create_fm_map` function test passed
  - [x] **fm_segmented** not equal to expected values (needed to `sorted(self.sensors)`) 
- [x] `get_labeled_user_scheme` function test passed
- [x] `extract_detections_for_inference` function test passed
- [x] `extract_detections_for_train` function test passed
- [x] `extract_features` function test passed
- [x] `inference_pipeline` function test passed
- [x] `training_pipeline` function test passed
- [x] `FeatureRanker` class `ensemble_ranking` function test passed
- [x] Create `FeMoDataset` class
- [x] `FeMoDataset` --> `split_data` method
- [ ] `split_data` with custom `fpd_tpd_ratio`
- [ ] `split_data` by `participants` option