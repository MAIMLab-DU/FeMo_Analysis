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
  - [x] **fm_segmented** not equal to expected values (needed to sorted(self.sensors)) 
- [x] `get_labeled_user_scheme` function test passed
- [x] `extract_detections_for_inference` function test passed
- [x] `extract_detections_for_train` function test passed
- [x] `extract_features` function test passed
- [x] `inference_pipeline` function test passed
- [ ] `training_pipeline` function test passed
- [ ] Create `FeMoDataset` class


Add during build dataset
```

    def save_features(self, filename, data: dict) -> None:
        """Saves the extracted features (and labels) to a .csv file"""

        features = data.get('features', None)
        labels = data.get('labels', None)
        columns = data.get('columns', None)
        
        if features is None:
            self.logger.warning('No features provided')
            return
        else:
            features_df = pd.DataFrame(features, columns=columns)
            header = True if columns is not None else False
            if labels is not None:
                labels_df = pd.DataFrame(labels, columns=['label'])
                features_df = pd.concat([features_df, labels_df], axis=1)
            features_df.to_csv(filename, header=header, index=False)
```