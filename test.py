import yaml

with open('/workspaces/maimlab-mlops/src/configs/pipeline-cfg.yaml', 'r') as f:
    pipeline_cfg = yaml.safe_load(f)
print(pipeline_cfg)