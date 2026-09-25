# Models

Model files are not stored in git. Create them with the scripts:

```
models/
  whisper_base/
    tokenizer.json        # scripts/fetch_models.py
    npu/                  # scripts/fetch_models.py  (AI Hub precompiled QNN ONNX, Snapdragon X Elite)
    cpu/                  # scripts/export_cpu_baseline.py  (same model as float ONNX, CPU baseline)
```
