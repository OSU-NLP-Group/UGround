We used an in-house infra for the training, which can not be shared.

Here we mainly provide the yaml files to share the critical hyperparameters we used.

From our ablations, we found the following lessons that may be useful for future research:

- We didn't find substantial performance difference between whether using LoRA or not on ScreenSpot.
- We didn't find substantial performance difference between whether freezing or unfreezing ViT during SFT.
- We find that Qwen2-VL is very good at dealing with different image resolutions, which is a critical capability for visual grounding. We tried to train it with only a fixed resolution, and found that it still performs well on different resolutions.
- In our experiments, we find 1344*1344 does achieve the best performance on ScreenSpot. Larger or smaller resolutions do not lead to better performance.



