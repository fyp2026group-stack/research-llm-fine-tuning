# Prior local inference observation (user-reported)

A previous command on Ubuntu 22.04 reported:

- GPU: NVIDIA GeForce GTX 1660, 6 GB VRAM (compute capability 7.5)
- PyTorch: 2.14.1+cu126
- Transformers: 5.20.2
- bitsandbytes: 0.50.2
- Model: Qwen/Qwen2.5-1.5B-Instruct, quantized to 4-bit NF4
- Successful model loading, three toy question answers, and stage-1 preflight pass
- Peak PyTorch GPU memory reserved: approximately 1.125 GiB

These values were provided by the researcher and are not reproduced by GitHub CI. The tests related to borrowing policy, not network diagnosis. The conflicting policy case did not explicitly abstain. These observations establish only local **inference feasibility**, not fine-tuning feasibility or any scientific accuracy claim.

The exact pretrained model revision used for the previous run was not recorded and must be pinned for proper experimental comparisons.
