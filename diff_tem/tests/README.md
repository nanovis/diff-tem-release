# Data Files
All the data in this folder are saved on my drive
# Data File format
Contents in .pt files:
A dictionary containing {
    "learned_vvf_values": learned_vvf_values, # learned vvf_values, complex matrix
    "ref_vvf_values": ref_vvf_values, # reference vvf_values, complex matrix
    "learnable_gains": learnable_gains,
    "ref_avg": ref_avg, # reference avg images with only real values
    "ref_noise_free": ref_noise_free, # reference noise-free images with only real values
    "learned_denoised": learned_noise_free, # learned denoised images with only real values
    "ref_noisy": ref_noisy, # reference noisy images with only real values
    "training_settings": configs
}

where ("training_settings": configs) is a dictionary containing{
    "Adam_betas": adam_betas,
    "with_baseline": True,
    "ref_sample_num": ref_sample_num,
    "learning_sample_num": learning_sample_num,
    "loss": "spatial-focal-MSE-log_probs",
    "dose_per_img": dose_per_img,
}

The code for generating images from vvf_values
```python
def normalize(x):
    shape = x.shape
    x_view = x.view(shape[0], shape[1], -1)
    x_max, _ = torch.max(x_view.detach(), -1, True)
    x_min, _ = torch.min(x_view.detach(), -1, True)
    x_range = x_max - x_min
    x = (x_view - x_min) / x_range
    return x.reshape(*shape)

idx = 1
biases = learnable_gains
proj_vvf_value = learned_vvf_values.cpu()[idx]
proj_vvf_value.real = proj_vvf_value.abs()
ref_proj_vvf_value = ref_vvf_values.cpu()[idx]
proj_vvf_value0_after_mtf = detector.apply_mtf(proj_vvf_value * biases[0] + biases[1])
ref_proj_vvf_value0_noise_free = detector.apply_mtf(ref_proj_vvf_value)
ref_proj_vvf_value_after_quantization = detector.apply_quantization(ref_proj_vvf_value)
ref_proj_vvf_value_noisy = detector.apply_mtf(ref_proj_vvf_value_after_quantization)  # ref noisy
samples = []
for i in range(ref_sample_num):
    ref_proj_vvf_value_after_quantization = detector.apply_quantization(ref_proj_vvf_value)
    ref_proj_vvf_value_noisy_ = detector.apply_mtf(ref_proj_vvf_value_after_quantization)  # ref noisy
    samples.append(ref_proj_vvf_value_noisy_)
ref_proj_vvf_value_avg = torch.stack(samples).mean(dim=0)

ref_avg = normalize(ref_proj_vvf_value_avg.real.unsqueeze(0).unsqueeze(0))
ref_noise_free = normalize(ref_proj_vvf_value0_noise_free.real.unsqueeze(0).unsqueeze(0))
learned_noise_free = normalize(proj_vvf_value0_after_mtf.real.unsqueeze(0).unsqueeze(0))
ref_noisy = normalize(ref_proj_vvf_value_noisy.real.unsqueeze(0).unsqueeze(0))
```
