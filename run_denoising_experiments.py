import torch
import taichi as ti
from torch import optim
from diff_tem.simulation_grad import Simulation
from focal_frequency_loss import FocalFrequencyLoss as FFL
import os
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

def normalize(x):
    shape = x.shape
    x_view = x.view(shape[0], shape[1], -1)
    x_max, _ = torch.max(x_view.detach(), -1, True)
    x_min, _ = torch.min(x_view.detach(), -1, True)
    x_range = x_max - x_min
    x = (x_view - x_min) / x_range
    return x.reshape(*shape)

if __name__ == "__main__":
    os.chdir("./diff_tem/tests/")
    doses = [700]
    ref_sample_nums = [2]
    ti_backend = ti.cpu
    learning_sample_num = 50
    for dose in doses:
        file_path = f"./input_TMV_files/input_TMV_{dose}.txt"
        for ref_sample_num in ref_sample_nums:
            torch.set_default_dtype(torch.float64)
            ti.init(arch=ti_backend)
            simulation = Simulation.construct_from_file(file_path)
            simulation.finish_init_()
            detector = simulation.detectors[0]
            adam_betas = (0.8, 0.992)
            dose_per_img = simulation.electronbeam.dose_per_im  # need to specify dose_per_im in input_MTV.txt file
            configs = {
                "Adam_betas": adam_betas,
                "with_baseline": True,
                "ref_sample_num": ref_sample_num,
                "learning_sample_num": learning_sample_num,
                "loss": "spatial-focal-MSE-log_probs",
                "dose_per_img": dose_per_img,
            }

            ## NOTICE: THIS IS JUST AN EXAMPLE FOR LOGGING USING WANDB, YOU SHOULD CREATE YOUR WANDB PROJECT
            ## AND REPLACE THREE FIELDS: project, entity, id IN wandb.init function !!!!!!
            date_exp = "22052023_03"
            run = wandb.init(project="diff_tem",
                             entity="kaust-nanovis",
                             id=f"""NN_{date_exp}_VVF_values_mse_{ref_sample_num}_dose_per_img={dose_per_img}_unavg""",
                             notes=str(configs),
                             reinit=True)
            use_cuda = True
            if use_cuda:
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

            ffl = FFL(loss_weight=1.0, alpha=1.0)
            ref_results, predetector_vvfs, snode_trees = simulation.generate_micrographs()
            ref_predetector_vvfs = predetector_vvfs[0]  # a list of VVF, length = tilt_num

            # clean Taichi mem
            for snode_tree in snode_trees:
                snode_tree.destroy()
            # clean Taichi mem on older version of Taichi
            ti.reset()

            # set single precision in learning to speed up
            learning_dtype = torch.float32
            learning_complex_dtype = torch.cfloat
            torch.set_default_dtype(learning_dtype)
            # get references
            ref_results = []
            for vvf in ref_predetector_vvfs:
                vvf_value_after_quantization, _log_probs, _poisson = detector.apply_quantization_differentiable(
                    vvf.values, ref_sample_num)
                vvf_values_after_mtf = [detector.apply_mtf(vvf_value_after_quantization[i]) for i in
                                        range(ref_sample_num)]
                ref_results.append(torch.stack(vvf_values_after_mtf))

            ref_results = torch.stack(ref_results) \
                .to(device).to(learning_complex_dtype)  # (tilt_num, sample_num, *resolution)
            repeat = learning_sample_num // ref_sample_num
            ref_results = ref_results.repeat(1, repeat, 1, 1)
            assert ref_results.shape[1] == learning_sample_num
            # random vvf values to be learned
            learnable_vvf_values = []
            for vvf in ref_predetector_vvfs:
                vvf_values = vvf.values
                rand_vvf_values = torch.rand_like(vvf_values, device=device, dtype=learning_complex_dtype) * 30
                rand_vvf_values.requires_grad_(True)
                learnable_vvf_values.append(rand_vvf_values)

            lr = 1.0
            optimizer = optim.Adam(learnable_vvf_values, lr=lr,
                                   betas=adam_betas)  # lr increase in later iterations
            iterations = 1
            i = 0
            ref_vvf_values = torch.stack([vvf.values for vvf in ref_predetector_vvfs]).to(device).to(
                learning_complex_dtype)

            ref_results_real = normalize(ref_results.real)

            # can run this cell multiple times
            trange = tqdm(range(iterations))
            for _ in trange:
                optimizer.zero_grad()
                pred_results = []
                log_likelihoods = []
                for vvf_value in learnable_vvf_values:
                    vvf_value_after_quantization, log_likelihood, poisson = detector.apply_quantization_differentiable(
                        vvf_value, learning_sample_num)
                    vvf_values_after_mtf = [detector.apply_mtf(vvf_value_after_quantization[i]) for i in
                                            range(learning_sample_num)]
                    pred_results.append(torch.stack(vvf_values_after_mtf))
                    log_likelihoods.append(log_likelihood)

                pred_results = torch.stack(pred_results)  # (tilt_num, learning_sample_num, *resolution)
                pred_results_real = normalize(pred_results.real)
                log_likelihoods = torch.stack(log_likelihoods)  # (tilt_num, learning_sample_num, *resolution)
                diff = ref_results_real - pred_results_real
                se = (diff ** 2)
                se_mean = se.detach().mean(dim=(-1, -2), keepdim=True)
                loss = ((se - se_mean) * log_likelihoods).mean()
                loss.backward()
                optimizer.step()
                learned_vvf = torch.stack([vvf_val.detach() for vvf_val in learnable_vvf_values])
                vvf_diff = (((learned_vvf.detach().real.abs()) - ref_vvf_values.real) ** 2).mean().sqrt()
                trange.set_description(f"Loss = {loss.detach()}, vvf_diff = {vvf_diff}, {ref_sample_num=}, {dose=}")
                ffl_metric = ffl(pred_results_real.detach(), ref_results_real)
                wandb.log({
                    "Train/Loss": loss.detach(),
                    "Train/vvf_diff": vvf_diff,
                    "Train/FFL": ffl_metric,
                })
                i += 1

            learned_vvf_values = learned_vvf.detach()
            ref_vvf_real_values = ref_vvf_values
            idx = 1
            proj_vvf_value0 = learned_vvf_values.cpu()[idx]
            proj_vvf_value0.real = proj_vvf_value0.abs()
            ref_proj_vvf_value0 = ref_vvf_real_values.cpu()[idx]
            proj_vvf_value0_after_mtf = detector.apply_mtf(proj_vvf_value0)
            ref_proj_vvf_value0_noise_free = detector.apply_mtf(ref_proj_vvf_value0)
            ref_proj_vvf_value0_after_quantization = detector.apply_quantization(ref_proj_vvf_value0)
            ref_proj_vvf_value0_noisy = detector.apply_mtf(ref_proj_vvf_value0_after_quantization)  # ref noisy
            for j in range(10):
                samples = []
                for i in range(ref_sample_num):
                    ref_proj_vvf_value0_after_quantization = detector.apply_quantization(ref_proj_vvf_value0)
                    ref_proj_vvf_value0_noisy_ = detector.apply_mtf(ref_proj_vvf_value0_after_quantization)  # ref noisy
                    samples.append(ref_proj_vvf_value0_noisy_)
                ref_proj_vvf_value0_avg = torch.stack(samples).mean(dim=0)

                ref_avg = normalize(ref_proj_vvf_value0_avg.real.unsqueeze(0).unsqueeze(0))
                ref_noise_free = normalize(ref_proj_vvf_value0_noise_free.real.unsqueeze(0).unsqueeze(0))
                learned_noise_free = normalize(proj_vvf_value0_after_mtf.real.unsqueeze(0).unsqueeze(0))
                ref_noisy = normalize(ref_proj_vvf_value0_noisy.real.unsqueeze(0).unsqueeze(0))

                ref_avg_ffl = ffl(ref_avg, ref_noise_free)  # ref avg vs. ref noise free
                pred_noise_free_fll = ffl(learned_noise_free, ref_noise_free)  # learned noise free vs. ref noise free
                noisy_ffl = ffl(ref_noisy, ref_noise_free)  # noisy vs. ref noise free
                if pred_noise_free_fll.item() >= ref_avg_ffl.item() and j < 9:
                    continue
                fig, ax = plt.subplots(1, 5, figsize=(24, 5), dpi=250)
                ax[0].imshow(learned_noise_free[0, 0].numpy(), cmap="gray")  # learned noise free
                ax[0].set_title("learned denoised")
                ax[1].imshow(ref_noise_free[0, 0].numpy(), cmap="gray")  # ref noise free
                ax[1].set_title("ref noise free")
                ax[2].imshow(ref_noisy[0, 0].numpy(), cmap="gray")  # ref noisy
                ax[2].set_title("ref noisy")
                ax[3].imshow(ref_avg[0, 0].numpy(), cmap="gray")
                ax[3].set_title("ref avg")
                ax[4].bar(["learned_denoised", "ref_avg", "ref_noisy"],
                          [pred_noise_free_fll.item(), ref_avg_ffl.item(), noisy_ffl.item()])
                ax[4].set_title("Freq Loss Comparison (vs. Ref Noise Free)")
                ax[4].bar_label(ax[4].containers[0], fmt='%.5f')
                fig.suptitle(f"Comparison ({ref_sample_num=})")
                fig.savefig(
                    f"./learn_vvf_values_images/comparison_{ref_sample_num=}_{dose_per_img}_normalized_unavg.png")
                save_data = {
                    "learned_vvf_values": learned_vvf_values,
                    "ref_vvf_values": ref_vvf_real_values,
                    "ref_avg": ref_avg,
                    "ref_noise_free": ref_noise_free,
                    "learned_denoised": learned_noise_free,
                    "ref_noisy": ref_noisy,
                    "training_settings": configs
                }
                torch.save(save_data, f"./learn_vvf_values_images/with_normalization"
                                      f"/run_data_{ref_sample_num=}_{dose_per_img=}_normalized_unavg.pt")
                break

            run.finish()
