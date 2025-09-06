import torch
import matplotlib.pyplot as plt
from snntorch import spikeplot as splt
import numpy as np


def plot_neuron_mem_traces(in_curr_traces, mem_traces, out_spk_traces, num_time_steps, thr_line=1, title=None, vline=False, slack=0.2):
    in_curr_t = torch.from_numpy(np.array(in_curr_traces))
    mem_t = torch.from_numpy(np.array(mem_traces))
    out_spk_t = torch.from_numpy(np.array(out_spk_traces))

    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input current
    ax[0].plot(in_curr_t, linestyle="dashed", marker=".", c="tab:orange", label="$I_{in}$")
    ax[0].set_ylim([torch.min(in_curr_t).item()-slack, torch.max(in_curr_t).item()+slack])
    ax[0].set_xlim([0, num_time_steps-1])
    ax[0].set_ylabel("Input Current $I_{in}$")
    #ax[0].set_ylabel("Before Neuron Update")
    if title:
        ax[0].set_title(title)
    #ax[0].legend()



    # Plot membrane potential
    ax[1].plot(mem_t, linestyle="dashed", marker=".", c="tab:blue")
    ax[1].set_ylim([torch.min(mem_t).item()-slack, torch.max(mem_t).item()+slack])
    ax[1].set_ylabel("Membrane Potential $U_{mem}^{next}$\n(post reset)")
    #if thr_line!=None:
    ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(out_spk_t, ax[2], s=400, c="black", marker="|")
    if vline:
        ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    plt.ylabel("Output spikes")
    plt.yticks([])


    plt.savefig("v_mem_traces.svg", format="svg")

    plt.show()



import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_neuron_compare(
    gpu_in_curr=None, gpu_mem=None, gpu_spk=None,
    npu_mem=None, npu_spk=None,
    num_time_steps=None, thr_line=1, title=None, slack=0.2
):
    fig_rows = 4 if (gpu_mem is not None and npu_mem is not None) else 3
    fig, ax = plt.subplots(fig_rows, figsize=(8, 7), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1, 0.4, 0.6][:fig_rows]})

    if gpu_in_curr is not None:
        in_curr_t = torch.as_tensor(np.array(gpu_in_curr))
        ax[0].plot(in_curr_t, linestyle="dashed", marker=".", c="tab:orange", label="GPU $I_{in}$")
        ax[0].set_ylim([in_curr_t.min().item()-slack, in_curr_t.max().item()+slack])
        ax[0].set_ylabel("Input Current $I_{in}$\n(GPU only)")
        ax[0].legend()
        if title:
            ax[0].set_title(title)

    # --- Membrane potentials ---
    row = 1 if gpu_in_curr is not None else 0
    if gpu_mem is not None:
        mem_gpu_t = torch.as_tensor(np.array(gpu_mem))
        ax[row].plot(mem_gpu_t, linestyle="dashed", marker=".", c="b", label="GPU $U_{mem}^{next}$")
    if npu_mem is not None:
        mem_npu_t = torch.as_tensor(np.array(npu_mem))
        ax[row].plot(mem_npu_t, linestyle="dotted", marker="x", c="r", label="NPU $U_{mem}^{next}$")

    if gpu_mem is not None or npu_mem is not None:
        combined = []
        if gpu_mem is not None: combined.append(mem_gpu_t)
        if npu_mem is not None: combined.append(mem_npu_t)
        min_val = min([c.min().item() for c in combined])
        max_val = max([c.max().item() for c in combined])
        ax[row].set_ylim([min_val - slack, max_val + slack])
        ax[row].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        ax[row].set_ylabel("Membrane Potential $U_{mem}^{next}$\n(Post Reset)")
        ax[row].legend()

    # --- Output spikes ---
    row += 1
    if gpu_spk is not None:
        ax[row].scatter(np.where(gpu_spk)[0], np.ones_like(np.where(gpu_spk)[0]),
                        marker="|", color="b", label="GPU Spikes", s=200)
    if npu_spk is not None:
        ax[row].scatter(np.where(npu_spk)[0], np.ones_like(np.where(npu_spk)[0])*0.5,
                        marker="|", color="r", label="NPU Spikes", s=200)
    ax[row].set_ylabel("Spikes")
    ax[row].set_yticks([])
    ax[row].legend()

    # --- Error plot (only if both GPU & NPU mem available) ---
    if gpu_mem is not None and npu_mem is not None:
        row += 1
        err = mem_gpu_t - mem_npu_t
        ax[row].plot(err, linestyle="dashed", marker=".", c="k")
        ax[row].set_ylabel("NPU vs GPU Membrane\nPotential Difference")
        ax[row].axhline(0, color="black", linewidth=1, alpha=0.5)

    ax[row].set_xlabel("Time step")
    plt.tight_layout()
    plt.savefig("mem_traces_comp.svg")
    plt.show()







v_th = 1


time_step_arr = [
0,
1,
2,
3,
4,
5,
6,
7,
8,
9,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
2,
2,
2,
2,
2]


npu_v_mem = [
0.117647,
0.882353,
0.176471,
0.882353,
-0.352941,
-0.294118,
-0.588235,
-0.411765,
-0.176471,
-0.470588,
-0.764706,
-1.000000,
0.647059,
1.411765,
2.117647,
0.470588,
-0.352941,
0.000000,
-0.117647,
-1.529412,
-2.529412,
-3.470588,
-2.941176,
-2.823529,
-2.529412]

npu_out_spk = [
0,
0,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
1,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0]

npu_mem_t = torch.from_numpy(np.array(npu_v_mem))
npu_out_spk_t = torch.from_numpy(np.array(npu_out_spk))
#in_curr_t = torch.from_numpy(np.array(in_curr))

#plot_neuron_mem_traces(in_curr_t, v_mem_t, out_spk_t, 25, thr_line=v_th, title="NPU: 784x56x56x56x10\nActivity of Neuron 45 in Layer 0 (Test sample 7)")





#gpu_in_curr_traces =  [-1.3277696371078491, -3.614259719848633, -2.4972097873687744, -2.193068265914917, -2.614798069000244, -1.0794700384140015, -1.0794700384140015, -1.1968574523925781, -1.0794700384140015, -1.0794700384140015, -1.0794700384140015, -1.0794700384140015, -1.0794700384140015, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526, -0.5650309324264526]
#gpu_mem_traces =  [-1.3277696371078491, -4.875640869140625, -7.129068374633789, -8.965682983398438, -11.132196426391602, -11.655056953430176, -12.151774406433105, -12.741043090820312, -13.18346118927002, -13.603757858276367, -14.003040313720703, -14.38235855102539, -14.742711067199707, -14.570606231689453, -14.407106399536133, -14.251782417297363, -14.10422420501709, -13.964043617248535, -13.830872535705566, -13.704360008239746, -13.584173202514648, -13.469995498657227, -13.361526489257812, -13.2584810256958, -13.160588264465332]
#gpu_out_spk_traces =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#gpu_in_curr_t = torch.from_numpy(np.array(gpu_in_curr_traces))
#gpu_mem_t = torch.from_numpy(np.array(gpu_mem_traces))
#gpu_out_spk_t = torch.from_numpy(np.array(gpu_out_spk_traces))


#plot_cur_mem_spk(gpu_in_curr_t, gpu_mem_t, gpu_out_spk_t, thr_line=v_th, title="GPU")




gpu_in_curr_traces =  [0.13456737995147705, 0.7681397795677185, 0.3855193257331848, 0.7030320167541504, -1.14237380027771, 0.11637416481971741, -0.3099183142185211, 0.14886660873889923, 0.19499672949314117, -0.2447560876607895, -0.32330241799354553, -0.28454914689064026, 1.5767323970794678, 1.8427287340164185, 1.7133312225341797, -0.5321261882781982, -0.7516416311264038, 0.3340177834033966, -0.13212652504444122, -1.4026563167572021, -1.0203237533569336, -1.0769243240356445, 0.3695564270019531, -0.005098741501569748, 0.1344054490327835]
gpu_mem_traces =  [0.13456737995147705, 0.8959788084030151, 0.23669922351837158, 0.92789626121521, -0.26087236404418945, -0.13145458698272705, -0.4348001778125763, -0.2641935646533966, -0.055987149477005005, -0.2979438900947571, -0.6063491106033325, -0.8605808019638062, 0.7591806650161743, 1.563950538635254, 2.1990842819213867, 0.5570038557052612, -0.22248798608779907, 0.12265419960021973, -0.015605032444000244, -1.4174811840057373, -2.3669309616088867, -3.3255085945129395, -2.7896766662597656, -2.6552915573120117, -2.3881216049194336]
gpu_out_spk_traces =  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

gpu_in_curr_t = torch.from_numpy(np.array(gpu_in_curr_traces))
gpu_mem_t = torch.from_numpy(np.array(gpu_mem_traces))
gpu_out_spk_t = torch.from_numpy(np.array(gpu_out_spk_traces))


#plot_neuron_mem_traces(gpu_in_curr_t, gpu_mem_t, gpu_out_spk_t, 25, thr_line=v_th, title="GPu: 784x56x56x56x10\nActivity of Neuron 45 in Layer 0 (Test sample 7)")


plot_neuron_compare(
    gpu_in_curr=gpu_in_curr_t,
    gpu_mem=gpu_mem_t,
    gpu_spk=gpu_out_spk_t,
    npu_mem=npu_mem_t,
    npu_spk=npu_out_spk_t,
    num_time_steps=25,
    title=f"LIF membrane potential traces comparison (GPU vs NPU)\nModel 784x56x56x56x10, Neuron 45, Layer 0 (Test Sample 7)"
)
