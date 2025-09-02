import torch
import matplotlib.pyplot as plt
from snntorch import spikeplot as splt
def plot_cur_mem_spk(cur, mem_prev, mem_nxt, spk, thr_line, vline=False, title=False, ylim_max2=2, ylim_max3=1.25):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input current
  ax[0].plot(cur, c="tab:orange", label="$I_{in}$")
  #ax[0].plot(mem_prev, c="tab:blue", label="$U_{mem}$")
  #sum_of_two = cur + mem_prev
  #ax[0].plot(sum_of_two, c="tab:brown", label="$I_{in}")
  ax[0].set_ylim([-1, 1])
  ax[0].set_xlim([0, 24])
  ax[0].set_ylabel("Input Current $I_{in}$")
  #ax[0].set_ylabel("Before Neuron Update")
  if title:
    ax[0].set_title(title)
  #ax[0].legend()




  ## Plot membrane potential
  #ax[1].plot(mem_prev)
  #ax[1].plot(v_th)
  #ax[1].set_ylim([0, 2])
  #ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
  #if thr_line:
    #ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  #plt.xlabel("Time step")

  # Plot membrane potential
  ax[1].plot(mem_nxt)
  ax[1].set_ylim([-2, 1.25])
  ax[1].set_ylabel("Membrane Potential $U_{mem}^{next}$\n(post reset)")
  #if thr_line!=None:
  ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  plt.xlabel("Time step")

  # Plot output spike using spikeplot
  splt.raster(spk, ax[2], s=400, c="black", marker="|")
  if vline:
    ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.ylabel("Output spikes")
  plt.yticks([])


  plt.savefig("v_mem_traces.svg", format="svg")


  plt.show()


input_spike = [
    [

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

[

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

]




import numpy as np

weights = np.load("784x64x64x10/model_params/fc_lif_layer_1_weights.npy")
biases = np.load("784x64x64x10/model_params/fc_lif_layer_1_biases.npy")

input_spike_np = np.array(input_spike)



weighted_sum = input_spike_np @ weights.T + biases

print
print("in_curr", weighted_sum.shape)
for in_spk_array in weighted_sum:
    print(in_spk_array[2])
#print(weighted_sum)
#np.savetxt("in_curr.csv", weighted_sum, 
#              delimiter = ",")


#weights: (64, 64)
#bias: (64)
#input: (25, 64)

in_curr = [
0.3969413079,
0.5152336881,
0.1412772015,
0.8472154811,
0.4808907583,
-0.5608451962,
-0.5608451962,
-0.2135998905,
-0.5532134473,
-0.3280763775,
0.0611436516,
0.6373419166,
0.7670819312,
0.7098436914,
0.7098436914,
0.1527248286,
0.3511507176,
0.3511507176,
-0.2364952005,
-0.2364952005,
-0.2364952005,
-0.2364952005,
-0.2364952005,
-0.2364952005,
-0.2364952005
]

v_mem_next = [
0.411765,
0.882353,
0.941176,
0.764706,
0.235294,
-0.352941,
-0.882353,
-1.058824,
-1.529412,
-1.823529,
-1.647059,
-0.941176,
-0.117647,
0.588235,
0.294118,
0.411765,
0.764706,
0.058824,
-0.176471,
-0.411765,
-0.588235,
-0.764706,
-0.941176,
-1.117647,
-1.294118
]


out_spk = [
0,
0,
0,
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
0,
1,
0,
0,
1,
0,
0,
0,
0,
0,
0,
0]
#v_th = np.ones(len(v_mem_next))
v_th = 1

v_mem_prev = [
0.000000,
0.411765,
0.882353,
0.941176,
0.764706,
0.235294,
-0.352941,
-0.882353,
-1.058824,
-1.529412,
-1.823529,
-1.647059,
-0.941176,
-0.117647,
0.588235,
0.294118,
0.411765,
0.764706,
0.058824,
-0.176471,
-0.411765,
-0.588235,
-0.764706,
-0.941176,
-1.117647]


v_mem_prev_t = torch.from_numpy(np.array(v_mem_prev))
v_mem_next_t = torch.from_numpy(np.array(v_mem_next))
out_spk_t = torch.from_numpy(np.array(out_spk))
in_curr_t = torch.from_numpy(np.array(in_curr))

plot_cur_mem_spk(in_curr_t, v_mem_prev_t, v_mem_next_t, out_spk_t, thr_line=v_th, title="Activity of Neuron[2] in Layer 1 in 784x64x64x10")



## Generate Plots
#fig, ax = plt.subplots(4, figsize=(8,6), sharex=True,
                    #gridspec_kw = {'height_ratios': [1, 1, 1, 0.4]})



#cur = in_curr_t
#mem_prev = v_mem_prev_t
#mem_nxt = v_mem_next_t
#spk = out_spk_t
#thr_line=v_th
#vline=False
#title=False
# Generate Plots
#fig, ax = plt.subplots(4, figsize=(8,6), sharex=True,
                        #gridspec_kw = {'height_ratios': [1]})
## Plot input current
#ax[0].plot(cur, c="tab:orange")
#ax[0].set_ylim([-1, 1])
#ax[0].set_xlim([0, 24])
#ax[0].set_ylabel("Input Current ($I_{in}$)")
#if title:
    #ax[0].set_title(title)

## Plot membrane potential
#ax[1].plot(mem_prev)
#ax[1].plot(v_th)
#ax[1].set_ylim([0, 2])
#ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
#if thr_line:
    #ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
#plt.xlabel("Time step")





#plt.show()





## Generate Plots
#fig, ax = plt.subplots(4, figsize=(8,6), sharex=True,
                        #gridspec_kw = {'height_ratios': [1, 1, 0.4]})
## Plot input current
#ax[0].plot(cur, c="tab:orange")
#ax[0].set_ylim([-1, 1])
#ax[0].set_xlim([0, 24])
#ax[0].set_ylabel("Input Current ($I_{in}$)")
#if title:
#ax[0].set_title(title)

## Plot membrane potential
#ax[1].plot(mem_prev)
#ax[1].plot(v_th)
#ax[1].set_ylim([0, 2])
#ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
#if thr_line:
#ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
#plt.xlabel("Time step")

## Plot membrane potential
#ax[2].plot(mem_nxt)
#ax[2].plot(v_th)
#ax[2].set_ylim([0, 1.25])
#ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
#if thr_line:
#ax[2].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
#plt.xlabel("Time step")

## Plot output spike using spikeplot
#splt.raster(spk, ax[3], s=400, c="black", marker="|")
#if vline:
#ax[3].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
#plt.ylabel("Output spikes")
#plt.yticks([])

#plt.show()
