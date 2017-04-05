#!/usr/bin/env python
# -*- coding=utf-8 -*-

import sys
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pymatgen.io.vaspio.vasp_output import Vasprun
from pymatgen.electronic_structure.core import Spin

import style

if __name__ == "__main__":
    # read data
    # ---------

    # kpoints labels
    labels = [ r"$L$", r"$\Gamma$", r"$X$", r"$U,K$", r"$\Gamma$" ]

    # density of states
    dosrun = Vasprun("./si_bands/vasprun.xml")
    spd_dos = dosrun.complete_dos.get_spd_dos()

    # bands
    run = Vasprun("./si_bands/vasprun.xml", parse_projected_eigen = True)
    bands = run.get_band_structure("./si_bands/KPOINTS",
                                   line_mode = True,
                                   efermi = dosrun.efermi)

    # set up matplotlib plot
    # ----------------------

    # general options for plot


    # set up 2 graph with aspec ration 2/1
    # plot 1: bands diagram
    gs = GridSpec(1, 2, width_ratios=[2,0])
#    fig = plt.figure(figsize=(11.69, 8.27))
    fig = plt.figure(figsize=plt.figaspect(.5))
    fig.suptitle(r"Band diagram of silicon", fontsize=22)
    ax = plt.subplot(gs[0])
#    ax2 = plt.subplot(gs[1]) # , sharey=ax1)

    # set ylim for the plot
    # ---------------------
    emin = 1e100
    emax = -1e100
    for spin in bands.bands.keys():
        for b in range(bands.nb_bands):
            emin = min(emin, min(bands.bands[spin][b]))
            emax = max(emax, max(bands.bands[spin][b]))

    emin -= bands.efermi + 1
    emax -= bands.efermi - 1
    ax.set_ylim(emin, emax)

    # Band Diagram
    # ------------
    name = "Si"
    pbands = bands.get_projections_on_elts_and_orbitals({name: ["s", "p", "d"]})

    # compute s, p, d normalized contributions
    contrib = np.zeros((bands.nb_bands, len(bands.kpoints), 3))
    for b in xrange(bands.nb_bands):
        for k in range(len(bands.kpoints)):
            sc = pbands[Spin.up][b][k][name]["s"]**2
            pc = pbands[Spin.up][b][k][name]["p"]**2
            dc = pbands[Spin.up][b][k][name]["d"]**2
            tot = sc + pc + dc
            if tot != 0.0:
                contrib[b, k, 0] = sc / tot
                contrib[b, k, 1] = pc / tot
                contrib[b, k, 2] = dc / tot

    # plot bands using rgb mapping
#    pal = style.seaborn.color_palette("coolwarm",8)

    cold_hot = np.array([[  5,   48,   97],
        [ 33,  102,  172],
        [ 67,  147,  195],
        [146,  197,  222],
        #[209,  229,  240],
        [247,  247,  247],
        #[254,  219,  199],
        [244,  165,  130],
        [214,   96,   77],
        [178,   24,   43],
        [103,    0,   31]])/256.

    pal = style.sns.blend_palette(cold_hot, 8)
    #pal = style.seaborn.dark_palette("skyblue", 8, reverse=True)
#    with style.seaborn.color_palette("coolwarm"):
    band = np.array(bands.bands[Spin.up]) - bands.efermi
    x = np.arange(len(bands.kpoints))
    for b in xrange(bands.nb_bands):
        plt.plot(x,band[b], color=pal[b])

    # style
    ax.set_xlabel(r"k-points")
    ax.set_ylabel(r"$E - E_f$   /   eV")
#    ax1.grid()

    # fermi level at 0


    # labels
    nlabs = len(labels)
    step = len(bands.kpoints) / (nlabs - 1)
    for i, lab in enumerate(labels):
        ax.vlines(i * step, emin, emax, "k", lw=1)
    ax.set_xticks([i * step for i in range(nlabs)])
    ax.set_xticklabels(labels)

    ax.set_xlim(0, len(bands.kpoints))
    #ax.set_title("Bands diagram")

    plt.subplots_adjust(wspace=0)
    style.sns.despine()
    style.sns.set_style()
    ax.hlines(y=0, xmin=0, xmax=len(bands.kpoints), color="k", lw=1)
#    plt.show()
    plt.savefig(os.path.join("../","figures",sys.argv[0].strip(".py") + ".pdf"), format="pdf")

