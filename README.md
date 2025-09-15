# JWST Spectral Stacking Internship

Repository holding code relating to automated analysis, stacking and fitting of NIRSpec spectra, as well as calculation of elemental abundances from results of the said fitting. The code was developed as part of a summer 2025 internship in the GalEv group at the Max Planck Institute for Extraterrestial Physics. 

All module-level functions, classes and attributes have short description of their purpose. (For further help with interpretation/modifications of the code I recommend turning to LLM alike Gemini, which I've found capable of making sense of most of the code's functions.) 

## Recreating internship results

Instructions bellow assume standard Linux/bash and IPython sessions. Total run time will be probably about ~2 days.

##### Install required libraries

Assuming standard `python3` installation with packages managed by `pip`.

    $ pip install numpy, scipy, astropy, matplotlib, pyneb, ppxf

The plotting code also pressumes an existing latex instalation in the environment (alike e.g. TeX Live).

##### Constructing folders and fetching data

Create project folder(s) and enter it:

    $ mkdir stacking stacking/Code stacking/Data stacking/Plots
    $ cd stacking

Fetch the DJA NIRSpec catalogue (assuming version `4.4`) and unzip it:

    $ wget https://s3.amazonaws.com/msaexp-nirspec/extractions/dja_msaexp_emission_lines_v4.4.csv.gz
    $ gzip -dk dja_msaexp_emission_lines_v4.4.csv.gz

Enter `Code` folder and fetch Git project (assuming the repository is public - which it currently isn't) and open IPython:

    $ cd Plots
    $ git clone https://github.com/vachaj11/omnistack.git
    $ python3
    ...
    >>>

##### Downloading and preparing spectra

Import relevant module, construct local representation of the catalogue (`../catalog_v4.json`), download and process (e.g. trim noisy edges) all spectral data:

    >>> import download as dl
    >>> dl.mini_reduction("../dja_msaexp_emission_lines_v4.4.csv", "../catalog_v4.json")

##### Calculating line fluxes and abundances for individual sources

(This process is very memory intensive and usually consumes up to ~10 GB of RAM while running.)

Import relevant modules, open local catalogue, filter it for medium resolution and quality spectra, calculate/reconstruct line fluxes for individual spectra, calculate abundances for individual spectra, save local catalogue and its subset with individual sources information:

    >>> import paramite as pr
    >>> import catalog
    >>> f = catalog.fetch_json("../catalog_v4.json")["sources"]
    >>> ff = catalog.rm_bad(f)
    >>> ffm = [s for s in ffm if s["grat"][0]=="g"]
    >>> upd_ffm = pr.calculate_fluxes(ffm)
    >>> ffmu = pr.calculate_indiv_lines(upd_ffm)
    >>> catalog.save_as_json({"sources": f}, "../catalog_v4.json")
    >>> catalog.save_as_json({"sources": ffmu}, "../catalog_indiv4.json")

##### Calculating continuum-subtracted spectra

(This step is essential only to plotting in `cumul_plot.py`, otherwise can be skipped. Requires about ~10 hours.)

Imports ppxf fitting module, calculate ppxf fits and related continua, calculate continua approximated through iterative clipping, subtract continua from spectral data:

    >>> import ppxf_fit as pf
    >>> pf.ppxf_fitting_multi(f, bi = "../Data/Npy_v4/", bo = "../Data/Continuum_v4/", bs = "../Data/Subtracted_v4/")
    >>> _ = pf.smooth_to_cont(f, bi = "../Data/Npy_v4/", bo = "../Data/Continuum_v4_b/", bs = "../Data/Subtracted_v4_b/")

##### Calculated and plot results presented in internship summary

Import relevant modules, run main plotting functions (this will create relevant `.pdf` plots in `../Plots/`):

    >>> import joint_fit as jf
    >>> import cumul_plot as cp
    >>> import joint_par as jp
    >>> jf.main()
    >>> cp.main()
    >>> jp.main()
