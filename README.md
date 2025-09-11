# JWST Spectral Stacking Internship

Repository of badly documented and in parts very chaotic code.

## Recreating internship results

Instructions bellow assume standard Linux/bash and IPython sessions. Total run time will be probably about ~1 day.

##### Install required libraries

Assuming standard `python3` installation with packages managed by `pip`.

    $ pip install numpy, scipy, astropy, matplotlib, pyneb, ppxf, 

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

Import relevant libraries, construct local representation of the catalogue (`../catalog_v4.json`), download and process (e.g. trim noisy edges) all spectral data:

    >>> import download as dl
    >>> import paramite as pr
    >>> import catalog
    >>> dl.mini_reduction("../dja_msaexp_emission_lines_v4.4.csv", "../catalog_v4.json")

##### Calculating line fluxes and abundances for individual sources

Open local catalogue, filter it for medium resolution and quality spectra, calculate/reconstruct line fluxes for individual spectra, calculate abundances for individual spectra, save local catalogue and its subset with individual sources information:

    >>> f = catalog.fetch_json("../catalog_v4.json")["sources"]
    >>> ff = catalog.rm_bad(f)
    >>> ffm = [s for s in ffm if s["grat"][0]=="g"]
    >>> upd_ffm = pr.calculate_fluxes(ffm)
    >>> ffmu = pr.calculate_indiv_lines(upd_ffm)
    >>> catalog.save_as_json({"sources": f}, "../catalog_v4.json")
    >>> catalog.save_as_json({"sources": ffmu}, "../catalog_indiv4.json")

##### Calculating continuum-subtracted spectra

(This is not essential step to the results and can be skipped. Requires about ~10 hours.)

    >>> import ppxf_fit as pf
    >>> ...

##### Calculated and plot results presented in internship summary

Import relevant librarie, run main plotting functions (this will create relevant `.pdf` plots in `../Plots/`):

    >>> import joint_fit as jf
    >>> import cumul_plot as cp
    >>> import joint_par as jp
    >>> jf.main()
    >>> cp.main()
    >>> jp.main()
