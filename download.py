import csv
import os
import time
import warnings

import astropy.io.fits as fits
import numpy as np

warnings.filterwarnings("ignore")
import urllib.request
from multiprocessing import Manager, Process, cpu_count

"""
import grizli
from grizli import utils
"""

import catalog
import spectr

typeint = ["ndup", "uid", "srcid", "nGr", "grade", "nRef"]
typefloat = [
    "ra",
    "dec",
    "zfit",
    "z",
    "sn50",
    "wmin",
    "wmax",
    "Lya",
    "Ha",
    "OIII",
    "L_Ha",
    "L_OIII",
]
typedel = ["HST", "NIRCam", "slit", "FITS", "Fnu", "Flam"]

typeint_c = {
    "srcid": "srcid",
    "grade": "grade",
}
typefloat_c = {
    "ra": "ra",
    "dec": "dec",
    "zfit": "z_best",
    "z": "zgrade",
    "sn50": "sn50",
    "flux50": "flux50",
    "err50": "err50",
    "wmin": "wmin",
    "wmax": "wmax",
    "Lya": "line_lya",
    "SII": "line_sii",
    "Ha": "line_ha",
    "OII": "line_oii",
    "OIII": "line_oiii",
    "SIII_63": "line_siii_6314",
    "SIII_90": "line_siii_9068",
    "NII_65": "line_nii_6549",
    "NII_66": "line_nii_6584",
    "phot_Av": "phot_Av",
    "phot_mass": "phot_mass",
    "phot_restU": "phot_restU",
    "phot_restV": "phot_restV",
    "phot_restJ": "phot_restJ",
    "z_phot": "z_phot",
    "phot_LHa": "phot_LHa",
    "phot_LOIII": "phot_LOIII",
    "phot_LOII": "phot_LOII",
}
typestr_c = {
    "file": "file",
    "root": "root",
    "comment": "comment",
    "grat": "grating",
}


def construct_dict(path):
    fil = open(path, "r")
    cs = csv.DictReader(fil, delimiter=",", quotechar='"')
    gall = []
    for row in cs:
        gal = dict(row)
        for k in typedel:
            if k in gal.keys():
                gal.pop(k)
        for k in gal:
            if gal[k] == "":
                gal[k] = None
            elif k in typeint:
                gal[k] = int(gal[k])
            elif k in typefloat:
                gal[k] = float(gal[k])
        gall.append(gal)
    return gall


def construct_dict_n(path):
    fil = open(path, "r")
    cs = csv.DictReader(fil, delimiter=",", quotechar='"')
    gall = []
    for row in cs:
        gal = dict(row)
        galn = dict()
        for k, v in typeint_c.items():
            if gal[v] != "":
                galn[k] = int(gal[v])
            else:
                galn[k] = None
        for k, v in typefloat_c.items():
            if gal[v] != "":
                galn[k] = float(gal[v])
            else:
                galn[k] = None
        for k, v in typestr_c.items():
            if gal[v] != "":
                galn[k] = gal[v]
            else:
                galn[k] = None
        gall.append(galn)
    return gall


'''
def download_spectra():
    BASE_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions/"
    PATH_TO_FILE = BASE_URL + "{root}/{file}"

    nrs = utils.read_catalog("../catalog.csv")

    print("By grade:")
    un = utils.Unique(nrs["grade"])

    print("By source:")
    root = utils.Unique(nrs["root"])

    print("By SNR:")
    print("N\t value")
    print("=\t =====")
    data = nrs["sn50"].data
    rang = [(0, 0.5), (0.5, 1), (1, 3), (3, 8), (8, 20), (20, 10**6)]
    dic = {(-(10**6), 0): 0}
    for i in rang:
        dic[i] = 0
    for i in data:
        t = False
        for r in rang:
            if r[0] < i < r[1]:
                dic[r] += 1
                t = True
                break
        if not t:
            dic[(-(10**6), 0)] += 1
    for v in dic:
        print(f"{dic[v]}\t {v}")

    ind = 1

    """
    try:
        inp1 = int(input("From entry: "))
    except:
        inp1 = None
    try:
        inp2 = int(input("To entry: "))
    except:
        inp2 = None
    for row in nrs[inp1:inp2]:
        print(f'\r\033[K ({ind} out of {inp2-inp1}) File: {row["file"]}', end=" ")
        url = PATH_TO_FILE.format(**row)
        pathl = "/home/vachaj11/Documents/MPE/2025/Fits/" + str(row["root"] + "/")
        if not os.path.exists(pathl):
            os.makedirs(pathl)
        urllib.request.urlretrieve(url, filename=pathl + str(row["file"]))
        ind += 1
    """
'''


def downloadall(sources, start_in=0):
    a = sources
    proc = cpu_count()
    active = []
    latest = start_in - 1
    while latest < len(a) - 1 or len(active) > 0:
        pr_no = min(proc, len(a) - 1 - latest)
        if len(active) < pr_no:
            for i in range(pr_no - len(active)):
                latest += 1
                args = (a[latest],)
                t = Process(target=download, args=args)
                t.start()
                print(f"\rRunning spectra {latest} \tout of {len(a)-1}", end="")
                active.append(t)
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.5)


def download(source):
    sp = spectr.get_spectrum_fits(
        source, base="https://s3.amazonaws.com/msaexp-nirspec/extractions/"
    )
    spectr.save_npy(source, sp, base="../Data/Npy/")


def move_around():
    a = catalog.fetch_json("../catalog.json")["sources"]
    for s in a:
        spectrum = spectr.get_spectrum(s)
        base = "../Data/Npy/"
        spectr.save_npy(s, spectrum, base=base)
        spectr.rm_npy(s)


def match_cosmos(sources):
    path = "../COSMOS25.fits"
    x = fits.open(path)
    ins = x[1].data["id"]
    ras = x[1].data["ra"]
    des = x[1].data["dec"]
    zes = x[2].data["zfinal"]
    zis = list(zip(ins, ras, des, zes))
    matched = []
    souf = [
        True if 149 < s["ra"] < 151 and 1 < s["dec"] < 3 else False for s in sources
    ]
    for ind, (s, f) in enumerate(zip(sources, souf)):
        s["COSMOS25_match"] = []
        if f and s["z"] is not None:
            for i, r, d, z in zis:
                if (
                    abs(s["ra"] - r) < 0.0005
                    and abs(s["dec"] - d) < 0.0005
                    and abs(s["z"] - z) < 0.2
                ):
                    s["COSMOS25_match"].append(int(i))
            if s["COSMOS25_match"]:
                matched.append(s)
            print(f'{s["srcid"]} ({ind} out of {len(sources)})')
    x.close()
    return matched


cosmos_val = {
    "cos_z": [2, "zfinal", lambda x: float(x)],
    "cos_age": [2, "age_minchi2", lambda x: float(x)],
    "cos_mass": [2, "mass_minchi2", lambda x: float(np.exp(x * np.log(10)))],
    "cos_sfr": [2, "sfr_minchi2", lambda x: float(np.exp(x * np.log(10)))],
    "cos_mass_2": [4, "mass", lambda x: float(x)],
    "cos_sfr_2": [4, "sfr_inst", lambda x: float(x)],
    "cos_met_2": [4, "metallicity", lambda x: float(x)],
}


def extract_cosmos(sources):
    path = "../COSMOS25.fits"
    x = fits.open(path)
    for s in sources:
        if "COSMOS25_match" in s.keys() and len(s["COSMOS25_match"]) == 1:
            ind = s["COSMOS25_match"][0]
            for k, v in cosmos_val.items():
                s[k] = v[2](x[v[0]].data[v[1]][ind])
        else:
            for k, v in cosmos_val.items():
                s[k] = None


def add_post_hoc(sources, pathp="../dja_msaexp_2.csv", values=None):
    dic = construct_dict(pathp)
    v = [
        "phot_Av",
        "phot_mass",
        "phot_restU",
        "phot_restV",
        "phot_restJ",
        "z_phot",
        "phot_LHa",
        "phot_LOIII",
        "phot_LOII",
    ]
    values = v if values is None else values
    for i, source in enumerate(sources):
        sid = source["file"]
        g = None
        for s in dic:
            if sid in [s["file"], s["file"].replace("-v4_", "-v3_")]:
                g = s
                break
        if g is not None:
            for v in values:
                source[v] = g[v]
        else:
            for v in values:
                source[v] = None
        print(f"\r{i} {type(g)}", end="")
    return sources
    
def trim_edges(sources, bi = '../Data/Npy_legacy/', bo='../Data/Npy/', **kwargs):
    vss = []
    for l, s in enumerate(sources):
        sp0 = spectr.get_spectrum(s, base=bi)
        i, w, spr = spectr.iden_bad_e(sp0, **kwargs)
        if i:
            vss.append([w, sp0, spr])
            spectr.save_npy(s, spr, base=bo)
        print('\r'+str(l), end='')  
    return vss
    
def update_ranges(sources, base='../Data/Npy/'):
    for s in sources:
        sp = spectr.get_spectrum(s, base=base)
        if sp is not None:
            s['range'] = spectr.useful_range(sp)
        else:
            s['range'] = []
    return sources
    
def degrade_high(sources, bi='../Data/Npy/',bo='../Data/Npy_med/'):
    mpx = {
        'g140h': 0.00060000,
        'g235h': 0.00100952,
        'g395h': 0.00170476,
    }
    adj = 0
    for l, s in enumerate(sources):
        print('\r'+str(l), end='') 
        if s['grat'][-1]=='h':
            sp0 = spectr.get_spectrum(s, base=bi)
            if sp0 is not None:
                spr = spectr.degrade_spectrum(sp0, mpx[s['grat']])
                spectr.save_npy(s, spr, base=bo)
                adj+=1
            s['grat_orig'] = s['grat']
            s['grat'] = s['grat'][:-1]+'m'
        elif 'grat_orig' not in s.keys():
            s['grat_orig'] = s['grat']
    print('\n'+str(adj))
    
