import os
import warnings

warnings.filterwarnings("ignore")
import urllib.request
from multiprocessing import Manager, Process, cpu_count

import grizli
from grizli import utils

import catalog
import spectr


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


def move_around():
    a = catalog.fetch_json("../catalog.json")["sources"]
    for s in a:
        spectrum = spectr.get_spectrum_n(s)
        base = "../Data/Npy/"
        spectr.save_npy(s, spectrum, base=base)
        spectr.rm_npy(s)
        
def add_photometry(sources, pathp):
    dic = catalog.construct_dict(pathp)
    print("here")
    values = [
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
    for i, source in enumerate(sources):
        sid = source["file"]
        g = None
        for s in dic:
            if sid == s["file"].replace("-v4_", "-v3_"):
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
