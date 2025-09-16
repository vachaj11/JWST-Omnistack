"""Holds methods for reconstruction of individual sources line-fluxes and abundances via a sub-sampling and stacking methodology."""

import time
from multiprocessing import Manager, Process, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import (
    coo_array,
    csc_array,
    csr_array,
    dok_array,
    eye_array,
    lil_array,
    load_npz,
    save_npz,
)
from scipy.sparse.linalg import LinearOperator, eigsh

import abundc as ac
import catalog


def sep_inds(ln, no):
    """Create bin indices for a given number of elements and approximate desired bin size."""
    if ln <= no:
        return [0, ln]
    binn = ln // no
    extr = ln % no
    bina = [0] * binn
    for i in range(extr):
        bina[i % len(bina)] += 1
    strs = [0]
    for i in range(binn - 1):
        strs.append(strs[-1] + no + bina[i])
    return strs + [ln]


def par_params(srss, method, nwg=0.3):
    """Legacy calculation method, saving calculated parameters to individual catalog entries. Which turns out unsuitet for paralizing."""
    nwg = nwg if (0 <= nwg <= 1) else 1
    dicv = [[i, sg] for i, sg in enumerate(srss)]
    manag = Manager()
    vals = manag.dict()
    proc = cpu_count()
    prls = [[] for i in range(proc)]
    for i, d in dicv:
        prls[i % len(prls)].append([i, d])
    active = []
    for p in prls:
        args = (p, method, vals)
        t = Process(target=pro_params, args=args)
        t.start()
        active.append(t)
    while active:
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        print(f"\r\033[KFinished {len(vals)} out of {len(srss)}.", end="")
        time.sleep(0.01)
    for i, srs in dicv:
        for s in srs:
            if (v := s.get("_p")) is not None and np.isfinite(v):
                if (vx := vals[i]) is not None and np.isfinite(vx):
                    s["_p"] = (1 - nwg) * v + nwg * vx
                else:
                    s["_p"] = v
            else:
                if (vx := vals[i]) is not None and np.isfinite(vx):
                    s["_p"] = vx
                else:
                    s["_p"] = None


def pro_params(isrs, method, vals):
    """Single-process function called by legacy calculation method."""
    for i, srs in isrs:
        v = method(srs)
        vals[i] = v


def ded_params(sources, method, no=50, sbin=5, nwg=0.3, ite=None):
    """Legacy function preparing inputs for legacy calulation method."""
    defs = []
    nefs = []
    no = min(no, int(len(sources) / 20))
    inds = sep_inds(len(sources), no)
    nb = len(inds) - 1
    ite = 5 * int(nb / sbin) if ite is None else ite
    par_params([[s] for s in sources], method)
    for i, s in enumerate(sources):
        par = s["_p"]  # method([s])
        if par is not None and np.isfinite(par):
            defs.append(s)
        else:
            s["_p"] = None
            nefs.append(s)
    defs.sort(key=lambda x: x["_p"])
    mid = int(len(defs) / 2)
    init = defs[:mid] + nefs + defs[mid:]
    for s in init:
        s["_p"] = defs[mid]["_p"] if s["_p"] is None else s["_p"]
    while ite > 0:
        rang = (
            (
                max(init, key=lambda x: x["_p"])["_p"]
                - min(init, key=lambda x: x["_p"])["_p"]
            )
            / nb
            * sbin
        )
        for s in init:
            s["_r"] = s["_p"] * np.random.uniform(-rang, rang)
        init.sort(key=lambda x: x["_r"])
        sini = [init[inds[i] : inds[i + 1]] for i in range(len(inds) - 1)]
        par_params(sini, method, nwg=nwg)
        ite -= 1


def pro_fluxes(sources, lt, r_lis, M, vals, **kwargs):
    """Calculation method called for each process. Fits lines to given stacks of spectra and saves results."""
    for i in range(len(r_lis)):
        row = M[i].todok()
        srs = [sources[k] for k in row.keys()]
        val = ac.fit_lines(srs, lt[1], lt[0], dwidth=lt[2], typ="mean", **kwargs)
        vals[r_lis[i]] = val


def cal_fluxes(sources, l_tuple, M, **kwargs):
    """Calculates fluxes in a specified region/lines for a provided catalog of spectra and matrix specifying its sub-stacking."""
    manag = Manager()
    vals = manag.dict()
    proc = cpu_count()
    prls = [[] for i in range(proc)]
    for i in range(M.shape[0]):
        prls[i % len(prls)].append(i)
    srs = []
    active = []
    for p in prls:
        args = (sources, l_tuple, p, M[p], vals)
        t = Process(target=pro_fluxes, args=args, kwargs=kwargs)
        t.start()
        active.append(t)
    while active:
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        print(f"\r\033[KFinished {len(vals)} out of {M.shape[0]}.", end="")
        time.sleep(0.01)
    j_vals = {k: [] for k in l_tuple[0].keys()}
    for k in l_tuple[0].keys():
        for i in range(M.shape[0]):
            j_vals[k].append(vals[i][k])
    return j_vals


def art_fluxes(sources, l_tuple, n_one=50, n_sam=None, save=None, **kwargs):
    """For a provided catalog of spectra randomly choses subsamples, stacks spectra within them and fit them with lines, and returns all results in joint linear-algebraic format."""
    if type(l_tuple[0]) is not dict:
        l_tuple[0] = {"tmp": l_tuple[0]}
    sources = catalog.rm_bad(sources)
    sources = catalog.filter_zranges(sources, [[min(l_tuple[1]), max(l_tuple[1])]])
    n_sou = len(sources)
    n_sam = n_sou * 2 if n_sam is None else n_sam
    M = lil_array((n_sam, n_sou), dtype="uint8")
    col_ind = []
    for i in range(n_sam):
        col_ind += list(np.random.choice(n_sou, size=n_one, replace=False))
        print(f"\r\033[K{i} out of {n_sam}", end="")

    data = [1] * (n_one * n_sam)
    row_ind = []
    for i in range(n_sam):
        row_ind += [i] * n_one
    M = coo_array(
        (data, (row_ind, col_ind)), shape=(n_sam, n_sou), dtype="uint8"
    ).tocsr()
    del row_ind
    del col_ind
    del data
    fluxes = cal_fluxes(sources, l_tuple, M, **kwargs)

    flubs = dict()
    for k, v in fluxes.items():
        flux = v.copy()
        for i in range(len(v)):
            flux[i] *= M[i].sum()
        flubs[k] = np.array(flux)
    if save is not None:
        save_npz(f"../M{save}.npz", M)
        np.save(f"../F{save}.npy", flubs)
        np.save(f"../S{save}.npy", sources)
    return M, flubs, sources


def ind_fluxes(sources, l_tuple):
    """Calculate fluxes values in specified region/lines for all sources in passed catalogue."""
    M = eye_array(len(sources), dtype="uint8", format="csr")
    fluxes = cal_fluxes(sources, l_tuple, M)
    return fluxes


def PART(M, fluxes, c_ite=25, lam=0.05, t_f=None):
    """Implementation of the Algebraic Reconstruction Technique for solving the linear-algebraic problem fluxes = M @ x."""
    M = M if type(M) is csr_array else csr_array(M)
    gval = dict()
    noi = int(M.shape[0] * c_ite)
    for nam, vals in fluxes.items():
        vals = np.array(vals)
        init = np.nanmean(vals) * M.sum() / M.shape[0]
        guess = np.ones(M.shape[1]) * init
        guess1 = guess.copy()
        gchang = [np.nan]
        t_i = time.time()
        for u in range(noi):
            i = np.random.randint(0, M.shape[0])
            guess += lam * (vals[i] - M[i].dot(guess)) * M[i] / M[i].sum() ** 2
            guess[guess < 0] = 0
            if u % 128 == 0:
                print(
                    f"\r\033[KFinished {u} out of {noi}. Convergence {gchang[-1]:.2e}.",
                    end="",
                )
            if u % -(-noi // 1000) == 0:
                guess0 = guess1
                guess1 = guess.copy()
                chang = np.nanpercentile(
                    np.nan_to_num(
                        np.where((p := (guess1 - guess0) / guess0) > 0, p, np.nan),
                        nan=np.nan,
                        posinf=np.nan,
                        neginf=np.nan,
                    ),
                    75,
                )
                if len(gchang) > 20 and chang < 0.001:
                    break
                gchang.append(chang)
                print(f" {chang:.f}")
            if t_f is not None and t_f < (time.time() - t_i):
                break
        print(gchang)
        gval[nam] = guess
    return gval


def MART(M, fluxes, c_ite=25, lam=0.01, t_f=None):
    """Implementation of the Multiplicative Algebraic Reconstruction Technique for solving the linear-algebraic problem fluxes = M @ x."""
    M = M if type(M) is csr_array else csr_array(M)
    gval = dict()
    noi = int(M.shape[0] * c_ite)
    for nam, vals in fluxes.items():
        vals = np.array(vals)
        init = np.nanmean(vals) * M.sum() / M.shape[0]
        guess = np.ones(M.shape[1]) * init
        guess1 = guess.copy()
        gchang = [np.nan]
        t_i = time.time()
        for u in range(noi):
            i = np.random.randint(0, M.shape[0])
            base = vals[i] / M[i].dot(guess)
            al = lam * M[i].toarray()
            guess *= np.exp(al * np.log(base))
            if u % 128 == 0:
                print(
                    f"\r\033[KFinished {u} out of {noi}. Convergence {gchang[-1]:.2e}.",
                    end="",
                )
            if u % -(-noi // 1000) == 0:
                guess0 = guess1
                guess1 = guess.copy()
                chang = np.nanpercentile(
                    np.nan_to_num(
                        np.where((p := (guess1 - guess0) / guess0) > 0, p, np.nan),
                        nan=np.nan,
                        posinf=np.nan,
                        neginf=np.nan,
                    ),
                    75,
                )
                if len(gchang) > 20 and chang < 0.001:
                    break
                gchang.append(chang)
            if t_f is not None and t_f < (time.time() - t_i):
                break
        print(gchang)
        gval[nam] = guess
    return gval


def MLEM(M, fluxes, c_ite=0.1, t_f=None):
    """Implementation of the Maximum-Likelihood Expectation-Maximization technique for solving the linear-algebraic problem fluxes = M @ x."""
    M = M if type(M) is csc_array else csc_array(M)
    gval = dict()
    noi = int(M.shape[0] * c_ite)
    for nam, vals in fluxes.items():
        vals = np.array(vals)
        init = np.nanmean(vals) * M.sum() / M.shape[0]
        guess = np.ones(M.shape[1]) * init
        guess1 = guess.copy()
        gchang = [np.nan]
        t_i = time.time()
        for u in range(noi):
            base = vals / (M @ guess)
            co = M.T @ base
            coe = co / M.sum(axis=0)
            guess *= coe
            print(
                f"\r\033[KFinished {u} out of {noi}. Convergence {gchang[-1]:.2e}.",
                end="",
            )
            if u % -(-noi // 1000) == 0:
                guess0 = guess1
                guess1 = guess.copy()
                chang = np.nanpercentile(
                    np.nan_to_num(
                        np.where((p := (guess1 - guess0) / guess0) > 0, p, np.nan),
                        nan=np.nan,
                        posinf=np.nan,
                        neginf=np.nan,
                    ),
                    75,
                )
                if len(gchang) > 20 and chang < 0.001:
                    break
                gchang.append(chang)
            if t_f is not None and t_f < (time.time() - t_i):
                break
        print(gchang)
        gval[nam] = guess
    return gval


def OSEM(M, fluxes, c_ite=0.1, N=16, t_f=None):
    """Implementation of the Ordered Subset Expectation-Maximization technique for solving the linear-algebraic problem fluxes = M @ x."""
    M = M if type(M) is csc_array else csc_array(M)
    gval = dict()
    inds = [[] for i in range(N)]
    noi = int(M.shape[0] * c_ite)
    for i in range(M.shape[0]):
        inds[i % len(inds)].append(i)
    for nam, vals in fluxes.items():
        vals = np.array(vals)
        Ms = []
        vss = []
        for ind in inds:
            Ms.append(M[ind, :])
            vss.append(vals[ind])
        init = np.nanmean(vals) * M.sum() / M.shape[0]
        guess = np.ones(M.shape[1]) * init
        guess1 = guess.copy()
        gchang = [np.nan]
        t_i = time.time()
        for u in range(noi):
            for i in range(N):
                base = vss[i] / (Ms[i] @ guess)
                co = Ms[i].T @ base
                coe = co / Ms[i].sum(axis=0)
                guess *= coe
            print(
                f"\r\033[KFinished {u} out of {noi}. Convergence {gchang[-1]:.2e}.",
                end="",
            )
            if u % -(-noi // 1000) == 0:
                guess0 = guess1
                guess1 = guess.copy()
                chang = np.nanpercentile(
                    np.nan_to_num(
                        np.where((p := (guess1 - guess0) / guess0) > 0, p, np.nan),
                        nan=np.nan,
                        posinf=np.nan,
                        neginf=np.nan,
                    ),
                    75,
                )
                if len(gchang) > 20 and chang < 0.001:
                    break
                gchang.append(chang)
            if t_f is not None and t_f < (time.time() - t_i):
                break
        print(gchang)
        gval[nam] = guess
    return gval


def FIST(M, fluxes, c_ite=0.1, lam=None, t_f=None):
    """Implementation of the Fast Iterative Shrinkage-Thresholding Algorithm for solving the linear-algebraic problem fluxes = M @ x."""
    M = M if type(M) is csr_array else csr_array(M)
    if lam is None:
        mATA = lambda v: M.T @ (M @ v)
        mOP = LinearOperator((M.shape[1], M.shape[1]), matvec=mATA)
        e = eigsh(mOP, k=1, which="LM", return_eigenvectors=False)[0]
        lam = 1 / e / 10
    gval = dict()
    noi = int(M.shape[0] * c_ite)
    for nam, vals in fluxes.items():
        vals = np.array(vals)
        init = np.nanmean(vals) * M.sum() / M.shape[0]
        gx9 = np.ones(M.shape[1]) * init
        gy0 = gx9
        t0 = 1
        guess1 = gx9.copy()
        gchang = [np.nan]
        t_i = time.time()
        for u in range(noi):
            gx0 = gy0 - lam * (M.T @ (M @ gy0 - vals))
            gx0[gx0 < 0] = 0
            t1 = (1 + np.sqrt(1 + 4 * t0**2)) / 2
            gy1 = gx0 + (t0 - 1) / t1 * (gx0 - gx9)
            gy0, gx9, t0 = gy1, gx0, t1
            print(
                f"\r\033[KFinished {u} out of {noi}. Convergence {gchang[-1]:.2e}.",
                end="",
            )
            if u % -(-noi // 1000) == 0:
                guess0 = guess1
                guess1 = gx9.copy()
                chang = np.nanpercentile(
                    np.nan_to_num(
                        np.where((p := (guess1 - guess0) / guess0) > 0, p, np.nan),
                        nan=np.nan,
                        posinf=np.nan,
                        neginf=np.nan,
                    ),
                    75,
                )
                if (
                    len(gchang) > 20
                    and chang < 0.001  # np.nanpercentile(gchang[1:10], 20) / 200
                ):
                    break
                gchang.append(chang)
                if t_f is not None and t_f < (time.time() - t_i):
                    break
        print(gchang)
        gval[nam] = gx9
    return gval


def calculate_fluxes(
    sources,
    tups=tuple(ac.core_lines.values()),
    method=lambda M, f: OSEM(M, f, t_f=1800),
    n_one=200,
    n_sam=250000,
):
    """Centralised function to reconstruct individual line-fluxes covered by the provided spectra. Includes both the steps of calculating fluxes in sub-sampled stacks and subsequent linear-algebraic reconstruction of individual fluxes. Finally saves calculated values as entries of catalogue of spectra."""
    for tup in tups:
        M, fl, so = art_fluxes(sources, tup, n_one=n_one, n_sam=n_sam)
        so_fl = method(M, fl)
        for nam, vs in so_fl.items():
            nam = "rec_" + str(nam)
            for i, s in enumerate(so):
                s[nam] = vs[i]
    return sources


def calculate_indiv_lines(sources, new=True, direct=True):
    """For catalogue of spectra with assumed pre-calculated and saved individual fluxes values: 1. identifies unique sources and creates a catalog with combined fluxes values from individual spectra, and 2. for each unique entry in the catalogue calculates all abundance measurements as available."""
    uniq = catalog.unique(sources)
    vals = ac.indiv_stat(ac.tem_den_red, uniq, calib=None, rec=True)
    for i in range(len(uniq)):
        uniq[i]["rec_tem_den"] = vals[i][:2]
    abun = dict()
    values = dict()
    if new:
        sul, nit, oxy = ac.Sulphur_new, ac.Nitrogen_new, ac.Oxygen_new
    else:
        sul, nit, oxy = ac.Sulphur, ac.Nitrogen, ac.Oxygen
    for k, v in sul.items():
        abun["rec_S_" + k + "_n" * new] = v
    for k, v in nit.items():
        abun["rec_N_" + k + "_n" * new] = v
    for k, v in oxy.items():
        abun["rec_O_" + k + "_n" * new] = v
    if direct:
        abun["direct"] = ac.abundances
    for k, f in abun.items():
        values[k] = ac.indiv_stat(f, uniq, calib=None, rec=True)
    skeys = list(values.keys())[:-1] if direct else values.keys()
    for i in range(len(uniq)):
        for k in skeys:
            uniq[i][k] = values[k][i]
    if direct:
        for i in range(len(uniq)):
            uniq[i]["rec_O_Dir" + "_n" * new] = [
                [values["direct"][i]["O"]],
                [values["direct"][i]["O"]],
            ]
            uniq[i]["rec_N_Dir" + "_n" * new] = [
                [values["direct"][i]["N"]],
                [values["direct"][i]["N"]],
            ]
            uniq[i]["rec_S_Dir" + "_n" * new] = [
                [values["direct"][i]["S"]],
                [values["direct"][i]["S"]],
            ]
    return uniq
