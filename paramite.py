import time
from multiprocessing import Manager, Process, cpu_count

import numpy as np


def sep_inds(ln, no):
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


def update_par(srss, method, nwg=0.3):
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
        t = Process(target=update_pro, args=args)
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


def update_pro(isrs, method, vals):
    for i, srs in isrs:
        v = method(srs)
        vals[i] = v


def ded_params(sources, method, no=50, sbin=5, nwg=0.3, ite=None):
    defs = []
    nefs = []
    no = min(no, int(len(sources) / 20))
    inds = sep_inds(len(sources), no)
    nb = len(inds) - 1
    ite = 5 * int(nb / sbin) if ite is None else ite
    update_par([[s] for s in sources], method)
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
        update_par(sini, method, nwg=nwg)
        ite -= 1
