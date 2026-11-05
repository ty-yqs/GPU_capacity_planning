"""Reproduce the A01 GPU capacity calculations and plot data.

Run with the bundled Python runtime (numpy and pandas required).  The random
seed and all simulation sizes are fixed.  BLAS thread limits are set before
importing numpy to keep the assignment inexpensive to run.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
LATEX = DATA / "latex"
RNG = np.random.default_rng(20261105)
REPS = 60000
BATCH = 400

requests = pd.read_csv(DATA / "request_count.csv", encoding="utf-8-sig")
service = pd.read_csv(DATA / "service_time.csv", encoding="utf-8-sig")
requests["date"] = requests.timestamp.str[:10]
peak = requests[(requests.day_type == "weekday") & (requests.period == "peak")].copy()
assert len(requests) == 20160 and len(peak) == 3000 and len(service) == 5600
assert (requests.text_requests + requests.image_requests == requests.total_requests).all()

counts = peak[["text_requests", "image_requests"]].to_numpy(dtype=np.int32)
total = counts.sum(axis=1)
tpool = service.loc[service.request_type == "text", "gpu_seconds"].to_numpy()
ipool = service.loc[service.request_type == "image", "gpu_seconds"].to_numpy()


def summary(x):
    x = np.asarray(x)
    return {"n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "variance": float(x.var(ddof=1)), "median": float(np.median(x)),
            "q95": float(np.quantile(x, .95)), "q99": float(np.quantile(x, .99)),
            "max": float(x.max()), "second_moment": float(np.mean(x*x))}


def ci_day_means(col):
    x = peak.groupby("date")[col].mean().to_numpy()
    # 95% Student t interval, df=9, treating days as independent blocks.
    half = 2.262157 * x.std(ddof=1) / math.sqrt(len(x))
    return [float(x.mean()-half), float(x.mean()+half)]


def dispersion_test(x, reps=2000):
    """Parametric-bootstrap p value for var/mean under a fitted iid Poisson."""
    lam = x.mean()
    observed = x.var(ddof=1) / lam
    exceed = 0
    for _ in range(reps):
        y = RNG.poisson(lam, size=len(x))
        if y.var(ddof=1) / y.mean() >= observed:
            exceed += 1
    return {"statistic": float(observed), "exceedances": exceed,
            "replications": reps, "p_upper_bound": float((exceed+1)/(reps+1))}


def box(x):
    q1, med, q3 = np.quantile(x, [.25, .5, .75])
    iqr = q3-q1
    lo = x[x >= q1-1.5*iqr].min()
    hi = x[x <= q3+1.5*iqr].max()
    return [float(z) for z in (lo,q1,med,q3,hi)]


def plot_data():
    out = LATEX / "fig_data"
    out.mkdir(exist_ok=True)
    periods = ["night", "morning", "daytime", "peak", "late"]
    boxes = {}
    for kind in ["weekday", "weekend"]:
        for period in periods:
            d = requests[(requests.day_type == kind) & (requests.period == period)]
            if len(d):
                boxes[kind+"_"+period] = {"text": box(d.text_requests.to_numpy()),
                                            "image": box(d.image_requests.to_numpy()),
                                            "n": len(d)}
    for typ, pool, edges in [("text",tpool,np.arange(0,6.25,.25)),
                             ("image",ipool,np.arange(0,42,2))]:
        hist, e = np.histogram(pool, bins=edges)
        # Density per unit GPU-second, so bins of unequal width are comparable.
        with (out / f"{typ}_hist.dat").open("w") as f:
            f.write("x y\n")
            for n, a, b in zip(hist,e[:-1],e[1:]):
                f.write(f"{(a+b)/2:.4f} {n/len(pool)/(b-a):.7f}\n")
    lines=[r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.1cm},width=.47\textwidth,height=6.0cm,",
           r"xmin=.5,xmax=5.5,xtick={1,2,3,4,5},xticklabels={夜间,上午,日间,高峰,深夜},",
           r"tick label style={font=\scriptsize},label style={font=\small},",
           r"boxplot/draw direction=y,boxplot/box extend=.23,grid=major,grid style={gray!15}]" ]
    for typ,title,ymax in [("text","文本请求数",160),("image","图像请求数",20)]:
        lines.append(rf"\nextgroupplot[title={{{title}}},ylabel={{次/分钟}},ymin=0,ymax={ymax}]")
        for j,period in enumerate(periods,1):
            for kind,offset,color in [("weekday",-.16,"blue!40"),("weekend",.16,"orange!50")]:
                lo,q1,med,q3,hi=boxes[kind+"_"+period][typ]
                lines.append(r"\addplot+[boxplot prepared={"+
                             f"draw position={j+offset:.2f},lower whisker={lo:g},lower quartile={q1:g},"+
                             f"median={med:g},upper quartile={q3:g},upper whisker={hi:g}"+
                             r"},fill="+color+r",draw=black!60] coordinates {};")
    lines.extend([r"\end{groupplot}",r"\end{tikzpicture}"])
    (out / "boxes.tex").write_text("\n".join(lines)+"\n")
    return boxes


def workload_components(lam_t, lam_i, var_t, var_i, cov):
    mt,mi = tpool.mean(),ipool.mean()
    vt,vi = tpool.var(ddof=1),ipool.var(ddof=1)
    text_mean,image_mean = lam_t*mt,lam_i*mi
    text_var = lam_t*vt + var_t*mt*mt
    image_var = lam_i*vi + var_i*mi*mi
    cross = 2*cov*mt*mi
    return {"text_mean":text_mean,"image_mean":image_mean,
            "mean":text_mean+image_mean,"text_variance":text_var,
            "image_variance":image_var,"covariance_variance":cross,
            "variance":text_var+image_var+cross,
            "sd":math.sqrt(text_var+image_var+cross)}


def sample_workloads(kind):
    results = np.empty(REPS)
    lam_t,lam_i = counts.mean(axis=0)
    for start in range(0,REPS,BATCH):
        n = min(BATCH, REPS-start)
        if kind == "poisson":
            ns = RNG.poisson([lam_t,lam_i],size=(n,2))
        else:
            if kind == "week1":
                source = counts[:1500]
            elif kind == "week2":
                source = counts[1500:]
            else:
                source = counts
            ns = source[RNG.integers(len(source),size=n)].copy()
            if kind == "growth":
                ns += RNG.poisson(.30*ns)
            elif kind == "image35":
                all_n = ns.sum(axis=1)
                ni = RNG.binomial(all_n,.35)
                ns = np.column_stack((all_n-ni,ni))
            elif kind == "burst":
                # A stated sensitivity scenario: 5% of minutes have 50% extra
                # arrivals of both types, conditional on their sampled counts.
                shock = RNG.random(n) < .05
                ns[shock] += RNG.poisson(.50*ns[shock])
        w = np.zeros(n)
        for col,pool in enumerate((tpool,ipool)):
            max_count = int(ns[:,col].max())
            sampled = pool[RNG.integers(len(pool),size=(n,max_count))]
            w += (sampled * (np.arange(max_count)[None,:] < ns[:,col,None])).sum(axis=1)
        results[start:start+n] = w
    return results


def wilson(k,n):
    z=1.959964; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-radius,center+radius]


def mc_summary(w):
    ans={"mean":float(w.mean()),"sd":float(w.std(ddof=1)),
         "q99":float(np.quantile(w,.99)),"q995":float(np.quantile(w,.995))}
    ans["gpu"]={}
    for m in range(3,15):
        k=int(np.count_nonzero(w>48*m))
        ans["gpu"][str(m)]={"exceed":k,"probability":k/len(w),"wilson95":wilson(k,len(w))}
    ans["minimum_point"]=next(m for m in range(3,15) if ans["gpu"][str(m)]["probability"]<=.01)
    ans["minimum_conservative"]=next(m for m in range(3,15) if ans["gpu"][str(m)]["wilson95"][1]<=.01)
    return ans


def normal_tail(z):
    return .5*math.erfc(z/math.sqrt(2))


def main():
    n_t,n_i = counts[:,0],counts[:,1]
    cov = float(np.cov(n_t,n_i,ddof=1)[0,1])
    lam_t,lam_i = n_t.mean(),n_i.mean()
    poisson=workload_components(lam_t,lam_i,lam_t,lam_i,0)
    empirical=workload_components(lam_t,lam_i,n_t.var(ddof=1),n_i.var(ddof=1),cov)
    data={"seed":20261105,"mc_reps":REPS,"peak_minutes":len(peak),
          "peak_days":peak.date.nunique(),
          "counts":{"text":summary(n_t),"image":summary(n_i),"total":summary(total),
                    "correlation":float(np.corrcoef(n_t,n_i)[0,1]),"covariance":cov,
                    "image_fraction":float(n_i.sum()/total.sum()),
                    "lambda_ci_day_t":{"text":ci_day_means("text_requests"),
                                       "image":ci_day_means("image_requests")},
                    "poisson_dispersion_test":{"text":dispersion_test(n_t),
                                               "image":dispersion_test(n_i)}},
          "service":{},"boxes":plot_data(),
          "workload":{"poisson_independent":poisson,"empirical_joint_counts":empirical}}
    for name,pool in [("text",tpool),("image",ipool)]:
        d=service[service.request_type==name]
        cold=d[d.cold_start==1].gpu_seconds.to_numpy()
        warm=d[d.cold_start==0].gpu_seconds.to_numpy()
        data["service"][name]={"all":summary(pool),"warm":summary(warm),
                               "cold":summary(cold),"cold_probability":len(cold)/len(pool),
                               "cold_mean_contribution":len(cold)/len(pool)*cold.mean(),
                               "warm_mean_contribution":len(warm)/len(pool)*warm.mean()}
    data["normal_tail"]={str(m):normal_tail((48*m-poisson["mean"])/poisson["sd"])
                         for m in range(3,9)}
    data["simulation"]={}
    for kind in ("poisson","current","growth","image35","burst","week1","week2"):
        print("Simulating",kind,flush=True)
        data["simulation"][kind]=mc_summary(sample_workloads(kind))
    (ROOT/"results.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"counts":data["counts"],"workload":data["workload"],
                      "normal_tail":data["normal_tail"],
                      "simulation":data["simulation"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
