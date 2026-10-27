"""Publication figures for PRAMIGO (static, print-ready PDFs).

* draw_multiomic_network  - the attention network (network_multiomic.pdf): hairline edges coloured by
                            omic, samples as condition-coloured rings, shaded modules (connected
                            components), a few labelled features per omic and module.
* draw_program_network    - the Leiden program network (leiden_cluster_network.pdf): one pie per program
                            (omic composition), edge width = summed inter-program weight, a label block
                            (program, size, top members) beside every node.
* draw_program_members    - one program's top members as a signed bar chart (weight x sigma).
* bh_fdr, square_frame, place_labels - shared helpers.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Wedge
from matplotlib.transforms import Bbox

INK, INK2, MUTED = "#0b0b0b", "#4a4a4a", "#8a8a8a"
CROSS_EDGE = "#8c8c8c"  # edges between features of two different omics


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (NaNs are left as NaN)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    q = pv[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.minimum(q, 1)
    out[ok] = tmp
    return out


def square_frame(ax, lw=0.9):
    """Square plotting area, black frame on all four sides, no grid."""
    ax.grid(False)
    ax.set_box_aspect(1)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(lw)


def place_labels(ax, fig, items, obstacles, fontsize=5.6, scale=1.0, pad=5, italic=False):
    """Put each label on the free side of its node, pushed outward until it overlaps neither another
    label nor a node; long moves get a thin leader line. items = [((x, y), text), ...]."""
    if not items:
        return
    fig.canvas.draw()
    to_px = ax.transData.transform
    obst = to_px(np.asarray(obstacles, float))
    centre = obst.mean(axis=0)
    renderer = fig.canvas.get_renderer()
    placed = []
    for (x, y), text in items:
        p = to_px((x, y))
        direction = p - centre
        direction = direction / (np.linalg.norm(direction) or 1)
        angles = np.deg2rad([0, 25, -25, 50, -50, 80, -80, 115, -115, 150, -150, 180])
        best = None
        for r in (scale * v for v in (7, 12, 18, 26, 36, 48, 62, 78, 95)):
            for a in angles:
                c, s = np.cos(a), np.sin(a)
                d = np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])
                anchor = p + r * d
                t = ax.text(*ax.transData.inverted().transform(anchor), text, fontsize=fontsize,
                            ha="left" if d[0] >= 0 else "right", va="center",
                            fontstyle="italic" if italic else "normal")
                bb = t.get_window_extent(renderer).expanded(1.08, 1.25)
                t.remove()
                hit_label = any(bb.overlaps(o) for o in placed)
                hit_node = np.any((obst[:, 0] > bb.x0 - pad) & (obst[:, 0] < bb.x1 + pad) &
                                  (obst[:, 1] > bb.y0 - pad) & (obst[:, 1] < bb.y1 + pad))
                if not hit_label and not hit_node:
                    best = (anchor, d, r, bb)
                    break
            if best:
                break
        if best is None:
            anchor, d, r, bb = p + 12 * scale * direction, direction, 12 * scale, None
        else:
            anchor, d, r, bb = best
        ax.annotate(text, (x, y), xytext=ax.transData.inverted().transform(anchor), fontsize=fontsize,
                    color=INK, ha="left" if d[0] >= 0 else "right", va="center", zorder=6,
                    fontstyle="italic" if italic else "normal",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
                    arrowprops=None if r <= 12 * scale else dict(arrowstyle="-", lw=0.4, color=MUTED,
                                                                 shrinkA=0, shrinkB=2))
        if bb is not None:
            placed.append(bb)


def _trunc(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


# ---------------------------------------------------------------------------
# Attention network
# ---------------------------------------------------------------------------
def _packed_layout(G, seed=3):
    """Spring layout per connected component (as cll_network_final: weight=None, k = 2.6/sqrt(n)),
    each component scaled to a radius proportional to sqrt(its size) and packed in rows, largest
    first. A single layout of the whole graph would fling small components far away and shrink
    the main module to a speck."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    n_max = len(comps[0])
    blocks = []
    for comp in comps:
        H = G.subgraph(comp)
        if len(H) == 1:
            p = {next(iter(H)): np.zeros(2)}
        else:
            p = nx.spring_layout(H, weight=None, k=2.6 / np.sqrt(len(H)), iterations=1000, seed=seed)
        X = np.array(list(p.values()), float)
        X -= X.mean(axis=0)
        X /= (np.abs(X).max() or 1)
        r = np.sqrt(len(H) / n_max)
        blocks.append(({v: X[i] * r for i, v in enumerate(p)}, r))
    pos, x, y, row_h = {}, 0.0, 0.0, 0.0
    width = 2.0 * blocks[0][1] + 1.2  # main module plus a column of small ones
    for p, r in blocks:
        if x > 0 and x + 2 * r > width:
            x, y, row_h = 0.0, y - row_h - 0.25, 0.0
        for v, xy in p.items():
            pos[v] = xy + np.array([x + r, y - r])
        x += 2 * r + 0.25
        row_h = max(row_h, 2 * r)
    X = np.array(list(pos.values()))
    X -= (X.min(axis=0) + X.max(axis=0)) / 2
    X /= (np.abs(X).max() or 1)
    return {v: X[i] for i, v in enumerate(pos)}


def draw_multiomic_network(nodes, edges, omic_names, omic_colors, omic_markers, cond_colors,
                           top_n, footer_note, n_labels=5, lw_range=None, seed=3):
    """nodes: DataFrame(node, label, type, condition) with type = an omic name or 'sample';
    edges: DataFrame(source, target, weight > 0). Returns the figure."""
    G = nx.Graph()
    for r in nodes.itertuples():
        G.add_node(r.node, label=str(r.label), type=r.type, condition=str(r.condition) if r.type == "sample" else "")
    for r in edges.itertuples():
        G.add_edge(r.source, r.target, weight=float(r.weight), lw=float(np.log(r.weight)))

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.89)
    ax.set_axis_off()
    conds = sorted({G.nodes[v]["condition"] for v in G if G.nodes[v]["type"] == "sample"})
    title = "–".join(omic_names) + "–sample attention"
    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "no edge passes the cutoff", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
        fig.text(0.06, 0.93, title, fontsize=9.5, fontweight="bold", color=INK, va="bottom")
        return fig

    pos = _packed_layout(G, seed)
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))

    # Modules (connected components) as shaded hulls, tinted by their dominant omic.
    from scipy.spatial import ConvexHull
    hull_pts = []
    for comp in nx.connected_components(G):
        if len(comp) < 3:
            continue
        pts = np.array([pos[v] for v in comp])
        try:
            hull = ConvexHull(pts)
        except Exception:  # collinear points
            continue
        types = [G.nodes[v]["type"] for v in comp if G.nodes[v]["type"] != "sample"]
        dom = max(set(types), key=types.count) if types else None
        col = omic_colors[dom] if dom is not None and types.count(dom) >= 0.75 * len(types) else MUTED
        c = pts[hull.vertices].mean(axis=0)
        poly = c + (pts[hull.vertices] - c) * 1.10
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec="none", alpha=0.06, zorder=0))
        ax.add_patch(Polygon(poly, closed=True, fc="none", ec=col, alpha=0.45, lw=0.6, ls=(0, (3, 2)), zorder=0))
        hull_pts.extend(poly)

    # Edges: colour = omic (grey between two omics), width/opacity = log weight.
    lws = np.array([d["lw"] for *_, d in G.edges(data=True)])
    lo, hi = lw_range or (lws.min(), lws.max())
    span = (hi - lo) or 1.0
    boost = float(np.clip(np.sqrt(4000 / G.number_of_edges()), 1, 3.5))

    def edge_kind(u, v):
        t = {G.nodes[u]["type"], G.nodes[v]["type"]} - {"sample"}
        return t.pop() if len(t) == 1 else "cross"

    for u, v, d in sorted(G.edges(data=True), key=lambda e: e[2]["lw"]):
        a = float(np.clip((d["lw"] - lo) / span, 0, 1))
        k = edge_kind(u, v)
        ax.plot(*zip(pos[u], pos[v]), color=CROSS_EDGE if k == "cross" else omic_colors[k],
                lw=(0.10 + 0.55 * a) * min(boost, 2),
                alpha=min(0.9, ((0.05 if k == "cross" else 0.10) + 0.40 * a) * boost),
                zorder=1, solid_capstyle="round")

    def sizes(vs, lo_, span_):
        m = max(deg[v] for v in vs)
        return [lo_ + span_ * np.sqrt(deg[v] / m) for v in vs]

    for t in omic_names:
        vs = [v for v in G if G.nodes[v]["type"] == t]
        if vs:
            ax.scatter([pos[v][0] for v in vs], [pos[v][1] for v in vs], s=sizes(vs, 5, 20),
                       marker=omic_markers[t], c=omic_colors[t], edgecolors="white", linewidths=0.3, zorder=3)
    samples = [v for v in G if G.nodes[v]["type"] == "sample"]
    if samples:
        s_size = dict(zip(samples, sizes(samples, 10, 46)))
        for cond in conds:
            vs = [v for v in samples if G.nodes[v]["condition"] == cond]
            ax.scatter([pos[v][0] for v in vs], [pos[v][1] for v in vs], s=[s_size[v] for v in vs], marker="o",
                       c="white", edgecolors=cond_colors[cond], linewidths=1.4, zorder=4)

    ax.set_aspect("equal")
    allxy = np.array(list(pos.values()) + hull_pts)
    ax.set_xlim(allxy[:, 0].min() - 0.10, allxy[:, 0].max() + 0.10)
    ax.set_ylim(allxy[:, 1].min() - 0.10, allxy[:, 1].max() + 0.10)

    # Labels: the n_labels features per omic with the highest summed weight, in every module.
    items = []
    for comp in nx.connected_components(G):
        for t in omic_names:
            vs = sorted([v for v in comp if G.nodes[v]["type"] == t], key=lambda v: wdeg[v], reverse=True)
            items += [(pos[v], _trunc(G.nodes[v]["label"], 28)) for v in vs[:n_labels]]
    place_labels(ax, fig, items, obstacles=[pos[v] for v in G], fontsize=5.8)

    n_cond = {c: sum(G.nodes[v]["condition"] == c for v in samples) for c in conds}
    n_omic = {t: sum(G.nodes[v]["type"] == t for v in G) for t in omic_names}
    fig.text(0.06, 0.935, title, fontsize=9.5, fontweight="bold", color=INK, va="bottom")
    fig.text(0.06, 0.915, f"Top {top_n} features per omic · "
             + " · ".join(f"{n_omic[t]} {t}" for t in omic_names)
             + f" · {len(samples)} samples (" + ", ".join(f"{n_cond[c]} {c}" for c in conds) + ")"
             + f" · {G.number_of_edges():,} edges", fontsize=6.3, color=INK2, va="bottom")
    handles = [Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec=cond_colors[c], mew=1.4,
                      label=f"Sample, {c} (n = {n_cond[c]})") for c in conds]
    handles += [Line2D([], [], ls="", marker=omic_markers[t], ms=5, mfc=omic_colors[t], mec="white", label=t)
                for t in omic_names]
    handles += [Line2D([], [], color=omic_colors[t], lw=1.2, label=f"{t} edge") for t in omic_names]
    if len(omic_names) > 1:
        handles.append(Line2D([], [], color=CROSS_EDGE, lw=1.2, label="cross-omic edge"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, fontsize=6.2,
                     bbox_to_anchor=(0.5, 0.045), handletextpad=0.4, columnspacing=1.3, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.5, 0.04, footer_note, ha="center", va="top", fontsize=5.3, color=MUTED, wrap=True)
    return fig


# ---------------------------------------------------------------------------
# Program network
# ---------------------------------------------------------------------------
def _styled_block(ax, x, y, lines, ha, va, offset, fs=6.0, bold_first=True):
    lh = fs * 1.28
    n = len(lines)
    texts = []
    for k, line in enumerate(lines):
        dy = {"bottom": (n - 1 - k) * lh + lh / 2, "top": -(k * lh + lh / 2), "center": ((n - 1) / 2 - k) * lh}[va]
        t = ax.annotate(line, (x, y), xytext=(offset[0], offset[1] + dy), textcoords="offset points", ha=ha,
                        va="center", fontsize=fs, color=INK, zorder=6, annotation_clip=False,
                        fontweight="bold" if (k == 0 and bold_first) else "normal")
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
        texts.append(t)
    return texts


def draw_program_network(programs, pos, edges, omic_names, omic_colors, title, subtitle):
    """programs: list of dict(key, name, size, counts={omic: n}, top=[member labels]);
    pos: {key: (x, y)}; edges: list of (key_a, key_b, weight). Returns the figure."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.90)
    info = {p["key"]: p for p in programs}
    xy = np.array([pos[p["key"]] for p in programs], float)
    ext = max(np.ptp(xy[:, 0]) if len(xy) > 1 else 0, np.ptp(xy[:, 1]) if len(xy) > 1 else 0, 1.0)

    def radius(k):
        return ext * (0.028 + 0.015 * np.log1p(info[k]["size"]))

    if edges:
        wmax = max(w for *_, w in edges) or 1.0
        for a, b, w in sorted(edges, key=lambda e: e[2]):
            ax.plot(*zip(pos[a], pos[b]), color="#9a9a9a", alpha=0.35, lw=0.25 + 3.0 * w / wmax,
                    solid_capstyle="round", zorder=1)
    for p in programs:
        x, y = pos[p["key"]]
        r = radius(p["key"])
        start = 90.0
        cuts = []
        for t in omic_names:
            n = p["counts"].get(t, 0)
            if n == 0:
                continue
            end = start + 360.0 * n / p["size"]
            ax.add_patch(Wedge((x, y), r, start, end, fc=omic_colors[t], ec="none", zorder=3))
            cuts.append(start)
            start = end
        if len(cuts) > 1:  # white seams between slices
            for ang in cuts:
                t_ = np.deg2rad(ang)
                ax.plot([x, x + r * np.cos(t_)], [y, y + r * np.sin(t_)], color="white", lw=0.8, zorder=3.5)
        ax.add_patch(Circle((x, y), r, fc="none", ec=INK, lw=0.6, zorder=4))
    pad = 0.55 * ext
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Label blocks beside the nodes: first side that overlaps no node and no other block.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    circles = {}
    for k in info:
        cx, cy = ax.transData.transform(pos[k])
        rp = ax.transData.transform((pos[k][0] + radius(k), pos[k][1]))[0] - cx
        circles[k] = Bbox([[cx - rp, cy - rp], [cx + rp, cy + rp]])

    def area(a, b):
        w = min(a.x1, b.x1) - max(a.x0, b.x0)
        h = min(a.y1, b.y1) - max(a.y0, b.y0)
        return w * h if w > 0 and h > 0 else 0.0

    xmid = float(np.mean(xy[:, 0]))
    placed = []
    for p in sorted(programs, key=lambda p: -p["size"]):
        k = p["key"]
        x, y = pos[k]
        r = radius(k)
        lines = [f"{p['name']} · n = {p['size']}"] + [_trunc(m, 30) for m in p["top"]]
        spots = {"above": ((x, y + r), "center", "bottom", (0, 3)), "below": ((x, y - r), "center", "top", (0, -3)),
                 "right": ((x + r, y), "left", "center", (5, 0)), "left": ((x - r, y), "right", "center", (-5, 0))}
        outer, inner = ("right", "left") if x >= xmid else ("left", "right")
        best = None
        for side in (outer, "above", "below", inner):
            anchor, ha, va, off = spots[side]
            texts = _styled_block(ax, *anchor, lines, ha=ha, va=va, offset=off)
            tight = Bbox.union([t.get_window_extent(renderer) for t in texts])
            gap = 8 * fig.dpi / 72
            box = Bbox([[tight.x0 - gap, tight.y0 - gap / 2], [tight.x1 + gap, tight.y1 + gap / 2]])
            cost = sum(area(box, b) for kk, b in circles.items() if kk != k) + sum(area(box, b) for b in placed)
            if best is None or cost < best[0]:
                if best:
                    for t in best[1]:
                        t.remove()
                best = (cost, texts, box)
            else:
                for t in texts:
                    t.remove()
            if cost <= 0:
                break
        placed.append(best[2])

    fig.text(0.06, 0.955, title, fontsize=10, fontweight="bold", color=INK, va="bottom")
    fig.text(0.06, 0.947, subtitle, fontsize=6, color=INK2, va="top")
    fig.legend(handles=[Patch(fc=omic_colors[t], label=t) for t in omic_names], loc="lower center",
               ncol=len(omic_names), frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.02),
               title="program composition", title_fontsize=6.5)
    return fig


# ---------------------------------------------------------------------------
# Program members (signed weight)
# ---------------------------------------------------------------------------
def draw_program_members(ax, labels, values, omics, omic_names, omic_markers, title, n_rows=10, fs=7.0):
    """Members sorted from most positive (top) to most negative weight, bars coloured RdBu_r by
    value; members of every omic but the first are marked with that omic's (black) marker.
    Returns the power of ten the values were divided by."""
    df = pd.DataFrame({"label": labels, "val": values, "omic": omics}).sort_values("val", ascending=False)
    df = df.head(n_rows)
    vmax = float(np.abs(df.val).max()) if len(df) else 1.0
    e = int(np.floor(np.log10(vmax))) if vmax > 0 else 0
    v = df.val.to_numpy() / 10.0 ** e
    lim = float(np.abs(v).max()) * 1.08 if len(v) else 1.0
    norm = matplotlib.colors.TwoSlopeNorm(0, -lim, lim)
    y = n_rows - 1 - np.arange(len(df))
    ax.barh(y, v, color=[plt.get_cmap("RdBu_r")(norm(x)) for x in v], edgecolor="black", linewidth=0.6, height=0.72)
    ax.axvline(0, color="black", linewidth=0.7)
    for t in omic_names[1:]:
        m = (df.omic == t).to_numpy()
        if m.any():
            ax.scatter(v[m] / 2, y[m], s=9, marker=omic_markers[t], color="black", zorder=3)
    lab = [_trunc(x, 22) for x in df.label]
    dup = pd.Series(lab).duplicated(keep=False)
    cnt = {}
    for i, l_ in enumerate(lab):
        if dup[i]:
            cnt[l_] = cnt.get(l_, 0) + 1
            lab[i] = f"{l_} ({cnt[l_]})"
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=fs)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlim(-lim, lim)
    ax.tick_params(axis="x", labelsize=fs - 0.5, length=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.set_title(title, fontsize=fs + 1, fontweight="bold", loc="left", pad=4)
    square_frame(ax)
    return e
