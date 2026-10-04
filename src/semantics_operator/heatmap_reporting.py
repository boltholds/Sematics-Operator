"""Full-resolution PNG heatmaps and an offline HTML gallery for input pairs."""

import html
import json
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import torch
from safetensors.torch import save_file

from .heatmap_directions import anchor_deltas, compare_directions
from .heatmaps import DEFAULT_TEMPLATE, aligned_matrices, compare_pair


def _plotting():
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ValueError('Heatmaps require matplotlib: pip install -e ".[plots]"') from error
    return Figure, FigureCanvasAgg


def _triptych(path, arrays, labels, title, scales, *, ylabel="Aligned token row"):
    Figure, Canvas = _plotting()
    n, width = arrays[0].shape
    # At least one output pixel per hidden coordinate; no PCA, binning or smoothing.
    figure = Figure(figsize=(max(14, width / 140 + 3), max(8, 0.52 * n + 3)), dpi=140)
    Canvas(figure)
    axes = figure.subplots(3, 1)
    figure.suptitle(title, fontsize=12)
    from matplotlib import colormaps

    cmap = colormaps["RdBu_r"].copy()
    cmap.set_bad("#d4d7db")
    for ax, values, heading, limit in zip(
        axes,
        arrays,
        ("A", "B", "Delta = B - A"),
        (scales["activation_max_abs"], scales["activation_max_abs"], scales["delta_max_abs"]),
        strict=True,
    ):
        bound = limit if limit > 0 else 1.0
        view = ax.imshow(
            np.ma.masked_invalid(values.numpy()),
            cmap=cmap,
            vmin=-bound,
            vmax=bound,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        note = (
            " (no comparable values)"
            if not torch.isfinite(values).any()
            else (" (all values zero)" if limit == 0 else "")
        )
        ax.set_title(heading + note, loc="left")
        ax.set_yticks(range(n), labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Hidden coordinate (original order)")
        figure.colorbar(view, ax=ax, fraction=0.018, pad=0.015).set_label("Activation")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.canvas.draw()
    plot_width = min(ax.get_window_extent().width for ax in axes)
    if plot_width < width:
        # Text/colourbars consume canvas pixels. Measure the actual plotting area.
        figure.set_size_inches(
            figure.get_figwidth() + (width - plot_width) / figure.dpi + 1,
            figure.get_figheight(),
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path)
    figure.clear()


def _metrics_plot(path, layers, title):
    Figure, Canvas = _plotting()
    figure = Figure(figsize=(11, 4.5), dpi=140)
    Canvas(figure)
    axes = figure.subplots(1, 2)
    indices = [layer["index"] for layer in layers]
    axes[0].plot(indices, [layer["anchor_metrics"]["delta_rms"] for layer in layers], "o-")
    axes[0].set_ylabel("RMS(B - A)")
    axes[1].plot(indices, [layer["anchor_metrics"]["cosine"] for layer in layers], "o-")
    axes[1].set_ylabel("Cosine(A, B)")
    axes[1].set_ylim(-1.05, 1.05)
    for ax in axes:
        ax.set_xlabel("Block index (zero-based)")
        ax.grid(alpha=0.25)
    figure.suptitle(title + " | last shared suffix token")
    figure.tight_layout()
    figure.savefig(path)
    figure.clear()


def _row_labels(pair):
    labels = []
    for row in pair["alignment"]:
        parts = []
        for side in ("a", "b"):
            pos = row[side]
            token = pair["inputs"][side]["tokens"][pos] if pos is not None else "—"
            # Full tokens remain available in HTML/JSON; keep image labels compact.
            token = token.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            parts.append(f"{side.upper()}:{pos if pos is not None else '-'} {token[:18]}")
        labels.append(" | ".join(parts) + " [" + row["kind"] + "]")
    return labels


def _render_pair(folder, pair, tensors):
    relative = pair["directory"]
    directory = folder / relative
    directory.mkdir()
    pair["tensor_file"] = f"{relative}/activations.safetensors"
    save_file(
        tensors,
        folder / pair["tensor_file"],
        metadata={
            "a_b": "raw [token, hidden] block outputs, before final model norm",
            "delta": "B - A on report alignment rows; NaN = unmatched token, not zero",
        },
    )
    words = [pair["inputs"][s]["word"] for s in ("a", "b")]
    title = f"A: {words[0][:60]} | B: {words[1][:60]}"
    labels = _row_labels(pair)
    for layer in pair["layers"]:
        key = layer["key"]
        arrays = aligned_matrices(tensors[key + ".a"], tensors[key + ".b"], pair["alignment"])
        layer["images"] = []
        # Paginate long contexts so every row stays readable and rendering is bounded.
        for page, start in enumerate(range(0, len(labels), 24), start=1):
            filename = f"{relative}/{key}-page-{page:03d}.png"
            _triptych(
                folder / filename,
                [a[start : start + 24] for a in arrays],
                labels[start : start + 24],
                f"{title}\n{layer['module']} | rows {start}–{min(start + 24, len(labels)) - 1}",
                pair["scales"],
            )
            layer["images"].append(filename)
    pair["overview_image"] = pair["metrics_image"] = None
    if pair["anchor"] is not None:
        a = torch.stack(
            [tensors[layer["key"] + ".a"][pair["anchor"]["a"]] for layer in pair["layers"]]
        )
        b = torch.stack(
            [tensors[layer["key"] + ".b"][pair["anchor"]["b"]] for layer in pair["layers"]]
        )
        pair["overview_image"] = f"{relative}/overview.png"
        _triptych(
            folder / pair["overview_image"],
            (a, b, b - a),
            [str(layer["index"]) for layer in pair["layers"]],
            title + "\nLast shared suffix token at every block",
            pair["scales"],
            ylabel="Block index",
        )
        pair["metrics_image"] = f"{relative}/metrics.png"
        _metrics_plot(folder / pair["metrics_image"], pair["layers"], title)


def _direction_plot(path, cross):
    Figure, Canvas = _plotting()
    comparisons = cross["comparisons"]
    columns = len(comparisons)
    values = np.array(
        [
            [layer["cosine"][c["left"]][c["right"]] for c in comparisons]
            for layer in cross["layers"]
        ],
        dtype=float,
    )
    figure = Figure(figsize=(max(6, 1.1 * columns + 2), max(4, 0.36 * len(values) + 2)), dpi=140)
    Canvas(figure)
    ax = figure.subplots()
    from matplotlib import colormaps

    cmap = colormaps["RdBu_r"].copy()
    cmap.set_bad("#d4d7db")
    view = ax.imshow(
        np.ma.masked_invalid(values),
        vmin=-1,
        vmax=1,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(columns), [f"{c['left'] + 1} vs {c['right'] + 1}" for c in comparisons])
    ax.set_yticks(range(len(values)), [str(l["index"]) for l in cross["layers"]])
    ax.set_xlabel("Pair indices (see report for words and anchor positions)")
    ax.set_ylabel("Block index (zero-based)")
    ax.set_title("Cross-pair delta directions: cosine(B - A, B - A)")
    for row in range(len(values)):
        for col in range(columns):
            value = values[row, col]
            label = f"{value:.2f}" if np.isfinite(value) else "N/A"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if np.isfinite(value) and abs(value) > 0.65 else "black",
            )
    figure.colorbar(view, ax=ax, pad=0.03).set_label("Cosine of delta directions")
    figure.tight_layout()
    figure.savefig(path)
    figure.clear()


def _direction_gallery(cross):
    esc = html.escape
    parts = [
        '<section id="directions"><h2>Сходство направлений дельт между парами</h2>',
        (
            "<p>Сравнивается cos(Δ₁, Δ₂) на последнем общем токене, где Δ = B − A. "
            "+1 — одно направление, 0 — ортогональные направления, −1 — противоположные. "
            "N/A — нет сопоставимого токена, нулевая дельта или сигнал не выше порога повторного прогона. "
            "Нулевая дельта не считается совпадением направлений.</p><ol>"
        ),
    ]
    for pair in cross["pairs"]:
        parts.append(
            f"<li>{esc(pair['label'])}; позиции A/B: {esc(str(pair['anchor_positions']))}; "
            f"число токенов: {esc(str(pair['token_counts']))}; ID общего токена: {pair['anchor_token_id']}</li>"
        )
    parts.append("</ol>")
    parts.extend(f"<p><strong>{esc(w)}</strong></p>" for w in cross["warnings"])
    parts.append(
        "<p>Порядок слов во всех языках должен задавать одинаковое смысловое направление. "
        "Шкала фиксирована: от −1 до +1. Смена токенизации и позиций может влиять на сходство. "
        "Высокое сходство само по себе не доказывает причинную роль или универсальность операции.</p>"
    )
    parts.append(
        f'<a href="{cross["image"]}"><img src="{cross["image"]}" alt="Сходство направлений дельт по блокам"></a>'
    )
    if cross["tensor_file"]:
        parts.append(
            f'<p><a href="{cross["tensor_file"]}">Векторы дельт для самостоятельного анализа</a></p>'
        )
    parts.append("</section>")
    return parts


def _gallery(report):
    esc = lambda value: html.escape(str(value), quote=True)
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Сравнение активаций по слоям</title>",
        "<style>body{font:16px system-ui;max-width:1600px;margin:32px auto;padding:0 20px;color:#202735;background:#f5f7fa}img{max-width:100%;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}table{border-collapse:collapse}td,th{border:1px solid #ccd3db;padding:6px}details,section{margin:24px 0}a{color:#195daf}</style>",
        "<h1>Активации A, B и Δ = B − A</h1>",
        (
            "<p>Выход каждого блока до финальной нормализации модели. Все координаты сохранены в исходном порядке. "
            "Шкалы A/B и отдельная шкала Δ фиксированы для всех слоёв одной пары; между парами шкалы могут отличаться. "
            "Красный — положительное значение, синий — отрицательное, белый — ноль, серый — нет сопоставимого токена. "
            "Полноразмерную картинку можно открыть нажатием.</p>"
        ),
        "<p>Различие активаций показывает зависимость представления от входа. Для проверки причинной роли нужен отдельный опыт с вмешательством.</p>",
        '<p><a href="report.json">Полный отчёт JSON</a></p><nav>',
    ]
    for i, pair in enumerate(report["pairs"]):
        name = " / ".join(pair["inputs"][s]["word"] for s in ("a", "b"))
        parts.append(f'<a href="#pair-{i}">{esc(name)}</a> · ')
    parts.append("</nav>")
    if report.get("cross_pair_directions"):
        parts.extend(_direction_gallery(report["cross_pair_directions"]))
    for i, pair in enumerate(report["pairs"]):
        parts.append(f'<section id="pair-{i}"><h2>Пара {i + 1}</h2>')
        for side in ("a", "b"):
            inp = pair["inputs"][side]
            parts.append(
                f"<h3>{side.upper()}: {esc(inp['word'])}</h3><pre>{esc(inp['text'])}</pre>"
            )
            parts.append(
                f"<details><summary>Фактические токены {side.upper()}</summary><pre>{esc(json.dumps(inp, ensure_ascii=False, indent=2))}</pre></details>"
            )
        parts.append(
            f"<p>Формат: {esc(pair['prompt_format'])}. Повтор A → A, max |Δ|: {pair['repeat_a_max_abs']:.6g}. "
            f'<a href="{pair["tensor_file"]}">Матрицы safetensors</a></p>'
        )
        parts.extend(f"<p><strong>{esc(w)}</strong></p>" for w in pair["warnings"])
        parts.append(
            "<p>prefix/suffix — одинаковые ID токенов; replacement — позиционное сравнение изменённого фрагмента одинаковой длины. "
            "Это не гарантия семантического соответствия. При разной длине фрагмента строки unmatched не имеют дельты.</p>"
        )
        for field in ("overview_image", "metrics_image"):
            if pair[field]:
                path = pair[field]
                parts.append(
                    f'<a href="{path}"><img src="{path}" alt="{field}" loading="lazy"></a>'
                )
        for layer in pair["layers"]:
            parts.append(
                f"<details open><summary>Блок {layer['index']}: {esc(layer['module'])}</summary>"
            )
            for path in layer["images"]:
                parts.append(
                    f'<a href="{path}"><img src="{path}" alt="A, B и дельта блока {layer["index"]}" loading="lazy"></a>'
                )
            parts.append("</details>")
        parts.append("</section>")
    parts.append("</html>")
    return "\n".join(parts)


def run_heatmaps(
    lm,
    cfg,
    *,
    pairs=None,
    template=DEFAULT_TEMPLATE,
    prompt_format="raw",
    layers=None,
    progress=lambda _: None,
):
    _plotting()  # Fail with installation instructions before model forwards.
    pairs = [("холодно", "жарко")] if pairs is None else pairs
    if not pairs:
        raise ValueError("At least one pair is required")
    folder = cfg.output_dir / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-heatmap-" + uuid4().hex[:8]
    )
    report = {
        "experiment": "word_pair_heatmaps_v1",
        "model": {
            "profile": cfg.profile,
            "path": str(cfg.model_path),
            "device": str(lm.device),
            "dtype": str(next(lm.model.parameters()).dtype),
        },
        "template": template,
        "capture": "full block output before final model normalization",
        "delta": "B - A; unmatched rows are NaN in tensors and grey in PNG",
        "coordinate_reduction": "none",
        "pairs": [],
        "cross_pair_directions": None,
        "limitations": [
            "Descriptive input contrast; no learned operator or causal intervention.",
            "Hidden coordinates are basis-dependent, not named concepts.",
            "Token alignment is lexical/positional, not a semantic alignment algorithm.",
        ],
    }
    directions = []
    for i, pair in enumerate(pairs):
        progress(f"Heatmap pair {i + 1}/{len(pairs)}: {pair[0]!a} -> {pair[1]!a}")
        item, tensors = compare_pair(
            lm, pair, template=template, prompt_format=prompt_format, layers=layers
        )
        folder.mkdir(parents=True, exist_ok=True)
        item["directory"] = f"pair-{i + 1:03d}"
        progress(
            f"Rendering {len(item['layers'])} blocks; repeat-A max delta = {item['repeat_a_max_abs']:.6g}"
        )
        _render_pair(folder, item, tensors)
        report["pairs"].append(item)
        directions.append(anchor_deltas(item, tensors))
        # Checkpoint each pair, and release tensors before processing the next pair.
        (folder / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (folder / "index.html").write_text(_gallery(report), encoding="utf-8")
        del tensors
    if len(pairs) > 1:
        cross = compare_directions(report["pairs"], directions)
        cross["image"] = "cross_pair_cosines.png"
        compact = {
            f"pair_{i:03d}.layer_{index:03d}.delta": value
            for i, pair in enumerate(directions)
            for index, value in pair.items()
        }
        cross["tensor_file"] = "anchor_deltas.safetensors" if compact else None
        if compact:
            save_file(
                compact,
                folder / cross["tensor_file"],
                metadata={
                    "delta": "B - A at each pair's last shared suffix token",
                    "indices": "zero-based pair and block indices",
                },
            )
        _direction_plot(folder / cross["image"], cross)
        report["cross_pair_directions"] = cross
        (folder / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (folder / "index.html").write_text(_gallery(report), encoding="utf-8")
    return folder
