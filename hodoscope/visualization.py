#!/usr/bin/env python3
"""
Visualization functions for trajectory embeddings.
Creates a unified interactive visualization with method switcher dropdown,
density heatmap overlay, and FPS-based flagging.
"""

import base64
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.neighbors import KernelDensity
from bokeh.plotting import figure, save
from bokeh.models import (ColumnDataSource, CustomJS, TextInput, Div, HoverTool,
                          LinearColorMapper, ColorBar, BasicTicker,
                          Select, Slider, Spacer, InlineStyleSheet)
from bokeh.layouts import column, row
from bokeh.resources import CDN
from bokeh.io import output_file
from bokeh.palettes import RdBu11
from bokeh.events import DocumentReady, ValueSubmit

from .sampling import (
    ALL_PLOT_METHODS,
    SAMPLING_METHOD_DISPLAY_NAMES,
    PlotData,
    collect_plot_data,
    compute_projection,
    compute_bandwidth,
    compute_kde_densities,
    compute_fps_ranks,
)


DEFAULT_COLORS = [
    '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
    '#FF97FF', '#FECB52', '#2CA02C',
    '#636EFA', '#EF553B', '#1F77B4',  # blue/red last — conflict with RdBu heatmap
]

SIDEBAR_HTML = """
<style>
  body {
    background: #f0f2f5;
    margin: 0;
    padding: 20px 24px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  #detail-panel {
    position: fixed; top: 0; right: 0; width: 440px; height: 100vh;
    background: #ffffff; overflow-y: auto;
    box-shadow: -3px 0 12px rgba(0,0,0,0.10);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px; line-height: 1.5;
    display: none; z-index: 9999; box-sizing: border-box;
    padding: 0;
  }
  #detail-panel .panel-header {
    position: sticky; top: 0; background: #f8f9fa;
    padding: 16px 20px 12px 20px;
    border-bottom: 1px solid #e8e8e8;
    z-index: 1;
  }
  #detail-panel .panel-header h3 {
    margin: 0; font-size: 15px; font-weight: 600; color: #333;
  }
  #detail-panel .close-btn {
    position: absolute; top: 12px; right: 16px; cursor: pointer;
    font-size: 22px; color: #999; line-height: 1;
  }
  #detail-panel .close-btn:hover { color: #333; }
  #detail-panel .panel-body { padding: 16px 20px 20px 20px; }
  #detail-panel .meta-section { margin-bottom: 12px; }
  #detail-panel .meta-row {
    display: flex; padding: 5px 0; font-size: 13px;
    border-bottom: 1px solid #f0f0f0;
  }
  #detail-panel .meta-label {
    width: 85px; flex-shrink: 0;
    font-weight: 600; color: #666;
  }
  #detail-panel .meta-value { color: #333; word-break: break-all; }
  #detail-panel .section-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: #888; margin: 16px 0 8px 0;
  }
  #detail-panel pre {
    white-space: pre-wrap; word-wrap: break-word;
    background: #f7f8fa; padding: 12px; border-radius: 6px;
    border: 1px solid #eef0f2;
    width: 100%; box-sizing: border-box;
    font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
    font-size: 12px; line-height: 1.5; color: #333;
    margin: 0;
  }
  #detail-panel .section-header {
    display: flex; align-items: center; justify-content: space-between;
    margin: 16px 0 8px 0;
  }
  #detail-panel .section-header .section-title { margin: 0; }
  #detail-panel .copy-btn {
    background: none; border: 1px solid #ddd; border-radius: 4px;
    cursor: pointer; padding: 2px 6px; color: #888;
    font-size: 12px; line-height: 1; display: flex; align-items: center; gap: 4px;
  }
  #detail-panel .copy-btn:hover { background: #f0f0f0; color: #555; border-color: #bbb; }
  #detail-panel .copy-btn.copied { color: #22c55e; border-color: #22c55e; }
  #drag-handle {
    position: fixed; top: 0; width: 6px; height: 100vh;
    cursor: col-resize; z-index: 10000; background: transparent;
    display: none;
  }
  #drag-handle:hover, #drag-handle.active { background: #636EFA; }
  #detail-panel .nav-bar {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 0 4px 0; margin-bottom: 8px;
    border-bottom: 1px solid #f0f0f0;
  }
  #detail-panel .nav-btn {
    background: #f0f2f5; border: 1px solid #ddd; border-radius: 4px;
    cursor: pointer; padding: 3px 10px; color: #555; font-size: 12px;
    font-weight: 500; line-height: 1.4;
  }
  #detail-panel .nav-btn:hover { background: #e4e6ea; border-color: #bbb; color: #333; }
  #detail-panel .nav-btn svg { vertical-align: -2px; }
  #detail-panel .nav-status {
    font-size: 11px; color: #aaa; font-style: italic;
  }
  #detail-panel .turn-nav-btn {
    background: #f7f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    cursor: pointer;
    padding: 1px 6px;
    margin-left: 6px;
    color: #666;
    font-size: 11px;
    line-height: 1.3;
  }
  #detail-panel .turn-nav-btn:hover {
    background: #eceff3;
    border-color: #c9d0d8;
    color: #333;
  }
  #detail-panel .context-toggle {
    display: flex; align-items: center; justify-content: space-between;
    margin: 16px 0 8px 0; cursor: pointer; user-select: none;
  }
  #detail-panel .context-toggle .section-title { margin: 0; }
  #detail-panel .context-toggle .toggle-left {
    display: flex; align-items: center; gap: 6px;
  }
  #detail-panel .toggle-arrow {
    display: inline-block; font-size: 10px; color: #888;
    transition: transform 0.15s ease;
  }
  #detail-panel .toggle-arrow.open { transform: rotate(90deg); }
</style>
<div id="drag-handle"></div>
<div id="detail-panel">
  <div class="panel-header">
    <span class="close-btn" onclick="document.getElementById('detail-panel').style.display='none';document.getElementById('drag-handle').style.display='none';window._hodoCurrentPoint=null;if(window._hodoHighlight){window._hodoHighlight.data={x:[],y:[],color:[]};window._hodoHighlight.change.emit()}">&times;</span>
    <h3>Turn Details</h3>
  </div>
  <div class="panel-body" id="detail-content">
    <p style="color:#999">Click a point to see details.</p>
  </div>
</div>
<script>
  (function() {
    var panel = document.getElementById('detail-panel');
    var handle = document.getElementById('drag-handle');
    function syncHandle() {
      handle.style.right = panel.offsetWidth + 'px';
    }
    var dragging = false;
    handle.addEventListener('mousedown', function(e) {
      dragging = true; handle.classList.add('active'); e.preventDefault();
    });
    document.addEventListener('mousemove', function(e) {
      if (!dragging) return;
      var w = window.innerWidth - e.clientX;
      if (w < 200) w = 200;
      if (w > window.innerWidth - 100) w = window.innerWidth - 100;
      panel.style.width = w + 'px';
      syncHandle();
    });
    document.addEventListener('mouseup', function() {
      if (dragging) { dragging = false; handle.classList.remove('active'); }
    });
    new MutationObserver(function() {
      if (panel.style.display !== 'none') syncHandle();
    }).observe(panel, {attributes: true, attributeFilter: ['style']});
    // Navigation: sources/renderers are set by callbacks; lazy fallback scans
    // Bokeh models by deterministic names for first-click reliability.
    window._hodoSources = window._hodoSources || null;
    window._hodoRenderers = window._hodoRenderers || null;
    window._hodoHighlight = window._hodoHighlight || null;
    window._hodoModeSource = window._hodoModeSource || null;
    window._hodoDensitySelector = window._hodoDensitySelector || null;
    window._hodoMode = (typeof window._hodoMode === 'number') ? window._hodoMode : 1;
    window._hodoCurrentPoint = window._hodoCurrentPoint || null;  // {s: sourceIdx, i: pointIdx}
    if (typeof window._hodoContextCollapsed === 'undefined') window._hodoContextCollapsed = true;
    window._hodoEnsureNavState = function() {
      if (window._hodoSources && window._hodoSources.length > 0) return true;
      if (typeof Bokeh === 'undefined' || !Bokeh.index) return false;
      try {
        var sourceByIdx = {};
        var rendererByIdx = {};
        var views = Object.values(Bokeh.index);
        for (var v = 0; v < views.length; v++) {
          var view = views[v];
          var doc = view && view.model && view.model.document;
          if (!doc || !doc._all_models || typeof doc._all_models.values !== 'function') continue;
          var models = Array.from(doc._all_models.values());
          for (var m = 0; m < models.length; m++) {
            var model = models[m];
            var name = (model && typeof model.name === 'string') ? model.name : '';
            var sm = /^hodo_source_(\\d+)$/.exec(name);
            if (sm && model.data && model.data['summary']) {
              sourceByIdx[parseInt(sm[1], 10)] = model;
            }
            var rm = /^hodo_renderer_(\\d+)$/.exec(name);
            if (rm && model.data_source && model.data_source.data && model.data_source.data['summary']) {
              rendererByIdx[parseInt(rm[1], 10)] = model;
            }
          }
        }
        var idxs = Object.keys(sourceByIdx).map(function(k) { return parseInt(k, 10); }).sort(function(a, b) { return a - b; });
        if (idxs.length === 0) return false;
        var initSources = [];
        var initRenderers = [];
        for (var i = 0; i < idxs.length; i++) {
          var idx = idxs[i];
          initSources.push(sourceByIdx[idx]);
          initRenderers.push(rendererByIdx[idx] || null);
        }
        window._hodoSources = initSources;
        window._hodoRenderers = initRenderers;
        console.log('[hodo eye] lazy init', {
          sourceCount: initSources.length,
          rendererCount: initRenderers.filter(function(r) { return !!r; }).length,
        });
        return true;
      } catch (e) {
        console.log('[hodo eye] lazy init failed', e);
        return false;
      }
    };
    window._hodoNav = function(action) {
      var sources = window._hodoSources;
      var renderers = window._hodoRenderers;
      if ((!sources || sources.length === 0) && window._hodoEnsureNavState) {
        window._hodoEnsureNavState();
        sources = window._hodoSources;
        renderers = window._hodoRenderers;
      }
      if (action === 'status' || action === 'panel') {
        console.log('[hodo eye] click', {
          action: action,
          hasSources: !!sources,
          hasRenderers: !!renderers,
          currentPoint: window._hodoCurrentPoint,
          mode: window._hodoMode,
        });
      }
      if (!sources) return;
      var mode = (typeof window._hodoMode === 'number') ? window._hodoMode : 1;
      if (window._hodoModeSource &&
          window._hodoModeSource.data &&
          window._hodoModeSource.data['active'] &&
          window._hodoModeSource.data['active'].length > 0 &&
          typeof window._hodoModeSource.data['active'][0] === 'number') {
        mode = window._hodoModeSource.data['active'][0];
      }
      var sentinel = 1000000000;
      var target = null;
      var cur = window._hodoCurrentPoint;

      if (action === 'turn_prev' || action === 'turn_next') {
        if (!cur || !sources[cur.s] || !sources[cur.s].data) return;
        var curData0 = sources[cur.s].data;
        var trajUuids0 = curData0['trajectory_uuid'];
        var turnIds0 = curData0['turn_id'];
        if (!trajUuids0 || !turnIds0 || cur.i >= trajUuids0.length) return;
        var trajUuid = trajUuids0[cur.i];

        var parseTurn = function(v) {
          if (typeof v === 'number' && isFinite(v)) return v;
          var sVal = String(v == null ? '' : v).trim();
          if (/^-?\\d+$/.test(sVal)) return parseInt(sVal, 10);
          var m = /-?\\d+/.exec(sVal);
          if (m) return parseInt(m[0], 10);
          return NaN;
        };

        var trajPoints = [];
        for (var ts = 0; ts < sources.length; ts++) {
          var td = sources[ts].data;
          var tTrajUuid = td['trajectory_uuid'];
          var tTurn = td['turn_id'];
          if (!tTrajUuid || !tTurn) continue;
          for (var ti = 0; ti < tTrajUuid.length; ti++) {
            if (tTrajUuid[ti] === trajUuid) {
              trajPoints.push({
                s: ts,
                i: ti,
                turnRaw: String(tTurn[ti]),
                turnNum: parseTurn(tTurn[ti]),
              });
            }
          }
        }
        if (trajPoints.length === 0) return;
        trajPoints.sort(function(a, b) {
          var aNum = isFinite(a.turnNum);
          var bNum = isFinite(b.turnNum);
          if (aNum && bNum && a.turnNum !== b.turnNum) return a.turnNum - b.turnNum;
          if (aNum !== bNum) return aNum ? -1 : 1;
          if (a.turnRaw !== b.turnRaw) return a.turnRaw < b.turnRaw ? -1 : 1;
          if (a.s !== b.s) return a.s - b.s;
          return a.i - b.i;
        });

        var tIdx = -1;
        for (var tj = 0; tj < trajPoints.length; tj++) {
          if (trajPoints[tj].s === cur.s && trajPoints[tj].i === cur.i) {
            tIdx = tj;
            break;
          }
        }
        if (action === 'turn_next') {
          target = (tIdx === -1) ? trajPoints[0] : trajPoints[(tIdx + 1) % trajPoints.length];
        } else {
          target = (tIdx === -1) ? trajPoints[trajPoints.length - 1] : trajPoints[(tIdx - 1 + trajPoints.length) % trajPoints.length];
        }
      } else {
        // Build list of active points (alpha > 0.05 covers both modes)
        // Skip categories hidden via legend (renderer.visible === false)
        var active = [];
        for (var s = 0; s < sources.length; s++) {
          if (renderers && renderers[s] && !renderers[s].visible) continue;
          var data = sources[s].data;
          var alphas = data['alpha'];
          var xs = data['x'];
          var ys = data['y'];
          var ranks = data['fps_rank'];
          if (!alphas || !xs || !ys) continue;
          for (var i = 0; i < alphas.length; i++) {
            if (alphas[i] > 0.05) {
              var rankVal = (ranks && ranks.length > i) ? ranks[i] : 1000000000;
              active.push({s: s, i: i, x: xs[i], y: ys[i], rank: rankVal});
            }
          }
        }
        if (active.length === 0) return;
        if (action === 'status' || action === 'panel') {
          console.log('[hodo eye] active points', {
            action: action,
            mode: mode,
            activeCount: active.length,
            minRank: active[0].rank,
          });
        }
        active.sort(function(a, b) {
          if (a.rank !== b.rank) return a.rank - b.rank;
          if (a.s !== b.s) return a.s - b.s;
          return a.i - b.i;
        });
        var curIdx = -1;
        if (cur) {
          for (var k = 0; k < active.length; k++) {
            if (active[k].s === cur.s && active[k].i === cur.i) { curIdx = k; break; }
          }
        }
        if (action === 'panel') {
          // Detail-panel eye: nearest visible point when current selection exists,
          // otherwise random visible point.
          var curX = null;
          var curY = null;
          if (cur && sources[cur.s] && sources[cur.s].data) {
            var curData = sources[cur.s].data;
            if (curData['x'] && curData['y'] && cur.i < curData['x'].length) {
              curX = curData['x'][cur.i];
              curY = curData['y'][cur.i];
            }
          }
          if (curX !== null && curY !== null) {
            var best = null;
            var bestDist = Infinity;
            for (var n = 0; n < active.length; n++) {
              var cand = active[n];
              if (cand.s === cur.s && cand.i === cur.i) continue;
              var dx = cand.x - curX;
              var dy = cand.y - curY;
              var dist = dx * dx + dy * dy;
              if (dist < bestDist) {
                bestDist = dist;
                best = cand;
              }
            }
            target = best || active[Math.floor(Math.random() * active.length)];
          } else {
            target = active[Math.floor(Math.random() * active.length)];
          }
        } else if (action === 'status') {
          if (mode === 0) {
            // Rank/suggest mode status-eye: random among global lowest-rank
            // visible points across all visible classes.
            var minRank = sentinel;
            for (var r = 0; r < active.length; r++) {
              if (active[r].rank < minRank) minRank = active[r].rank;
            }
            var top = [];
            for (var t = 0; t < active.length; t++) {
              if (active[t].rank === minRank) top.push(active[t]);
            }
            console.log('[hodo eye] status-rank candidates', {
              minRank: minRank,
              candidateCount: top.length,
              candidates: top,
            });
            if (top.length === 0) {
              target = active[0];
            } else if (top.length > 1 && cur) {
              var topOthers = [];
              for (var u = 0; u < top.length; u++) {
                if (!(top[u].s === cur.s && top[u].i === cur.i)) topOthers.push(top[u]);
              }
              var pickFrom = topOthers.length > 0 ? topOthers : top;
              target = pickFrom[Math.floor(Math.random() * pickFrom.length)];
            } else {
              target = top[Math.floor(Math.random() * top.length)];
            }
          } else {
            // Status-eye in search mode: random visible point (not nearest-neighbor).
            if (active.length > 1 && curIdx !== -1) {
              var activeOthers = [];
              for (var a = 0; a < active.length; a++) {
                if (!(active[a].s === cur.s && active[a].i === cur.i)) activeOthers.push(active[a]);
              }
              var randomPool = activeOthers.length > 0 ? activeOthers : active;
              target = randomPool[Math.floor(Math.random() * randomPool.length)];
            } else {
              target = active[Math.floor(Math.random() * active.length)];
            }
          }
        } else {
          // Prev/Next: keep class first, switch class only at class boundaries.
          var classBuckets = {};
          var classOrder = [];
          for (var b = 0; b < active.length; b++) {
            var cls = active[b].s;
            if (!classBuckets[cls]) {
              classBuckets[cls] = [];
              classOrder.push(cls);
            }
            classBuckets[cls].push(active[b]);
          }
          classOrder.sort(function(a, b2) { return a - b2; });
          for (var c = 0; c < classOrder.length; c++) {
            var clsKey = classOrder[c];
            classBuckets[clsKey].sort(function(p1, p2) {
              if (p1.rank !== p2.rank) return p1.rank - p2.rank;
              return p1.i - p2.i;
            });
          }
          var currentClass = (cur && classBuckets[cur.s]) ? cur.s : classOrder[0];
          var classPoints = classBuckets[currentClass];
          var classPos = classOrder.indexOf(currentClass);
          var classIdx = -1;
          if (cur) {
            for (var cp = 0; cp < classPoints.length; cp++) {
              if (classPoints[cp].s === cur.s && classPoints[cp].i === cur.i) {
                classIdx = cp;
                break;
              }
            }
          }

          if (action === 'next') {
            if (classIdx === -1) {
              target = classPoints[0];
            } else if (classIdx < classPoints.length - 1) {
              target = classPoints[classIdx + 1];
            } else {
              var nextClass = classOrder[(classPos + 1) % classOrder.length];
              target = classBuckets[nextClass][0];
            }
          } else {
            if (classIdx === -1) {
              target = classPoints[classPoints.length - 1];
            } else if (classIdx > 0) {
              target = classPoints[classIdx - 1];
            } else {
              var prevClass = classOrder[(classPos - 1 + classOrder.length) % classOrder.length];
              var prevPoints = classBuckets[prevClass];
              target = prevPoints[prevPoints.length - 1];
            }
          }
        }
      }
      if (!target) return;
      // Clear all selections, then set the target
      for (var s2 = 0; s2 < sources.length; s2++) {
        sources[s2].selected.indices = [];
      }
      sources[target.s].selected.indices = [target.i];
      sources[target.s].selected.change.emit();
      if (action === 'status' || action === 'panel') {
        console.log('[hodo eye] target', {
          action: action,
          target: target,
        });
      }
      // Defensive: ensure pane is visible even if callback dispatch is delayed.
      var panel = document.getElementById('detail-panel');
      var handle = document.getElementById('drag-handle');
      if (panel) panel.style.display = 'block';
      if (handle) handle.style.display = 'block';
      setTimeout(function() {
        if (sources[target.s]) sources[target.s].selected.change.emit();
      }, 0);
    };
    window._hodoCopy = function(btn) {
      var header = btn.closest('.section-header') || btn.closest('.context-toggle');
      var pre = header ? header.nextElementSibling : null;
      if (!pre) return;
      navigator.clipboard.writeText(pre.textContent).then(function() {
        btn.classList.add('copied');
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>Copied';
        setTimeout(function() {
          btn.classList.remove('copied');
          btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy';
        }, 1500);
      });
    };
  })();
</script>
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@dataclass
class MethodProjection:
    """Pre-computed projection data for a single dim-reduction method."""
    name: str                             # e.g., "tsne"
    display_name: str                     # e.g., "t-SNE"
    X_2d: np.ndarray                      # (N, 2)
    x_range: tuple[float, float]          # padded bounds
    y_range: tuple[float, float]
    density_grids: list[list[float]]      # [n_categories][grid_size^2]
    density_gaps: list[float]             # (N,) own_density - mean(other_density)
    fps_ranks: list[int]                  # (N,) per-point FPS rank
    grid_size: int


def _make_scatter_sources(p, X_2d, data, default_alpha=0.6,
                          extra_columns=None, keep_none=False, **scatter_kw):
    """Create per-category scatter sources and renderers, add HoverTool.

    Args:
        p: Bokeh figure.
        X_2d: (N, 2) array of 2D coordinates.
        data: PlotData instance.
        default_alpha: Alpha for scatter points.
        extra_columns: Dict of {col_name: list} with len==N, partitioned per category.
        keep_none: If True, append None for empty categories (positional indexing).
        **scatter_kw: Extra kwargs for p.scatter (e.g. line_color, line_width).

    Returns:
        (all_sources, renderers, source_masks) — source_masks[i] is list of global indices
        for source i (None for empty categories when keep_none=True).
    """
    label_colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(data.type_names))]
    label_names = list(data.type_names)
    extra_columns = extra_columns or {}

    all_sources = []
    renderers = []
    source_masks = []
    for label_idx in range(len(data.type_names)):
        name = label_names[label_idx]
        color = label_colors[label_idx]
        mask = [i for i, l in enumerate(data.labels) if l == label_idx]
        if not mask:
            if keep_none:
                all_sources.append(None)
                renderers.append(None)
                source_masks.append(None)
            continue

        source_data = dict(
            x=[X_2d[i, 0] for i in mask],
            y=[X_2d[i, 1] for i in mask],
            alpha=[default_alpha] * len(mask),
            type_label=[name] * len(mask),
            trajectory_id=[data.trajectory_ids[i] for i in mask],
            trajectory_uuid=[data.trajectory_uuids[i] for i in mask],
            turn_id=[str(data.turn_ids[i]) for i in mask],
            summary=[data.summaries[i] for i in mask],
            _global_idx=list(mask),
        )
        for col_name, col_values in extra_columns.items():
            source_data[col_name] = [col_values[i] for i in mask]

        source = ColumnDataSource(data=source_data, name=f'hodo_source_{label_idx}')
        all_sources.append(source)
        source_masks.append(mask)

        scatter = p.scatter(
            'x', 'y', source=source,
            color=color, size=6, alpha=default_alpha,
            legend_label=name,
            selection_color=color, nonselection_alpha=0.15,
            name=f'hodo_renderer_{label_idx}',
            **scatter_kw,
        )
        renderers.append(scatter)

    valid_renderers = [r for r in renderers if r is not None]
    hover = HoverTool(tooltips=[
        ('type', '@type_label'),
        ('trajectory', '@trajectory_id'),
        ('turn', '@turn_id'),
        ('summary', '@summary'),
    ], renderers=valid_renderers)
    p.add_tools(hover)

    return all_sources, renderers, source_masks


def _add_tap_callback(sources, highlight_source=None, label_colors=None, renderers=None):
    """Wire up tap-to-detail-panel JS callback on all sources."""
    args = dict(sources=sources)
    if highlight_source:
        args['highlight_source'] = highlight_source
    if label_colors:
        args['label_colors'] = label_colors
    if renderers:
        args['renderers'] = renderers

    eye_svg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>'
    copy_svg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'

    tap_cb = CustomJS(args=args, code=f"""
        // Store sources/renderers globally for _hodoNav
        window._hodoSources = sources;
        if (typeof renderers !== 'undefined') window._hodoRenderers = renderers;
        if (typeof highlight_source !== 'undefined') window._hodoHighlight = highlight_source;

        let foundSelection = false;
        for (let s = 0; s < sources.length; s++) {{
            const source = sources[s];
            const idx = source.selected.indices;
            if (idx.length === 0) continue;
            foundSelection = true;
            const i = idx[0];
            const d = source.data;
            const esc = (v) => String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

            // Update highlight triangle
            if (typeof highlight_source !== 'undefined' && typeof label_colors !== 'undefined') {{
                highlight_source.data = {{
                    x: [d['x'][i]],
                    y: [d['y'][i]],
                    color: [label_colors[s % label_colors.length]],
                }};
                highlight_source.change.emit();
            }}

            // Track current point for navigation
            window._hodoCurrentPoint = {{s: s, i: i}};

            // Build nav bar: check if point is in active set (alpha + renderer visible)
            const alphas = d['alpha'];
            const fpsRanks = d['fps_rank'];
            const densityGaps = d['density_gap'];
            const sentinel = 1000000000;
            const rankVal = (fpsRanks && fpsRanks.length > i) ? fpsRanks[i] : null;
            const gapVal = (densityGaps && densityGaps.length > i) ? densityGaps[i] : null;
            let maxRankDisplay = 1;
            if (fpsRanks && fpsRanks.length > 0) {{
                let maxFinite = -1;
                for (let ri = 0; ri < fpsRanks.length; ri++) {{
                    const rv = fpsRanks[ri];
                    if (Number.isFinite(rv) && rv < sentinel && rv > maxFinite) maxFinite = rv;
                }}
                if (maxFinite >= 0) maxRankDisplay = maxFinite + 1;
            }}
            const rankText = (Number.isFinite(rankVal) && rankVal < sentinel)
                ? ('#' + (rankVal + 1))
                : ('#' + maxRankDisplay + '+');
            const rankBadge = (Number.isFinite(rankVal) && rankVal < sentinel)
                ? ('<span class="nav-status">(' + rankText + ')</span>')
                : ('<span class="nav-status">' + rankText + '</span>');
            const gapText = Number.isFinite(gapVal)
                ? ('(' + (gapVal >= 0 ? '+' : '-') + ') ' + Math.abs(gapVal).toExponential(3))
                : '(?) N/A';
            const rendererVisible = (typeof renderers === 'undefined') || !renderers[s] || renderers[s].visible;
            const isActive = rendererVisible && (alphas ? alphas[i] > 0.05 : true);
            let navHtml;
            if (isActive) {{
                navHtml = '<div class="nav-bar">' +
                    rankBadge +
                    '<button class="nav-btn" onclick="_hodoNav(\\x27prev\\x27)">&lsaquo; Prev</button>' +
                    '<button class="nav-btn" onclick="_hodoNav(\\x27next\\x27)">Next &rsaquo;</button>' +
                    '</div>';
            }} else {{
                navHtml = '<div class="nav-bar">' +
                    rankBadge +
                    '<span class="nav-status">Not in current set</span>' +
                    '<button class="nav-btn" onclick="_hodoNav(\\x27panel\\x27)" title="Jump to nearest visible point">{eye_svg}</button>' +
                    '</div>';
            }}

            const panel = document.getElementById('detail-panel');
            const content = document.getElementById('detail-content');
            content.innerHTML = navHtml +
                '<div class="meta-section">' +
                '<div class="meta-row"><span class="meta-label">Type</span><span class="meta-value">' + esc(d['type_label'][i]) + '</span></div>' +
                '<div class="meta-row"><span class="meta-label">Trajectory</span><span class="meta-value">' + esc(d['trajectory_id'][i]) + '</span></div>' +
                '<div class="meta-row"><span class="meta-label">Turn</span><span class="meta-value">' + esc(d['turn_id'][i]) +
                '<button class="turn-nav-btn" onclick="_hodoNav(\\x27turn_prev\\x27)">&lsaquo; Prev turn</button>' +
                '<button class="turn-nav-btn" onclick="_hodoNav(\\x27turn_next\\x27)">Next turn &rsaquo;</button>' +
                '</span></div>' +
                '<div class="meta-row"><span class="meta-label">FPS Order</span><span class="meta-value">' + esc(rankText) + '</span></div>' +
                '<div class="meta-row"><span class="meta-label">Density Gap</span><span class="meta-value">' + esc(gapText) + '</span></div>' +
                '</div>' +
                '<div class="section-header"><span class="section-title">Summary</span><button class="copy-btn" onclick="_hodoCopy(this)">{copy_svg}Copy</button></div>' +
                '<pre>' + esc(d['summary'][i]) + '</pre>' +
                '<div class="section-header"><span class="section-title">Original Action</span><button class="copy-btn" onclick="_hodoCopy(this)">{copy_svg}Copy</button></div>' +
                '<pre>' + esc(window._hodoActionTexts[d['_global_idx'][i]]) + '</pre>' +
                (function() {{
                    var tc = (window._hodoTaskContexts && d['trajectory_uuid']) ? (window._hodoTaskContexts[d['trajectory_uuid'][i]] || '') : '';
                    if (!tc) return '';
                    var collapsed = !!window._hodoContextCollapsed;
                    var arrowCls = collapsed ? 'toggle-arrow' : 'toggle-arrow open';
                    var preStyle = collapsed ? 'display:none' : '';
                    return '<div class="context-toggle" onclick="(function(el){{var pre=el.nextElementSibling;var arrow=el.querySelector(\\x27.toggle-arrow\\x27);if(pre.style.display===\\x27none\\x27){{pre.style.display=\\x27\\x27;arrow.classList.add(\\x27open\\x27);window._hodoContextCollapsed=false;}}else{{pre.style.display=\\x27none\\x27;arrow.classList.remove(\\x27open\\x27);window._hodoContextCollapsed=true;}}}})(this)">' +
                        '<span class="toggle-left"><span class="' + arrowCls + '">&#9654;</span><span class="section-title">Task Context</span></span>' +
                        '<button class="copy-btn" onclick="event.stopPropagation();_hodoCopy(this)">{copy_svg}Copy</button>' +
                        '</div>' +
                        '<pre style="' + preStyle + '">' + esc(tc) + '</pre>';
                }})();
            panel.style.display = 'block';
            document.getElementById('drag-handle').style.display = 'block';
            break;
        }}
        if (!foundSelection) {{
            const panel = document.getElementById('detail-panel');
            const handle = document.getElementById('drag-handle');
            if (panel) panel.style.display = 'none';
            if (handle) handle.style.display = 'none';
            if (typeof highlight_source !== 'undefined') {{
                highlight_source.data = {{x: [], y: [], color: []}};
                highlight_source.change.emit();
            }}
            window._hodoCurrentPoint = null;
        }}
    """)
    for source in sources:
        source.selected.js_on_change('indices', tap_cb)


def _make_search_widgets():
    """Create search widgets (no callbacks — wired by caller)."""
    search_input = TextInput(title='Search...', placeholder='Enter to search', width=180)
    search_mode = Select(
        title='Within:',
        value='both',
        options=[('both', 'All text'), ('summary', 'Summaries'), ('action', 'Actions')],
        width=120,
    )
    match_div = Div(
        text='',
        min_width=180,
        sizing_mode='stretch_width',
        styles={
            'font-family': 'monospace',
            'font-size': '13px',
            'color': '#666',
            'text-align': 'right',
            'white-space': 'nowrap',
        },
    )
    return search_input, search_mode, match_div


def _compress_data(data):
    """Gzip-compress a JSON-serializable object and return a base64 string."""
    js = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    compressed = gzip.compress(js.encode('utf-8'))
    return base64.b64encode(compressed).decode('ascii')


# JS snippet that decompresses gzip'd base64 data into window globals.
# Uses DecompressionStream (supported in Chrome 80+, Firefox 113+, Safari 16.4+).
_DECOMPRESS_JS = """
<script>
(async function() {
  async function _hodoDecode(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    var ds = new DecompressionStream('gzip');
    var writer = ds.writable.getWriter();
    writer.write(bytes);
    writer.close();
    var reader = ds.readable.getReader();
    var chunks = [];
    while (true) {
      var result = await reader.read();
      if (result.done) break;
      chunks.push(result.value);
    }
    var totalLen = 0;
    for (var c of chunks) totalLen += c.length;
    var merged = new Uint8Array(totalLen);
    var offset = 0;
    for (var c of chunks) { merged.set(c, offset); offset += c.length; }
    return JSON.parse(new TextDecoder().decode(merged));
  }
  %s
})();
</script>
""".strip()


def _save_with_sidebar(layout, html_path, title, external_data=None):
    """Save Bokeh layout to HTML and append the detail sidebar."""
    output_file(str(html_path), title=title)
    save(layout, resources=CDN)
    with open(html_path, 'a') as f:
        f.write(SIDEBAR_HTML)
        if external_data:
            assignments = []
            for var_name, data in external_data.items():
                b64 = _compress_data(data)
                assignments.append(
                    f'window.{var_name}=await _hodoDecode("{b64}");'
                )
            f.write('\n' + _DECOMPRESS_JS % '\n  '.join(assignments))


# ---------------------------------------------------------------------------
# Projection computation (delegates to sampling module)
# ---------------------------------------------------------------------------

def _compute_method_projection(
    data: PlotData,
    method: str,
    grid_size: int = 80,
    bandwidth: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> MethodProjection:
    """Compute projection + KDE grids + FPS ranks for one method."""
    X_2d = compute_projection(data.X, method, labels=data.labels)
    labels = data.labels
    type_names = data.type_names
    n_categories = len(type_names)

    # Padded bounds
    x_min, x_max = float(X_2d[:, 0].min()), float(X_2d[:, 0].max())
    y_min, y_max = float(X_2d[:, 1].min()), float(X_2d[:, 1].max())
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad

    # KDE grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if bandwidth is None:
        bandwidth = compute_bandwidth(X_2d)

    # Compute both grid densities (for heatmap) and point densities (for FPS)
    # in a single KDE fit per category to avoid double-fitting.
    density_grids = []
    point_densities = []
    for label_idx in range(n_categories):
        mask = labels == label_idx
        X_cat = X_2d[mask]
        if len(X_cat) > 0:
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(X_cat)
            density = np.exp(kde.score_samples(grid_points))
            density_points = np.exp(kde.score_samples(X_2d))
        else:
            density = np.zeros(len(grid_points))
            density_points = np.zeros(len(X_2d))
        density_grids.append(density.tolist())
        point_densities.append(density_points)

    # Per-point density gap: own density minus mean of other categories.
    density_gaps = np.zeros(len(X_2d), dtype=float)
    n_others = max(n_categories - 1, 1)
    for i in range(len(X_2d)):
        own_label = int(labels[i])
        own = point_densities[own_label][i]
        other_sum = sum(point_densities[j][i] for j in range(n_categories) if j != own_label)
        density_gaps[i] = own - (other_sum / n_others if n_categories > 1 else 0.0)

    # FPS ranking — pass precomputed point densities to avoid recomputing KDE
    fps_ranks = compute_fps_ranks(
        X_2d, labels, n_categories,
        point_densities=point_densities,
        alpha=alpha,
        beta=beta,
    )

    display_name = SAMPLING_METHOD_DISPLAY_NAMES.get(method, method.upper())

    return MethodProjection(
        name=method,
        display_name=display_name,
        X_2d=X_2d,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        density_grids=density_grids,
        density_gaps=density_gaps.tolist(),
        fps_ranks=fps_ranks,
        grid_size=grid_size,
    )


# ---------------------------------------------------------------------------
# Unified Bokeh builder
# ---------------------------------------------------------------------------

def _build_unified_bokeh(method_projections: list[MethodProjection], data: PlotData, html_path: Path):
    """Build a single interactive Bokeh plot with method switcher dropdown."""
    initial = method_projections[0]
    label_names = list(data.type_names)

    # Build methods_data JS dict: {method_name: {x: [...], y: [...], fps_ranks: [...],
    #   density_grids: [...], x_range: [lo, hi], y_range: [lo, hi], grid_size: int, display_name: str}}
    methods_data = {}
    for mp in method_projections:
        methods_data[mp.name] = {
            'x': mp.X_2d[:, 0].tolist(),
            'y': mp.X_2d[:, 1].tolist(),
            'density_gaps': mp.density_gaps,
            'fps_ranks': mp.fps_ranks,
            'density_grids': mp.density_grids,
            'x_range': list(mp.x_range),
            'y_range': list(mp.y_range),
            'grid_size': mp.grid_size,
            'display_name': mp.display_name,
        }

    # Initial density: compute for first category (default selection)
    n_categories = len(data.type_names)
    _sel = np.array(initial.density_grids[0])
    _sum_other = np.zeros_like(_sel)
    for _j in range(n_categories):
        if _j != 0:
            _sum_other += np.array(initial.density_grids[_j])
    _avg_other = _sum_other / max(n_categories - 1, 1)
    _diff = _sel - _avg_other
    initial_vmax = max(float(np.abs(_diff).max()), 1e-10)
    initial_density = _diff.reshape(initial.grid_size, initial.grid_size)

    # Create figure
    title = f'Action Summaries \u2014 {initial.display_name}'
    p = figure(
        title=title,
        width=1100, height=750,
        tools='pan,wheel_zoom,box_zoom,tap,reset,save',
        active_scroll='wheel_zoom',
        toolbar_location='above',
        x_range=initial.x_range,
        y_range=initial.y_range,
    )
    p.title.text_font_size = '15px'
    p.title.text_font_style = 'normal'
    p.background_fill_color = '#fafafa'
    p.border_fill_color = '#ffffff'
    p.outline_line_color = '#e0e0e0'
    p.grid.grid_line_color = '#eeeeee'
    p.grid.grid_line_alpha = 0.6
    p.xaxis.axis_label = ''
    p.yaxis.axis_label = ''

    # Heatmap image source
    heatmap_source = ColumnDataSource(data=dict(
        image=[initial_density],
        x=[initial.x_range[0]],
        y=[initial.y_range[0]],
        dw=[initial.x_range[1] - initial.x_range[0]],
        dh=[initial.y_range[1] - initial.y_range[0]],
    ))

    palette = list(RdBu11)
    color_mapper = LinearColorMapper(palette=palette, low=-initial_vmax, high=initial_vmax)

    p.image(
        image='image',
        source=heatmap_source,
        x='x', y='y', dw='dw', dh='dh',
        color_mapper=color_mapper,
        level='image',
        global_alpha=0.4,
    )

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title='\u2190 under \u00b7 over represented \u2192',
    )
    p.add_layout(color_bar, 'right')

    # Scatter points with fps_rank + density_gap
    fps_rank_full = initial.fps_ranks
    density_gap_full = initial.density_gaps
    all_sources, renderers, source_masks = _make_scatter_sources(
        p, initial.X_2d, data, default_alpha=0.7,
        extra_columns={'fps_rank': fps_rank_full, 'density_gap': density_gap_full},
        keep_none=True,
        line_color='white', line_width=0.5,
    )

    valid_sources = [s for s in all_sources if s is not None]
    valid_renderers = [r for r in renderers if r is not None]
    valid_masks = [m for m in source_masks if m is not None]

    # Highlight layer — triangle marker for currently selected point
    highlight_source = ColumnDataSource(data=dict(x=[], y=[], color=[]))
    p.scatter(
        'x', 'y', source=highlight_source,
        marker='inverted_triangle',
        fill_color='color', line_color='black', line_width=2,
        size=14, level='overlay',
    )

    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    p.legend.label_text_font_size = '11px'
    p.legend.background_fill_alpha = 0.85
    p.legend.border_line_alpha = 0.3
    p.legend.padding = 6
    p.legend.spacing = 2

    # --- Method switcher dropdown ---
    method_options = [(mp.name, mp.display_name) for mp in method_projections]
    compact_select = InlineStyleSheet(css=":host { min-width: 0 !important; }")
    method_selector = Select(
        title="Projection:",
        value=initial.name,
        options=method_options,
        width=90,
        stylesheets=[compact_select],
    )

    # current_method source for cross-callback communication
    current_method_source = ColumnDataSource(data=dict(method=[initial.name]))

    # Density selector
    density_selector = Select(
        title="Density overlay:",
        value="0",
        options=[(str(i), label_names[i]) for i in range(len(data.type_names))] + [("none", "None")],
        width=170,
        stylesheets=[compact_select],
    )

    # Density callback
    density_cb = CustomJS(
        args=dict(
            selector=density_selector,
            heatmap_source=heatmap_source,
            color_mapper=color_mapper,
            current_method_source=current_method_source,
            methods_data=methods_data,
            n_categories=len(data.type_names),
        ),
        code="""
        const selectedIdx = selector.value;
        const methodName = current_method_source.data['method'][0];
        const mdata = methods_data[methodName];
        const gridSize = mdata.grid_size;
        const totalSize = gridSize * gridSize;
        const density_grids = mdata.density_grids;
        const nCat = n_categories;

        const existingImage = heatmap_source.data['image'][0];
        let vmax = 1;

        if (selectedIdx === 'none') {
            for (let i = 0; i < totalSize; i++) {
                existingImage[i] = 0;
            }
        } else {
            const catIdx = parseInt(selectedIdx);
            const selectedDensity = density_grids[catIdx];
            vmax = 0;
            const numOthers = nCat - 1;
            for (let i = 0; i < totalSize; i++) {
                let sumOther = 0;
                for (let j = 0; j < nCat; j++) {
                    if (j !== catIdx) {
                        sumOther += density_grids[j][i];
                    }
                }
                const avgOther = numOthers > 0 ? sumOther / numOthers : 0;
                const diff = selectedDensity[i] - avgOther;
                existingImage[i] = diff;
                vmax = Math.max(vmax, Math.abs(diff));
            }
            vmax = Math.max(vmax, 1e-10);
        }

        color_mapper.low = -vmax;
        color_mapper.high = vmax;
        heatmap_source.change.emit();
        """,
    )
    density_selector.js_on_change('value', density_cb)

    # Mode controls — □/■ indicators in widget titles, mode tracked via data source
    # mode 0 = top picks, mode 1 = search
    mode_source = ColumnDataSource(data=dict(active=[1]))  # start in search mode
    fps_slider = Slider(start=1, end=100, value=50, step=1,
                        title='\u25a1 Suggest samples per group', width=250)
    search_input, search_mode_sel, match_div = _make_search_widgets()
    search_input.title = '\u25a0 Search...'
    match_div.text = (
        f'{len(data.labels)} total'
        ' <span style="cursor:pointer;vertical-align:-2px;opacity:0.6" '
        'title="Jump to random point" onclick="_hodoNav(\'status\')">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2">'
        '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
        '<circle cx="12" cy="12" r="3"/></svg></span>'
    )

    # Unified filter callback — auto-switches mode based on trigger (cb_obj)
    filter_cb = CustomJS(
        args=dict(
            mode_src=mode_source,
            slider=fps_slider,
            search=search_input,
            search_mode=search_mode_sel,
            match_div=match_div,
            sources=valid_sources,
            renderers=valid_renderers,
            default_alpha=0.7,
        ),
        code="""
        const eyeBtn = ' <span style="cursor:pointer;vertical-align:-2px;opacity:0.6" title="Jump to random point" onclick="_hodoNav(\\x27status\\x27)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></span>';
        // Keep nav globals in sync here; top-eye is rendered by this callback.
        window._hodoSources = sources;
        window._hodoRenderers = renderers;
        const mode = mode_src.data['active'];
        const prevMode = mode[0];
        const fromSlider = cb_obj === slider || (cb_obj && cb_obj.model === slider);
        const fromSearch = cb_obj === search || (cb_obj && cb_obj.model === search);
        const fromSearchMode = cb_obj === search_mode || (cb_obj && cb_obj.model === search_mode);

        // Auto-switch mode based on what the user interacted with
        if (fromSlider && mode[0] !== 0) mode[0] = 0;
        if ((fromSearch || fromSearchMode) && mode[0] !== 1) mode[0] = 1;

        // Update □/■ indicators only on mode transition (avoids re-render focus loss)
        if (mode[0] !== prevMode) {
            if (mode[0] === 0) {
                slider.title = "\\u25a0 Suggest samples per group";
                search.title = "\\u25a1 Search...";
            } else {
                slider.title = "\\u25a1 Suggest samples per group";
                search.title = "\\u25a0 Search...";
            }
        }
        window._hodoMode = mode[0];

        if (mode[0] === 1) {
            // Search mode
            const queryRaw = (search.value || '').trim();
            const modeVal = search_mode.value;
            let query = queryRaw.toLowerCase();
            let useRegex = false;
            let regex = null;
            let regexError = null;
            if (queryRaw.startsWith('re:')) {
                useRegex = true;
                const pattern = queryRaw.slice(3);
                try {
                    regex = new RegExp(pattern, 'i');
                } catch (e) {
                    regexError = e;
                }
            } else if (queryRaw.length >= 2 && queryRaw[0] === '/') {
                const lastSlash = queryRaw.lastIndexOf('/');
                if (lastSlash > 0) {
                    useRegex = true;
                    const pattern = queryRaw.slice(1, lastSlash);
                    let flags = queryRaw.slice(lastSlash + 1);
                    if (flags.indexOf('g') !== -1) flags = flags.replace(/g/g, '');
                    if (flags.indexOf('i') === -1) flags += 'i';
                    try {
                        regex = new RegExp(pattern, flags);
                    } catch (e) {
                        regexError = e;
                    }
                }
            }
            let count = 0;
            for (let s = 0; s < sources.length; s++) {
                const source = sources[s];
                const rendererVisible = !renderers[s] || renderers[s].visible;
                const summaries = source.data['summary'];
                const globalIdxs = source.data['_global_idx'];
                const actionTexts = window._hodoActionTexts || [];
                const newAlpha = [];
                for (let i = 0; i < summaries.length; i++) {
                    let match;
                    const actionText = actionTexts[globalIdxs[i]] || '';
                    if (!queryRaw) {
                        match = true;
                    } else if (regexError) {
                        match = false;
                    } else if (useRegex && regex) {
                        const summaryText = summaries[i] || '';
                        if (modeVal === 'summary') {
                            match = regex.test(summaryText);
                        } else if (modeVal === 'action') {
                            match = regex.test(actionText);
                        } else {
                            match = regex.test(summaryText) || regex.test(actionText);
                        }
                    } else if (modeVal === 'summary') {
                        match = summaries[i].toLowerCase().indexOf(query) !== -1;
                    } else if (modeVal === 'action') {
                        match = actionText.toLowerCase().indexOf(query) !== -1;
                    } else {
                        match = summaries[i].toLowerCase().indexOf(query) !== -1 || actionText.toLowerCase().indexOf(query) !== -1;
                    }
                    newAlpha.push(match ? default_alpha : 0.02);
                    if (match && rendererVisible) count++;
                }
                source.data['alpha'] = newAlpha;
                source.change.emit();
                renderers[s].glyph.fill_alpha = {field: 'alpha'};
                renderers[s].glyph.line_alpha = {field: 'alpha'};
            }
            if (regexError && queryRaw) {
                match_div.text = 'Invalid regex' + eyeBtn;
            } else {
                match_div.text = (queryRaw ? count + (count === 1 ? ' match' : ' matches') : count + ' total') + eyeBtn;
            }
        } else {
            // Top samples mode
            const threshold = slider.value;
            let shown = 0;
            for (let s = 0; s < sources.length; s++) {
                const source = sources[s];
                const rendererVisible = !renderers[s] || renderers[s].visible;
                const ranks = source.data['fps_rank'];
                const n = ranks.length;
                const newAlpha = [];
                for (let i = 0; i < n; i++) {
                    if (ranks[i] < threshold) {
                        newAlpha.push(0.9);
                        if (rendererVisible) shown++;
                    } else {
                        newAlpha.push(0);
                    }
                }
                source.data['alpha'] = newAlpha;
                source.change.emit();
                renderers[s].glyph.fill_alpha = {field: 'alpha'};
                renderers[s].glyph.line_alpha = {field: 'alpha'};
            }
            match_div.text = shown + ' shown' + eyeBtn;
        }
        """,
    )
    fps_slider.js_on_change('value', filter_cb)
    search_input.js_on_change('value', filter_cb)
    search_mode_sel.js_on_change('value', filter_cb)
    submit_cb = CustomJS(
        args=dict(mode_src=mode_source, search=search_input, slider=fps_slider),
        code="""
        // Enter in search box should always switch to search mode and re-run filter,
        // even if query text did not change.
        mode_src.data['active'][0] = 1;
        mode_src.change.emit();
        window._hodoMode = 1;
        slider.title = "\\u25a1 Suggest samples per group";
        search.title = "\\u25a0 Search...";

        const q = search.value || '';
        search.value = q + ' ';
        search.value = q;
        """,
    )
    search_input.js_on_event(ValueSubmit, submit_cb)

    legend_count_cb = CustomJS(
        args=dict(
            mode_src=mode_source,
            search=search_input,
            match_div=match_div,
            sources=valid_sources,
            renderers=valid_renderers,
        ),
        code="""
        const eyeBtn = ' <span style="cursor:pointer;vertical-align:-2px;opacity:0.6" title="Jump to random point" onclick="_hodoNav(\\x27status\\x27)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></span>';
        window._hodoSources = sources;
        window._hodoRenderers = renderers;
        const mode = mode_src.data['active'][0];
        if (mode === 1) {
            const query = search.value.toLowerCase().trim();
            let count = 0;
            for (let s = 0; s < sources.length; s++) {
                if (renderers[s] && !renderers[s].visible) continue;
                const alpha = sources[s].data['alpha'] || [];
                for (let i = 0; i < alpha.length; i++) {
                    if (alpha[i] > 0.05) count++;
                }
            }
            match_div.text = (query ? count + (count === 1 ? ' match' : ' matches') : count + ' total') + eyeBtn;
        } else {
            let shown = 0;
            for (let s = 0; s < sources.length; s++) {
                if (renderers[s] && !renderers[s].visible) continue;
                const alpha = sources[s].data['alpha'] || [];
                for (let i = 0; i < alpha.length; i++) {
                    if (alpha[i] > 0.05) shown++;
                }
            }
            match_div.text = shown + ' shown' + eyeBtn;
        }
        """,
    )
    for renderer in valid_renderers:
        renderer.js_on_change('visible', legend_count_cb)

    # Tap callback
    label_colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(data.type_names))]
    _add_tap_callback(valid_sources, highlight_source, label_colors, valid_renderers)

    init_nav_cb = CustomJS(
        args=dict(
            sources=valid_sources,
            renderers=valid_renderers,
            highlight_source=highlight_source,
            mode_src=mode_source,
            density_selector=density_selector,
        ),
        code="""
        window._hodoSources = sources;
        window._hodoRenderers = renderers;
        window._hodoHighlight = highlight_source;
        window._hodoModeSource = mode_src;
        window._hodoDensitySelector = density_selector;
        if (mode_src && mode_src.data && mode_src.data['active'] && mode_src.data['active'].length > 0) {
            window._hodoMode = mode_src.data['active'][0];
        }
        """,
    )
    p.js_on_event(DocumentReady, init_nav_cb)
    p.js_on_event(DocumentReady, filter_cb)

    # --- Method switcher callback ---
    method_switch_cb = CustomJS(
        args=dict(
            method_selector=method_selector,
            methods_data=methods_data,
            sources=valid_sources,
            renderers=valid_renderers,
            source_masks=valid_masks,
            heatmap_source=heatmap_source,
            color_mapper=color_mapper,
            fig=p,
            density_selector=density_selector,
            current_method_source=current_method_source,
            mode_src=mode_source,
            fps_slider=fps_slider,
            search_input=search_input,
            n_categories=len(data.type_names),
            highlight_source=highlight_source,
        ),
        code="""
        // Clear highlight on method switch
        highlight_source.data = {x: [], y: [], color: []};
        highlight_source.change.emit();
        window._hodoCurrentPoint = null;

        const methodName = method_selector.value;
        const mdata = methods_data[methodName];
        const allX = mdata.x;
        const allY = mdata.y;
        const allGaps = mdata.density_gaps;
        const allFps = mdata.fps_ranks;

        // Update current method tracker
        current_method_source.data['method'] = [methodName];
        current_method_source.change.emit();

        // Swap x/y/fps_rank in each scatter source using source_masks
        for (let s = 0; s < sources.length; s++) {
            const source = sources[s];
            const mask = source_masks[s];
            const n = mask.length;
            const newX = new Array(n);
            const newY = new Array(n);
            const newGap = new Array(n);
            const newFps = new Array(n);
            for (let i = 0; i < n; i++) {
                const gi = mask[i];
                newX[i] = allX[gi];
                newY[i] = allY[gi];
                newGap[i] = allGaps[gi];
                newFps[i] = allFps[gi];
            }
            source.data['x'] = newX;
            source.data['y'] = newY;
            source.data['density_gap'] = newGap;
            source.data['fps_rank'] = newFps;
            source.change.emit();
        }

        // Update figure ranges
        fig.x_range.start = mdata.x_range[0];
        fig.x_range.end = mdata.x_range[1];
        fig.y_range.start = mdata.y_range[0];
        fig.y_range.end = mdata.y_range[1];

        // Update heatmap bounds
        heatmap_source.data['x'] = [mdata.x_range[0]];
        heatmap_source.data['y'] = [mdata.y_range[0]];
        heatmap_source.data['dw'] = [mdata.x_range[1] - mdata.x_range[0]];
        heatmap_source.data['dh'] = [mdata.y_range[1] - mdata.y_range[0]];

        // Update figure title
        fig.title.text = 'Action Summaries \\u2014 ' + mdata.display_name;

        // Recompute density heatmap with new method's grids
        const selectedIdx = density_selector.value;
        const gridSize = mdata.grid_size;
        const totalSize = gridSize * gridSize;
        const density_grids = mdata.density_grids;
        const nCat = n_categories;
        const existingImage = heatmap_source.data['image'][0];
        let vmax = 1;

        if (selectedIdx === 'none') {
            for (let i = 0; i < totalSize; i++) {
                existingImage[i] = 0;
            }
        } else {
            const catIdx = parseInt(selectedIdx);
            const selectedDensity = density_grids[catIdx];
            vmax = 0;
            const numOthers = nCat - 1;
            for (let i = 0; i < totalSize; i++) {
                let sumOther = 0;
                for (let j = 0; j < nCat; j++) {
                    if (j !== catIdx) {
                        sumOther += density_grids[j][i];
                    }
                }
                const avgOther = numOthers > 0 ? sumOther / numOthers : 0;
                const diff = selectedDensity[i] - avgOther;
                existingImage[i] = diff;
                vmax = Math.max(vmax, Math.abs(diff));
            }
            vmax = Math.max(vmax, 1e-10);
        }
        color_mapper.low = -vmax;
        color_mapper.high = vmax;
        heatmap_source.change.emit();

        // Re-apply current mode filter
        search_input.value = '';
        const mode = mode_src.data['active'][0];
        window._hodoMode = mode;
        if (mode === 0) {
            fps_slider.title = "\\u25a0 Suggest samples per group";
            search_input.title = "\\u25a1 Search...";
            const threshold = fps_slider.value;
            for (let s = 0; s < sources.length; s++) {
                const source = sources[s];
                const ranks = source.data['fps_rank'];
                const n = ranks.length;
                const newAlpha = [];
                for (let i = 0; i < n; i++) {
                    newAlpha.push(ranks[i] < threshold ? 0.9 : 0);
                }
                source.data['alpha'] = newAlpha;
                source.change.emit();
                renderers[s].glyph.fill_alpha = {field: 'alpha'};
                renderers[s].glyph.line_alpha = {field: 'alpha'};
            }
        } else {
            fps_slider.title = "\\u25a1 Suggest samples per group";
            search_input.title = "\\u25a0 Search...";
            for (let s = 0; s < sources.length; s++) {
                const source = sources[s];
                const n = source.data['x'].length;
                source.data['alpha'] = Array(n).fill(0.7);
                source.change.emit();
                renderers[s].glyph.fill_alpha = 0.7;
                renderers[s].glyph.line_alpha = 0.7;
            }
        }
        """,
    )
    method_selector.js_on_change('value', method_switch_cb)

    slider_css = InlineStyleSheet(css="""
        :host { margin: 0; }
        :host .noUi-target { height: 12px; margin-top: 10px; }
        :host .noUi-connect { background: linear-gradient(45deg, #fff, #dfe3fbf7) !important; }
    """)
    fps_slider.stylesheets = [slider_css]
    fps_slider.align = 'center'
    match_div.align = 'end'
    controls_row = row(
        method_selector, Spacer(width=8),
        density_selector, Spacer(width=16),
        search_input, Spacer(width=8),
        search_mode_sel, Spacer(width=16),
        column(fps_slider, margin=(6, 0, 0, 0)),
        match_div,
        width=1100,
    )

    bokeh_layout = column(
        controls_row,
        Spacer(height=4),
        p,
    )

    # Build external data for HTML size optimization:
    # - task_context deduped by trajectory_uuid (all turns in a trajectory share context)
    # - action_text stored as flat array indexed by _global_idx
    task_context_map = {}
    for uuid, tc in zip(data.trajectory_uuids, data.task_contexts):
        if tc and uuid not in task_context_map:
            task_context_map[uuid] = tc
    external_data = {
        '_hodoActionTexts': list(data.action_texts),
        '_hodoTaskContexts': task_context_map,
    }

    _save_with_sidebar(bokeh_layout, html_path, 'Trajectory Explorer - Hodoscope',
                       external_data=external_data)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def visualize_action_summaries(
    summaries_by_type: dict,
    output_file: str | Path | None = None,
    methods: list = None,
    grid_size: int = 80,
    bandwidth: float = None,
    alpha: float = 1.0,
    beta: float = 0.1,
):
    """Visualize action summary embeddings with a unified interactive plot.

    Produces a single HTML file with:
    - A dropdown to switch between dim-reduction methods
    - Interactive density heatmap overlay
    - FPS-based point flagging
    - Search/filter controls
    - Click-to-inspect detail panel

    Args:
        summaries_by_type: Dict with trajectory type names as keys.
                          Order matters - last key is rendered on top.
        output_file: Path for the output HTML file. If None, generates
                     trajectory_explorer_{YYYYMMDD_HHMMSS}.html in CWD.
        methods: List of methods to use. Options: 'pca', 'tsne', 'umap', 'trimap', 'pacmap'.
                 Default: ['tsne'].
        grid_size: Resolution of density heatmap grid (default: 80).
        bandwidth: KDE bandwidth. If None, auto-computed using Scott's rule.
        alpha: FPS distance exponent (higher = more spatial spread).
        beta: FPS density gap floor (negative gaps mapped to [0, beta],
            positive gaps to [beta, 1]).
    """
    from datetime import datetime

    if methods is None:
        methods = ['tsne']
    methods = [m.lower() for m in methods]

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = Path(f"trajectory_explorer_{ts}.html")
    else:
        html_path = Path(output_file)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    data = collect_plot_data(summaries_by_type)

    # Print counts by type
    counts_str = ', '.join([f"{(data.labels==i).sum()} {data.type_names[i]}" for i in range(len(data.type_names))])
    print(f"\nAction summaries: {len(data.X)} total ({counts_str})")

    # Compute projections
    projections = []
    for method in methods:
        display = SAMPLING_METHOD_DISPLAY_NAMES.get(method, method.upper())
        print(f"Computing {display}...")
        try:
            mp = _compute_method_projection(data, method, grid_size=grid_size, bandwidth=bandwidth, alpha=alpha, beta=beta)
            projections.append(mp)
        except Exception as e:
            print(f"  WARNING: {display} failed: {e}, skipping")

    if not projections:
        print("WARNING: No projection methods succeeded, skipping visualization")
        return

    _build_unified_bokeh(projections, data, html_path)
    print(f"Saved visualization to {html_path}")
    return html_path
