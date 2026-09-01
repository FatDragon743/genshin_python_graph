const $ = (sel) => document.querySelector(sel);

const ui = {
  statusText: $("#status-text"),
  dbText: $("#db-text"),
  luckPanel: $("#panel-luck"),
  probPanel: $("#panel-prob"),
  log: $("#log"),
  btnLogin: $("#btn-login"),
  btnSync: $("#btn-sync"),
  btnRefresh: $("#btn-refresh"),
  btnLogout: $("#btn-logout"),
  modal: $("#qr-modal"),
  qrImg: $("#qr-img"),
  qrMsg: $("#qr-msg"),
  btnQrCancel: $("#btn-qr-cancel"),
  pityInput: $("#pity-input"),
  btnRecalc: $("#btn-recalc"),
  probMetrics: $("#prob-metrics"),
  histMetrics: $("#hist-metrics"),
  zoneRow: $("#zone-row"),
  charTimeline: $("#char-timeline"),
  probPoolTabs: $("#prob-pool-tabs"),
  probPoolRule: $("#prob-pool-rule"),
  probSectionTitle: $("#prob-section-title"),
  fourMetrics: $("#four-metrics"),
  fourRuleTip: $("#four-rule-tip"),
  fourSectionTitle: $("#four-section-title"),
  fourTimeline: $("#four-timeline"),
};

let qrJobId = null;
let qrTimer = null;
let logTimer = null;
let syncWatch = null;
let luckData = null;
let activePoolKey = "character";
const charts = {
  singleBase: null,
  singleSoft: null,
  cum: null,
  hist: null,
  pmf: null,
  ecdf: null,
  zones: null,
  seq: null,
  ff: null,
  fourSingle: null,
  fourCum: null,
  fourHist: null,
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.detail || data.message || res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

function setAccount(status) {
  if (!status) return;
  if (status.logged_in) {
    ui.statusText.textContent = `已登录 · UID ${status.uid}（${status.nickname || "—"}）`;
    ui.btnLogin.textContent = "重新扫码";
    ui.btnSync.disabled = false;
    ui.btnLogout.disabled = false;
  } else {
    ui.statusText.textContent = "未登录（仍可分析本地库）";
    ui.btnLogin.textContent = "米游社扫码";
    ui.btnSync.disabled = true;
    ui.btnLogout.disabled = true;
  }
  const db = status.db;
  if (db && !db.error) {
    const rows = db.rows != null ? `${db.rows} 条` : "行数待分析";
    const span =
      db.time_start && db.time_end ? ` · ${db.time_start} ~ ${db.time_end}` : "";
    ui.dbText.textContent = `本地库 ${db.file} · ${rows}${span}`;
  } else if (db && db.error) {
    ui.dbText.textContent = `本地库读取失败: ${db.error}`;
  } else {
    ui.dbText.textContent = "暂无本地库";
  }
}

async function refreshStatus() {
  const status = await api("/api/status");
  setAccount(status);
  return status;
}

async function refreshLogs() {
  try {
    const data = await api("/api/logs");
    ui.log.textContent = (data.lines || []).join("\n");
    ui.log.scrollTop = ui.log.scrollHeight;
  } catch (_) {}
}

function pct(v) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function num(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function shortName(name) {
  if (!name) return "?";
  return name.length > 4 ? name.slice(0, 4) : name;
}

function winBar(on, off) {
  const total = (on || 0) + (off || 0);
  if (!total) {
    return `<div class="win-bar empty"><span>暂无小保底样本</span></div>`;
  }
  const winPct = (on / total) * 100;
  return `<div class="win-bar" title="仅统计小保底 50/50，大保底不计入">
    <div class="win-bar-track">
      <div class="win-bar-on" style="width:${winPct}%"></div>
    </div>
    <div class="win-bar-meta">
      <span class="on">不歪 ${on}</span>
      <span class="rate">${winPct.toFixed(1)}%</span>
      <span class="off">歪 ${off}</span>
    </div>
  </div>`;
}

function renderPool(pool) {
  const lt = pool.long_term;
  const rc = pool.recent;
  const recentN = luckData?.recent_window || 10;

  const timeline = [
    {
      name: "?",
      pity: pool.current_pity,
      is_off: false,
      is_guaranteed: false,
      current: true,
    },
    ...pool.timeline,
  ];

  const hitsHtml = timeline
    .map((h) => {
      const cls = ["hit"];
      if (h.current) cls.push("current");
      if (h.is_off) cls.push("off");
      else if (h.is_guaranteed && pool.track_5050) cls.push("guaranteed");
      let stamp = "";
      if (h.is_off) stamp = `<span class="stamp">歪</span>`;
      else if (h.is_guaranteed && pool.track_5050)
        stamp = `<span class="stamp guaranteed">保</span>`;
      return `<div class="${cls.join(" ")}" title="${h.name || ""} ${h.time || ""}">
        ${stamp}
        <div class="name">${shortName(h.name)}</div>
        <div class="pity">${h.pity}</div>
      </div>`;
    })
    .join("");

  function sliceCard(stats, title, tip, tone) {
    const metrics = [
      ["出金", stats.five_count || 0],
      ["均垫", num(stats.avg_pity)],
    ];
    if (pool.track_5050) {
      metrics.push(["不歪:歪", `${stats.on_count ?? 0}:${stats.off_count ?? 0}`]);
      metrics.push(["不歪率", pct(stats.win_rate)]);
      if (title === "长期") metrics.push(["均UP", num(stats.avg_per_up, 0)]);
      if (title === "近期" && (stats.lose_streak || 0) >= 1)
        metrics.push(["连歪", stats.lose_streak]);
    }
    return `<div class="horizon-card ${tone}">
      <div class="horizon-head">
        <div class="horizon-badge" style="background:${stats.luck_color || "#9e9e9e"}">
          <span class="horizon-kicker">${title}</span>
          <strong>${stats.luck_label || "—"}</strong>
        </div>
        <p class="horizon-tip">${tip}</p>
      </div>
      <div class="horizon-metrics">
        ${metrics
          .map(([k, v]) => `<div class="hero-metric"><b>${v}</b><small>${k}</small></div>`)
          .join("")}
      </div>
      ${pool.track_5050 ? winBar(stats.on_count, stats.off_count) : ""}
    </div>`;
  }

  return `<article class="pool-stage" data-key="${pool.key}">
    <div class="pool-stage-head">
      <h2 class="pool-title">${pool.title}</h2>
      <div class="pity-chips">
        <span>当前垫刀 <strong>${pool.current_pity}</strong></span>
        ${
          pool.track_5050
            ? pool.awaiting_guaranteed
              ? `<span class="chip-guaranteed">大保底</span>`
              : `<span class="chip-soft">小保底</span>`
            : ""
        }
        <span class="chip-soft">总抽 ${lt.total_pulls || 0}</span>
      </div>
    </div>
    <div class="horizon-grid">
      ${sliceCard(lt, "长期", "全历史样本 · 更稳定的欧非底色", "long")}
      ${sliceCard(rc, "近期", `最近 ${recentN} 金 · 手感/情绪波动`, "recent")}
    </div>
    <div class="timeline-head">出金轨迹（新 → 旧）· 「保」为大保底不计入 50/50</div>
    <div class="timeline">${hitsHtml}</div>
  </article>`;
}

function renderLuck(data) {
  luckData = data;
  if (!data.pools || !data.pools.length) {
    ui.luckPanel.innerHTML = `<p class="empty-hint">暂无数据</p>`;
    return;
  }
  if (!data.pools.some((p) => p.key === activePoolKey)) {
    activePoolKey = data.pools[0].key;
  }
  const cache = data.cache || {};
  const cacheTip = cache.hit
    ? `缓存命中 · ${data.elapsed_ms ?? "—"}ms`
    : `已重算 · ${data.elapsed_ms ?? "—"}ms`;

  const tabs = data.pools
    .map((p) => {
      const lt = p.long_term || {};
      const rc = p.recent || {};
      return `<button type="button" class="pool-tab ${p.key === activePoolKey ? "active" : ""}" data-pool="${p.key}">
        <span class="pool-tab-title">${p.title}</span>
        <span class="pool-tab-dual">
          <span class="pool-tab-luck" style="background:${lt.luck_color || "#9e9e9e"}" title="长期">长 ${lt.luck_label || "—"}</span>
          <span class="pool-tab-luck soft" style="background:${rc.luck_color || "#9e9e9e"}" title="近期">近 ${rc.luck_label || "—"}</span>
        </span>
      </button>`;
    })
    .join("");

  const pool = data.pools.find((p) => p.key === activePoolKey) || data.pools[0];
  ui.luckPanel.innerHTML = `
    <div class="pool-tabs" role="tablist">${tabs}</div>
    <p class="cache-tip">${cacheTip} · 数据未变直接读缓存</p>
    <div class="pool-pane" id="pool-pane">${renderPool(pool)}</div>
  `;
  ui.luckPanel.querySelectorAll(".pool-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      activePoolKey = btn.dataset.pool;
      renderLuck(luckData);
      const pane = $("#pool-pane");
      if (pane) {
        pane.classList.remove("pane-enter");
        void pane.offsetWidth;
        pane.classList.add("pane-enter");
      }
    });
  });
  if (data.status) setAccount(data.status);
}

function destroyChart(key) {
  if (charts[key]) {
    charts[key].destroy();
    charts[key] = null;
  }
}

function chartDefaults(title) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    layout: { padding: { top: 4, right: 8, bottom: 4, left: 4 } },
    plugins: {
      title: {
        display: true,
        text: title,
        font: { size: 16, weight: "600" },
        padding: { top: 4, bottom: 12 },
        color: "#1c2430",
      },
      legend: {
        position: "top",
        align: "center",
        labels: {
          boxWidth: 16,
          boxHeight: 16,
          padding: 14,
          font: { size: 14, family: "'Noto Sans SC', sans-serif" },
          color: "#1c2430",
        },
      },
      tooltip: {
        titleFont: { size: 14 },
        bodyFont: { size: 13 },
      },
    },
    scales: {
      x: {
        ticks: { font: { size: 12 }, color: "#546e7a", maxRotation: 0 },
        grid: { color: "rgba(15,20,25,0.06)" },
      },
      y: {
        ticks: {
          font: { size: 13 },
          color: "#546e7a",
          callback(v) {
            const n = Number(v);
            if (!Number.isFinite(n)) return v;
            const t = Math.round(n * 1000) / 1000;
            return Number.isInteger(t) ? `${t}` : `${parseFloat(t.toFixed(2))}`;
          },
        },
        grid: { color: "rgba(15,20,25,0.06)" },
      },
    },
  };
}

function chartOpts(title, extra = {}) {
  const base = chartDefaults(title);
  const out = {
    ...base,
    ...extra,
    plugins: {
      ...base.plugins,
      ...(extra.plugins || {}),
      title: { ...base.plugins.title, ...((extra.plugins || {}).title || {}) },
      legend: { ...base.plugins.legend, ...((extra.plugins || {}).legend || {}) },
    },
  };
  if (extra.scales) {
    out.scales = {
      x: { ...base.scales.x, ...(extra.scales.x || {}) },
      y: { ...base.scales.y, ...(extra.scales.y || {}) },
    };
    if (extra.scales.x?.ticks) {
      out.scales.x.ticks = { ...base.scales.x.ticks, ...extra.scales.x.ticks };
    }
    if (extra.scales.y?.ticks) {
      out.scales.y.ticks = { ...base.scales.y.ticks, ...extra.scales.y.ticks };
    }
  }
  return out;
}

function renderProbPoolTabs(pools) {
  if (!ui.probPoolTabs) return;
  const list = pools || [];
  ui.probPoolTabs.innerHTML = list
    .map((p) => {
      const lt = p.long_term || {};
      const rc = p.recent || {};
      return `<button type="button" class="pool-tab ${p.key === activePoolKey ? "active" : ""}" data-pool="${p.key}">
        <span class="pool-tab-title">${p.title}</span>
        <span class="pool-tab-dual">
          <span class="pool-tab-luck" style="background:${lt.luck_color || "#9e9e9e"}" title="长期">长 ${lt.luck_label || "—"}</span>
          <span class="pool-tab-luck soft" style="background:${rc.luck_color || "#9e9e9e"}" title="近期">近 ${rc.luck_label || "—"}</span>
        </span>
      </button>`;
    })
    .join("");
  ui.probPoolTabs.querySelectorAll(".pool-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (activePoolKey === btn.dataset.pool) return;
      activePoolKey = btn.dataset.pool;
      loadProbability();
      if (luckData) renderLuck(luckData);
    });
  });
}

function renderFourStar(data) {
  const four = data.four_star || {};
  const title = data.pool_title || "卡池";
  if (ui.fourSectionTitle) {
    ui.fourSectionTitle.textContent = `${title} · 四星十抽保底`;
  }
  if (ui.fourRuleTip) {
    const basePct =
      four.base_rate != null ? `${(four.base_rate * 100).toFixed(1)}%` : "5.1%";
    const softPct =
      four.soft_rate != null ? `${(four.soft_rate * 100).toFixed(1)}%` : "56.1%";
    ui.fourRuleTip.textContent =
      `规则：1–8 抽 ${basePct} · 第 ${four.soft_pity ?? 9} 抽约 ${softPct} · 第 ${four.hard_pity ?? 10} 抽必出四星及以上` +
      (four.note ? ` · ${four.note}` : "") +
      ` · 五星也会重置四星计数`;
  }
  if (ui.fourMetrics) {
    ui.fourMetrics.innerHTML = [
      ["当前紫垫", four.current_pity ?? 0],
      ["下一抽概率", pct(four.next_rate)],
      ["期望还需", num(four.expected_pulls, 2)],
      ["距第9抽抬升", four.pulls_to_soft ?? "—"],
      ["距十抽保底", four.pulls_to_hard ?? "—"],
      ["50%出紫还需", four["pulls_to_50%"] ?? "—"],
      ["历史四星+", four.count ?? 0],
      ["历史均间隔", four.avg_interval ?? "—"],
    ]
      .map(([k, v]) => `<div class="metric"><b>${v ?? "—"}</b><small>${k}</small></div>`)
      .join("");
  }

  const curve = four.curve || { x: [], single: [], cumulative: [] };
  const softN = four.soft_pity || 9;
  const hardN = four.hard_pity || 10;

  destroyChart("fourSingle");
  const elFs = $("#chart-four-single");
  if (elFs) {
    charts.fourSingle = new Chart(elFs, {
      type: "bar",
      data: {
        labels: (curve.x || []).map((n) => `第${n}抽`),
        datasets: [
          {
            label: "单抽出紫(及以上)概率",
            data: (curve.single || []).map((v) => +(v * 100).toFixed(2)),
            backgroundColor: (curve.x || []).map((n) =>
              n >= hardN ? "#7e57c2" : n >= softN ? "#ab47bc" : "#ce93d8"
            ),
            borderWidth: 0,
          },
        ],
      },
      options: chartOpts("从当前紫垫起 · 单抽出四星及以上概率", {
        plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 100, ticks: { callback: (v) => v + "%" } } },
      }),
    });
  }

  destroyChart("fourCum");
  const elFc = $("#chart-four-cum");
  if (elFc) {
    charts.fourCum = new Chart(elFc, {
      type: "line",
      data: {
        labels: (curve.x || []).map((n) => `再抽${n - (four.current_pity || 0)}次`),
        datasets: [
          {
            label: "累积出紫概率",
            data: (curve.cumulative || []).map((v) => +(v * 100).toFixed(2)),
            borderColor: "#6a1b9a",
            backgroundColor: "rgba(106,27,154,0.12)",
            fill: true,
            tension: 0.2,
            pointRadius: 4,
            borderWidth: 2.5,
          },
        ],
      },
      options: chartOpts("从当前紫垫起 · 累积出四星及以上概率", {
        scales: { y: { min: 0, max: 100, ticks: { callback: (v) => v + "%" } } },
      }),
    });
  }

  destroyChart("fourHist");
  const hist = four.hist || { labels: [], counts: [] };
  const elFh = $("#chart-four-hist");
  if (elFh) {
    charts.fourHist = new Chart(elFh, {
      type: "bar",
      data: {
        labels: hist.labels,
        datasets: [
          {
            label: "次数",
            data: hist.counts,
            backgroundColor: (hist.labels || []).map((lb) => {
              const n = Number(lb);
              if (n >= 10) return "#7e57c2";
              if (n >= 9) return "#ab47bc";
              return "#ce93d8";
            }),
          },
        ],
      },
      options: chartOpts("历史四星及以上间隔分布（抽数）", {
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
      }),
    });
  }

  if (ui.fourTimeline) {
    const recent = four.recent || [];
    ui.fourTimeline.innerHTML =
      `<div class="timeline-head" style="padding:12px 0 4px">近期四星及以上（新 → 旧）</div>` +
      (recent
        .map(
          (h, i) =>
            `<div class="char-row"><span>${i + 1}. ${h.name || "—"}</span><b>${h.pity ?? "—"} 抽</b></div>`
        )
        .join("") || `<p class="empty-hint">暂无四星记录</p>`);
  }
}

function renderProbability(data) {
  if (data.status) setAccount(data.status);
  if (data.pools) renderProbPoolTabs(data.pools);

  const rule = data.pool_rule || {};
  const soft = (data.curve && data.curve.soft_pity) || rule.soft_pity || 74;
  const hard = (data.curve && data.curve.hard_pity) || rule.hard_pity || 90;
  const title = data.pool_title || "卡池";
  if (ui.probSectionTitle) {
    ui.probSectionTitle.textContent = `${title} · 当前垫刀出金概率`;
  }
  if (ui.probPoolRule) {
    const basePct = rule.base_rate != null ? `${(rule.base_rate * 100).toFixed(1)}%` : "—";
    ui.probPoolRule.textContent =
      `${title}规则：基础 ${basePct} · 软保底 ${soft} · 硬保底 ${hard}` +
      (rule.featured_note ? ` · ${rule.featured_note}` : "") +
      (data.cache?.hit ? ` · 缓存 ${data.elapsed_ms ?? "—"}ms` : ` · 已重算 ${data.elapsed_ms ?? "—"}ms`);
  }
  ui.pityInput.max = String(hard - 1);
  ui.pityInput.value = data.pulls;

  const s = data.stats || {};
  const ins = data.insights || {};

  ui.probMetrics.innerHTML = [
    ["当前垫刀", s.current_pulls],
    ["当前出金率", pct(s.current_prob)],
    ["下一抽出金率", pct(s.next_pull_prob)],
    ["距软保底", s.pulls_to_soft_pity],
    ["距硬保底", s.pulls_to_hard_pity],
    ["期望还需", num(s.expected_pulls, 1)],
    ["50%出金还需", s["pulls_to_50%"]],
    ["90%出金还需", s["pulls_to_90%"]],
  ]
    .map(([k, v]) => `<div class="metric"><b>${v ?? "—"}</b><small>${k}</small></div>`)
    .join("");

  renderFourStar(data);

  const delta = ins.avg_vs_theo;
  const deltaTxt =
    delta == null ? "—" : delta > 0 ? `偏高 ${delta}` : delta < 0 ? `偏低 ${Math.abs(delta)}` : "持平";

  ui.histMetrics.innerHTML = [
    ["样本金", ins.n ?? 0],
    ["均垫", ins.avg ?? "—"],
    ["中位垫", ins.median ?? "—"],
    ["标准差", ins.std ?? "—"],
    ["P25 / P75", `${ins.p25 ?? "—"} / ${ins.p75 ?? "—"}`],
    ["理论期望", ins.theo_expected ?? "—"],
    ["均垫vs理论", deltaTxt],
    ["软保底占比", pct(ins.soft_pity_rate)],
  ]
    .map(([k, v]) => `<div class="metric"><b>${v}</b><small>${k}</small></div>`)
    .join("");

  const zones = ins.zones || {};
  ui.zoneRow.innerHTML = Object.values(zones)
    .map(
      (z) =>
        `<div class="zone-chip"><b>${z.count}</b><span>${z.label}</span><small>${pct(z.rate)}</small></div>`
    )
    .join("");

  const hist = data.history || {};
  const chars = hist.characters || [];
  const counts = hist.wish_counts || [];
  ui.charTimeline.innerHTML =
    `<div class="timeline-head" style="padding:12px 0 4px">出金序列（旧 → 新）</div>` +
    (chars
      .map(
        (name, i) =>
          `<div class="char-row"><span>${i + 1}. ${name || "—"}</span><b>${counts[i] ?? "—"} 抽</b></div>`
      )
      .join("") || `<p class="empty-hint">暂无五星历史</p>`);

  const curve = data.curve || { x: [], single: [], cumulative: [] };
  const fine = ins.fine_hist || { labels: [], counts: [] };

  const cmp = ins.compare_pmf || { x: [], theoretical: [], empirical: [], theo_cdf: [] };
  const ecdf = ins.ecdf || { x: [], y: [] };
  const seq = ins.sequence || { index: [], pity: [], rolling_avg: [] };
  const ff = ins.fifty_fifty_roll || { index: [], win_rate: [] };

  destroyChart("singleBase");
  destroyChart("singleSoft");
  const xs = curve.x || [];
  const ys = (curve.single || []).map((v) => +(v * 100).toFixed(4));
  const baseIdx = [];
  const softIdx = [];
  xs.forEach((x, i) => {
    if (x < soft) baseIdx.push(i);
    else softIdx.push(i);
  });
  // 若当前已进入软保底，基础段可能为空：用软保底前 1 点占位说明
  const baseLabels = baseIdx.map((i) => xs[i]);
  const baseData = baseIdx.map((i) => ys[i]);
  const softLabels = softIdx.map((i) => xs[i]);
  const softData = softIdx.map((i) => ys[i]);
  const rawBaseMax = Math.max(1.5, ...(baseData.length ? baseData : [0.6]), 0.6) * 1.35;
  const baseMax = Math.round(rawBaseMax * 10) / 10; // 避免 2.0250000000000004
  const fmtPct = (v) => `${Number(v).toFixed(Number.isInteger(Number(v)) ? 0 : 1)}%`;

  const elBase = $("#chart-single-base");
  const elSoft = $("#chart-single-soft");
  if (elBase) {
    charts.singleBase = new Chart(elBase, {
      type: "bar",
      data: {
        labels: baseLabels.length ? baseLabels : [`<${soft}`],
        datasets: [
          {
            label: "基础段概率",
            data: baseData.length ? baseData : [0],
            backgroundColor: "#81c7c1",
            borderWidth: 0,
          },
        ],
      },
      options: chartOpts(`基础段放大（<${soft}抽，纵轴 0–${baseMax.toFixed(1)}%）`, {
        plugins: { legend: { display: false } },
        scales: {
          y: {
            min: 0,
            max: baseMax,
            ticks: { callback: fmtPct },
          },
        },
      }),
    });
  }
  if (elSoft) {
    charts.singleSoft = new Chart(elSoft, {
      type: "bar",
      data: {
        labels: softLabels.length ? softLabels : [soft],
        datasets: [
          {
            label: "软/硬保底段",
            data: softData.length ? softData : [0],
            backgroundColor: (softLabels.length ? softLabels : [soft]).map((x) =>
              x >= hard ? "#c47a2c" : "#e57373"
            ),
            borderWidth: 0,
          },
        ],
      },
      options: chartOpts(`软保底爬升（≥${soft}抽，纵轴 0–100%）`, {
        plugins: { legend: { display: false } },
        scales: {
          y: { min: 0, max: 100, ticks: { callback: (v) => v + "%" } },
        },
      }),
    });
  }

  destroyChart("cum");
  charts.cum = new Chart($("#chart-cum"), {
    type: "line",
    data: {
      labels: curve.x,
      datasets: [
        {
          label: "累积出金概率",
          data: curve.cumulative.map((v) => +(v * 100).toFixed(3)),
          borderColor: "#1f6f6a",
          backgroundColor: "rgba(31,111,106,0.12)",
          fill: true,
          tension: 0.2,
          pointRadius: 2,
        },
      ],
    },
    options: chartOpts("从当前垫刀起 · 累积出金概率", {
      scales: { y: { min: 0, max: 100, ticks: { callback: (v) => v + "%" } } },
    }),
  });

  destroyChart("hist");
  charts.hist = new Chart($("#chart-hist"), {
    type: "bar",
    data: {
      labels: fine.labels,
      datasets: [
        {
          label: "次数",
          data: fine.counts,
          backgroundColor: fine.labels.map((lb) => {
            const hi = Number(String(lb).split("-")[1] || 0);
            if (hi >= hard) return "#c47a2c";
            if (hi >= soft) return "#e57373";
            if (hi >= (ins.soft_pity ? soft - 14 : 60)) return "#90caf9";
            return "#81c7c1";
          }),
        },
      ],
    },
    options: chartOpts("历史出金垫数分布（5 抽一档）", {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 45, minRotation: 0, font: { size: 11 } } },
        y: { beginAtZero: true, ticks: { stepSize: 1 } },
      },
    }),
  });

  destroyChart("pmf");
  charts.pmf = new Chart($("#chart-pmf"), {
    type: "line",
    data: {
      labels: cmp.x,
      datasets: [
        {
          label: "理论概率",
          data: (cmp.theoretical || []).map((v) => +(v * 100).toFixed(4)),
          borderColor: "#78909c",
          borderDash: [5, 4],
          pointRadius: 0,
          tension: 0.15,
          borderWidth: 2,
        },
        {
          label: "你的经验频率",
          data: (cmp.empirical || []).map((v) => +(v * 100).toFixed(4)),
          borderColor: "#c47a2c",
          backgroundColor: "rgba(196,122,44,0.12)",
          fill: true,
          pointRadius: 0,
          tension: 0.15,
          borderWidth: 2.5,
        },
      ],
    },
    options: chartOpts("垫数概率质量：理论 vs 经验", {
      scales: { y: { beginAtZero: true, ticks: { callback: (v) => v + "%" } } },
    }),
  });

  destroyChart("ecdf");
  const theoCdfByPity = {};
  (cmp.x || []).forEach((x, i) => {
    theoCdfByPity[x] = cmp.theo_cdf[i];
  });
  const theoAtEmp = (ecdf.x || []).map((x) => {
    const v = theoCdfByPity[x];
    return v == null ? null : +(v * 100).toFixed(2);
  });
  charts.ecdf = new Chart($("#chart-ecdf"), {
    type: "line",
    data: {
      labels: ecdf.x,
      datasets: [
        {
          label: "经验累积分布",
          data: (ecdf.y || []).map((v) => +(v * 100).toFixed(2)),
          borderColor: "#ef6c00",
          pointRadius: 3,
          borderWidth: 2.5,
          stepped: true,
        },
        {
          label: "理论累积分布",
          data: theoAtEmp,
          borderColor: "#546e7a",
          borderDash: [4, 4],
          pointRadius: 0,
          borderWidth: 2,
          spanGaps: true,
        },
      ],
    },
    options: chartOpts("累积分布 CDF（出金垫数）", {
      scales: {
        x: { title: { display: true, text: "垫数", font: { size: 13 } } },
        y: { min: 0, max: 100, ticks: { callback: (v) => v + "%" } },
      },
    }),
  });

  destroyChart("zones");
  const zoneVals = Object.values(zones);
  charts.zones = new Chart($("#chart-zones"), {
    type: "doughnut",
    data: {
      labels: zoneVals.map((z) => z.label),
      datasets: [
        {
          data: zoneVals.map((z) => z.count),
          backgroundColor: ["#66bb6a", "#42a5f5", "#ef5350", "#c47a2c"],
        },
      ],
    },
    options: chartOpts("出金区间占比", {
      plugins: {
        legend: {
          position: "right",
          labels: { font: { size: 14 }, padding: 16, boxWidth: 18, boxHeight: 18 },
        },
      },
      scales: { x: { display: false }, y: { display: false } },
    }),
  });

  destroyChart("seq");
  charts.seq = new Chart($("#chart-seq"), {
    type: "line",
    data: {
      labels: seq.index,
      datasets: [
        {
          label: "每次出金垫数",
          data: seq.pity,
          borderColor: "#1f6f6a",
          backgroundColor: "rgba(31,111,106,0.08)",
          fill: false,
          pointRadius: 4,
          borderWidth: 2,
          tension: 0.2,
        },
        {
          label: "滚动均垫(5)",
          data: seq.rolling_avg,
          borderColor: "#c47a2c",
          borderDash: [6, 3],
          pointRadius: 0,
          borderWidth: 2.5,
          tension: 0.25,
        },
      ],
    },
    options: chartOpts("出金垫数时间序列（旧→新）", {
      scales: {
        y: {
          title: { display: true, text: "垫数", font: { size: 13 } },
          suggestedMin: 0,
          suggestedMax: 90,
        },
      },
    }),
  });

  destroyChart("ff");
  const track5050 = rule.track_5050 !== false && data.pool !== "permanent";
  charts.ff = new Chart($("#chart-5050"), {
    type: "line",
    data: {
      labels: track5050 ? ff.index : [],
      datasets: track5050
        ? [
            {
              label: data.pool === "weapon" ? "累计「非常驻」率(近似)" : "累计小保底不歪率",
              data: (ff.win_rate || []).map((v) => +(v * 100).toFixed(2)),
              borderColor: "#2e7d32",
              backgroundColor: "rgba(46,125,50,0.1)",
              fill: true,
              tension: 0.2,
              pointRadius: 3,
              borderWidth: 2.5,
            },
            {
              label: data.pool === "weapon" ? "参考线 75%" : "理论 50%",
              data: (ff.index || []).map(() => (data.pool === "weapon" ? 75 : 50)),
              borderColor: "#9e9e9e",
              borderDash: [4, 4],
              pointRadius: 0,
              borderWidth: 2,
            },
          ]
        : [
            {
              label: "常驻池无 UP/50-50",
              data: [],
              borderColor: "#9e9e9e",
            },
          ],
    },
    options: chartOpts(
      track5050
        ? data.pool === "weapon"
          ? "武器池出金非歪走势（相对常驻武器；定轨未单独建模）"
          : "小保底不歪率走势（仅 50/50，大保底不计）"
        : "常驻池：无大小保底 / UP 机制",
      {
        scales: { y: { min: 0, max: 100, ticks: { callback: (v) => v + "%" } } },
      }
    ),
  });
}


async function loadLuck() {
  try {
    const data = await api("/api/luck");
    renderLuck(data);
  } catch (e) {
    ui.luckPanel.innerHTML = `<p class="empty-hint">${e.message}</p>`;
  }
}

async function loadProbability(pulls) {
  try {
    const params = new URLSearchParams();
    params.set("pool", activePoolKey || "character");
    if (pulls != null) params.set("pulls", String(pulls));
    const data = await api(`/api/probability?${params.toString()}`);
    if (data.pool) activePoolKey = data.pool;
    renderProbability(data);
  } catch (e) {
    ui.probMetrics.innerHTML = `<p class="empty-hint">${e.message}</p>`;
  }
}

async function refreshAll() {
  ui.btnRefresh.disabled = true;
  try {
    await refreshStatus();
    await Promise.all([loadLuck(), loadProbability()]);
    await refreshLogs();
  } finally {
    ui.btnRefresh.disabled = false;
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  const showLuck = name === "luck";
  ui.luckPanel.classList.toggle("active", showLuck);
  ui.probPanel.classList.toggle("active", !showLuck);
  const panel = showLuck ? ui.luckPanel : ui.probPanel;
  panel.classList.remove("panel-enter");
  void panel.offsetWidth;
  panel.classList.add("panel-enter");
}

function closeQr() {
  if (qrTimer) clearInterval(qrTimer);
  qrTimer = null;
  if (qrJobId) {
    api(`/api/login/qr/${qrJobId}/cancel`, { method: "POST" }).catch(() => {});
  }
  qrJobId = null;
  ui.modal.classList.add("hidden");
}

async function startQr() {
  ui.modal.classList.remove("hidden");
  ui.qrImg.removeAttribute("src");
  ui.qrMsg.textContent = "正在生成二维码…";
  const { job_id } = await api("/api/login/qr/start", { method: "POST" });
  qrJobId = job_id;
  qrTimer = setInterval(async () => {
    try {
      const st = await api(`/api/login/qr/${job_id}`);
      ui.qrMsg.textContent = st.message || "";
      if (st.qr_png_b64) {
        ui.qrImg.src = `data:image/png;base64,${st.qr_png_b64}`;
      }
      if (st.status === "ok") {
        clearInterval(qrTimer);
        qrTimer = null;
        ui.modal.classList.add("hidden");
        if (st.account) setAccount(st.account);
        else await refreshStatus();
        await refreshAll();
      } else if (st.status === "error" || st.status === "aborted") {
        clearInterval(qrTimer);
        qrTimer = null;
        if (st.status === "error") ui.qrMsg.textContent = st.error || "失败";
      }
    } catch (e) {
      ui.qrMsg.textContent = e.message;
    }
  }, 800);
}

async function startSync() {
  ui.btnSync.disabled = true;
  try {
    await api("/api/sync", { method: "POST" });
    if (syncWatch) clearInterval(syncWatch);
    let n = 0;
    syncWatch = setInterval(async () => {
      n += 1;
      await refreshLogs();
      await refreshStatus();
      if (n >= 90) {
        clearInterval(syncWatch);
        ui.btnSync.disabled = false;
      }
      if ((ui.log.textContent || "").includes("同步完成")) {
        clearInterval(syncWatch);
        ui.btnSync.disabled = false;
        await refreshAll();
      }
      if ((ui.log.textContent || "").includes("同步失败")) {
        clearInterval(syncWatch);
        ui.btnSync.disabled = false;
      }
    }, 1000);
  } catch (e) {
    alert(e.message);
    ui.btnSync.disabled = false;
  }
}

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => switchTab(t.dataset.tab));
});
ui.btnLogin.addEventListener("click", () => startQr().catch((e) => alert(e.message)));
ui.btnQrCancel.addEventListener("click", closeQr);
ui.btnLogout.addEventListener("click", async () => {
  await api("/api/logout", { method: "POST" });
  await refreshStatus();
});
ui.btnSync.addEventListener("click", () => startSync());
ui.btnRefresh.addEventListener("click", () => refreshAll());
ui.btnRecalc.addEventListener("click", () => {
  const v = Number(ui.pityInput.value);
  loadProbability(Number.isFinite(v) ? v : undefined);
});

(async function init() {
  await refreshStatus();
  await refreshLogs();
  logTimer = setInterval(refreshLogs, 2500);
  await refreshAll();
})();
