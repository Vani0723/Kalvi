// Kalvi front end: live captions, lecture library and transcript replay.
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { status: null, lectures: [], current: null, recording: false, liveId: null };

  // ---------------------------------------------------------------- helpers
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clock = (s) => {
    s = Math.max(0, Math.floor(s || 0));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    const mm = String(m).padStart(2, "0"), ss = String(sec).padStart(2, "0");
    return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  };
  const deviceName = (d) => (d === "npu" ? "NPU" : d === "cpu" ? "CPU" : "mock");
  let toastTimer;
  const toast = (msg) => {
    const t = $("toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), 4000);
  };
  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || res.statusText);
    return body;
  }

  // ------------------------------------------------------------------ status
  async function loadStatus() {
    const s = await api("/api/status");
    state.status = s;
    state.recording = s.recording;
    state.liveId = s.lecture_id;
    const badge = $("device");
    badge.className = `device-badge ${s.device}`;
    $("device-label").textContent = s.device_label;
    const sel = $("language");
    sel.innerHTML = Object.entries(s.languages)
      .map(([code, name]) => `<option value="${code}" ${code === s.language ? "selected" : ""}>${esc(name)}</option>`)
      .join("");
    renderRecordButton();
  }

  // ---------------------------------------------------------------- lectures
  async function loadLectures() {
    state.lectures = await api("/api/lectures");
    const list = $("lecture-list");
    if (!state.lectures.length) {
      list.innerHTML = `<li class="empty">No lectures yet. Record one or import a recording.</li>`;
      return;
    }
    list.innerHTML = state.lectures
      .map((l) => {
        const date = new Date(l.started_at * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
        const live = l.id === state.liveId ? " · recording" : "";
        return `<li><button data-id="${l.id}" class="${state.current && state.current.id === l.id ? "active" : ""}">
          <span class="l-title">${esc(l.title)}</span>
          <span class="l-meta">${date} · ${clock(l.duration_s)} · ${l.segment_count} captions${live}</span>
        </button></li>`;
      })
      .join("");
  }

  async function openLecture(id) {
    if (id === state.liveId) return showLive();
    const lecture = await api(`/api/lectures/${id}`);
    state.current = lecture;
    $("live-view").hidden = true;
    $("lecture-view").hidden = false;
    $("lv-title").textContent = lecture.title;
    const when = new Date(lecture.started_at * 1000).toLocaleString();
    const dur = lecture.segments.length ? lecture.segments[lecture.segments.length - 1].end_s : 0;
    $("lv-meta").textContent = `${when} · ${clock(dur)} · transcribed on ${deviceName(lecture.device)}`;
    $("player").src = lecture.audio_path ? `/api/lectures/${id}/audio` : "";
    $("search").value = "";
    $("search-count").textContent = "";
    $("import-status").hidden = true;
    renderTranscript();
    loadLectures();
  }

  function renderTranscript() {
    const q = $("search").value.trim().toLowerCase();
    const segs = state.current ? state.current.segments : [];
    let hits = 0;
    $("transcript").innerHTML = segs
      .map((s) => {
        let text = esc(s.text);
        const match = !q || s.text.toLowerCase().includes(q);
        if (q && match) {
          hits += 1;
          const re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
          text = text.replace(re, (m) => `<mark>${m}</mark>`);
        }
        return `<li data-start="${s.start_s}" data-end="${s.end_s}" class="${match ? "" : "hidden"}">
          <span class="ts">${clock(s.start_s)}</span><span>${text}</span></li>`;
      })
      .join("");
    $("search-count").textContent = q ? `${hits} match${hits === 1 ? "" : "es"}` : "";
    if (!segs.length) $("transcript").innerHTML = `<li class="muted">No captions yet.</li>`;
  }

  function showLive() {
    state.current = null;
    $("lecture-view").hidden = true;
    $("live-view").hidden = false;
    loadLectures();
  }

  // --------------------------------------------------------------- recording
  function renderRecordButton() {
    const btn = $("record");
    btn.classList.toggle("recording", state.recording);
    btn.setAttribute("aria-label", state.recording ? "Stop recording" : "Start recording");
    $("record-hint").textContent = state.recording ? "Recording. Press again to stop." : "Press record when the lecture starts.";
    if (!state.recording) {
      $("speaking").textContent = "Idle";
      $("speaking").classList.remove("on");
      $("meter-fill").style.width = "0";
    }
  }

  async function toggleRecord() {
    const btn = $("record");
    btn.disabled = true;
    try {
      if (state.recording) {
        await api("/api/record/stop", { method: "POST" });
        state.recording = false;
        toast("Lecture saved.");
      } else {
        $("captions").innerHTML = "";
        $("timer").textContent = "00:00";
        const lecture = await api("/api/record/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: $("title").value.trim() || null, language: $("language").value }),
        });
        state.recording = true;
        state.liveId = lecture.id;
        if (!$("title").value.trim()) $("title").value = lecture.title;
      }
    } catch (err) {
      toast(err.message);
    } finally {
      btn.disabled = false;
      renderRecordButton();
      loadLectures();
    }
  }

  function addCaption(seg, device) {
    const box = $("captions");
    box.querySelector(".caption.pending")?.remove();
    const el = document.createElement("div");
    el.className = "caption";
    el.innerHTML = `<span class="ts">${clock(seg.start_s)}</span>
      <div><div class="text">${esc(seg.text)}</div>
      <div class="meta">${(seg.latency_ms / 1000).toFixed(2)} s on ${deviceName(device)}${seg.language ? " · " + esc(seg.language) : ""}</div></div>`;
    box.appendChild(el);
    box.scrollTop = box.scrollHeight;
  }

  function showPending(start) {
    const box = $("captions");
    if (box.querySelector(".caption.pending")) return;
    const el = document.createElement("div");
    el.className = "caption pending";
    el.innerHTML = `<span class="ts">${clock(start)}</span><div class="text">Transcribing…</div>`;
    box.appendChild(el);
    box.scrollTop = box.scrollHeight;
  }

  // ---------------------------------------------------------------- import
  async function importFile(file) {
    const form = new FormData();
    form.append("file", file);
    form.append("title", file.name.replace(/\.[^.]+$/, ""));
    form.append("language", $("language").value);
    toast(`Importing ${file.name}…`);
    try {
      const { lecture_id } = await api("/api/import", { method: "POST", body: form });
      await openLecture(lecture_id);
      $("import-status").hidden = false;
      $("import-status").textContent = "Transcribing on the device…";
    } catch (err) {
      toast(err.message);
    }
  }

  // -------------------------------------------------------------- websocket
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      switch (ev.type) {
        case "level":
          $("meter-fill").style.width = `${Math.min(100, ev.rms * 900)}%`;
          $("speaking").textContent = ev.speaking ? "Speech detected" : "Listening";
          $("speaking").classList.toggle("on", ev.speaking);
          $("timer").textContent = clock(ev.elapsed_s);
          break;
        case "transcribing":
          if (ev.start_s !== undefined && state.recording) showPending(ev.start_s);
          break;
        case "segment":
          if (ev.segment.lecture_id === state.liveId && state.recording) addCaption(ev.segment, ev.device);
          if (state.current && ev.segment.lecture_id === state.current.id) {
            state.current.segments.push(ev.segment);
            renderTranscript();
          }
          break;
        case "import_progress":
          if (state.current && ev.lecture_id === state.current.id) {
            $("import-status").hidden = false;
            $("import-status").textContent = `Transcribing on the device… ${clock(ev.done_s)} of ${clock(ev.total_s)}`;
          }
          break;
        case "import_done":
          if (state.current && ev.lecture_id === state.current.id) $("import-status").textContent = "Transcription complete.";
          loadLectures();
          break;
        case "status":
          state.recording = ev.recording;
          if (!ev.recording) state.liveId = null;
          renderRecordButton();
          loadLectures();
          break;
        case "error":
          toast(ev.message);
          break;
      }
    };
    ws.onclose = () => setTimeout(connect, 1500);
  }

  // ------------------------------------------------------------- benchmarks
  async function loadBenchmarks() {
    const data = await api("/api/benchmarks").catch(() => ({ results: [] }));
    if (!data.results || !data.results.length) return;
    const by = Object.fromEntries(data.results.map((r) => [r.device, r]));
    const npu = by.npu, cpu = by.cpu;
    const stats = [];
    if (npu) {
      stats.push([`${npu.encoder_ms} ms`, "Encoder latency on NPU"]);
      stats.push([`${npu.tokens_per_s}`, "Decoder tokens/s on NPU"]);
      stats.push([`${npu.real_time_factor}`, "Real-time factor on NPU"]);
    }
    if (npu && cpu && npu.total_ms_mean) stats.push([`${(cpu.total_ms_mean / npu.total_ms_mean).toFixed(1)}x`, "Faster than CPU"]);
    if (!npu && cpu) stats.push([`${cpu.real_time_factor}`, "Real-time factor on CPU"]);
    $("bench-body").innerHTML = `<div class="bench-grid">${stats
      .map(([v, l]) => `<div class="stat"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`)
      .join("")}</div>`;
    $("bench-card").hidden = false;
  }

  // ------------------------------------------------------------------ events
  $("record").addEventListener("click", toggleRecord);
  $("new-lecture").addEventListener("click", () => {
    showLive();
    if (!state.recording) {
      $("title").value = "";
      $("captions").innerHTML = "";
      $("title").focus();
    }
  });
  $("lecture-list").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-id]");
    if (btn) openLecture(Number(btn.dataset.id));
  });
  $("import-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) importFile(file);
    e.target.value = "";
  });
  $("search").addEventListener("input", renderTranscript);
  $("transcript").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-start]");
    if (!li || !$("player").src) return;
    $("player").currentTime = Number(li.dataset.start);
    $("player").play();
  });
  $("player").addEventListener("timeupdate", () => {
    const t = $("player").currentTime;
    document.querySelectorAll("#transcript li[data-start]").forEach((li) => {
      li.classList.toggle("playing", t >= Number(li.dataset.start) && t < Number(li.dataset.end));
    });
  });
  $("lv-delete").addEventListener("click", async () => {
    if (!state.current || !confirm(`Delete "${state.current.title}" and its recording?`)) return;
    try {
      await api(`/api/lectures/${state.current.id}`, { method: "DELETE" });
      toast("Lecture deleted.");
      showLive();
    } catch (err) {
      toast(err.message);
    }
  });

  loadStatus().catch((err) => toast(`Could not reach Kalvi: ${err.message}`));
  loadLectures();
  loadBenchmarks();
  connect();
})();
