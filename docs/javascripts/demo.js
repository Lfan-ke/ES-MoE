// The models run in the reader's browser: the checkpoints are public, the pictures they drop in are not.
(function () {
  const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.all.min.js";
  const VIEWERS = [
    ["Netron", "https://netron.app/?url="],
    ["Wetron", "https://wetron.app/?url="],
  ];
  const SAMPLES = { coco: ["bus.jpg", "zidane.jpg"], drone: ["aerial.jpg"] };
  const COLOURS = ["#e8a33d", "#3e7c8c", "#b3574d", "#6c8a3e", "#6c5ce0", "#2f8f83"];
  // One colour per block, so a reader tells the blocks apart at a glance; inside a block the top-k
  // experts keep that colour and the rest go grey.
  const BLOCKS = ["#e4572e", "#e8a33d", "#1f8a80", "#6c5ce0", "#b3574d", "#3e7c8c"];

  const TEXT = {
    zh: {
      group: "模型组",
      coco: "COCO 80 类",
      drone: "VisDrone 10 类",
      pick: "选模型",
      sample: "样例图",
      upload: "选择图片",
      run: "重跑",
      running: "推理中",
      loading: "加载模型",
      block: "块",
      none: "这个模型没有 ES-MoE 块。",
      chosen: "选中",
      absent: "站点上还没有模型文件，本地预览时属正常。",
      hint: "图片只在你的浏览器里处理，不会上传。推理也是使用你浏览器运行环境的本地计算能力进行推理。",
      found: (count) => `${count} 个目标`,
      two: "最多同时比较三个模型。",
    },
    en: {
      group: "Group",
      coco: "COCO, 80 classes",
      drone: "VisDrone, 10 classes",
      pick: "Models",
      sample: "Samples",
      upload: "Choose an image",
      run: "Run again",
      running: "Running",
      loading: "Loading",
      block: "Block",
      none: "This model has no ES-MoE block.",
      chosen: "chosen",
      absent: "The site has no model files yet, which is normal in a local preview.",
      hint: "Pictures stay in your browser and are never uploaded, and the inference runs on your own machine, inside the browser.",
      found: (count) => `${count} objects`,
      two: "Three models at a time.",
    },
  };

  // Three panels still fit the content column side by side; a fourth would be a scroll bar.
  const LIMIT = 3;
  // The strip keeps the samples and the reader's own pictures, the oldest dropping off the front.
  const STRIP = 5;
  const state = { index: [], group: "coco", chosen: [], image: null, base: "", assets: "", lang: "en", uploads: [] };
  let words = TEXT.en;
  let pending = null;
  let queue = Promise.resolve();
  const sessions = new Map();

  // The wasm runtime is one runtime: two panels building or running sessions at the same time break
  // each other, so every model waits its turn.
  function serial(task) {
    const next = queue.then(task, task);
    queue = next.catch(() => {});
    return next;
  }

  function library() {
    pending =
      pending ||
      new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = ORT;
        script.onload = () => {
          // Cross-origin isolation is what multiple threads would need, and a static host cannot set it.
          ort.env.wasm.numThreads = 1;
          resolve();
        };
        script.onerror = reject;
        document.head.appendChild(script);
      });
    return pending;
  }

  async function session(model) {
    if (!sessions.has(model.id)) {
      await library();
      const url = state.base + model.file;
      const create = async () => {
        try {
          return await ort.InferenceSession.create(url, { executionProviders: ["webgpu", "wasm"] });
        } catch {
          return ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
        }
      };
      sessions.set(model.id, create());
    }
    return sessions.get(model.id);
  }

  function letterbox(image, size) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const scale = Math.min(size / image.width, size / image.height);
    const width = Math.round(image.width * scale);
    const height = Math.round(image.height * scale);
    const dx = Math.floor((size - width) / 2);
    const dy = Math.floor((size - height) / 2);
    const context = canvas.getContext("2d");
    context.fillStyle = "#727272";
    context.fillRect(0, 0, size, size);
    context.drawImage(image, dx, dy, width, height);
    return { canvas, scale, dx, dy };
  }

  function tensor(canvas, size) {
    const { data } = canvas.getContext("2d").getImageData(0, 0, size, size);
    const plane = size * size;
    const values = new Float32Array(3 * plane);
    for (let index = 0, pixel = 0; index < plane; index += 1, pixel += 4) {
      values[index] = data[pixel] / 255;
      values[index + plane] = data[pixel + 1] / 255;
      values[index + 2 * plane] = data[pixel + 2] / 255;
    }
    return new ort.Tensor("float32", values, [1, 3, size, size]);
  }

  function overlap(a, b) {
    const x = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
    const y = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
    const shared = x * y;
    return shared / (a[2] * a[3] + b[2] * b[3] - shared);
  }

  function detect(output, classes, conf = 0.25, iou = 0.45) {
    const [, rows, anchors] = output.dims;
    const found = [];
    for (let anchor = 0; anchor < anchors; anchor += 1) {
      let best = 0;
      let score = 0;
      for (let row = 4; row < rows; row += 1) {
        const value = output.data[row * anchors + anchor];
        if (value > score) {
          score = value;
          best = row - 4;
        }
      }
      if (score < conf) continue;
      const cx = output.data[anchor];
      const cy = output.data[anchors + anchor];
      const width = output.data[2 * anchors + anchor];
      const height = output.data[3 * anchors + anchor];
      found.push({ box: [cx - width / 2, cy - height / 2, width, height], score, label: classes[best], index: best });
    }
    found.sort((left, right) => right.score - left.score);
    const kept = [];
    for (const candidate of found) {
      if (!kept.some((other) => other.index === candidate.index && overlap(other.box, candidate.box) > iou)) {
        kept.push(candidate);
      }
    }
    return kept;
  }

  function paint(canvas, image, found, pad) {
    const scale = 720 / image.width;
    canvas.width = 720;
    canvas.height = Math.round(image.height * scale);
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const size = Math.max(11, Math.round(canvas.width / 52));
    context.lineWidth = Math.max(1.5, canvas.width / 480);
    context.font = `${size}px "Source Sans 3", system-ui, sans-serif`;
    context.textBaseline = "top";
    for (const item of found) {
      const x = ((item.box[0] - pad.dx) / pad.scale) * scale;
      const y = ((item.box[1] - pad.dy) / pad.scale) * scale;
      const width = (item.box[2] / pad.scale) * scale;
      const height = (item.box[3] / pad.scale) * scale;
      const colour = COLOURS[item.index % COLOURS.length];
      context.strokeStyle = colour;
      context.strokeRect(x, y, width, height);
      const caption = `${item.label} ${item.score.toFixed(2)}`;
      const top = Math.max(0, y - size - 4);
      context.fillStyle = colour;
      context.fillRect(x, top, context.measureText(caption).width + 8, size + 4);
      context.fillStyle = "#ffffff";
      context.fillText(caption, x + 4, top + 2);
    }
  }

  function gates(probabilities, blocks) {
    if (!blocks.length) return `<p class="es-demo__none">${words.none}</p>`;
    return blocks
      .map((block, index) => {
        const values = Array.from(probabilities[index]);
        const order = [...values.keys()].sort((left, right) => values[right] - values[left]);
        const chosen = new Set(order.slice(0, block.top_k));
        const rows = values
          .map((value, expert) => {
            return `
            <div class="es-gate ${chosen.has(expert) ? "es-gate--on" : "es-gate--off"}" style="--share:${value.toFixed(3)}">
              <span>k${block.kernels[expert]}</span>
              <div class="es-gate__track"><i style="width:${(value * 100).toFixed(1)}%"></i></div>
              <em>${value.toFixed(3)}</em>
            </div>`;
          })
          .join("");
        const picked = order
          .slice(0, block.top_k)
          .map((expert) => `k${block.kernels[expert]}`)
          .join(" + ");
        return `<figure class="es-demo__block" style="--gate:${BLOCKS[index % BLOCKS.length]}">
          <figcaption>${words.block} ${index + 1} · top-${block.top_k} · ${words.chosen} ${picked}</figcaption>
          ${rows}
        </figure>`;
      })
      .join("");
  }

  async function run(model, panel) {
    const status = panel.querySelector(".es-demo__cost");
    status.textContent = words.loading;
    const active = await session(model);
    status.textContent = words.running;
    const pad = letterbox(state.image, model.imgsz);
    const started = performance.now();
    const outputs = await active.run({ images: tensor(pad.canvas, model.imgsz) });
    const elapsed = Math.round(performance.now() - started);
    const found = detect(outputs.pred, model.classes);
    paint(panel.querySelector("canvas"), state.image, found, pad);
    panel.querySelector(".es-demo__gates").innerHTML = gates(
      model.blocks.map((_block, index) => outputs[`probs${index}`].data),
      model.blocks,
    );
    status.textContent = `${elapsed} ms · ${words.found(found.length)}`;
  }

  function panels(host) {
    const shown = host.querySelector(".es-demo__panels");
    // The chosen models share one row and the pictures shrink to fit, so they stay comparable.
    shown.style.setProperty("--panels", String(state.chosen.length || 1));
    shown.innerHTML = state.index
      .filter((model) => state.chosen.includes(model.id))
      .map((model) => {
        const address = new URL(state.base + model.file, location.href).href;
        const viewers = VIEWERS.map(
          ([name, prefix]) => `<a href="${prefix}${address}" target="_blank" rel="noopener">${name}</a>`,
        ).join(" · ");
        return `<section class="es-demo__panel" data-model="${model.id}">
          <header><strong>${model.label[state.lang]}</strong><span class="es-demo__cost"></span></header>
          <canvas></canvas>
          <div class="es-demo__gates"></div>
          <footer>
            <span class="es-demo__meta" title="${model.source}">${model.source}</span>
            <span class="es-demo__note">${model.imgsz}px · ${(model.parameters / 1e6).toFixed(2)}M</span>
            <span class="es-demo__viewers">${viewers}</span>
          </footer>
        </section>`;
      })
      .join("");
  }

  function controls(host) {
    const groups = ["coco", "drone"]
      .map(
        (group) =>
          `<button type="button" class="es-demo__tab${state.group === group ? " es-demo__tab--on" : ""}" data-group="${group}">${words[group]}</button>`,
      )
      .join("");
    const models = state.index
      .filter((model) => model.group === state.group)
      .map(
        (model) =>
          `<label class="es-demo__choice"><input type="checkbox" value="${model.id}"${state.chosen.includes(model.id) ? " checked" : ""}>${model.label[state.lang]}</label>`,
      )
      .join("");
    const shots = [...SAMPLES[state.group].map((name) => state.assets + name), ...state.uploads];
    const samples = shots
      .slice(Math.max(0, shots.length - STRIP))
      .map((src) => {
        const here = state.image && state.image.src === src ? " es-demo__sample--on" : "";
        return `<img class="es-demo__sample${here}" src="${src}" alt="" loading="lazy">`;
      })
      .join("");
    host.querySelector(".es-demo__controls").innerHTML = `
      <div class="es-demo__row"><b>${words.group}</b>${groups}</div>
      <div class="es-demo__row"><b>${words.pick}</b>${models}</div>
      <div class="es-demo__row"><b>${words.sample}</b>${samples}
        <label class="es-demo__upload">${words.upload}<input type="file" accept="image/*" hidden></label>
        <button type="button" class="es-demo__run">${words.run}</button>
      </div>
      <p class="es-demo__hint">${words.hint}</p>`;
  }

  function show(host, source) {
    const image = new Image();
    image.onload = () => {
      state.image = image;
      controls(host);
      panels(host);
      host.querySelectorAll(".es-demo__panel").forEach((panel) => {
        const model = state.index.find((item) => item.id === panel.dataset.model);
        serial(() => run(model, panel)).catch((error) => {
          panel.querySelector(".es-demo__cost").textContent = String(error).slice(0, 90);
        });
      });
    };
    image.crossOrigin = "anonymous";
    image.src = source;
  }

  function chosen(host) {
    return [...host.querySelectorAll(".es-demo__choice input:checked")].map((box) => box.value);
  }

  function wire(host) {
    host.addEventListener("click", (event) => {
      const tab = event.target.closest(".es-demo__tab");
      if (tab) {
        state.group = tab.dataset.group;
        state.chosen = state.index.filter((model) => model.group === state.group).map((model) => model.id).slice(0, 2);
        controls(host);
        host.querySelector(".es-demo__panels").innerHTML = "";
        return;
      }
      const sample = event.target.closest(".es-demo__sample");
      if (sample) show(host, sample.src);
      else if (event.target.closest(".es-demo__run") && state.image) show(host, state.image.src);
    });
    host.addEventListener("change", (event) => {
      if (event.target.type === "checkbox") {
        const picked = chosen(host);
        state.chosen = picked.slice(0, LIMIT);
        if (picked.length > LIMIT) {
          controls(host);
          host.querySelector(".es-demo__hint").textContent = words.two;
        }
        if (state.image) show(host, state.image.src);
      } else if (event.target.files && event.target.files[0]) {
        const source = URL.createObjectURL(event.target.files[0]);
        state.uploads.push(source);
        show(host, source);
      }
    });
  }

  async function boot() {
    const host = document.querySelector("#esmoe-demo");
    if (!host || host.dataset.ready) return;
    host.dataset.ready = "1";
    state.lang = (document.documentElement.lang || "").startsWith("zh") ? "zh" : "en";
    words = TEXT[state.lang];
    // The models sit beside the version directories, so the path differs between a version, a
    // language and a local preview; derive it from the version segment when there is one.
    const inside = location.pathname.match(/^(.*?\/)(dev|latest|\d+\.\d+(?:\.\d+)?)\//);
    state.base = inside ? `${inside[1]}models/` : host.dataset.models;
    // i18n keeps one copy of the assets, at the version's root rather than under the language.
    state.assets = host.dataset.assets;
    host.innerHTML = '<div class="es-demo__controls"></div><div class="es-demo__panels"></div>';
    try {
      const response = await fetch(state.base + "index.json");
      state.index = response.ok ? await response.json() : [];
    } catch {
      state.index = [];
    }
    if (!state.index.length) {
      host.innerHTML = `<p class="es-demo__none">${words.absent}</p>`;
      return;
    }
    state.chosen = state.index.filter((model) => model.group === state.group).map((model) => model.id).slice(0, 2);
    controls(host);
    wire(host);
  }

  document$.subscribe(boot);
})();
