// The models run in the reader's browser: the checkpoints are public, the pictures they drop in are not.
(function () {
  const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.all.min.js";
  // Netron opens a model straight from its address; Wetron reads local files only, so it gets the
  // plain address and the file beside it.
  const NETRON = "https://netron.app/?url=";
  const WETRON = "https://wetron.app/";
  // The run button is a train: it rocks under the pointer and pulls away while a model is working.
  const TRAIN = '<svg viewBox="0 0 1024 1024" aria-hidden="true" focusable="false"><path d="M128 170.666667a42.666667 42.666667 0 1 0 0 85.333333h95.317333l15.36 122.709333q2.389333 19.114667-10.368 33.536-12.714667 14.421333-32 14.421334H128a42.666667 42.666667 0 0 0 0 85.333333h68.352a128 128 0 0 0 126.976-143.872L309.333333 256H362.666667q9.685333 0 22.357333 2.389333A277.333333 277.333333 0 0 0 661.333333 512h128c7.04 0 14.08-0.256 21.034667-0.810667Q853.333333 557.44 853.333333 576q0 26.496-18.773333 45.226667-18.730667 18.773333-45.226667 18.773333H128a42.666667 42.666667 0 1 0 0 85.333333h661.333333a149.333333 149.333333 0 0 0 149.333334-149.333333c0-44.245333-37.461333-97.877333-94.08-151.381333a338.432 338.432 0 0 0-3.370667-4.608l-1.152 0.341333C704.810667 294.613333 466.56 170.666667 362.666667 170.666667H128zm397.568 199.765333q-35.882667-35.882667-48.896-81.28l5.162667 2.176q88.106667 38.058667 177.834666 97.237333q29.141333 19.2 54.528 38.101334H661.333333q-79.530667 0-135.765333-56.234667zM128 768a42.666667 42.666667 0 1 0 0 85.333333h682.666667a42.666667 42.666667 0 1 0 0-85.333333H128z"/></svg>';
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
      ready: "跑一遍",
      empty: "请选择一个模型",
      waiting: "点一张样例图，或上传自己的图片",
      block: "块",
      none: "这个模型没有 ES-MoE 块。",
      chosen: "选中",
      absent: "站点上还没有模型文件，本地预览时属正常。",
      hint: "图片只在你的浏览器里处理，不会上传。推理也是使用你浏览器运行环境的本地计算能力进行推理。",
      found: (count) => `${count} 个目标`,
      limit: "最多同时比较三个模型。",
      file: "模型文件",
      broken: "这张图片打不开，换一张试试。",
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
      ready: "Run",
      empty: "Pick a model",
      waiting: "Pick a sample or upload a picture",
      block: "Block",
      none: "This model has no ES-MoE block.",
      chosen: "chosen",
      absent: "The site has no model files yet, which is normal in a local preview.",
      hint: "Pictures stay in your browser and are never uploaded, and the inference runs on your own machine, inside the browser.",
      found: (count) => `${count} objects`,
      limit: "Three models at a time.",
      file: "Model file",
      broken: "That picture would not open; try another.",
    },
  };

  // Three panels still fit the content column side by side; a fourth would be a scroll bar.
  const LIMIT = 3;
  const START = 2;
  // The strip keeps the samples and the reader's own pictures, the oldest dropping off the front.
  const STRIP = 5;
  const state = {
    index: [],
    group: "coco",
    chosen: [],
    image: null,
    base: "",
    assets: "",
    lang: "en",
    uploads: [],
    notice: "",
    run: 0,
  };
  let words = TEXT.en;
  let pending = null;
  let queue = Promise.resolve();
  const sessions = new Map();
  const providers = new Map();
  let hardware;

  // WebGPU tells a page its adapter's vendor and architecture and nothing about the backend behind it,
  // so the label carries what can be known and stays quiet about the rest.
  async function machine() {
    if (hardware !== undefined) return hardware;
    hardware = "";
    try {
      const adapter = navigator.gpu && (await navigator.gpu.requestAdapter());
      const info = adapter && adapter.info;
      hardware = [info && info.vendor, info && info.architecture].filter(Boolean).join(" ");
    } catch {
      hardware = "";
    }
    return hardware;
  }

  // WebAssembly answers for itself: SIMD by validating a module that needs it, threads by the shared
  // memory only a cross-origin-isolated page is given.
  function processor() {
    const simd = WebAssembly.validate(
      new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98,
        11,
      ]),
    );
    const threads = typeof SharedArrayBuffer !== "undefined" && self.crossOriginIsolated;
    return [simd && "SIMD", threads && "threads"].filter(Boolean).join(", ");
  }

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
      // An NPU first where the browser offers one, then the GPU, then the processor.
      const create = async () => {
        const tries = [];
        if (navigator.ml) tries.push([[{ name: "webnn", deviceType: "npu" }], () => "WebNN(npu)"]);
        if (navigator.gpu) {
          tries.push([["webgpu"], async () => { const found = await machine(); return found ? `WebGPU(${found})` : "WebGPU"; }]);
        }
        tries.push([["wasm"], () => (processor() ? `WASM(${processor()})` : "WASM")]);
        let last;
        for (const [wanted, name] of tries) {
          try {
            const ready = await ort.InferenceSession.create(url, { executionProviders: wanted });
            providers.set(model.id, await name());
            return ready;
          } catch (error) {
            last = error;
          }
        }
        throw last;
      };
      // A failed load must not be remembered, or running again could never recover from it.
      sessions.set(
        model.id,
        create().catch((error) => {
          sessions.delete(model.id);
          throw error;
        }),
      );
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
      const label = classes[best] ?? String(best);
      found.push({ box: [cx - width / 2, cy - height / 2, width, height], score, label, index: best });
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

  // Top-k picks the experts, then inference drops any whose renormalised share falls under the
  // block's threshold, the leading one excepted. The panel marks what actually ran.
  function picked(values, block) {
    const order = [...values.keys()].sort((left, right) => values[right] - values[left]);
    const top = order.slice(0, block.top_k);
    const total = top.reduce((sum, expert) => sum + values[expert], 0) || 1;
    const threshold = block.threshold || 0;
    return top.filter((expert, place) => place === 0 || values[expert] / total >= threshold);
  }

  function gates(probabilities, blocks) {
    if (!blocks.length) return `<p class="es-demo__none">${words.none}</p>`;
    return blocks
      .map((block, index) => {
        const values = Array.from(probabilities[index]);
        const order = picked(values, block);
        const chosen = new Set(order);
        const rows = values
          .map((value, expert) => {
            return `
            <div class="es-gate ${chosen.has(expert) ? "es-gate--on" : "es-gate--off"}" style="--share:${value.toFixed(3)}">
              <span>k${block.kernels[expert] ?? expert + 1}</span>
              <div class="es-gate__track"><i style="width:${(value * 100).toFixed(1)}%"></i></div>
              <em>${value.toFixed(3)}</em>
            </div>`;
          })
          .join("");
        const names = order.map((expert) => `k${block.kernels[expert] ?? expert + 1}`).join(" + ");
        return `<figure class="es-demo__block" style="--gate:${BLOCKS[index % BLOCKS.length]}">
          <figcaption>${words.block} ${index + 1} · top-${block.top_k} · ${words.chosen} ${names}</figcaption>
          ${rows}
        </figure>`;
      })
      .join("");
  }

  // Three draft lines behind the train, each on its own beat so they never move in lockstep.
  function drafts() {
    return [0, 1, 2]
      .map(() => {
        const beat = (0.9 + Math.random() * 0.9).toFixed(2);
        const wait = (Math.random() * 0.8).toFixed(2);
        return `<i style="--beat:${beat}s;--wait:${wait}s"></i>`;
      })
      .join("");
  }

  // A first run fetches about ten megabytes and starts a runtime, so the button says where it is.
  function working(host, text) {
    const button = host.querySelector(".es-demo__run");
    if (!button) return;
    button.toggleAttribute("data-busy", Boolean(text));
    button.querySelector(".es-demo__state").textContent = text || "";
  }

  async function run(model, panel, host) {
    const status = panel.querySelector(".es-demo__cost");
    const name = model.label[state.lang];
    status.textContent = words.loading;
    working(host, `${words.loading} · ${name}`);
    const active = await session(model);
    const provider = providers.get(model.id);
    status.textContent = words.running;
    working(host, [`${words.running} · ${name}`, provider].filter(Boolean).join(" · "));
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
    if (!state.chosen.length || !state.image) {
      shown.innerHTML = `<p class="es-demo__empty">${state.chosen.length ? words.waiting : words.empty}</p>`;
      return;
    }
    shown.innerHTML = state.index
      .filter((model) => state.chosen.includes(model.id))
      .map((model) => {
        const address = new URL(state.base + model.file, location.href).href;
        const viewers =
          `<a href="${NETRON}${address}" target="_blank" rel="noopener">Netron</a> · ` +
          `<a href="${WETRON}" target="_blank" rel="noopener">Wetron</a> · ` +
          `<a href="${address}" download>${words.file}</a>`;
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
    const shots = [
      ...SAMPLES[state.group].map((name) => new URL(state.assets + name, location.href).href),
      ...state.uploads,
    ];
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
        <button type="button" class="es-demo__run" title="${words.ready}" aria-label="${words.ready}"><span class="es-demo__train">${TRAIN}${drafts()}</span><span class="es-demo__spin"></span><span class="es-demo__state"></span></button>
      </div>
      <p class="es-demo__hint">${state.notice || words.hint}</p>`;
    state.notice = "";
  }

  function show(host, source) {
    // Switching picture or models mid-queue: the older run has nothing left to draw into.
    const generation = (state.run += 1);
    working(host, words.loading);
    const image = new Image();
    image.onerror = () => {
      state.notice = words.broken;
      controls(host);
    };
    image.onload = () => {
      if (generation !== state.run) return;
      state.image = image;
      controls(host);
      panels(host);
      host.querySelectorAll(".es-demo__panel").forEach((panel) => {
        const model = state.index.find((item) => item.id === panel.dataset.model);
        serial(() => (generation === state.run ? run(model, panel, host) : null)).catch((error) => {
          panel.querySelector(".es-demo__cost").textContent = String(error).slice(0, 90);
          working(host, "");
        });
      });
      serial(() => working(host, ""));
    };
    image.crossOrigin = "anonymous";
    image.src = source;
  }

  function wire(host) {
    host.addEventListener("click", (event) => {
      const tab = event.target.closest(".es-demo__tab");
      if (tab) {
        state.group = tab.dataset.group;
        state.chosen = state.index
          .filter((model) => model.group === state.group)
          .map((model) => model.id)
          .slice(0, START);
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
        // Kept in the order they were ticked, so the oldest gives way once the row is full.
        const id = event.target.value;
        const wanted = event.target.checked ? [...state.chosen, id] : state.chosen.filter((other) => other !== id);
        if (wanted.length > LIMIT) state.notice = words.limit;
        state.chosen = wanted.slice(-LIMIT);
        controls(host);
        // The row always says where it stands: which model is missing, or which picture.
        if (!state.chosen.length || !state.image) panels(host);
        else show(host, state.image.src);
      } else if (event.target.files && event.target.files[0]) {
        const source = URL.createObjectURL(event.target.files[0]);
        state.uploads.push(source);
        while (state.uploads.length > STRIP) URL.revokeObjectURL(state.uploads.shift());
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
    state.chosen = state.index
      .filter((model) => model.group === state.group)
      .map((model) => model.id)
      .slice(0, START);
    controls(host);
    panels(host);
    wire(host);
  }

  document$.subscribe(boot);
})();
