export class BrowserPose {
  constructor() {
    this.worker = new Worker(new URL("./pose-worker.js", import.meta.url));
    this.pending = new Map();
    this.nextId = 0;
    this.closed = false;
    this.worker.onmessage = ({ data }) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.pending.delete(data.id);
      if (data.type === "error") pending.reject(new Error(data.message));
      else pending.resolve(data);
    };
    this.worker.onerror = () => this.close(new Error("관절 분석을 실행할 수 없습니다. 브라우저의 하드웨어 가속을 켜고 다시 시도해 주세요."));
  }
  request(type, data = {}, transfer = [], timeout = 30_000) {
    if (this.closed) return Promise.reject(new Error("Pose worker closed"));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.close(new Error("관절 분석 응답이 없습니다. 연결 상태를 확인하고 다시 시도해 주세요."));
      }, timeout);
      this.pending.set(id, { resolve, reject, timer });
      try { this.worker.postMessage({ id, type, ...data }, transfer); }
      catch (error) { clearTimeout(timer); this.pending.delete(id); reject(error); }
    });
  }
  async init() {
    this.info = await this.request("init", {}, [], 90_000);
    if (/swiftshader|llvmpipe|software/i.test(this.info.renderer || "")) {
      const error = new Error("하드웨어 GPU를 사용할 수 없습니다. 브라우저 하드웨어 가속을 켜고 다시 시도해 주세요.");
      this.close(error); throw error;
    }
    return this.info;
  }
  async detect(source, timestampMs) {
    const bitmap = await createImageBitmap(source);
    if (this.closed) { bitmap.close(); throw new Error("Pose worker closed"); }
    return this.request("frame", { bitmap, timestampMs }, [bitmap]);
  }
  close(error = new Error("Pose worker closed")) {
    if (this.closed) return;
    this.closed = true;
    this.worker.terminate();
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(error); }
    this.pending.clear();
  }
}
