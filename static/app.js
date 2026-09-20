"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { user: null, csrf: "", dashboard: null, stream: null, socket: null, sending: false, timer: null, mode: null, closing: false };
const modeNames = { basic: "모아뛰기", alternating: "번갈아뛰기", double: "이중뛰기" };

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.csrf && options.method && options.method !== "GET") headers["X-CSRF-Token"] = state.csrf;
  const response = await fetch(path, { credentials: "same-origin", ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || "요청을 처리하지 못했습니다.");
  return body;
}

function showToast(message) {
  const toast = $("#toast"); toast.textContent = message; toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 2600);
}

function showApp(user) {
  state.user = user; state.csrf = user.csrfToken || state.csrf;
  $("#login-view").hidden = true; $("#app-shell").hidden = false;
  $("#user-badge").textContent = (user.displayName || user.username).slice(0, 1);
  $("#admin-tab").hidden = user.role !== "admin";
  $("#welcome-heading").textContent = `${user.displayName}님, 리듬을 선택하세요.`;
  for (const mode of Object.keys(modeNames)) {
    const card = document.querySelector(`[data-mode-card="${mode}"]`);
    const allowed = Boolean(user.permissions[mode]);
    card.classList.toggle("is-locked", !allowed);
    card.querySelector(".locked-copy").hidden = allowed;
  }
  loadDashboard();
}

function showLogin() {
  state.user = null; state.csrf = "";
  $("#login-view").hidden = false; $("#app-shell").hidden = true; $("#workout-view").hidden = true;
}

async function loadSession() {
  try { showApp(await api("/api/auth/me")); } catch { showLogin(); }
}

function formatTime(total) {
  const value = Math.max(0, Math.round(total || 0));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function localDate(value) {
  return new Intl.DateTimeFormat("ko-KR", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

async function loadDashboard() {
  try {
    const data = await api("/api/dashboard"); state.dashboard = data;
    $("#summary-count").textContent = data.summary.count.toLocaleString("ko-KR");
    $("#summary-sessions").textContent = data.summary.sessions.toLocaleString("ko-KR");
    $("#summary-duration").textContent = `${Math.round(data.summary.duration / 60)}분`;
    const body = $("#history-body"); body.replaceChildren();
    $("#history-empty").hidden = data.recent.length > 0;
    data.recent.forEach((item) => {
      const row = document.createElement("tr");
      [item.user, modeNames[item.mode] || item.mode, `${item.count}회`, formatTime(item.duration), localDate(item.startedAt)].forEach((text) => {
        const cell = document.createElement("td"); cell.textContent = text; row.appendChild(cell);
      });
      body.appendChild(row);
    });
  } catch (error) { showToast(error.message); }
}

function navigate(target) {
  $$(".page-view").forEach((view) => { view.hidden = view.id !== `${target}-view`; });
  $$(".nav-tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.nav === target));
  if (target === "admin") loadAdmin(); else loadDashboard();
}

function permissionToggle(user, key, label, apiField = key) {
  const wrapper = document.createElement("label");
  const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = Boolean(user.permissions[key]);
  checkbox.disabled = user.role === "admin";
  checkbox.addEventListener("change", async () => {
    try { await api(`/api/admin/users/${user.id}`, { method: "PATCH", body: JSON.stringify({ [`can_${apiField}`]: checkbox.checked }) }); showToast("권한을 저장했습니다."); }
    catch (error) { checkbox.checked = !checkbox.checked; showToast(error.message); }
  });
  wrapper.append(checkbox, document.createTextNode(` ${label}`)); return wrapper;
}

async function loadAdmin() {
  if (state.user.role !== "admin") return navigate("dashboard");
  try {
    const [users, logs] = await Promise.all([api("/api/admin/users"), api("/api/admin/audit")]);
    const list = $("#user-list"); list.replaceChildren();
    users.forEach((user) => {
      const row = document.createElement("div"); row.className = "user-row";
      const identity = document.createElement("div"); const name = document.createElement("strong"); name.textContent = user.displayName;
      const meta = document.createElement("small"); meta.textContent = `${user.username} · ${user.role === "admin" ? "관리자" : "사용자"}${user.active ? "" : " · 비활성"}`;
      identity.append(name, meta);
      const toggles = document.createElement("div"); toggles.className = "permission-toggles";
      toggles.append(permissionToggle(user, "basic", "모아"), permissionToggle(user, "alternating", "교대"), permissionToggle(user, "double", "이중"), permissionToggle(user, "history", "기록", "view_history"));
      row.append(identity, toggles); list.appendChild(row);
    });
    const audits = $("#audit-list"); audits.replaceChildren();
    logs.slice(0, 16).forEach((log) => {
      const item = document.createElement("div"); item.className = "audit-item";
      const action = document.createElement("strong"); action.textContent = log.action;
      const info = document.createElement("span"); info.textContent = `${log.actor} · ${localDate(log.createdAt)}`;
      item.append(action, info); audits.appendChild(item);
    });
  } catch (error) { showToast(error.message); }
}

async function startWorkout(mode) {
  if (!state.user.permissions[mode]) return;
  state.mode = mode; state.closing = false;
  $("#workout-mode-name").textContent = modeNames[mode]; $("#workout-view").hidden = false; $("#result-overlay").hidden = true;
  $("#live-count").textContent = "0"; $("#live-time").textContent = "00:00"; $("#connection-state").textContent = "카메라 연결 중";
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
    const video = $("#camera-source"); video.srcObject = state.stream; await video.play();
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    state.socket = new WebSocket(`${protocol}://${location.host}/ws/count/${mode}`); state.socket.binaryType = "blob";
    state.socket.onopen = () => { $("#connection-state").textContent = "분석 서버 연결됨"; };
    state.socket.onmessage = handleSocketMessage;
    state.socket.onclose = (event) => {
      state.sending = false;
      if (!state.closing && event.code !== 1000) { showToast(event.reason || "측정 연결이 종료되었습니다."); cleanupWorkout(); }
    };
    state.socket.onerror = () => showToast("분석 서버에 연결할 수 없습니다.");
  } catch (error) { showToast(error.name === "NotAllowedError" ? "카메라 사용 권한을 허용해 주세요." : "카메라를 시작할 수 없습니다."); cleanupWorkout(); }
}

async function handleSocketMessage(event) {
  if (event.data instanceof Blob) {
    const bitmap = await createImageBitmap(event.data); const canvas = $("#processed-canvas");
    const context = canvas.getContext("2d"); context.drawImage(bitmap, 0, 0, canvas.width, canvas.height); bitmap.close();
    state.sending = false; window.setTimeout(sendFrame, 15); return;
  }
  const message = JSON.parse(event.data);
  if (message.type === "ready") { sendFrame(); return; }
  if (message.type === "state") { updateWorkoutState(message); return; }
  if (message.type === "complete") { state.closing = true; showResult(message); }
}

function sendFrame() {
  if (state.sending || !state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  const video = $("#camera-source"); if (!video.videoWidth) return window.setTimeout(sendFrame, 100);
  const canvas = $("#capture-canvas"); const context = canvas.getContext("2d", { alpha: false });
  context.save(); context.translate(canvas.width, 0); context.scale(-1, 1); context.drawImage(video, 0, 0, canvas.width, canvas.height); context.restore();
  canvas.toBlob((blob) => { if (blob && state.socket?.readyState === WebSocket.OPEN) { state.sending = true; state.socket.send(blob); } }, "image/jpeg", .78);
}

function updateWorkoutState(message) {
  $("#live-count").textContent = message.count; $("#live-time").textContent = formatTime(message.elapsed);
  $("#body-state").textContent = message.ready ? "인식됨" : "위치 조정";
  $("#camera-guide").hidden = message.ready;
  const labels = { SEARCHING: "자세 찾는 중", COUNTDOWN: "준비", COUNTING: "측정 중" };
  $("#phase-label").textContent = labels[message.phase] || message.phase;
  const countdown = $("#countdown-overlay"); countdown.hidden = message.phase !== "COUNTDOWN";
  if (!countdown.hidden) countdown.querySelector("span").textContent = Math.max(1, Math.ceil(message.countdown));
}

function stopWorkout() {
  if (state.socket?.readyState === WebSocket.OPEN) { state.closing = true; state.socket.send(JSON.stringify({ type: "stop" })); }
  else cleanupWorkout();
}

function showResult(message) {
  $("#result-count").textContent = message.count; $("#result-detail").textContent = `${modeNames[state.mode]} · ${formatTime(message.duration)}`;
  $("#result-overlay").hidden = false; stopMedia();
}

function stopMedia() { state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null; state.sending = false; }
function cleanupWorkout() { state.closing = true; stopMedia(); if (state.socket && state.socket.readyState < WebSocket.CLOSING) state.socket.close(); state.socket = null; $("#workout-view").hidden = true; $("#result-overlay").hidden = true; loadDashboard(); }

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault(); $("#login-error").textContent = "";
  try {
    const user = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ username: $("#login-username").value, password: $("#login-password").value }) });
    $("#login-form").reset(); showApp(user);
  } catch (error) { $("#login-error").textContent = error.message; }
});
$("#logout-button").addEventListener("click", async () => { try { await api("/api/auth/logout", { method: "POST" }); } finally { showLogin(); } });
$$(`[data-nav]`).forEach((button) => button.addEventListener("click", () => navigate(button.dataset.nav)));
$$(`[data-mode]`).forEach((button) => button.addEventListener("click", () => startWorkout(button.dataset.mode)));
$("#stop-workout").addEventListener("click", stopWorkout); $("#workout-back").addEventListener("click", stopWorkout);
$("#result-close").addEventListener("click", cleanupWorkout);
$$(`[data-size]`).forEach((button) => button.addEventListener("click", () => { $$("[data-size]").forEach((item) => item.classList.toggle("is-active", item === button)); $("#workout-layout").className = `workout-layout size-${button.dataset.size}`; }));
$("#fullscreen-button").addEventListener("click", () => document.fullscreenElement ? document.exitFullscreen() : $("#workout-view").requestFullscreen());
$("#open-create-user").addEventListener("click", () => $("#user-dialog").showModal());
$("#close-user-dialog").addEventListener("click", () => $("#user-dialog").close());
$("#user-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const form = new FormData(event.currentTarget);
  const payload = { username: form.get("username"), display_name: form.get("display_name"), password: form.get("password"), role: "member", can_basic: form.has("can_basic"), can_alternating: form.has("can_alternating"), can_double: form.has("can_double"), can_view_history: form.has("can_view_history") };
  try { await api("/api/admin/users", { method: "POST", body: JSON.stringify(payload) }); event.currentTarget.reset(); $("#user-dialog").close(); showToast("사용자를 추가했습니다."); loadAdmin(); }
  catch (error) { $("#user-form-error").textContent = error.message; }
});
window.addEventListener("beforeunload", stopMedia);
loadSession();
