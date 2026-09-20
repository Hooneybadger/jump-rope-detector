"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {
  user: null, csrf: "", dashboard: null, stream: null, socket: null, sending: false,
  mode: null, duration: 60, workoutId: null, closing: false, frameTimer: null, frameSentAt: 0,
  selectedWorkouts: new Set(), selectedUsers: new Set(),
};
const FRAME_INTERVAL = 1000 / 12;
const poseConnections = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[24,26],[26,28],[28,30],[30,32]];
const modeNames = { basic: "모아뛰기", alternating: "번갈아뛰기", double: "이중뛰기" };
const statusNames = { completed: "완료", interrupted: "중단", running: "측정 중" };
const authHeadings = { login: "다시 시작해 볼까요?", signup: "나만의 기록을 시작하세요.", "reset-request": "비밀번호를 다시 설정하세요.", "reset-confirm": "새 비밀번호를 정하세요." };

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
  window.setTimeout(() => toast.classList.remove("is-visible"), 2800);
}

function showAuthPanel(name) {
  $$(`[data-auth-panel]`).forEach((panel) => { panel.hidden = panel.dataset.authPanel !== name; });
  $("#auth-heading").textContent = authHeadings[name];
}

function showApp(user) {
  state.user = user; state.csrf = user.csrfToken || state.csrf;
  $("#login-view").hidden = true; $("#app-shell").hidden = false;
  $("#user-badge").textContent = (user.displayName || user.username).slice(0, 1);
  $("#user-name").textContent = user.displayName || user.username;
  $("#admin-tab").hidden = user.role !== "admin";
  $("#welcome-heading").innerHTML = `${escapeHtml(user.displayName)}님,<br>오늘도 뛰어볼까요?`;
  for (const mode of Object.keys(modeNames)) {
    const card = document.querySelector(`[data-mode-card="${mode}"]`); const allowed = Boolean(user.permissions[mode]);
    card.classList.toggle("is-locked", !allowed); card.querySelector(".locked-copy").hidden = allowed;
  }
  loadDashboard();
}

function escapeHtml(value) {
  const span = document.createElement("span"); span.textContent = value || ""; return span.innerHTML;
}

function showLogin() {
  state.user = null; state.csrf = ""; $("#login-view").hidden = false; $("#app-shell").hidden = true; $("#workout-view").hidden = true;
  showAuthPanel("login");
}

async function loadSession() {
  const resetToken = new URLSearchParams(location.search).get("reset");
  if (resetToken) { $("#reset-token").value = resetToken; showAuthPanel("reset-confirm"); }
  try { showApp(await api("/api/auth/me")); } catch { if (!resetToken) showLogin(); }
}

function formatTime(total) {
  const value = Math.max(0, Math.round(total || 0));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function localDate(value) { return new Intl.DateTimeFormat("ko-KR", { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value)); }

function actionButton(action, label, id, className = "") {
  const button = document.createElement("button"); button.type = "button"; button.dataset.action = action; button.dataset.id = id;
  button.className = `record-action ${className}`.trim(); button.textContent = label; return button;
}

async function openRecord(id) {
  try {
    const item = await api(`/api/workouts/${id}`);
    $("#record-title").textContent = `${modeNames[item.mode] || item.mode} 기록`;
    $("#record-user").textContent = item.user; $("#record-count").textContent = `${item.count.toLocaleString("ko-KR")}회`;
    $("#record-duration").textContent = formatTime(item.duration); $("#record-status").textContent = statusNames[item.status] || item.status;
    $("#record-started").textContent = localDate(item.startedAt); $("#record-download").href = `/api/workouts/${item.id}/pdf`;
    $("#record-dialog").showModal();
  } catch (error) { showToast(error.message); }
}

async function deleteRecord(id) {
  if (!await askConfirmation("이 측정 기록을 삭제할까요? 삭제한 기록은 복구할 수 없습니다.")) return;
  try { await api(`/api/workouts/${id}`, { method: "DELETE" }); showToast("측정 기록을 삭제했습니다."); await loadDashboard(); }
  catch (error) { showToast(error.message); }
}

function updateHistorySelection() {
  const checkboxes = $$(`[data-record-select]:not(:disabled)`);
  const selected = checkboxes.filter((item) => item.checked);
  state.selectedWorkouts = new Set(selected.map((item) => Number(item.value)));
  const selectAll = $("#history-select-all");
  selectAll.disabled = checkboxes.length === 0;
  selectAll.checked = checkboxes.length > 0 && selected.length === checkboxes.length;
  selectAll.indeterminate = selected.length > 0 && selected.length < checkboxes.length;
  $("#history-selection-count").textContent = selected.length ? `${selected.length}개 선택` : "선택 없음";
  $("#delete-selected-records").disabled = selected.length === 0;
}

async function deleteSelectedRecords() {
  const ids = [...state.selectedWorkouts];
  if (!ids.length || !await askConfirmation(`선택한 측정 기록 ${ids.length}개를 삭제할까요? 삭제 후에는 복구할 수 없습니다.`)) return;
  const button = $("#delete-selected-records"); button.disabled = true; button.textContent = "삭제 중…";
  try {
    const result = await api("/api/workouts/bulk-delete", { method: "POST", body: JSON.stringify({ ids }) });
    showToast(`측정 기록 ${result.deleted}개를 삭제했습니다.`); await loadDashboard();
  } catch (error) { showToast(error.message); updateHistorySelection(); }
  finally { button.textContent = "선택 기록 삭제"; }
}

async function loadDashboard() {
  try {
    const data = await api("/api/dashboard"); state.dashboard = data;
    $("#summary-count").textContent = data.summary.count.toLocaleString("ko-KR");
    $("#summary-sessions").textContent = data.summary.sessions.toLocaleString("ko-KR");
    $("#summary-duration").textContent = `${Math.round(data.summary.duration / 60)}분`;
    const body = $("#history-body"); body.replaceChildren(); $("#history-empty").hidden = data.recent.length > 0;
    state.selectedWorkouts.clear();
    data.recent.forEach((item) => {
      const row = document.createElement("tr");
      const selectCell = document.createElement("td"); selectCell.className = "check-cell";
      const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.value = item.id; checkbox.dataset.recordSelect = "";
      checkbox.disabled = item.status === "running"; checkbox.setAttribute("aria-label", `${modeNames[item.mode] || item.mode} 기록 선택`);
      checkbox.addEventListener("change", updateHistorySelection); selectCell.appendChild(checkbox); row.appendChild(selectCell);
      [item.user, modeNames[item.mode] || item.mode, `${item.count.toLocaleString("ko-KR")}회`, formatTime(item.duration), statusNames[item.status] || item.status, localDate(item.startedAt)].forEach((text, index) => { const cell = document.createElement("td"); cell.textContent = text; if (index === 4) cell.className = `status status-${item.status}`; row.appendChild(cell); });
      const actions = document.createElement("td"); actions.className = "record-actions";
      const download = document.createElement("a"); download.className = "record-action"; download.href = `/api/workouts/${item.id}/pdf`; download.textContent = "PDF"; download.setAttribute("download", "");
      actions.append(actionButton("view", "상세", item.id), download, actionButton("delete", "삭제", item.id, "is-danger")); row.appendChild(actions);
      body.appendChild(row);
    });
    updateHistorySelection();
  } catch (error) { showToast(error.message); }
}

function navigate(target) {
  $$(".page-view").forEach((view) => { view.hidden = view.id !== `${target}-view`; });
  $$(".nav-tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.nav === target));
  if (target === "admin") loadAdmin(); else loadDashboard();
}

function permissionToggle(user, key, label, apiField = key) {
  const wrapper = document.createElement("label"); const checkbox = document.createElement("input");
  checkbox.type = "checkbox"; checkbox.checked = Boolean(user.permissions[key]); checkbox.disabled = user.role === "admin";
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
    const list = $("#user-list"); list.replaceChildren(); state.selectedUsers.clear();
    users.forEach((user) => {
      const row = document.createElement("div"); row.className = "user-row"; const identity = document.createElement("div");
      const selector = document.createElement("input"); selector.type = "checkbox"; selector.value = user.id; selector.dataset.userSelect = ""; selector.className = "user-select";
      selector.disabled = user.id === state.user.id; selector.setAttribute("aria-label", `${user.displayName} 사용자 선택`); selector.addEventListener("change", updateUserSelection);
      const name = document.createElement("strong"); name.textContent = user.displayName; const meta = document.createElement("small");
      meta.textContent = `${user.username}${user.email ? ` · ${user.email}` : ""} · ${user.role === "admin" ? "관리자" : "사용자"}${user.active ? "" : " · 비활성"}`; identity.append(name, meta);
      const toggles = document.createElement("div"); toggles.className = "permission-toggles";
      toggles.append(permissionToggle(user, "basic", "모아"), permissionToggle(user, "alternating", "교대"), permissionToggle(user, "double", "이중"), permissionToggle(user, "history", "기록", "view_history"));
      row.append(selector, identity, toggles); list.appendChild(row);
    });
    updateUserSelection();
    const audits = $("#audit-list"); audits.replaceChildren();
    logs.slice(0, 16).forEach((log) => { const item = document.createElement("div"); item.className = "audit-item"; const action = document.createElement("strong"); action.textContent = log.action; const info = document.createElement("span"); info.textContent = `${log.actor} · ${localDate(log.createdAt)}`; item.append(action, info); audits.appendChild(item); });
  } catch (error) { showToast(error.message); }
}

function updateUserSelection() {
  const checkboxes = $$(`[data-user-select]:not(:disabled)`);
  const selected = checkboxes.filter((item) => item.checked);
  state.selectedUsers = new Set(selected.map((item) => Number(item.value)));
  const selectAll = $("#user-select-all");
  selectAll.disabled = checkboxes.length === 0;
  selectAll.checked = checkboxes.length > 0 && selected.length === checkboxes.length;
  selectAll.indeterminate = selected.length > 0 && selected.length < checkboxes.length;
  $("#delete-selected-users").disabled = selected.length === 0;
}

async function deleteSelectedUsers() {
  const ids = [...state.selectedUsers];
  if (!ids.length || !await askConfirmation(`선택한 사용자 ${ids.length}명을 삭제할까요? 사용자의 측정 기록도 함께 삭제됩니다.`)) return;
  const button = $("#delete-selected-users"); button.disabled = true; button.textContent = "삭제 중…";
  try {
    const result = await api("/api/admin/users/bulk-delete", { method: "POST", body: JSON.stringify({ ids }) });
    showToast(`사용자 ${result.deleted}명을 삭제했습니다.`); await loadAdmin();
  } catch (error) { showToast(error.message); updateUserSelection(); }
  finally { button.textContent = "선택 사용자 삭제"; }
}

function openProfile() {
  const user = state.user; if (!user) return;
  const name = user.displayName || user.username;
  $("#profile-avatar").textContent = name.slice(0, 1); $("#profile-name").textContent = name;
  $("#profile-role").textContent = user.role === "admin" ? "관리자" : "회원";
  $("#profile-username").textContent = user.username; $("#profile-email").textContent = user.email || "등록된 이메일 없음";
  const permissions = $("#profile-permissions"); permissions.replaceChildren();
  Object.entries(modeNames).forEach(([key, label]) => { if (user.permissions[key]) { const item = document.createElement("span"); item.textContent = label; permissions.appendChild(item); } });
  $("#profile-dialog").showModal();
}

let confirmResolver = null;
function askConfirmation(message) {
  $("#confirm-message").textContent = message; $("#confirm-dialog").showModal();
  return new Promise((resolve) => { confirmResolver = resolve; });
}

function resolveConfirmation(value) {
  $("#confirm-dialog").close(); const resolve = confirmResolver; confirmResolver = null; if (resolve) resolve(value);
}

function openWorkoutSetup(mode) {
  if (!state.user.permissions[mode]) return;
  state.mode = mode; $("#setup-mode-name").textContent = modeNames[mode]; $("#setup-error").textContent = "";
  $("#workout-setup-dialog").showModal();
}

async function startWorkout() {
  state.closing = false; state.workoutId = null;
  $("#workout-mode-name").textContent = modeNames[state.mode]; $("#workout-view").hidden = false; $("#result-overlay").hidden = true;
  $("#live-count").textContent = "0"; $("#live-time").textContent = "00:00"; $("#target-time").textContent = formatTime(state.duration); $("#remaining-time").textContent = formatTime(state.duration); $("#time-progress").style.width = "100%";
  $("#connection-state").textContent = "카메라 연결 중"; $("#camera-guide").hidden = false; $("#phase-label").textContent = "자세 찾는 중";
  try { if (!document.fullscreenElement && $("#workout-view").requestFullscreen) await $("#workout-view").requestFullscreen(); } catch { showToast("브라우저 메뉴에서도 전체 화면을 켤 수 있습니다."); }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 }, facingMode: "user" }, audio: false });
    const video = $("#camera-source"); video.srcObject = state.stream; await video.play();
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    state.socket = new WebSocket(`${protocol}://${location.host}/ws/count/${state.mode}?duration=${state.duration}`);
    state.socket.onopen = () => { $("#connection-state").textContent = "실시간 분석 연결됨"; };
    state.socket.onmessage = handleSocketMessage;
    state.socket.onclose = (event) => { state.sending = false; if (!state.closing && event.code !== 1000) { showToast(event.reason || "측정 연결이 종료되었습니다."); cleanupWorkout(); } };
    state.socket.onerror = () => showToast("분석 서버에 연결할 수 없습니다.");
  } catch (error) { showToast(error.name === "NotAllowedError" ? "카메라 사용 권한을 허용해 주세요." : "카메라를 시작할 수 없습니다."); cleanupWorkout(); }
}

async function handleSocketMessage(event) {
  const message = JSON.parse(event.data);
  if (message.type === "ready") { state.workoutId = message.workoutId; scheduleFrame(); return; }
  if (message.type === "state") {
    const roundTrip = Math.max(0, performance.now() - state.frameSentAt); state.sending = false;
    updateWorkoutState(message); drawPose(message.landmarks || []);
    $("#connection-state").textContent = `분석 ${Math.round(message.processingMs || 0)}ms · 왕복 ${Math.round(roundTrip)}ms`;
    if (!(message.phase === "COUNTING" && message.elapsed >= state.duration)) scheduleFrame(Math.max(0, FRAME_INTERVAL - roundTrip));
    return;
  }
  if (message.type === "complete") { state.closing = true; state.workoutId = message.workoutId; showResult(message); }
}

function scheduleFrame(delay = 0) {
  window.clearTimeout(state.frameTimer); state.frameTimer = window.setTimeout(sendFrame, delay);
}

function sendFrame() {
  if (state.sending || !state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  const video = $("#camera-source"); if (!video.videoWidth) return scheduleFrame(100);
  const canvas = $("#capture-canvas"); const context = canvas.getContext("2d", { alpha: false });
  context.drawImage(video, 0, 0, canvas.width, canvas.height); state.sending = true;
  canvas.toBlob((blob) => {
    if (blob && state.socket?.readyState === WebSocket.OPEN) {
      try { state.frameSentAt = performance.now(); state.socket.send(blob); }
      catch { state.sending = false; scheduleFrame(FRAME_INTERVAL); }
    }
    else { state.sending = false; scheduleFrame(FRAME_INTERVAL); }
  }, "image/jpeg", .62);
}

function drawPose(landmarks) {
  const canvas = $("#processed-canvas"); const context = canvas.getContext("2d"); context.clearRect(0, 0, canvas.width, canvas.height);
  if (landmarks.length < 33) return;
  context.lineWidth = 5; context.lineCap = "round"; context.lineJoin = "round"; context.strokeStyle = "rgba(200,255,61,.92)";
  poseConnections.forEach(([from, to]) => {
    const a = landmarks[from]; const b = landmarks[to]; if (!a || !b || a[2] < .45 || b[2] < .45) return;
    context.beginPath(); context.moveTo(a[0] * canvas.width, a[1] * canvas.height); context.lineTo(b[0] * canvas.width, b[1] * canvas.height); context.stroke();
  });
  context.fillStyle = "#ffffff";
  landmarks.forEach((point, index) => { if (index < 11 || point[2] < .55) return; context.beginPath(); context.arc(point[0] * canvas.width, point[1] * canvas.height, 5, 0, Math.PI * 2); context.fill(); });
}

function updateWorkoutState(message) {
  $("#live-count").textContent = message.count; $("#live-time").textContent = formatTime(message.elapsed);
  const remaining = Math.max(0, state.duration - message.elapsed); $("#remaining-time").textContent = formatTime(remaining);
  $("#time-progress").style.width = `${Math.max(0, (remaining / state.duration) * 100)}%`;
  $("#body-state").textContent = message.ready ? "인식됨" : "위치 조정"; $("#camera-guide").hidden = message.ready;
  const labels = { SEARCHING: "자세 찾는 중", COUNTDOWN: "준비", COUNTING: "측정 중" }; $("#phase-label").textContent = labels[message.phase] || message.phase;
  const countdown = $("#countdown-overlay"); countdown.hidden = message.phase !== "COUNTDOWN"; if (!countdown.hidden) countdown.querySelector("span").textContent = Math.max(1, Math.ceil(message.countdown));
}

function stopWorkout() { if (state.socket?.readyState === WebSocket.OPEN) { state.closing = true; state.socket.send(JSON.stringify({ type: "stop" })); } else cleanupWorkout(); }

function showResult(message) {
  $("#result-count").textContent = message.count; $("#result-detail").textContent = `${modeNames[state.mode]} · ${formatTime(message.duration)}`;
  $("#result-overlay").hidden = false; stopMedia();
}

function stopMedia() {
  window.clearTimeout(state.frameTimer); state.frameTimer = null; state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null; state.sending = false;
  const canvas = $("#processed-canvas"); canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}
async function cleanupWorkout() {
  state.closing = true; stopMedia(); if (state.socket && state.socket.readyState < WebSocket.CLOSING) state.socket.close(); state.socket = null;
  $("#workout-view").hidden = true; $("#result-overlay").hidden = true; if (document.fullscreenElement) await document.exitFullscreen().catch(() => {}); loadDashboard();
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const formElement = event.currentTarget; $("#login-error").textContent = "";
  try { const user = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ username: $("#login-username").value, password: $("#login-password").value }) }); formElement.reset(); showApp(user); }
  catch (error) { $("#login-error").textContent = error.message; }
});

$("#signup-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const formElement = event.currentTarget; const form = new FormData(formElement); $("#signup-error").textContent = "";
  if (form.get("password") !== form.get("password_confirm")) { $("#signup-error").textContent = "비밀번호가 서로 다릅니다."; return; }
  try { const user = await api("/api/auth/signup", { method: "POST", body: JSON.stringify({ username: form.get("username"), email: form.get("email"), display_name: form.get("display_name"), password: form.get("password") }) }); formElement.reset(); showApp(user); }
  catch (error) { $("#signup-error").textContent = error.message; }
});

$("#reset-request-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const form = new FormData(event.currentTarget); $("#reset-request-error").textContent = ""; $("#reset-request-message").textContent = "";
  try { const result = await api("/api/auth/password-reset/request", { method: "POST", body: JSON.stringify({ email: form.get("email") }) }); $("#reset-request-message").textContent = result.message; if (result.developmentToken) { $("#reset-token").value = result.developmentToken; showAuthPanel("reset-confirm"); } }
  catch (error) { $("#reset-request-error").textContent = error.message; }
});

$("#reset-confirm-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const form = new FormData(event.currentTarget); $("#reset-confirm-error").textContent = "";
  if (form.get("password") !== form.get("password_confirm")) { $("#reset-confirm-error").textContent = "비밀번호가 서로 다릅니다."; return; }
  try { await api("/api/auth/password-reset/confirm", { method: "POST", body: JSON.stringify({ token: form.get("token"), password: form.get("password") }) }); history.replaceState({}, "", location.pathname); showAuthPanel("login"); showToast("비밀번호를 변경했습니다. 새 비밀번호로 로그인하세요."); }
  catch (error) { $("#reset-confirm-error").textContent = error.message; }
});

$$(`[data-auth-target]`).forEach((button) => button.addEventListener("click", () => showAuthPanel(button.dataset.authTarget)));
$("#logout-button").addEventListener("click", async () => { try { await api("/api/auth/logout", { method: "POST" }); } finally { showLogin(); } });
$$(`[data-nav]`).forEach((button) => button.addEventListener("click", () => navigate(button.dataset.nav)));
$$(`[data-mode]`).forEach((button) => button.addEventListener("click", () => openWorkoutSetup(button.dataset.mode)));
$$(`[data-duration]`).forEach((button) => button.addEventListener("click", () => { $("#workout-duration").value = button.dataset.duration; $$(`[data-duration]`).forEach((item) => item.classList.toggle("is-active", item === button)); }));
$("#workout-duration").addEventListener("input", () => $$(`[data-duration]`).forEach((button) => button.classList.toggle("is-active", button.dataset.duration === $("#workout-duration").value)));
$("#close-setup-dialog").addEventListener("click", () => $("#workout-setup-dialog").close());
$("#workout-setup-form").addEventListener("submit", (event) => { event.preventDefault(); const duration = Number($("#workout-duration").value); if (!Number.isInteger(duration) || duration < 10 || duration > 3600) { $("#setup-error").textContent = "10초부터 3,600초 사이로 입력해 주세요."; return; } state.duration = duration; $("#workout-setup-dialog").close(); startWorkout(); });
$("#stop-workout").addEventListener("click", stopWorkout); $("#workout-back").addEventListener("click", stopWorkout);
$("#result-close").addEventListener("click", async () => { await cleanupWorkout(); const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches; window.scrollTo({ top: $(".record-section").offsetTop - 90, behavior: reducedMotion ? "auto" : "smooth" }); });
$("#result-download").addEventListener("click", () => { if (state.workoutId) window.location.assign(`/api/workouts/${state.workoutId}/pdf`); });
$("#fullscreen-button").addEventListener("click", () => document.fullscreenElement ? document.exitFullscreen() : $("#workout-view").requestFullscreen());
$("#history-body").addEventListener("click", (event) => { const target = event.target.closest("button[data-action]"); if (!target) return; if (target.dataset.action === "view") openRecord(target.dataset.id); if (target.dataset.action === "delete") deleteRecord(target.dataset.id); });
$("#history-select-all").addEventListener("change", (event) => { $$(`[data-record-select]:not(:disabled)`).forEach((item) => { item.checked = event.currentTarget.checked; }); updateHistorySelection(); });
$("#delete-selected-records").addEventListener("click", deleteSelectedRecords);
$("#close-record-dialog").addEventListener("click", () => $("#record-dialog").close());
$("#profile-button").addEventListener("click", openProfile); $("#close-profile-dialog").addEventListener("click", () => $("#profile-dialog").close());
$("#confirm-cancel").addEventListener("click", () => resolveConfirmation(false)); $("#confirm-accept").addEventListener("click", () => resolveConfirmation(true));
$("#confirm-dialog").addEventListener("cancel", (event) => { event.preventDefault(); resolveConfirmation(false); });

$("#open-create-user").addEventListener("click", () => $("#user-dialog").showModal()); $("#close-user-dialog").addEventListener("click", () => $("#user-dialog").close());
$("#user-select-all").addEventListener("change", (event) => { $$(`[data-user-select]:not(:disabled)`).forEach((item) => { item.checked = event.currentTarget.checked; }); updateUserSelection(); });
$("#delete-selected-users").addEventListener("click", deleteSelectedUsers);
$("#user-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const formElement = event.currentTarget; const form = new FormData(formElement);
  const payload = { username: form.get("username"), email: form.get("email") || null, display_name: form.get("display_name"), password: form.get("password"), role: "member", can_basic: form.has("can_basic"), can_alternating: form.has("can_alternating"), can_double: form.has("can_double"), can_view_history: form.has("can_view_history") };
  try { await api("/api/admin/users", { method: "POST", body: JSON.stringify(payload) }); formElement.reset(); $("#user-dialog").close(); showToast("사용자를 추가했습니다."); loadAdmin(); }
  catch (error) { $("#user-form-error").textContent = error.message; }
});

window.addEventListener("beforeunload", stopMedia); loadSession();
