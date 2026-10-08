"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {
  user: null, csrf: "", dashboard: null, stream: null, socket: null, sending: false,
  mode: null, duration: 60, countdown: 3, workoutId: null, closing: false, frameTimer: null, frameSentAt: 0,
  frameInterval: 1000 / 20, phase: "SEARCHING", lastCount: 0, lastElapsed: 0, timeline: [], resultShown: false,
  selectedWorkouts: new Set(), selectedUsers: new Set(),
};
const captureCanvas = $("#capture-canvas");
const captureContext = captureCanvas.getContext("2d", { alpha: false });
const poseCanvas = $("#processed-canvas");
const poseContext = poseCanvas.getContext("2d");
const poseConnections = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[24,26],[26,28],[28,30],[30,32]];
const modeNames = { basic: "모아뛰기", alternating: "번갈아뛰기", double: "이중뛰기" };
const cheerAssets = {
  basic: {
    animation: "/cheer-basic.gif", poster: "/cheer-basic-poster.png",
    alt: "모아뛰기를 하는 남성 전신 애니메이션", caption: "두 발의 가벼운 리듬을 유지해요.",
  },
  alternating: {
    animation: "/cheer-alternating.gif", poster: "/cheer-alternating-poster.png",
    alt: "번갈아뛰기를 하는 남성 전신 애니메이션", caption: "좌우 발을 고르게 바꾸며 리듬을 이어가요.",
  },
  double: {
    animation: "/cheer-double.gif", poster: "/cheer-double-poster.png",
    alt: "이중뛰기를 하는 남성 전신 애니메이션", caption: "손목은 빠르게, 착지는 가볍게 이어가요.",
  },
};
const statusNames = { completed: "완료", interrupted: "중단", running: "측정 중" };
const sexNames = { male: "남성", female: "여성" };
const AVATAR_SOURCE_LIMIT = 20 * 1024 * 1024;
const phaseLabels = { SEARCHING: "자세 찾는 중", COUNTDOWN: "준비", COUNTING: "측정 중", DONE: "측정 종료" };
const phaseOrder = ["SEARCHING", "COUNTDOWN", "COUNTING", "DONE"];
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const authHeadings = { login: "다시 시작해 볼까요?", signup: "나만의 기록을 시작하세요.", "reset-request": "비밀번호를 다시 설정하세요.", "reset-confirm": "새 비밀번호를 정하세요." };
let dashboardEntered = false;

function playDashboardEntry() {
  if (dashboardEntered || !window.gsap || reducedMotion()) return;
  dashboardEntered = true;
  window.gsap.from([".arena-header", ".lanes-section .section-title", ".lane"], {
    opacity: 0, y: 18, duration: 0.7, stagger: 0.07, ease: "power3.out", clearProps: "transform,opacity",
  });
}

function bindLanePreview() {
  $$(".lane").forEach((lane) => {
    const image = lane.querySelector(".lane-figure img");
    const play = () => { if (!reducedMotion() && !lane.classList.contains("is-locked")) image.src = image.dataset.motion; };
    const rest = () => { image.src = image.dataset.poster; };
    lane.addEventListener("pointerenter", play); lane.addEventListener("pointerleave", rest);
    lane.addEventListener("focusin", play); lane.addEventListener("focusout", rest);
  });
}

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
  renderAvatar($("#user-badge"), user);
  $("#user-name").textContent = user.displayName || user.username;
  $("#admin-tab").hidden = user.role !== "admin";
  $("#welcome-heading").innerHTML = `${escapeHtml(user.displayName || user.username)}님,<br>오늘 기록을 시작해 볼까요?`;
  for (const mode of Object.keys(modeNames)) {
    const card = document.querySelector(`[data-mode-card="${mode}"]`); const allowed = Boolean(user.permissions[mode]);
    card.classList.toggle("is-locked", !allowed); card.querySelector(".locked-copy").hidden = allowed;
  }
  renderBodyProfileCta(user);
  window.requestAnimationFrame(playDashboardEntry);
  loadDashboard();
  if (user.profile && user.profile.status === "pending" && !$("#body-profile-dialog").open) openBodyProfile(true);
}

function renderAvatar(target, user) {
  if (user.avatarUrl) {
    const image = document.createElement("img"); image.src = user.avatarUrl; image.alt = ""; target.replaceChildren(image);
  } else {
    target.replaceChildren(document.createTextNode((user.displayName || user.username).slice(0, 1)));
  }
}

function durationLabel(seconds) {
  const minutes = Math.floor(seconds / 60); const rest = seconds % 60;
  return [minutes ? `${minutes}분` : "", rest || !minutes ? `${rest}초` : ""].filter(Boolean).join(" ");
}

function renderBodyProfileCta(user) {
  const profile = user.profile || {}; const advice = profile.recommendation;
  $("#open-body-profile span").textContent = profile.status === "completed" ? "맞춤 프로필 설정" : "맞춤 프로필 만들기";
  $("#body-profile-note").textContent = advice
    ? `추천 측정 시간 ${durationLabel(advice.seconds)} · 하루 권장 운동량 기준`
    : profile.status === "completed"
      ? "나이, 키, 몸무게를 입력하면 하루 권장 운동량에 맞춘 측정 시간을 추천해 드려요."
      : "프로필을 만들면 하루 권장 운동량에 맞춘 측정 시간을 추천해 드려요.";
}

let bodyProfileFirstRun = false;
function openBodyProfile(firstRun = false) {
  bodyProfileFirstRun = firstRun;
  const profile = state.user.profile || {}; const completed = profile.status === "completed";
  $("#body-profile-title").textContent = completed ? "맞춤 프로필 설정" : "맞춤 프로필 만들기";
  $("#save-body-profile").textContent = completed ? "맞춤 프로필 저장" : "맞춤 프로필 생성";
  $("#skip-body-profile").textContent = firstRun ? "건너뛰기" : "취소";
  // A saved profile with no sex and no body data was saved as "선택 안 함"; a new profile starts unselected.
  const hasBodyData = [profile.age, profile.heightCm, profile.weightKg].some((value) => value != null);
  const chosenSex = profile.sex || (completed && !hasBodyData ? "" : null);
  $$('#body-profile-form input[name="sex"]').forEach((input) => { input.checked = input.value === chosenSex; });
  $("#body-age").value = profile.age ?? ""; $("#body-height").value = profile.heightCm ?? ""; $("#body-weight").value = profile.weightKg ?? "";
  syncBodyFields();
  $("#body-profile-error").textContent = "";
  $("#body-profile-dialog").showModal();
}

const bodyInputs = () => [$("#body-age"), $("#body-height"), $("#body-weight")];

function syncBodyFields() {
  const declined = Boolean($('#body-profile-form input[name="sex"][value=""]:checked'));
  bodyInputs().forEach((input) => { input.disabled = declined; if (declined) input.value = ""; });
  $("#body-fields-hint").hidden = !declined;
}

function readOptionalNumber(input, label, integer) {
  const raw = input.value.trim(); const min = Number(input.min); const max = Number(input.max);
  const message = `${label}는 ${min}~${max} 사이${integer ? "의 정수" : ""}로 입력해 주세요.`;
  if (input.validity.badInput) throw new Error(message);
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) throw new Error(message);
  return value;
}

async function saveBodyProfile(event) {
  event.preventDefault();
  const error = $("#body-profile-error"); const button = $("#save-body-profile"); error.textContent = "";
  let payload;
  try {
    payload = {
      sex: new FormData(event.currentTarget).get("sex") || null,
      age: readOptionalNumber($("#body-age"), "나이", true),
      height_cm: readOptionalNumber($("#body-height"), "키", false),
      weight_kg: readOptionalNumber($("#body-weight"), "몸무게", false),
    };
  } catch (inputError) { error.textContent = inputError.message; return; }
  const label = button.textContent; button.disabled = true; button.textContent = "저장 중…";
  try {
    const user = await api("/api/auth/body-profile", { method: "PUT", body: JSON.stringify(payload) });
    $("#body-profile-dialog").close(); showApp(user);
    const advice = user.profile.recommendation;
    showToast(advice ? `맞춤 프로필을 저장했어요. 추천 측정 시간은 ${durationLabel(advice.seconds)}입니다.` : "맞춤 프로필을 저장했어요.");
  } catch (requestError) { error.textContent = requestError.message; }
  finally { button.disabled = false; button.textContent = label; }
}

async function skipBodyProfile() {
  const dialog = $("#body-profile-dialog");
  if (!bodyProfileFirstRun) { dialog.close(); return; }
  try {
    const user = await api("/api/auth/body-profile/skip", { method: "POST" });
    dialog.close(); showApp(user); showToast("훈련 화면의 '맞춤 프로필 만들기'에서 언제든 다시 만들 수 있어요.");
  } catch (requestError) { $("#body-profile-error").textContent = requestError.message; }
}

function renderBodySummary(user) {
  const profile = user.profile || {}; const advice = profile.recommendation;
  const values = {
    "#profile-sex": sexNames[profile.sex], "#profile-age": profile.age != null ? `${profile.age}세` : null,
    "#profile-height": profile.heightCm != null ? `${profile.heightCm}cm` : null, "#profile-weight": profile.weightKg != null ? `${profile.weightKg}kg` : null,
  };
  Object.entries(values).forEach(([selector, value]) => { const cell = $(selector); cell.textContent = value || "미입력"; cell.classList.toggle("is-empty", !value); });
  $("#edit-body-profile span").textContent = profile.status === "completed" ? "맞춤 프로필 설정" : "맞춤 프로필 만들기";
  $("#profile-recommend").textContent = advice
    ? `추천 측정 시간 ${durationLabel(advice.seconds)} · ${advice.reason}`
    : profile.status === "completed"
      ? "나이, 키, 몸무게를 입력하면 추천 측정 시간과 내 몸에 맞춘 칼로리를 볼 수 있어요."
      : "맞춤 프로필을 만들면 추천 측정 시간과 내 몸에 맞춘 칼로리를 볼 수 있어요.";
}

function applyAvatar(user) {
  state.user = user; renderAvatar($("#user-badge"), user); renderAvatar($("#profile-avatar"), user); $("#remove-avatar").hidden = !user.avatarUrl;
}

async function squareAvatar(file) {
  const bitmap = await createImageBitmap(file);
  const side = Math.min(bitmap.width, bitmap.height); const size = Math.min(256, side);
  const canvas = document.createElement("canvas"); canvas.width = size; canvas.height = size;
  const context = canvas.getContext("2d"); context.fillStyle = "#ffffff"; context.fillRect(0, 0, size, size);
  context.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, size, size);
  bitmap.close();
  const webp = await new Promise((resolve) => canvas.toBlob(resolve, "image/webp", 0.86));
  if (webp && webp.type === "image/webp") return webp;
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.88));
}

async function changeAvatar(event) {
  const input = event.currentTarget; const file = input.files && input.files[0]; input.value = "";
  const error = $("#avatar-error"); error.textContent = "";
  if (!file) return;
  if (!file.type.startsWith("image/")) { error.textContent = "사진 파일만 올릴 수 있습니다."; return; }
  if (file.size > AVATAR_SOURCE_LIMIT) { error.textContent = "20MB 이하의 사진을 골라 주세요."; return; }
  const button = $("#change-avatar"); button.disabled = true;
  try {
    let blob;
    try { blob = await squareAvatar(file); } catch { throw new Error("이 사진은 브라우저에서 읽을 수 없습니다. JPG나 PNG 사진을 골라 주세요."); }
    const response = await fetch("/api/auth/avatar", { method: "PUT", credentials: "same-origin", headers: { "Content-Type": blob.type, "X-CSRF-Token": state.csrf }, body: blob });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || "사진을 저장하지 못했습니다.");
    applyAvatar(body); showToast("프로필 사진을 바꿨습니다.");
  } catch (uploadError) { error.textContent = uploadError.message; }
  finally { button.disabled = false; }
}

async function removeAvatar() {
  $("#avatar-error").textContent = "";
  try { applyAvatar(await api("/api/auth/avatar", { method: "DELETE" })); showToast("프로필 사진을 삭제했습니다."); }
  catch (requestError) { $("#avatar-error").textContent = requestError.message; }
}

function escapeHtml(value) {
  const span = document.createElement("span"); span.textContent = value || ""; return span.innerHTML;
}

function showLogin() {
  state.user = null; state.csrf = ""; $("#login-view").hidden = false; $("#app-shell").hidden = true; $("#workout-view").hidden = true;
  $$("dialog[open]").forEach((dialog) => dialog.close());
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

// 34 -> 34초, 120 -> 2분, 85 -> 1분 25초
function renderDurationUnits(target, total) {
  const value = Math.max(0, Math.round(total || 0));
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  const parts = [];
  if (minutes) parts.push([minutes, "분"]);
  if (seconds || !minutes) parts.push([seconds, "초"]);
  target.replaceChildren(...parts.map(([number, unit]) => {
    const part = document.createElement("span");
    part.className = "duration-part";
    const digits = document.createElement("span");
    digits.className = "digits";
    digits.textContent = number.toLocaleString("ko-KR");
    const label = document.createElement("small");
    label.textContent = unit;
    part.append(digits, label);
    return part;
  }));
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
    renderDurationUnits($("#summary-duration"), data.summary.duration);
    const body = $("#history-body"); body.replaceChildren(); $("#history-empty").hidden = data.recent.length > 0;
    state.selectedWorkouts.clear();
    data.recent.forEach((item) => {
      const row = document.createElement("tr");
      const selectCell = document.createElement("td"); selectCell.className = "check-cell";
      const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.value = item.id; checkbox.dataset.recordSelect = "";
      checkbox.disabled = item.status === "running"; checkbox.setAttribute("aria-label", `${modeNames[item.mode] || item.mode} 기록 선택`);
      checkbox.addEventListener("change", updateHistorySelection); selectCell.appendChild(checkbox); row.appendChild(selectCell);
      [item.user, modeNames[item.mode] || item.mode, `${item.count.toLocaleString("ko-KR")}회`, formatTime(item.duration), statusNames[item.status] || item.status, localDate(item.startedAt)].forEach((text, index) => { const cell = document.createElement("td"); cell.textContent = text; if (index === 2 || index === 3) cell.className = "num"; if (index === 4) cell.className = `status status-${item.status}`; row.appendChild(cell); });
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
  renderAvatar($("#profile-avatar"), user); $("#remove-avatar").hidden = !user.avatarUrl; $("#avatar-error").textContent = "";
  $("#profile-name").textContent = name; renderBodySummary(user);
  $("#profile-role").textContent = user.role === "admin" ? "관리자" : "회원";
  $("#profile-username").value = user.username; $("#profile-display-name").value = user.displayName || ""; $("#profile-email").value = user.email || "";
  $("#profile-form-error").textContent = ""; $("#profile-form").querySelectorAll('input[type="password"]').forEach((input) => { input.value = ""; });
  $("#profile-form").querySelector("details").open = false;
  const permissions = $("#profile-permissions"); permissions.replaceChildren();
  Object.entries(modeNames).forEach(([key, label]) => { if (user.permissions[key]) { const item = document.createElement("span"); item.textContent = label; permissions.appendChild(item); } });
  $("#profile-dialog").showModal();
}

async function saveProfile(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget); const error = $("#profile-form-error"); const button = $("#save-profile");
  const newPassword = String(form.get("new_password") || ""); const confirmation = String(form.get("new_password_confirm") || "");
  error.textContent = "";
  if (newPassword !== confirmation) { error.textContent = "새 비밀번호가 서로 다릅니다."; return; }
  const payload = {
    display_name: String(form.get("display_name") || "").trim(),
    email: String(form.get("email") || "").trim() || null,
    current_password: String(form.get("current_password") || "") || null,
    new_password: newPassword || null,
  };
  button.disabled = true; button.textContent = "저장 중…";
  try {
    const user = await api("/api/auth/profile", { method: "PATCH", body: JSON.stringify(payload) });
    $("#profile-dialog").close(); showApp(user); showToast("프로필을 저장했습니다.");
  } catch (requestError) { error.textContent = requestError.message; }
  finally { button.disabled = false; button.textContent = "프로필 저장"; }
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
  const advice = state.user.profile && state.user.profile.recommendation;
  const recommended = $("#recommended-duration"); const hint = $("#recommend-hint");
  recommended.hidden = !advice; hint.hidden = !advice; recommended.dataset.duration = advice ? String(advice.seconds) : "";
  if (advice) {
    recommended.textContent = `추천 ${durationLabel(advice.seconds)}`;
    hint.textContent = `${advice.reason}. 한 번에 어렵다면 여러 번 나눠 뛰어도 됩니다.`;
    $("#workout-duration").value = String(advice.seconds);
  }
  $$(`[data-duration]`).forEach((button) => button.classList.toggle("is-active", button.dataset.duration === $("#workout-duration").value));
  $("#workout-setup-dialog").showModal();
}

function resetWorkoutScreen() {
  state.phase = "SEARCHING"; state.lastCount = 0; state.lastElapsed = 0; state.timeline = []; state.resultShown = false;
  $("#workout-mode-name").textContent = modeNames[state.mode]; $("#result-overlay").hidden = true;
  $("#live-count").textContent = "0"; $("#live-time").textContent = "00:00"; $("#live-pace").textContent = "0";
  $("#target-time").textContent = formatTime(state.duration); $("#remaining-time").textContent = formatTime(state.duration);
  $("#time-progress").style.transform = "scaleX(1)"; $("#ready-progress").style.transform = "scaleX(0)";
  const progress = $(".time-track"); progress.setAttribute("aria-valuemax", String(state.duration)); progress.setAttribute("aria-valuenow", String(state.duration));
  $("#connection-state").textContent = "카메라 연결 중"; $("#camera-guide").hidden = false; $("#countdown-overlay").hidden = true; $("#body-state").textContent = "대기";
  setPhase("SEARCHING");
  const cheer = $("#cheer-loop"); const cheerImage = cheer.querySelector("img"); cheer.hidden = false;
  const cheerAsset = cheerAssets[state.mode] || cheerAssets.basic;
  cheerImage.src = reducedMotion() ? cheerAsset.poster : cheerAsset.animation;
  cheerImage.alt = cheerAsset.alt; cheer.querySelector("figcaption").textContent = cheerAsset.caption;
}

function setPhase(phase) {
  state.phase = phase;
  const current = phaseOrder.indexOf(phase);
  $$(".phase-steps li").forEach((item, index) => {
    item.classList.toggle("is-current", index === current); item.classList.toggle("is-done", index < current);
    if (index === current) item.setAttribute("aria-current", "step"); else item.removeAttribute("aria-current");
  });
  const label = $("#phase-label"); label.textContent = phaseLabels[phase] || phase; label.classList.toggle("is-live", phase === "COUNTING");
}

async function startWorkout() {
  state.closing = false; state.workoutId = null; $("#workout-view").hidden = false; resetWorkoutScreen();
  try { if (!document.fullscreenElement && $("#workout-view").requestFullscreen) await $("#workout-view").requestFullscreen(); } catch { showToast("브라우저 메뉴에서도 전체 화면을 켤 수 있습니다."); }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 }, facingMode: "user" }, audio: false });
    const video = $("#camera-source"); video.srcObject = state.stream; await video.play();
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    state.socket = new WebSocket(`${protocol}://${location.host}/ws/count/${state.mode}?duration=${state.duration}&countdown=${state.countdown}`);
    state.socket.onopen = () => { $("#connection-state").textContent = "분석 서버 연결됨"; };
    state.socket.onmessage = handleSocketMessage;
    state.socket.onclose = handleSocketClose;
    state.socket.onerror = () => showToast("분석 서버에 연결할 수 없습니다.");
  } catch (error) { showToast(error.name === "NotAllowedError" ? "카메라 사용 권한을 허용해 주세요." : "카메라를 시작할 수 없습니다."); cleanupWorkout(); }
}

function handleSocketClose(event) {
  state.sending = false;
  if (state.closing || state.resultShown) return;
  if (state.phase === "COUNTING" && state.workoutId) {
    // The server stores the session as interrupted; show what was measured instead of dropping the user back.
    showResult({ workoutId: state.workoutId, mode: state.mode, count: state.lastCount, duration: Math.round(state.lastElapsed), targetDuration: state.duration, status: "interrupted", reason: "disconnected" });
    return;
  }
  if (event.code !== 1000) showToast(event.reason || "측정 연결이 종료되었습니다.");
  cleanupWorkout();
}

async function handleSocketMessage(event) {
  const message = JSON.parse(event.data);
  if (message.type === "ready") {
    state.workoutId = message.workoutId;
    if (message.analysisFps) state.frameInterval = 1000 / message.analysisFps;
    scheduleFrame(); return;
  }
  if (message.type === "state") {
    const roundTrip = Math.max(0, performance.now() - state.frameSentAt); state.sending = false;
    updateWorkoutState(message); drawPose(message.landmarks || []);
    $("#connection-state").textContent = `분석 ${Math.round(message.processingMs || 0)}ms · 왕복 ${Math.round(roundTrip)}ms`;
    // Keep streaming until the server announces the finish; the server owns the clock.
    if (!message.finished) scheduleFrame(Math.max(0, state.frameInterval - roundTrip));
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
  captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height); state.sending = true;
  captureCanvas.toBlob((blob) => {
    if (blob && state.socket?.readyState === WebSocket.OPEN) {
      try { state.frameSentAt = performance.now(); state.socket.send(blob); }
      catch { state.sending = false; scheduleFrame(state.frameInterval); }
    }
    else { state.sending = false; scheduleFrame(state.frameInterval); }
  }, "image/jpeg", .62);
}

function drawPose(landmarks) {
  const canvas = poseCanvas; const context = poseContext; context.clearRect(0, 0, canvas.width, canvas.height);
  if (landmarks.length < 33) return;
  context.lineWidth = 5; context.lineCap = "round"; context.lineJoin = "round"; context.strokeStyle = "rgba(240,122,60,.92)";
  poseConnections.forEach(([from, to]) => {
    const a = landmarks[from]; const b = landmarks[to]; if (!a || !b || a[2] < .45 || b[2] < .45) return;
    context.beginPath(); context.moveTo(a[0] * canvas.width, a[1] * canvas.height); context.lineTo(b[0] * canvas.width, b[1] * canvas.height); context.stroke();
  });
  context.fillStyle = "#eceff2";
  landmarks.forEach((point, index) => { if (index < 11 || point[2] < .55) return; context.beginPath(); context.arc(point[0] * canvas.width, point[1] * canvas.height, 5, 0, Math.PI * 2); context.fill(); });
}

function updateWorkoutState(message) {
  if (message.phase !== state.phase) setPhase(message.phase);
  const countOutput = $("#live-count");
  if (message.count !== state.lastCount) {
    countOutput.textContent = message.count;
    if (message.count > state.lastCount && !reducedMotion()) { countOutput.classList.remove("is-bumped"); void countOutput.offsetWidth; countOutput.classList.add("is-bumped"); }
    state.lastCount = message.count;
  }
  if (message.phase === "COUNTING") {
    state.lastElapsed = message.elapsed; state.timeline.push([message.elapsed, message.count]);
    $("#live-pace").textContent = message.elapsed >= 3 ? Math.round(message.count / message.elapsed * 60) : "0";
  }
  $("#live-time").textContent = formatTime(message.elapsed);
  const remaining = Math.max(0, state.duration - message.elapsed); $("#remaining-time").textContent = formatTime(remaining);
  $("#time-progress").style.transform = `scaleX(${Math.max(0, remaining / state.duration)})`;
  $(".time-track").setAttribute("aria-valuenow", String(Math.round(remaining)));
  $("#body-state").textContent = message.framed ? "인식됨" : message.readyProgress > 0 ? "확인 중" : "위치 조정";
  $("#ready-progress").style.transform = `scaleX(${message.framed && message.phase === "SEARCHING" ? Math.max(.06, message.readyProgress) : 0})`;
  $("#camera-guide").hidden = Boolean(message.framed) || message.phase !== "SEARCHING";
  const countdown = $("#countdown-overlay"); countdown.hidden = message.phase !== "COUNTDOWN";
  if (!countdown.hidden) {
    const digit = countdown.querySelector("span"); const next = String(Math.max(1, Math.ceil(message.countdown)));
    if (digit.textContent !== next) { digit.textContent = next; digit.classList.remove("is-tick"); void digit.offsetWidth; digit.classList.add("is-tick"); }
  }
}

function stopWorkout() { if (state.socket?.readyState === WebSocket.OPEN) { state.closing = true; state.socket.send(JSON.stringify({ type: "stop" })); } else cleanupWorkout(); }

function segmentCounts(timeline, duration, size) {
  const segments = Math.max(1, Math.ceil(Math.max(duration, 1) / size)); const ends = new Array(segments).fill(null);
  timeline.forEach(([elapsed, count]) => { const index = Math.min(segments - 1, Math.floor(elapsed / size)); ends[index] = count; });
  let previous = 0;
  return ends.map((value) => { const end = value === null ? previous : value; const delta = Math.max(0, end - previous); previous = end; return delta; });
}

function bestWindow(timeline, size) {
  let best = 0; let start = 0;
  for (let end = 0; end < timeline.length; end += 1) {
    while (timeline[end][0] - timeline[start][0] > size) start += 1;
    const base = start > 0 ? timeline[start - 1][1] : 0;
    best = Math.max(best, timeline[end][1] - base);
  }
  return best;
}

function renderRhythm(timeline, duration) {
  const chart = $("#result-chart"); chart.replaceChildren();
  const hasData = timeline.length > 1 && duration > 0;
  chart.hidden = !hasData; $("#result-chart-empty").hidden = hasData;
  if (!hasData) return;
  const counts = segmentCounts(timeline, duration, 5); const peak = Math.max(...counts, 1); const bestIndex = counts.indexOf(Math.max(...counts));
  chart.classList.toggle("is-dense", counts.length > 24);
  chart.setAttribute("aria-label", `5초 구간별 횟수: ${counts.join(", ")}회`);
  counts.forEach((value, index) => {
    const bar = document.createElement("div"); bar.className = `rhythm-bar${index === bestIndex && value > 0 ? " is-best" : ""}`;
    bar.style.height = `${Math.max(3, value / peak * 100)}%`; bar.style.setProperty("--i", index);
    const label = document.createElement("span"); label.textContent = value; bar.appendChild(label); chart.appendChild(bar);
  });
}

function showResult(message) {
  state.resultShown = true; setPhase("DONE"); stopMedia();
  const mode = message.mode || state.mode; const duration = Number(message.duration) || 0; const target = Number(message.targetDuration) || state.duration;
  const reason = message.reason || (message.status === "completed" ? "time" : "stopped");
  const statusText = { time: "목표 시간 완료", stopped: "직접 종료", not_started: "측정 시작 전 종료", disconnected: "연결 끊김, 중단 기록으로 저장" }[reason] || "측정 종료";
  const statusLine = $("#result-status");
  statusLine.querySelector("span").textContent = statusText; statusLine.classList.toggle("is-muted", reason !== "time");
  statusLine.querySelector("use").setAttribute("href", reason === "time" || reason === "stopped" ? "#i-check-circle" : "#i-warning-circle");
  $("#result-mode").textContent = modeNames[mode] || mode;
  $("#result-count").textContent = Number(message.count || 0).toLocaleString("ko-KR");
  $("#result-duration").textContent = formatTime(duration); $("#result-target").textContent = `/ ${formatTime(target)}`;
  $("#result-pace").textContent = duration > 0 ? Math.round(message.count / duration * 60) : "0";
  $("#result-best").textContent = bestWindow(state.timeline, 10);
  $("#result-detail").textContent = reason === "not_started" ? "전신 인식과 준비 시간이 끝나기 전에 종료되어 횟수를 세지 않았습니다." : reason === "disconnected" ? "연결이 끊기기 직전까지 센 횟수입니다." : "결과는 측정 기록에 자동 저장되었습니다.";
  $("#result-download").hidden = !message.workoutId;
  renderRhythm(state.timeline, duration);
  $("#cheer-loop").hidden = true; $("#countdown-overlay").hidden = true; $("#camera-guide").hidden = true; $("#result-overlay").hidden = false;
  $("#result-retry").focus({ preventScroll: true });
}

function stopMedia() {
  window.clearTimeout(state.frameTimer); state.frameTimer = null; state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null; state.sending = false;
  const video = $("#camera-source"); video.srcObject = null;
  poseContext.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
}

function closeSocket() {
  state.closing = true; if (state.socket && state.socket.readyState < WebSocket.CLOSING) state.socket.close(); state.socket = null;
}

async function cleanupWorkout() {
  closeSocket(); stopMedia();
  $("#workout-view").hidden = true; $("#result-overlay").hidden = true; if (document.fullscreenElement) await document.exitFullscreen().catch(() => {}); loadDashboard();
}

async function retryWorkout() {
  closeSocket(); stopMedia(); await startWorkout();
}

function syncFullscreenIcon() {
  const active = Boolean(document.fullscreenElement);
  $("#fullscreen-button use").setAttribute("href", active ? "#i-arrows-in" : "#i-arrows-out");
  $("#fullscreen-button").setAttribute("aria-label", active ? "전체 화면 끝내기" : "전체 화면 전환");
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
$$(`[data-countdown]`).forEach((button) => button.addEventListener("click", () => { $("#workout-countdown").value = button.dataset.countdown; $$(`[data-countdown]`).forEach((item) => item.classList.toggle("is-active", item === button)); }));
$("#workout-countdown").addEventListener("input", () => $$(`[data-countdown]`).forEach((button) => button.classList.toggle("is-active", button.dataset.countdown === $("#workout-countdown").value)));
$("#close-setup-dialog").addEventListener("click", () => $("#workout-setup-dialog").close());
$("#workout-setup-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const duration = Number($("#workout-duration").value); const countdown = Number($("#workout-countdown").value);
  if (!Number.isInteger(duration) || duration < 10 || duration > 3600) { $("#setup-error").textContent = "측정 시간은 10초부터 3,600초 사이로 입력해 주세요."; return; }
  if (!Number.isInteger(countdown) || countdown < 1 || countdown > 30) { $("#setup-error").textContent = "카운트다운은 1초부터 30초 사이로 입력해 주세요."; return; }
  state.duration = duration; state.countdown = countdown; $("#workout-setup-dialog").close(); startWorkout();
});
$("#stop-workout").addEventListener("click", stopWorkout); $("#workout-back").addEventListener("click", stopWorkout);
$("#result-close").addEventListener("click", async () => { await cleanupWorkout(); $(".record-section").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" }); });
$("#result-retry").addEventListener("click", retryWorkout);
$("#result-download").addEventListener("click", () => { if (state.workoutId) window.location.assign(`/api/workouts/${state.workoutId}/pdf`); });
$("#fullscreen-button").addEventListener("click", () => (document.fullscreenElement ? document.exitFullscreen() : $("#workout-view").requestFullscreen()).catch(() => showToast("전체 화면을 바꿀 수 없습니다.")));
document.addEventListener("fullscreenchange", syncFullscreenIcon);
$("#history-body").addEventListener("click", (event) => { const target = event.target.closest("button[data-action]"); if (!target) return; if (target.dataset.action === "view") openRecord(target.dataset.id); if (target.dataset.action === "delete") deleteRecord(target.dataset.id); });
$("#history-select-all").addEventListener("change", (event) => { $$(`[data-record-select]:not(:disabled)`).forEach((item) => { item.checked = event.currentTarget.checked; }); updateHistorySelection(); });
$("#delete-selected-records").addEventListener("click", deleteSelectedRecords);
$("#close-record-dialog").addEventListener("click", () => $("#record-dialog").close());
$("#profile-button").addEventListener("click", openProfile); $("#close-profile-dialog").addEventListener("click", () => $("#profile-dialog").close());
$("#profile-form").addEventListener("submit", saveProfile);
$("#open-body-profile").addEventListener("click", () => openBodyProfile(false));
$("#edit-body-profile").addEventListener("click", () => { $("#profile-dialog").close(); openBodyProfile(false); });
$("#body-profile-form").addEventListener("submit", saveBodyProfile);
$$('#body-profile-form input[name="sex"]').forEach((input) => input.addEventListener("change", syncBodyFields));
$("#skip-body-profile").addEventListener("click", skipBodyProfile);
$("#body-profile-dialog").addEventListener("cancel", (event) => { event.preventDefault(); skipBodyProfile(); });
$("#change-avatar").addEventListener("click", () => $("#avatar-input").click());
$("#avatar-input").addEventListener("change", changeAvatar);
$("#remove-avatar").addEventListener("click", removeAvatar);
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

window.addEventListener("beforeunload", stopMedia); bindLanePreview(); loadSession();
