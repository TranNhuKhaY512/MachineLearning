const input = document.getElementById("input");
const send = document.getElementById("send");
const chatbox = document.getElementById("chatbox");
const typing = document.getElementById("typing");
const live = document.getElementById("a11y-live");
const clearBtn = document.getElementById("clear");
const themeToggle = document.getElementById("themeToggle");
const idModal = document.getElementById("idModal");
const idInput = document.getElementById("idInput");
const idSubmit = document.getElementById("idSubmit");
const idCancel = document.getElementById("idCancel");
const newModal = document.getElementById("newModal");
const choiceModal = document.getElementById("choiceModal");
const choiceOld = document.getElementById("choiceOld");
const choiceNew = document.getElementById("choiceNew");
const newName = document.getElementById("newName");
const newAge = document.getElementById("newAge");
const newDisease = document.getElementById("newDisease");
const newUnderlying = document.getElementById("newUnderlying");
const newAdmissions = document.getElementById("newAdmissions");
const newSeverity = document.getElementById("newSeverity");
const newSubmit = document.getElementById("newSubmit");
const newCancel = document.getElementById("newCancel");

function autoResizeTextarea(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

function scrollToBottom() {
  chatbox.scrollTop = chatbox.scrollHeight;
}

function formatTime(date) {
  const d = date || new Date();
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  return `${h}:${m}`;
}

function createMessageElement(text, sender) {
  const li = document.createElement("li");
  li.className = `message message--${sender}`;

  const avatar = document.createElement("div");
  avatar.className = "message__avatar";
  avatar.textContent = sender === "user" ? "🙂" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = "message__bubble";
  bubble.textContent = text;

  const meta = document.createElement("div");
  meta.className = "message__meta";
  meta.textContent = formatTime();

  li.appendChild(sender === "user" ? bubble : avatar);
  li.appendChild(sender === "user" ? avatar : bubble);
  li.appendChild(meta);

  return li;
}

function addMessage(text, sender) {
  const el = createMessageElement(text, sender);
  chatbox.appendChild(el);
  scrollToBottom();
  if (live) live.textContent = (sender === "user" ? "Bạn: " : "AI: ") + text;
}

function setTyping(state) {
  if (!typing) return;
  typing.style.display = state ? "flex" : "none";
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";
  autoResizeTextarea(input);
  setTyping(true);

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    addMessage(data.reply || "(không có phản hồi)", "bot");
  } catch (err) {
    addMessage("❌ Lỗi kết nối server", "bot");
  } finally {
    setTyping(false);
  }
}

send.onclick = sendMessage;

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

input.addEventListener("input", () => autoResizeTextarea(input));

// Welcome and open choice modal for guided flow

// Removed quick-action chips per request

// Theme toggle with persistence
function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "dark") root.classList.add("theme-dark");
  else root.classList.remove("theme-dark");
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle("theme-dark");
  const theme = isDark ? "dark" : "light";
  try { localStorage.setItem("theme", theme); } catch {}
  if (themeToggle) themeToggle.textContent = isDark ? "☀️" : "🌙";
}

try {
  const saved = localStorage.getItem("theme");
  if (saved) applyTheme(saved);
  if (themeToggle) themeToggle.textContent = document.documentElement.classList.contains("theme-dark") ? "☀️" : "🌙";
} catch {}

if (themeToggle) themeToggle.addEventListener("click", toggleTheme);

// ID Modal logic
function openIdModal() {
  if (!idModal) return;
  idModal.classList.add("show");
  idModal.setAttribute("aria-hidden", "false");
  setTimeout(() => { if (idInput) idInput.focus(); }, 0);
}

function closeIdModal() {
  if (!idModal) return;
  idModal.classList.remove("show");
  idModal.setAttribute("aria-hidden", "true");
}

// New patient modal logic
function openNewModal() {
  if (!newModal) return;
  newModal.classList.add("show");
  newModal.setAttribute("aria-hidden", "false");
  setTimeout(() => { if (newAge) newAge.focus(); }, 0);
}

function closeNewModal() {
  if (!newModal) return;
  newModal.classList.remove("show");
  newModal.setAttribute("aria-hidden", "true");
}

async function submitNewModal() {
  const payload = {
    name: (newName?.value || "").trim(),
    age: Number((newAge?.value || "").trim()),
    disease: (newDisease?.value || "").trim(),
    underlying: (newUnderlying?.value || "").trim(),
    admissions: Number((newAdmissions?.value || "").trim()),
    severity: Number((newSeverity?.value || "0").trim())
  };

  if (!payload.age || !payload.disease || !payload.underlying || isNaN(payload.admissions)) {
    addMessage("Vui lòng nhập đủ Tuổi, Bệnh, Bệnh nền, Số lần nhập viện.", "bot");
    return;
  }

  try {
    setTyping(true);
    const res = await fetch("/predict_new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const { risk_percent, factors, summary } = data;
    let text = "Các yếu tố:\n";
    if (Array.isArray(factors)) {
      for (const f of factors) {
        text += `- ${f.name}: ${f.percent} %\n`;
      }
    }
    text += `------------------------------\nTổng nguy cơ tái nhập viện: ${risk_percent} %\n${summary || ""}`;
    addMessage(text, "bot");
  } catch (e) {
    addMessage("❌ Không tính được nguy cơ. Thử lại sau.", "bot");
  } finally {
    setTyping(false);
    closeNewModal();
  }
}

if (newSubmit) newSubmit.addEventListener("click", submitNewModal);
if (newCancel) newCancel.addEventListener("click", closeNewModal);
if (newModal) {
  newModal.addEventListener("click", (e) => { if (e.target === newModal) closeNewModal(); });
}

// Choice modal on load
function openChoice() {
  if (!choiceModal) return;
  choiceModal.classList.add("show");
  choiceModal.setAttribute("aria-hidden", "false");
}

function closeChoice() {
  if (!choiceModal) return;
  choiceModal.classList.remove("show");
  choiceModal.setAttribute("aria-hidden", "true");
}

if (choiceOld) choiceOld.addEventListener("click", () => { closeChoice(); openIdModal(); });
if (choiceNew) choiceNew.addEventListener("click", () => { closeChoice(); openNewModal(); });

// Show choice after initial greeting
window.addEventListener("load", () => {
  addMessage("Xin chào! Tôi là trợ lý y tế ảo. Bạn là bệnh nhân cũ hay bệnh nhân mới?", "bot");
  setTimeout(openChoice, 400);
});

function submitIdModal() {
  if (!idInput) return;
  const id = idInput.value.trim();
  if (id && /^\d+$/.test(id)) {
    input.value = id;
    autoResizeTextarea(input);
    closeIdModal();
    sendMessage();
  } else {
    addMessage("Vui lòng nhập ID hợp lệ (số).", "bot");
  }
}

if (idSubmit) idSubmit.addEventListener("click", submitIdModal);
if (idCancel) idCancel.addEventListener("click", closeIdModal);
if (idModal) {
  idModal.addEventListener("click", (e) => { if (e.target === idModal) closeIdModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && idModal.classList.contains("show")) closeIdModal(); });
}

// Clear history
if (clearBtn) {
  clearBtn.addEventListener("click", () => {
    chatbox.innerHTML = "";
    addMessage("Lịch sử đã được xóa.", "bot");
  });
}
