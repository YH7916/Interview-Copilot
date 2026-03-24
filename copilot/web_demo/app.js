const chatLog = document.getElementById("chat-log");
const composer = document.getElementById("composer");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const clearButton = document.getElementById("clear-button");
const newSessionButton = document.getElementById("new-session-button");
const healthPill = document.getElementById("health-pill");
const modelNote = document.getElementById("model-note");

function addMessage(role, content, tag = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const head = document.createElement("div");
  head.className = "message-head";

  const roleLabel = document.createElement("span");
  roleLabel.className = "role";
  roleLabel.textContent = role === "assistant" ? "Copilot" : "You";

  const tagLabel = document.createElement("span");
  tagLabel.className = "tag";
  tagLabel.textContent = tag || (role === "assistant" ? "Runtime" : "Input");

  const body = document.createElement("pre");
  body.className = "message-body";
  body.textContent = normalizeDisplayText(content);

  head.append(roleLabel, tagLabel);
  article.append(head, body);
  chatLog.append(article);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Health check failed");
    }
    healthPill.textContent = "Model Connected";
    healthPill.classList.add("ready");
    modelNote.textContent = `model: ${payload.model}`;
  } catch (error) {
    healthPill.textContent = "Runtime Error";
    modelNote.textContent = error.message;
  }
}

async function sendMessage(message, options = {}) {
  const tag = options.tag || (message.startsWith("/") ? "Command" : "Answer");
  const silentUser = options.silentUser || false;

  if (!silentUser) {
    addMessage("user", message, tag);
  }

  sendButton.disabled = true;
  sendButton.textContent = "Sending...";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Chat request failed");
    }
    addMessage("assistant", payload.response, options.responseTag || "Model");
  } catch (error) {
    addMessage("assistant", error.message, "Error");
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send";
  }
}

function resetTranscript(message, tag = "Reset") {
  chatLog.innerHTML = "";
  addMessage("assistant", message, tag);
}

function normalizeDisplayText(content) {
  return String(content)
    .replace(/- trace:\s+[A-Za-z]:[\\/].*?[\\/]data[\\/]traces[\\/]/g, "- trace: data/traces/")
    .replace(/[ \t]+\n/g, "\n")
    .trim();
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }
  messageInput.value = "";
  await sendMessage(message);
});

document.querySelectorAll(".chip").forEach((button) => {
  button.addEventListener("click", async () => {
    const message = button.dataset.message || "";
    if (!message) {
      return;
    }
    messageInput.value = message;
    messageInput.focus();
    await sendMessage(message);
  });
});

clearButton.addEventListener("click", () => {
  resetTranscript(
    "Transcript cleared. You can continue using the same runtime session or start a fresh one with New Session.",
    "Reset"
  );
});

newSessionButton.addEventListener("click", async () => {
  resetTranscript("Starting a fresh runtime session...", "Session");
  await sendMessage("/new", {
    silentUser: true,
    responseTag: "Runtime",
  });
});

fetchHealth();
