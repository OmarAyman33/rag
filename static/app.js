const messagesEl = document.getElementById("messages");
const composerEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const newChatBtn = document.getElementById("new-chat-btn");
const chatListEl = document.getElementById("chat-list");
const uploadBtn = document.getElementById("upload-btn");
const fileInputEl = document.getElementById("file-input");

const STORAGE_KEY = "rag_chats";
const VIEWPORT_MARGIN = 10;

messagesEl.addEventListener("mouseover", (e) => {
  const chip = e.target.closest(".citation");
  if (!chip) return;
  const tooltip = chip.querySelector(".citation-tooltip");
  if (!tooltip) return;

  const chipRect = chip.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();

  let left = chipRect.left + chipRect.width / 2 - tooltipRect.width / 2;
  left = Math.max(
    VIEWPORT_MARGIN,
    Math.min(left, window.innerWidth - tooltipRect.width - VIEWPORT_MARGIN)
  );

  let top = chipRect.top - tooltipRect.height - 8;
  if (top < VIEWPORT_MARGIN) {
    top = chipRect.bottom + 8;
  }

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
});

let activeSource = null;
let chats = loadChats();
let activeChatId = chats.length ? chats[0].id : null;

function loadChats() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveChats() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

function getActiveChat() {
  return chats.find((c) => c.id === activeChatId) || null;
}

function deriveTitle(text) {
  const trimmed = text.trim().replace(/\s+/g, " ");
  return trimmed.length > 42 ? trimmed.slice(0, 42) + "…" : trimmed;
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  newChatBtn.disabled = busy;
  for (const btn of chatListEl.querySelectorAll(".chat-list-item")) {
    btn.disabled = busy;
  }
}

function renderChatList() {
  chatListEl.innerHTML = "";
  if (!chats.length) {
    const empty = document.createElement("div");
    empty.className = "chat-list-empty";
    empty.textContent = "No chats yet";
    chatListEl.appendChild(empty);
    return;
  }
  for (const chat of chats) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chat-list-item" + (chat.id === activeChatId ? " active" : "");
    btn.textContent = chat.title || "New chat";
    btn.addEventListener("click", () => switchChat(chat.id));
    chatListEl.appendChild(btn);
  }
}

function switchChat(id) {
  if (activeSource) return; // don't switch mid-stream
  activeChatId = id;
  renderChatList();
  renderActiveChat();
}

function startNewChat() {
  if (activeSource) return;
  activeChatId = null;
  renderChatList();
  messagesEl.innerHTML = "";
  inputEl.focus();
}

newChatBtn.addEventListener("click", startNewChat);

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("visible"));
  setTimeout(() => {
    toast.classList.remove("visible");
    setTimeout(() => toast.remove(), 200);
  }, 3000);
}

function handleFileSelection(files) {
  if (!files.length) return;
  const names = files.map((f) => f.name).join(", ");
  showToast(`Selected ${files.length} file${files.length > 1 ? "s" : ""}: ${names} — upload isn't wired up yet.`);
}

uploadBtn.addEventListener("click", () => fileInputEl.click());
fileInputEl.addEventListener("change", () => {
  handleFileSelection([...fileInputEl.files]);
  fileInputEl.value = "";
});

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderAnswerHtml(text, citations) {
  const escaped = escapeHtml(text);
  return escaped.replace(/\[(\d+)\]/g, (match, n) => {
    const c = citations[n];
    if (!c) return match;
    const snippet = escapeHtml(c.text.slice(0, 220)) + (c.text.length > 220 ? "…" : "");
    return (
      `<span class="citation">[${n}]` +
      `<span class="citation-tooltip">` +
      `<span class="tooltip-source">${escapeHtml(c.source)} · chunk ${c.chunk_index}</span>` +
      `${snippet}` +
      `</span></span>`
    );
  });
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addUserMessageEl(text) {
  const el = document.createElement("div");
  el.className = "msg-user";
  el.textContent = text;
  messagesEl.appendChild(el);
  return el;
}

function addErrorMessage(container, message) {
  const el = document.createElement("div");
  el.className = "msg-error";
  el.textContent = message;
  container.appendChild(el);
}

function buildThinkingEl(open) {
  const thinkingEl = document.createElement("details");
  thinkingEl.className = "thinking";
  thinkingEl.open = open;
  const summaryEl = document.createElement("summary");
  summaryEl.innerHTML = `<span class="chevron">›</span> Thinking`;
  const thinkingBodyEl = document.createElement("div");
  thinkingBodyEl.className = "thinking-body";
  thinkingEl.appendChild(summaryEl);
  thinkingEl.appendChild(thinkingBodyEl);
  return { thinkingEl, thinkingBodyEl };
}

function renderActiveChat() {
  messagesEl.innerHTML = "";
  const chat = getActiveChat();
  if (!chat) return;

  for (const msg of chat.messages) {
    if (msg.role === "user") {
      addUserMessageEl(msg.text);
      continue;
    }

    const assistantEl = document.createElement("div");
    assistantEl.className = "msg-assistant";

    if (msg.thinking && msg.thinking.length) {
      const { thinkingEl, thinkingBodyEl } = buildThinkingEl(false);
      for (const entry of msg.thinking) {
        const qEl = document.createElement("div");
        qEl.innerHTML = `<div class="thinking-question">${escapeHtml(entry.question)}</div>`;
        for (const chunk of entry.chunks) {
          const srcEl = document.createElement("div");
          srcEl.className = "thinking-source";
          srcEl.textContent = `[${chunk.n}] ${chunk.source} · chunk ${chunk.chunk_index}`;
          qEl.appendChild(srcEl);
        }
        thinkingBodyEl.appendChild(qEl);
      }
      assistantEl.appendChild(thinkingEl);
    }

    const bodyEl = document.createElement("div");
    bodyEl.className = "msg-assistant-body";
    bodyEl.innerHTML = renderAnswerHtml(msg.text || "", msg.citations || {});
    assistantEl.appendChild(bodyEl);

    if (msg.error) {
      addErrorMessage(assistantEl, msg.error);
    }

    messagesEl.appendChild(assistantEl);
  }
  scrollToBottom();
}

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text || sendBtn.disabled) return;
  sendQuery(text);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composerEl.requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

function sendQuery(query) {
  let chat = getActiveChat();
  if (!chat) {
    chat = {
      id: crypto.randomUUID(),
      title: deriveTitle(query),
      messages: [],
    };
    chats.unshift(chat);
    activeChatId = chat.id;
  }
  chat.messages.push({ role: "user", text: query });

  const assistantMsg = { role: "assistant", text: "", citations: {}, thinking: [] };
  chat.messages.push(assistantMsg);
  saveChats();
  renderChatList();

  addUserMessageEl(query);
  inputEl.value = "";
  inputEl.style.height = "auto";
  setBusy(true);

  const assistantEl = document.createElement("div");
  assistantEl.className = "msg-assistant";

  const { thinkingEl, thinkingBodyEl } = buildThinkingEl(true);

  const bodyEl = document.createElement("div");
  bodyEl.className = "msg-assistant-body";
  const typingEl = document.createElement("div");
  typingEl.className = "typing-dots";
  typingEl.innerHTML = "<span></span><span></span><span></span>";
  bodyEl.appendChild(typingEl);

  assistantEl.appendChild(thinkingEl);
  assistantEl.appendChild(bodyEl);
  messagesEl.appendChild(assistantEl);
  scrollToBottom();

  let firstDelta = true;
  let sawAnyEvent = false;

  if (activeSource) {
    activeSource.close();
  }
  const source = new EventSource(`/api/chat?query=${encodeURIComponent(query)}`);
  activeSource = source;

  source.onmessage = (e) => {
    sawAnyEvent = true;
    let event;
    try {
      event = JSON.parse(e.data);
    } catch {
      return;
    }

    switch (event.type) {
      case "atomic_questions": {
        for (const q of event.questions) {
          assistantMsg.thinking.push({ question: q, chunks: [] });
          const qEl = document.createElement("div");
          qEl.dataset.question = q;
          qEl.innerHTML = `<div class="thinking-question">${escapeHtml(q)}</div>`;
          thinkingBodyEl.appendChild(qEl);
        }
        break;
      }
      case "retrieval": {
        const entry = assistantMsg.thinking.find((t) => t.question === event.question);
        if (entry) entry.chunks.push(...event.chunks);
        const qEl = [...thinkingBodyEl.children].find(
          (el) => el.dataset.question === event.question
        );
        if (qEl) {
          for (const chunk of event.chunks) {
            const srcEl = document.createElement("div");
            srcEl.className = "thinking-source";
            srcEl.textContent = `[${chunk.n}] ${chunk.source} · chunk ${chunk.chunk_index}`;
            qEl.appendChild(srcEl);
          }
        }
        break;
      }
      case "citation_map": {
        assistantMsg.citations = event.citations;
        break;
      }
      case "answer_delta": {
        if (firstDelta) {
          thinkingEl.open = false;
          bodyEl.innerHTML = "";
          firstDelta = false;
        }
        assistantMsg.text += event.text;
        bodyEl.innerHTML = renderAnswerHtml(assistantMsg.text, assistantMsg.citations);
        scrollToBottom();
        break;
      }
      case "error": {
        if (firstDelta) {
          bodyEl.innerHTML = "";
        }
        assistantMsg.error = event.message || "Something went wrong.";
        addErrorMessage(assistantEl, assistantMsg.error);
        break;
      }
      case "done": {
        source.close();
        activeSource = null;
        setBusy(false);
        saveChats();
        inputEl.focus();
        break;
      }
    }
  };

  source.onerror = () => {
    source.close();
    activeSource = null;
    setBusy(false);
    if (!sawAnyEvent || (firstDelta && !assistantMsg.text)) {
      bodyEl.innerHTML = "";
      assistantMsg.error = "Connection lost. Please try again.";
      addErrorMessage(assistantEl, assistantMsg.error);
    }
    saveChats();
  };
}

renderChatList();
renderActiveChat();
