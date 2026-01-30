const chatbox = document.getElementById("chatbox");
const userInput = document.getElementById("userInput");

let typingElement = null;

// VOICE SETTINGS
let selectedLang = "en-IN"; // English + Indian accent
let selectedVoiceName = ""; // auto pick

function stopBotVoice() {
  if (window.speechSynthesis) window.speechSynthesis.cancel();
}

function autoPickVoice() {
  const voices = window.speechSynthesis.getVoices();
  if (!voices || voices.length === 0) return;

  let best = voices.find(v => v.lang === selectedLang);
  if (!best) best = voices.find(v => v.lang.includes("en"));
  if (!best) best = voices[0];

  selectedVoiceName = best.name;
}

function speakText(text) {
  if (!window.speechSynthesis) return;
  stopBotVoice();

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = selectedLang;
  utter.rate = 1.0;
  utter.pitch = 1.0;
  utter.volume = 1.0;

  const voices = window.speechSynthesis.getVoices();
  const v = voices.find(x => x.name === selectedVoiceName);
  if (v) utter.voice = v;

  window.speechSynthesis.speak(utter);
}

window.speechSynthesis.onvoiceschanged = () => {
  autoPickVoice();
};

// UI
function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  if (sender === "bot") {
    const title = document.createElement("div");
    title.classList.add("msg-title");
    title.innerText = "Virtual Doctor";
    bubble.appendChild(title);
  }

  const body = document.createElement("div");
  body.innerText = text;
  bubble.appendChild(body);

  msg.appendChild(bubble);
  chatbox.appendChild(msg);
  chatbox.scrollTop = chatbox.scrollHeight;
}

function showTyping() {
  if (typingElement) return;

  typingElement = document.createElement("div");
  typingElement.classList.add("message", "bot");

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  const title = document.createElement("div");
  title.classList.add("msg-title");
  title.innerText = "Virtual Doctor";
  bubble.appendChild(title);

  const typingWrap = document.createElement("div");
  typingWrap.classList.add("typing-bubble");
  typingWrap.innerHTML = `<div class="typing-text">Typing</div>
                          <div class="typing-dots"><span></span><span></span><span></span></div>`;

  bubble.appendChild(typingWrap);
  typingElement.appendChild(bubble);

  chatbox.appendChild(typingElement);
  chatbox.scrollTop = chatbox.scrollHeight;
}

function hideTyping() {
  if (typingElement) {
    typingElement.remove();
    typingElement = null;
  }
}

// Send
async function sendMessage(customMessage = null) {
  const message = customMessage || userInput.value.trim();
  if (message === "") return;

  stopBotVoice();
  addMessage(message, "user");
  userInput.value = "";

  showTyping();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message })
    });

    const data = await response.json();
    hideTyping();

    // Virtual doctor questions
    if (data.virtual_mode && data.bot_question) {
      addMessage(data.bot_question, "bot");
      speakText(data.bot_question);
      return;
    }

    // Final result
    let reply = "";

    if (data.triage) {
      reply += `Triage Level: ${data.triage.level}\n${data.triage.advice}\n\n`;
    }

    reply += "Top Predictions:\n";
    if (data.top3) {
      data.top3.forEach((item, idx) => {
        reply += `${idx + 1}) ${item.disease} - ${item.confidence}%\n`;
      });
    }

    reply += `\nDoctor: ${data.doctor}\n\nAdvice:\n${data.precaution}`;

    addMessage(reply, "bot");
    speakText(reply);

  } catch (err) {
    hideTyping();
    addMessage("Server error. Please restart and try again.", "bot");
  }
}

userInput.addEventListener("keypress", function (e) {
  if (e.key === "Enter") sendMessage();
});
