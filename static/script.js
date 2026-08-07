const input   = document.getElementById("message");
const button  = document.getElementById("send-btn");
const chatBox = document.getElementById("chat-box");

let history = JSON.parse(sessionStorage.getItem("chat_history") || "[]");

// Configure marked for safe inline rendering
marked.setOptions({ breaks: true, gfm: true });

function setLoading(isLoading) {
    button.disabled = isLoading;
    input.disabled  = isLoading;
}

function appendMessage(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (role === "assistant") {
        // Render markdown for assistant messages
        bubble.innerHTML = marked.parse(text);
    } else {
        // Plain text for user messages
        bubble.textContent = text;
    }

    wrapper.appendChild(bubble);
    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;
    return wrapper;
}

function showTyping() {
    const wrapper = document.createElement("div");
    wrapper.className = "message assistant typing-indicator";
    wrapper.innerHTML = `
        <div class="message-bubble">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;
    return wrapper;
}

async function sendMessage() {
    const message = input.value.trim();
    if (!message) return;

    appendMessage("user", message);
    input.value = "";
    setLoading(true);

    const typingEl = showTyping();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, history }),
        });

        const data  = await response.json();
        const reply = data.response;

        typingEl.remove();
        appendMessage("assistant", reply);

        history.push({ role: "user",      content: message });
        history.push({ role: "assistant", content: reply   });
        if (history.length > 10) history = history.slice(-10);
        sessionStorage.setItem("chat_history", JSON.stringify(history));

    } catch (err) {
        typingEl.remove();
        appendMessage("assistant", "Something went wrong. Please try again.");
    } finally {
        setLoading(false);
        input.focus();
    }
}

input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !button.disabled) sendMessage();
});

button.addEventListener("click", sendMessage);