const input = document.getElementById("message");
const button = document.getElementById("send-btn");
const chatBox = document.getElementById("chat-box");

let history = JSON.parse(
    sessionStorage.getItem("chat_history") || "[]"
);

marked.setOptions({
    breaks: true,
    gfm: true
});


function setLoading(isLoading) {
    button.disabled = isLoading;
    input.disabled = isLoading;
}


function appendMessage(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (role === "assistant") {
        // Make sure we always have text to render
        const safeText = String(text || "I couldn't generate a response.");

        bubble.innerHTML = marked.parse(safeText);
    } else {
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
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

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
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                history: history
            })
        });


        // Check HTTP status
        if (!response.ok) {
            throw new Error(
                `Server returned ${response.status}`
            );
        }


        const data = await response.json();

        console.log("Chatbot response:", data);


        const reply = data.response;


        typingEl.remove();


        if (!reply) {

            appendMessage(
                "assistant",
                "I received an empty response from the server."
            );

        } else {

            appendMessage(
                "assistant",
                reply
            );

        }


        history.push({
            role: "user",
            content: message
        });

        history.push({
            role: "assistant",
            content: reply || ""
        });


        // Keep only the last 10 messages
        if (history.length > 10) {
            history = history.slice(-10);
        }


        sessionStorage.setItem(
            "chat_history",
            JSON.stringify(history)
        );


    } catch (err) {

        console.error(
            "Chat request failed:",
            err
        );


        typingEl.remove();


        appendMessage(
            "assistant",
            "Something went wrong. Please try again."
        );

    } finally {

        setLoading(false);

        input.focus();
    }
}


input.addEventListener(
    "keydown",
    (e) => {

        if (
            e.key === "Enter" &&
            !button.disabled
        ) {
            sendMessage();
        }

    }
);


button.addEventListener(
    "click",
    sendMessage
);