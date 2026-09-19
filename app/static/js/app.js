// static/js/app.js

const imageInput        = document.getElementById("imageInput");
const uploadArea        = document.getElementById("uploadArea");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const previewImg        = document.getElementById("previewImg");
const questionInput     = document.getElementById("questionInput");
const sendBtn           = document.getElementById("sendBtn");
const chatMessages      = document.getElementById("chatMessages");
const resetBtn          = document.getElementById("resetBtn");
const anchorBox         = document.getElementById("anchorBox");
const anchorText        = document.getElementById("anchorText");

let imageUploaded = false;

// ---------------------------------------------------------------------------
// Upload handling
// ---------------------------------------------------------------------------
uploadArea.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please upload an image file.");
        return;
    }

    // Preview immediately (client-side, no analysis)
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.classList.add("visible");
        uploadPlaceholder.style.display = "none";
    };
    reader.readAsDataURL(file);

    // Send to backend (stores in server-side session, no analysis)
    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch("/upload", {
            method: "POST",
            body: formData,
            credentials: "same-origin",   // <-- ensure session cookie is sent
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        imageUploaded = true;
        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.focus();

        // Hide anchor box on new upload
        anchorBox.classList.add("hidden");
        anchorText.textContent = "";

        // Clear chat, show hint
        chatMessages.innerHTML = "";
        addMessage("assistant", "Image loaded. Ask me a question about it.");

        if (data.preview) {
            previewImg.src = data.preview;
            previewImg.classList.add("visible");
            uploadPlaceholder.style.display = "none";
        }
    } catch (err) {
        console.error(err);
        alert("Upload failed. Is the Flask server running?");
    }
}

// ---------------------------------------------------------------------------
// Chat handling
// ---------------------------------------------------------------------------
sendBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question || !imageUploaded) return;

    addMessage("user", question);
    questionInput.value = "";
    sendBtn.disabled = true;

    const loadingEl = addMessage("assistant", "Analyzing...");

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
            credentials: "same-origin",   // <-- ensure session cookie is sent
        });

        const data = await res.json();
        loadingEl.remove();

        if (data.error) {
            addMessage("assistant", `Error: ${data.error}`);
        } else {
            const pressure = data.pressure_detected ? "pressure-alert" : "";
            addMessage("assistant", data.answer, pressure);

            // Display CAAP anchor once available
            if (data.anchor && anchorBox.classList.contains("hidden")) {
                anchorText.textContent = data.anchor;
                anchorBox.classList.remove("hidden");
            }
        }
    } catch (err) {
        loadingEl.remove();
        addMessage("assistant", "Request failed. Check the server console.");
        console.error(err);
    }

    sendBtn.disabled = false;
    questionInput.focus();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function addMessage(role, text, extraClass = "") {
    const msg = document.createElement("div");
    msg.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${extraClass}`.trim();
    bubble.textContent = text;

    msg.appendChild(bubble);
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return msg;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------
resetBtn.addEventListener("click", async () => {
    try {
        await fetch("/reset", {
            method: "POST",
            credentials: "same-origin",   // <-- ensure session cookie is sent
        });
    } catch (err) {
        console.error("Reset failed:", err);
    }

    imageUploaded = false;
    previewImg.src = "";
    previewImg.classList.remove("visible");
    uploadPlaceholder.style.display = "block";
    questionInput.disabled = true;
    sendBtn.disabled = true;
    questionInput.value = "";
    anchorBox.classList.add("hidden");
    anchorText.textContent = "";

    chatMessages.innerHTML = "";
    addMessage("assistant", "Upload an image on the left, then ask me anything about it.");
});