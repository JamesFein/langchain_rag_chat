const uploadStatus = document.getElementById('uploadStatus');
const chatbox = document.getElementById('chatbox');
const queryInput = document.getElementById('queryInput');

async function uploadFiles() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;

    if (files.length === 0) {
        uploadStatus.textContent = 'Please select files to upload.';
        return;
    }

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    uploadStatus.textContent = 'Uploading...';

    try {
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (response.ok) {
            uploadStatus.textContent = `Successfully uploaded: ${result.filenames.join(', ')}`;
        } else {
            uploadStatus.textContent = `Error: ${result.detail || 'Upload failed'}`;
        }
    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.textContent = 'Upload failed. See console for details.';
    }
}

async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query) return;

    appendMessage(query, 'user');
    queryInput.value = '';

    try {
        const response = await fetch('/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query }),
        });

        const result = await response.json();

        if (response.ok) {
            appendMessage(result.answer, 'bot');
        } else {
            appendMessage(`Error: ${result.detail || 'Failed to get answer'}`, 'bot');
        }
    } catch (error) {
        console.error('Chat error:', error);
        appendMessage('Failed to get answer. See console for details.', 'bot');
    }
}

function appendMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender + '-message');
    messageDiv.textContent = text;
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight; // Auto-scroll to bottom
}

// Allow sending message with Enter key
queryInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
