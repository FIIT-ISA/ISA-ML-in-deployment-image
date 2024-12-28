const uploadBtn = document.getElementById('upload-btn');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-input');
const chatBox = document.getElementById('chat-box');
const clearBtn = document.getElementById('clear-btn');

let lastSentFile = null;    // Keeps track of the last sent file
let currentFile = null;

const maxUploadImageWidth = 400;
const maxUploadImageHeight = 300;

// Initially disable the send button
sendBtn.disabled = true;

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    chatBox.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    chatBox.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    chatBox.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {chatBox.classList.add('dragover');}
function unhighlight(e) {chatBox.classList.remove('dragover');}

chatBox.addEventListener('drop', handleDrop, false);


uploadBtn.onclick = function() {
    fileInput.click();
};

fileInput.onchange = function(event) {
    const file = event.target.files[0];
    uploadFile(file)
};

sendBtn.onclick = function() {
    if (currentFile) {
        const formData = new FormData();
        formData.append('file', currentFile);
        lastSentFile = currentFile;

        fetch('http://localhost:8080/predict', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            const chatBox = document.getElementById('chat-box');

            const msgDiv = document.createElement('div');
            msgDiv.className = 'chat-message';
            msgDiv.innerHTML = `
            <img src="./static/images/model-icon.png" class="model-icon">
            <span>This is an image of a ${data.prediction}</span>`;
            chatBox.appendChild(msgDiv);

            chatBox.scrollTop = chatBox.scrollHeight;
            sendBtn.disabled = true;
        })
        .catch(error => console.error('Error:', error));
    } else {
        alert('Please select a file first.');
    }
};

clearBtn.onclick = function() {
    chatBox.innerHTML = '';
    lastSentFile = null; 
    sendBtn.disabled = true;
};

function handleDrop(e) {
    let files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function uploadFile(file){
    if (file) {
        /* Keep track of last uploaded image */
        if(file != lastSentFile)
            sendBtn.disabled = false;
        currentFile = file;

        /* Create a "User Response" div element with user icon and filename */
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chat-message';
        msgDiv.innerHTML = `
        <img src="./static/images/user-icon.png" class="user-icon">
        <span class="text-next-to-icon">${file.name}</span>`;
        chatBox.appendChild(msgDiv);

        /* Read file and view it in the chat-box */
        const reader = new FileReader();
        reader.onload = function() {
            /* Scale image to max size */
            scaleImage(reader.result).then(scaledImgSrc => {
                const imgDiv = document.createElement('div');
                imgDiv.className = 'chat-message';
                imgDiv.innerHTML = `<div class="chat-image"></div><img src="${scaledImgSrc}">`;
                chatBox.appendChild(imgDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }).catch(error => {
                console.error("Error scaling image:", error);
            })
        };
        reader.readAsDataURL(file);
    }
}

function scaleImage(readerResult){
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = function() {
            try{
                let width = img.width;
                let height = img.height;

                const scalingFactor = Math.min(1, maxUploadImageWidth / width, maxUploadImageHeight / height);
                width *= scalingFactor;
                height *= scalingFactor;

                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = width;
                canvas.height = height;

                ctx.drawImage(img, 0, 0, width, height)
                resolve(canvas.toDataURL());
            }
            catch (error) {
                reject(error)
            }
        };
        img.onerror = reject;
        img.src = readerResult;
    })
}