// Variables globales
let uploadedFiles = [];
let vectorDatabase = null;
let chatModel = null;
let chatHistory = [];
let chatSources = [];
let isProcessing = false;

// Elementos DOM - Se inicializarán cuando el DOM esté cargado
let fileInput, dropZone, fileList, createDatabaseBtn, databaseStatus, databaseStatusText;
let pdfPreviewContainer, pdfPreview, temperatureSlider, temperatureValue;
let maxTokensSlider, maxTokensValue, topKSlider, topKValue;
let initModelBtn, modelStatus, modelStatusText, chatMessages;
let chatInput, sendBtn, sourcesHeader, sourcesContent, sourcesList, sourcesLoader, exportBtn;

// Inicialización al cargar el DOM
document.addEventListener('DOMContentLoaded', () => {
    // Inicializar referencias a elementos DOM
    fileInput = document.getElementById('fileInput');
    dropZone = document.getElementById('dropZone');
    fileList = document.getElementById('fileList');
    createDatabaseBtn = document.getElementById('createDatabaseBtn');
    databaseStatus = document.getElementById('databaseStatus');
    databaseStatusText = document.getElementById('databaseStatusText');
    pdfPreviewContainer = document.getElementById('pdfPreviewContainer');
    pdfPreview = document.getElementById('pdfPreview');
    temperatureSlider = document.getElementById('temperatureSlider');
    temperatureValue = document.getElementById('temperatureValue');
    maxTokensSlider = document.getElementById('maxTokensSlider');
    maxTokensValue = document.getElementById('maxTokensValue');
    topKSlider = document.getElementById('topKSlider');
    topKValue = document.getElementById('topKValue');
    initModelBtn = document.getElementById('initModelBtn');
    modelStatus = document.getElementById('modelStatus');
    modelStatusText = document.getElementById('modelStatusText');
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendBtn = document.getElementById('sendBtn');
    sourcesHeader = document.getElementById('sourcesHeader');
    sourcesContent = document.getElementById('sourcesContent');
    sourcesList = document.getElementById('sourcesList');
    sourcesLoader = document.getElementById('sourcesLoader');
    exportBtn = document.getElementById('exportBtn');

    // Inicialización de sliders
    temperatureSlider.addEventListener('input', () => {
        temperatureValue.textContent = temperatureSlider.value;
    });

    maxTokensSlider.addEventListener('input', () => {
        maxTokensValue.textContent = maxTokensSlider.value;
    });

    topKSlider.addEventListener('input', () => {
        topKValue.textContent = topKSlider.value;
    });

    // Manejo de archivos
    fileInput.addEventListener('change', handleFileSelect);
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent-color)';
        dropZone.style.backgroundColor = 'rgba(0, 168, 232, 0.1)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--bg-light)';
        dropZone.style.backgroundColor = '';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--bg-light)';
        dropZone.style.backgroundColor = '';

        const files = e.dataTransfer.files;
        handleFiles(files);
    });

    // Botones de acción
    createDatabaseBtn.addEventListener('click', createVectorDatabase);
    initModelBtn.addEventListener('click', initializeModel);
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Acordeón de fuentes
    sourcesHeader.addEventListener('click', () => {
        sourcesContent.classList.toggle('show');
        const icon = sourcesHeader.querySelector('.fas');
        if (sourcesContent.classList.contains('show')) {
            icon.classList.replace('fa-chevron-down', 'fa-chevron-up');
        } else {
            icon.classList.replace('fa-chevron-up', 'fa-chevron-down');
        }
    });

    // Exportar chat
    exportBtn.addEventListener('click', exportChat);
});

// Funciones de manejo de archivos
function handleFileSelect(event) {
    const files = event.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type === 'application/pdf') {
            if (!uploadedFiles.some(f => f.name === file.name)) {
                uploadedFiles.push(file);
                displayFile(file);
            }
        } else {
            showNotification('Solo se permiten archivos PDF', 'error');
        }
    }
    updateCreateDatabaseButton();
}

function displayFile(file) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    fileItem.innerHTML = `
        <span class="file-item-name">${file.name}</span>
        <span class="file-remove" data-filename="${file.name}">
            <i class="fas fa-times"></i>
        </span>
    `;

    fileList.appendChild(fileItem);

    fileItem.querySelector('.file-remove').addEventListener('click', function () {
        const filename = this.getAttribute('data-filename');
        uploadedFiles = uploadedFiles.filter(f => f.name !== filename);
        fileItem.remove();
        updateCreateDatabaseButton();
    });

    fileItem.addEventListener('click', function (e) {
        // Ignorar si se hace clic en el botón de eliminar
        if (e.target.closest('.file-remove')) return;

        // Mostrar vista previa del PDF
        const blob = new Blob([file], { type: 'application/pdf' });
        const url = URL.createObjectURL(blob);
        pdfPreview.src = url;
        pdfPreviewContainer.classList.remove('hidden');
    });
}

function updateCreateDatabaseButton() {
    createDatabaseBtn.disabled = uploadedFiles.length === 0;
}

// Funciones API
function createVectorDatabase() {
    if (uploadedFiles.length === 0) {
        showNotification('Por favor, suba al menos un documento PDF', 'error');
        return;
    }

    databaseStatus.classList.remove('hidden', 'status-success', 'status-error');
    databaseStatus.classList.add('status-pending');
    databaseStatusText.textContent = 'Procesando documentos...';
    createDatabaseBtn.disabled = true;

    // Crear FormData para enviar archivos
    const formData = new FormData();
    uploadedFiles.forEach(file => {
        formData.append('files[]', file);
    });

    // Enviar solicitud al backend
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                databaseStatus.classList.remove('status-pending');
                databaseStatus.classList.add('status-success');
                databaseStatusText.textContent = data.message;
                vectorDatabase = {
                    files: data.files,
                    created: new Date().toISOString()
                };
                initModelBtn.disabled = false;
            } else {
                throw new Error(data.error || 'Error al crear la base de datos');
            }
        })
        .catch(error => {
            databaseStatus.classList.remove('status-pending');
            databaseStatus.classList.add('status-error');
            databaseStatusText.textContent = error.message;
            createDatabaseBtn.disabled = false;
        });
}

function initializeModel() {
    if (!vectorDatabase) {
        showNotification('Primero debe crear la base de datos vectorial', 'error');
        return;
    }

    const selectedModel = document.querySelector('input[name="model"]:checked').value;
    const temperature = parseFloat(temperatureSlider.value);
    const maxTokens = parseInt(maxTokensSlider.value);
    const topK = parseInt(topKSlider.value);

    modelStatus.classList.remove('hidden', 'status-success', 'status-error');
    modelStatus.classList.add('status-pending');
    modelStatusText.textContent = 'Inicializando modelo...';
    initModelBtn.disabled = true;

    // Enviar solicitud al backend
    fetch('/api/initialize_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model_name: selectedModel,
            temperature: temperature,
            max_tokens: maxTokens,
            top_k: topK
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                modelStatus.classList.remove('status-pending');
                modelStatus.classList.add('status-success');
                modelStatusText.textContent = data.message;

                chatModel = {
                    name: selectedModel,
                    temperature: temperature,
                    maxTokens: maxTokens,
                    topK: topK
                };

                // Habilitar chat
                chatInput.disabled = false;
                sendBtn.disabled = false;
                exportBtn.disabled = false;

                // Mensaje de bienvenida
                addBotMessage(`¡Chatbot listo! He procesado ${uploadedFiles.length} documento(s). ¿Qué te gustaría saber sobre ellos?`);
            } else {
                throw new Error(data.error || 'Error al inicializar el modelo');
            }
        })
        .catch(error => {
            modelStatus.classList.remove('status-pending');
            modelStatus.classList.add('status-error');
            modelStatusText.textContent = error.message;
            initModelBtn.disabled = false;
        });
}

function sendMessage() {
    if (isProcessing) return;

    const message = chatInput.value.trim();
    if (!message) return;

    if (!chatModel) {
        showNotification('Primero debe inicializar el chatbot', 'error');
        return;
    }

    addUserMessage(message);
    chatInput.value = '';

    isProcessing = true;
    sendBtn.disabled = true;
    chatInput.disabled = true;
    sourcesLoader.classList.remove('hidden');

    // Enviar solicitud al backend
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: message
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addBotMessage(data.answer);
                updateSources(data.sources);
            } else {
                throw new Error(data.error || 'Error al procesar la consulta');
            }
        })
        .catch(error => {
            addBotMessage(`Lo siento, ocurrió un error: ${error.message}`);
        })
        .finally(() => {
            isProcessing = false;
            sendBtn.disabled = false;
            chatInput.disabled = false;
            sourcesLoader.classList.add('hidden');
        });
}

// Funciones de visualización de mensajes
function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-user';
    messageDiv.innerHTML = `
        <p>${text}</p>
        <div class="message-meta">
            <span>${getCurrentTime()}</span>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    chatHistory.push({ role: 'user', content: text, timestamp: new Date().toISOString() });
}

function addBotMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-bot';
    messageDiv.innerHTML = `
        <p>${formatBotResponse(text)}</p>
        <div class="message-meta">
            <span>${getCurrentTime()}</span>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    chatHistory.push({ role: 'assistant', content: text, timestamp: new Date().toISOString() });
}

function formatBotResponse(text) {
    // Formatear texto con Markdown sencillo: negrita, cursiva, listas
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Gestión de fuentes
function updateSources(sources) {
    chatSources = sources;

    sourcesList.innerHTML = '';
    sources.forEach((source, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        sourceItem.innerHTML = `
            <div class="source-page">Página ${source.page} - ${source.filename}</div>
            <div class="source-text">${highlightRelevantText(source.content)}</div>
        `;
        sourcesList.appendChild(sourceItem);
    });

    if (!sourcesContent.classList.contains('show')) {
        sourcesHeader.click();
    }
}

function highlightRelevantText(text) {
    // Resaltar partes relevantes del texto
    const lastQuery = chatHistory.filter(msg => msg.role === 'user').pop()?.content || '';
    if (!lastQuery) return text;

    // Palabras clave de la consulta (eliminar palabras comunes)
    const keywords = lastQuery
        .toLowerCase()
        .replace(/[¿?.¡!,;:]/g, '')
        .split(' ')
        .filter(word =>
            word.length > 3 &&
            !['como', 'para', 'donde', 'cuando', 'porque', 'cuál', 'cual', 'este', 'esta', 'estos', 'estas'].includes(word)
        );

    // Resaltar palabras clave en el texto
    let highlightedText = text;
    keywords.forEach(keyword => {
        const regex = new RegExp(`(${keyword})`, 'gi');
        highlightedText = highlightedText.replace(regex, '<span class="highlight">$1</span>');
    });

    return highlightedText;
}

// Exportar conversación
function exportChat() {
    if (chatHistory.length === 0) {
        showNotification('No hay conversación para exportar', 'warning');
        return;
    }

    let exportText = '# Conversación con Chatbot RAG PDF\n';
    exportText += `Fecha: ${new Date().toLocaleDateString()}\n`;
    exportText += `Documentos analizados: ${vectorDatabase.files.join(', ')}\n\n`;

    chatHistory.forEach(msg => {
        const role = msg.role === 'user' ? 'Usuario' : 'Asistente';
        exportText += `## ${role} (${new Date(msg.timestamp).toLocaleString()})\n`;
        exportText += msg.content + '\n\n';
    });

    // Crear archivo para descargar
    const blob = new Blob([exportText], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'conversacion-chatbot-rag.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('Conversación exportada exitosamente', 'success');
}

// Notificaciones
function showNotification(message, type = 'info') {
    // Implementación simple con alert
    // En una implementación más completa, usar un sistema de notificaciones como toast
    alert(message);
}