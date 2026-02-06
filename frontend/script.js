/**
 * AI Video to Fluent Text - Frontend JavaScript
 */

// ============================================================
// Configuration
// ============================================================

const API_BASE_URL = 'http://localhost:8000';

// ============================================================
// DOM Elements
// ============================================================

const elements = {
    // Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    browseBtn: document.getElementById('browseBtn'),
    filePreview: document.getElementById('filePreview'),
    fileName: document.getElementById('fileName'),
    fileSize: document.getElementById('fileSize'),
    removeFile: document.getElementById('removeFile'),
    processBtn: document.getElementById('processBtn'),
    
    // Language options
    languageSelect: document.getElementById('languageSelect'),
    translateCheckbox: document.getElementById('translateCheckbox'),
    taskSelect: document.getElementById('taskSelect'),
    
    // Language info badges
    languageInfo: document.getElementById('languageInfo'),
    detectedLanguage: document.getElementById('detectedLanguage'),
    videoDuration: document.getElementById('videoDuration'),
    
    // Text input
    textInput: document.getElementById('textInput'),
    improveTextBtn: document.getElementById('improveTextBtn'),
    
    // Loading
    loadingSection: document.getElementById('loadingSection'),
    loadingText: document.getElementById('loadingText'),
    step1: document.getElementById('step1'),
    step2: document.getElementById('step2'),
    step3: document.getElementById('step3'),
    
    // Results
    resultsSection: document.getElementById('resultsSection'),
    originalText: document.getElementById('originalText'),
    cleanedText: document.getElementById('cleanedText'),
    improvedText: document.getElementById('improvedText'),
    resetBtn: document.getElementById('resetBtn'),
    
    // Error
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    retryBtn: document.getElementById('retryBtn'),
};

// ============================================================
// State
// ============================================================

let selectedFile = null;

// ============================================================
// Utility Functions
// ============================================================

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showSection(sectionToShow) {
    // Hide all sections
    elements.loadingSection.classList.add('hidden');
    elements.resultsSection.classList.add('hidden');
    elements.errorSection.classList.add('hidden');
    
    // Show the requested section
    if (sectionToShow) {
        sectionToShow.classList.remove('hidden');
    }
}

function resetSteps() {
    elements.step1.classList.remove('active', 'completed');
    elements.step2.classList.remove('active', 'completed');
    elements.step3.classList.remove('active', 'completed');
    elements.step1.classList.add('active');
}

function updateStep(stepNum) {
    if (stepNum >= 2) {
        elements.step1.classList.remove('active');
        elements.step1.classList.add('completed');
        elements.step2.classList.add('active');
    }
    if (stepNum >= 3) {
        elements.step2.classList.remove('active');
        elements.step2.classList.add('completed');
        elements.step3.classList.add('active');
    }
}

function showError(message) {
    elements.errorMessage.textContent = message;
    showSection(elements.errorSection);
}

function showResults(data) {
    // Display language and duration info if available
    if (elements.languageInfo) {
        const hasLanguageInfo = data.language_name || data.detected_language;
        const hasDurationInfo = data.duration_formatted;
        
        if (hasLanguageInfo || hasDurationInfo) {
            elements.languageInfo.classList.remove('hidden');
            
            if (elements.detectedLanguage && hasLanguageInfo) {
                elements.detectedLanguage.textContent = data.language_name || data.detected_language || 'Unknown';
                elements.detectedLanguage.parentElement.classList.remove('hidden');
            }
            
            if (elements.videoDuration && hasDurationInfo) {
                elements.videoDuration.textContent = data.duration_formatted;
                elements.videoDuration.parentElement.classList.remove('hidden');
            }
        } else {
            elements.languageInfo.classList.add('hidden');
        }
    }
    
    elements.originalText.textContent = data.original_text || data.originalText || 'No text found';
    elements.cleanedText.textContent = data.cleaned_text || data.cleanedText || 'No text found';
    elements.improvedText.textContent = data.improved_text || data.improvedText || 'No text found';
    showSection(elements.resultsSection);
}

// ============================================================
// API Calls
// ============================================================

async function processVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Add language options
    const language = elements.languageSelect?.value || 'auto';
    const translateToEnglish = elements.translateCheckbox?.checked || false;
    const task = elements.taskSelect?.value || 'Fix grammar in this text:';
    
    formData.append('language', language);
    formData.append('translate_to_english', translateToEnglish);
    formData.append('task', task);
    
    // Determine endpoint based on file type
    const isAudio = file.type.startsWith('audio/');
    const endpoint = isAudio ? '/api/upload-audio' : '/api/upload-video';
    
    // Use AbortController for timeout (15 minutes for long videos)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 900000); // 15 min timeout
    
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            let errorMsg = 'Failed to process file';
            try {
                const error = await response.json();
                errorMsg = error.detail || errorMsg;
            } catch (e) {
                errorMsg = `Server error: ${response.status} ${response.statusText}`;
            }
            throw new Error(errorMsg);
        }
        
        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
            throw new Error('Request timed out. The video may be too long. Try a shorter video or use the Colab notebook for large files.');
        }
        
        // Check for network errors
        if (error.message === 'Failed to fetch') {
            throw new Error('Cannot connect to server. Make sure the backend is running at ' + API_BASE_URL);
        }
        
        throw error;
    }
}

async function improveText(text) {
    const task = elements.taskSelect?.value || 'Fix grammar in this text:';
    
    // Use AbortController for timeout (5 minutes)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 min timeout
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/improve-text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                clean_first: true,
                task: task,
            }),
            signal: controller.signal,
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            let errorMsg = 'Failed to improve text';
            try {
                const error = await response.json();
                errorMsg = error.detail || errorMsg;
            } catch (e) {
                errorMsg = `Server error: ${response.status} ${response.statusText}`;
            }
            throw new Error(errorMsg);
        }
        
        return response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
            throw new Error('Request timed out. Try with shorter text.');
        }
        
        if (error.message === 'Failed to fetch') {
            throw new Error('Cannot connect to server. Make sure the backend is running at ' + API_BASE_URL);
        }
        
        throw error;
    }
}

// ============================================================
// Event Handlers
// ============================================================

// File selection
function handleFileSelect(file) {
    if (!file) return;
    
    selectedFile = file;
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatFileSize(file.size);
    elements.uploadArea.classList.add('hidden');
    elements.filePreview.classList.remove('hidden');
}

// Process video button
async function handleProcessVideo() {
    if (!selectedFile) return;
    
    // Show loading
    showSection(elements.loadingSection);
    resetSteps();
    elements.loadingText.textContent = 'Processing your file...';
    
    try {
        // Simulate step progress (actual processing is async)
        setTimeout(() => updateStep(2), 2000);
        setTimeout(() => updateStep(3), 5000);
        
        const result = await processVideo(selectedFile);
        showResults(result);
        
    } catch (error) {
        console.error('Processing error:', error);
        showError(error.message || 'Failed to process file. Please try again.');
    }
}

// Improve text button
async function handleImproveText() {
    const text = elements.textInput.value.trim();
    
    if (!text) {
        alert('Please enter some text to improve');
        return;
    }
    
    // Show loading
    showSection(elements.loadingSection);
    elements.loadingText.textContent = 'Improving your text...';
    elements.step1.classList.remove('active');
    elements.step3.classList.add('active');
    
    try {
        const result = await improveText(text);
        showResults(result);
        
    } catch (error) {
        console.error('Improvement error:', error);
        showError(error.message || 'Failed to improve text. Please try again.');
    }
}

// Remove selected file
function handleRemoveFile() {
    selectedFile = null;
    elements.fileInput.value = '';
    elements.filePreview.classList.add('hidden');
    elements.uploadArea.classList.remove('hidden');
}

// Reset to initial state
function handleReset() {
    handleRemoveFile();
    elements.textInput.value = '';
    showSection(null);
}

// Copy to clipboard
async function handleCopy(targetId) {
    const element = document.getElementById(targetId);
    if (!element) return;
    
    const text = element.textContent;
    
    try {
        await navigator.clipboard.writeText(text);
        
        // Show feedback
        const btn = document.querySelector(`[data-target="${targetId}"]`);
        const originalText = btn.textContent;
        btn.textContent = '✅ Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
        
    } catch (error) {
        console.error('Copy failed:', error);
    }
}

// Drag and drop
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
}

// ============================================================
// Event Listeners
// ============================================================

// File input
elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
elements.uploadArea.addEventListener('click', (e) => {
    if (e.target !== elements.browseBtn) {
        elements.fileInput.click();
    }
});
elements.fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop
elements.uploadArea.addEventListener('dragover', handleDragOver);
elements.uploadArea.addEventListener('dragleave', handleDragLeave);
elements.uploadArea.addEventListener('drop', handleDrop);

// Buttons
elements.removeFile.addEventListener('click', handleRemoveFile);
elements.processBtn.addEventListener('click', handleProcessVideo);
elements.improveTextBtn.addEventListener('click', handleImproveText);
elements.resetBtn.addEventListener('click', handleReset);
elements.retryBtn.addEventListener('click', handleReset);

// Copy buttons
document.querySelectorAll('.btn-copy').forEach(btn => {
    btn.addEventListener('click', () => {
        handleCopy(btn.dataset.target);
    });
});

// ============================================================
// Initialization
// ============================================================

console.log('🎬 AI Video to Fluent Text - Frontend loaded');
console.log('📍 API URL:', API_BASE_URL);
