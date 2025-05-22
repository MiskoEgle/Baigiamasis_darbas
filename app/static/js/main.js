// Show loading spinner
function showSpinner() {
    let spinner = document.createElement('div');
    spinner.className = 'spinner';
    document.body.appendChild(spinner);
    spinner.style.display = 'block';
}

// Hide loading spinner
function hideSpinner() {
    const spinner = document.querySelector('.spinner');
    if (spinner) {
        spinner.style.display = 'none';
        spinner.remove();
    }
}

// Show alert message
function showAlert(message, type = 'success') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.row'));
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Handle file upload
function handleFileUpload(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Check image dimensions
                if (img.width > 1000 || img.height > 1000) {
                    reject('Image dimensions too large. Please upload an image smaller than 1000x1000 pixels.');
                } else {
                    resolve(e.target.result);
                }
            };
            img.onerror = () => reject('Error loading image');
            img.src = e.target.result;
        };
        reader.onerror = () => reject('Error reading file');
        reader.readAsDataURL(file);
    });
}

// Update prediction display
function updatePredictionDisplay(prediction, confidence) {
    const predictionResult = document.getElementById('predictionResult');
    const predictionText = document.getElementById('predictionText');
    const progressBar = document.querySelector('.progress-bar');
    
    predictionResult.style.display = 'block';
    predictionText.textContent = `Predicted Class: ${prediction}`;
    
    progressBar.style.width = `${confidence}%`;
    progressBar.textContent = `${confidence.toFixed(2)}%`;
    
    // Add color based on confidence
    if (confidence >= 80) {
        progressBar.className = 'progress-bar bg-success';
    } else if (confidence >= 60) {
        progressBar.className = 'progress-bar bg-warning';
    } else {
        progressBar.className = 'progress-bar bg-danger';
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const mode = formData.get('mode');
    const endpoint = mode === 'predict' ? '/predict' : '/train';
    
    try {
        showSpinner();
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (mode === 'predict') {
            updatePredictionDisplay(result.prediction, result.probabilities[0][result.prediction] * 100);
        } else {
            showAlert('Training completed successfully!');
        }
    } catch (error) {
        console.error('Error:', error);
        showAlert(error.message || 'An error occurred while processing the request.', 'danger');
    } finally {
        hideSpinner();
    }
}

// Initialize event listeners
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const imageFile = document.getElementById('imageFile');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }
    
    if (imageFile) {
        imageFile.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    const result = await handleFileUpload(file);
                    const preview = document.getElementById('preview');
                    preview.src = result;
                    preview.style.display = 'block';
                } catch (error) {
                    showAlert(error, 'danger');
                    imageFile.value = '';
                }
            }
        });
    }
});

// Mokymo forma
const trainForm = document.getElementById('train-form');
const trainProgress = document.getElementById('train-progress');
const trainProgressBar = trainProgress.querySelector('.progress-bar');

trainForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(trainForm);
    
    try {
        // Rodyti progreso juostą
        trainProgress.style.display = 'block';
        trainProgressBar.style.width = '0%';
        
        // Siųsti užklausą
        const response = await fetch('/mokyti', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Klaida mokant modelį');
        }
        
        const data = await response.json();
        
        // Atnaujinti progreso juostą
        trainProgressBar.style.width = '100%';
        
        // Atnaujinti rezultatus
        updateResults(data.results);
        
        // Nukreipti į rezultatų puslapį
        window.location.href = '/rezultatai';
        
    } catch (error) {
        console.error('Klaida:', error);
        alert('Įvyko klaida mokant modelį: ' + error.message);
        trainProgress.style.display = 'none';
    }
});

// Testavimo forma
const testForm = document.getElementById('test-form');
const testResult = document.getElementById('test-result');

testForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(testForm);
    
    try {
        // Siųsti užklausą
        const response = await fetch('/testuoti', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Klaida testuojant modelį');
        }
        
        const data = await response.json();
        
        // Rodyti rezultatus
        testResult.style.display = 'block';
        testResult.innerHTML = `
            <h3>Prognozės:</h3>
            <p>KNN: ${data.knn}</p>
            <p>CNN: ${data.cnn}</p>
            <p>Transformer: ${data.transformer}</p>
        `;
        
    } catch (error) {
        console.error('Klaida:', error);
        alert('Įvyko klaida testuojant modelį: ' + error.message);
    }
});

// Atnaujinti rezultatus
function updateResults(results) {
    // KNN modelis
    document.getElementById('knn-accuracy').textContent = results.knn.accuracy.toFixed(4);
    document.getElementById('knn-f1').textContent = results.knn.f1.toFixed(4);
    
    // CNN modelis
    document.getElementById('cnn-accuracy').textContent = results.cnn.accuracy.toFixed(4);
    document.getElementById('cnn-f1').textContent = results.cnn.f1.toFixed(4);
    
    // Transformer modelis
    document.getElementById('transformer-accuracy').textContent = results.transformer.accuracy.toFixed(4);
    document.getElementById('transformer-f1').textContent = results.transformer.f1.toFixed(4);
    
    // Atnaujinti grafikus
    updateGraphs(results);
}

// Atnaujinti grafikus
function updateGraphs(results) {
    // KNN modelis
    document.getElementById('knn-confusion').src = '/static/knn_grafikai/konfuzijos_macica.png';
    document.getElementById('knn-classes').src = '/static/knn_grafikai/klasiu_grafikai.png';
    document.getElementById('knn-metrics').src = '/static/knn_grafikai/metriku_grafikai.png';
    
    // CNN modelis
    document.getElementById('cnn-confusion').src = '/static/cnn_grafikai/konfuzijos_macica.png';
    document.getElementById('cnn-classes').src = '/static/cnn_grafikai/klasiu_grafikai.png';
    document.getElementById('cnn-metrics').src = '/static/cnn_grafikai/metriku_grafikai.png';
    
    // Transformer modelis
    document.getElementById('transformer-confusion').src = '/static/transformer_grafikai/konfuzijos_macica.png';
    document.getElementById('transformer-classes').src = '/static/transformer_grafikai/klasiu_grafikai.png';
    document.getElementById('transformer-metrics').src = '/static/transformer_grafikai/metriku_grafikai.png';
    
    // Rodyti grafikus
    document.querySelectorAll('.graphs img').forEach(img => {
        img.style.display = 'block';
    });
}

// Stebėti mokymo progresą
async function checkProgress() {
    try {
        const response = await fetch('/progresas');
        const data = await response.json();
        
        if (data.status === 'in_progress') {
            trainProgressBar.style.width = `${data.progress}%`;
            setTimeout(checkProgress, 1000);
        } else if (data.status === 'completed') {
            trainProgressBar.style.width = '100%';
            setTimeout(() => {
                trainProgress.style.display = 'none';
            }, 1000);
        } else if (data.status === 'error') {
            throw new Error(data.message);
        }
        
    } catch (error) {
        console.error('Klaida:', error);
        alert('Įvyko klaida stebint progresą: ' + error.message);
        trainProgress.style.display = 'none';
    }
}

// Pradėti stebėti progresą kai pradedamas mokymas
trainForm.addEventListener('submit', () => {
    checkProgress();
}); 