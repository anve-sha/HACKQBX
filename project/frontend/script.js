document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadZone = document.getElementById('uploadZone');
    const imageInput = document.getElementById('imageInput');
    const uploadText = document.getElementById('uploadText');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = predictBtn.querySelector('.btn-text');
    const loader = predictBtn.querySelector('.loader');
    
    const resultsSection = document.getElementById('resultsSection');
    const originalImage = document.getElementById('originalImage');
    const segmentedImage = document.getElementById('segmentedImage');
    const scanLine = document.querySelector('.scan-line');
    
    // Metric Elements
    const iouScoreEl = document.getElementById('iouScore');
    const pixelAccuracyEl = document.getElementById('pixelAccuracy');
    const diceScoreEl = document.getElementById('diceScore');

    let currentFile = null;

    // --- Event Listeners for Drag and Drop ---
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-active');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-active');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-active');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    uploadZone.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // --- File Handling ---
    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }

        currentFile = file;
        uploadText.textContent = file.name;
        predictBtn.disabled = false;
        
        // Preview original image in results immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImage.src = e.target.result;
            // Also blank out previous results
            segmentedImage.src = '';
            resultsSection.classList.remove('hidden');
            
            // Reset metrics
            iouScoreEl.textContent = "0.00";
            pixelAccuracyEl.textContent = "0.00";
            diceScoreEl.textContent = "0.00";
        };
        reader.readAsDataURL(file);
    }

    // --- Predict Button ---
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State: Loading
        predictBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        segmentedImage.src = ''; // clear old
        scanLine.classList.remove('hidden'); // show scanner

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            // DEPLOYMENT NOTE: Change this to your Render URL (e.g., 'https://my-model-api.onrender.com') when deploying to Netlify
            const API_BASE_URL = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' 
                ? 'http://127.0.0.1:8000' 
                : 'https://YOUR-RENDER-APP-NAME.onrender.com';

            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.message || 'Error running inference');
            }

            const data = await response.json();

            // Display Image output
            segmentedImage.src = data.segmented_image;

            // Animate Metrics
            animateMetric(iouScoreEl, 0, data.iou_score, 1500);
            animateMetric(pixelAccuracyEl, 0, data.pixel_accuracy, 1500);
            animateMetric(diceScoreEl, 0, data.dice_score, 1500);

        } catch (error) {
            console.error('Prediction error:', error);
            alert('Failed to get prediction from backend. Ensure the API is running.');
        } finally {
            // UI State: Reset Loading
            predictBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            scanLine.classList.add('hidden'); // hide scanner
        }
    });

    // --- Utility: Animate Number Counting ---
    function animateMetric(element, start, end, duration) {
        let startTime = null;
        
        const step = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            
            // Easing out function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentVal = (start + (end - start) * easeOutQuart).toFixed(2);
            
            element.textContent = currentVal;
            
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                element.textContent = end.toFixed(2);
            }
        };
        window.requestAnimationFrame(step);
    }
});
