document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const imageUpload = document.getElementById('imageUpload');
    const predictBtn = document.getElementById('predictBtn');
    const imagePreview = document.getElementById('imagePreview');
    const resultsContainer = document.getElementById('resultsContainer');
    
    // Result display elements
    const mainPrediction = document.getElementById('mainPrediction');
    const confidenceScore = document.getElementById('confidenceScore');
    const uncertainty = document.getElementById('uncertainty');
    const clinicalReport = document.getElementById('clinicalReport');
    const gradCamImage = document.getElementById('gradCamImage');
    const spinner = document.getElementById('spinner');

    let file = null;

    // --- Event Listeners ---

    // Handle file selection
    imageUpload.addEventListener('change', (event) => {
        file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                resultsContainer.classList.add('hidden'); // Hide old results
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle prediction button click
    predictBtn.addEventListener('click', async () => {
        if (!file) {
            alert('Please select an image file first.');
            return;
        }

        // --- UI State Management ---
        spinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errText}`);
            }

            const data = await response.json();
            updateResultsUI(data);

        } catch (error) {
            console.error("Error:", error);
            mainPrediction.textContent = "An error occurred during prediction.";
            resultsContainer.classList.remove('hidden');
        } finally {
            // --- Reset UI State ---
            spinner.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    // --- Helper Functions ---

    /**
     * Updates the UI with the prediction results.
     * @param {object} data - The prediction data from the API.
     */
    function updateResultsUI(data) {
        if (data.error) {
            mainPrediction.textContent = `Error: ${data.error}`;
        } else {
            mainPrediction.textContent = data.prediction || 'N/A';
            confidenceScore.textContent = `Confidence: ${( (data.confidence || 0) * 100).toFixed(2)}%`;
            uncertainty.textContent = `Uncertainty: ${data.uncertainty_classification || 'N/A'}`;
            
            // Display the clinical report with preserved formatting
            clinicalReport.textContent = data.clinical_report || 'No report available.';
            
            // Display the Grad-CAM image
            if (data.grad_cam_image_b64) {
                gradCamImage.src = `data:image/jpeg;base64,${data.grad_cam_image_b64}`;
                gradCamImage.classList.remove('hidden');
            } else {
                gradCamImage.classList.add('hidden');
            }
        }
        resultsContainer.classList.remove('hidden');
    }
});