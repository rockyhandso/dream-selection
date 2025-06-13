document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('wicketPredictionForm');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Get form data
        const formData = {
            bowler: document.getElementById('bowler').value,
            batsmen: document.getElementById('batsmen').value.split(',').map(name => name.trim()),
            venue: document.getElementById('venue').value,
            pitch: document.getElementById('pitch').value,
            innings: document.getElementById('innings').value,
            overs: document.getElementById('overs').value
        };

        try {
            // Show loading state
            resultDiv.innerHTML = '<div class="alert alert-info">Predicting wickets...</div>';

            // Send prediction request
            const response = await fetch('/predict-wickets', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (response.ok) {
                // Display prediction results
                resultDiv.innerHTML = `
                    <div class="alert alert-success">
                        <h4>Prediction Results</h4>
                        <p>Predicted Wickets: ${data.predicted_wickets}</p>
                        <p>Confidence: ${data.confidence}%</p>
                        <p>Additional Insights:</p>
                        <ul>
                            ${data.insights.map(insight => `<li>${insight}</li>`).join('')}
                        </ul>
                    </div>
                `;
            } else {
                throw new Error(data.error || 'Failed to get prediction');
            }
        } catch (error) {
            resultDiv.innerHTML = `
                <div class="alert alert-danger">
                    Error: ${error.message}
                </div>
            `;
        }
    });
}); 