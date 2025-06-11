(function () {
    const form = document.getElementById('simulationForm');
    const resultBox = document.getElementById('result');
  
    form.addEventListener('submit', function (e) {
      e.preventDefault();
  
      const batsman = document.getElementById('batsman').value.trim();
      const bowlersText = document.getElementById('bowlers').value.trim();
      const venue = document.getElementById('venue').value.trim();
      const pitch = document.getElementById('pitch').value;
      const innings = document.getElementById('innings').value;
  
      if (!batsman || !bowlersText) {
        resultBox.innerHTML = `<p style="color:red;">Please enter all required fields.</p>`;
        return;
      }
  
      const bowlers = bowlersText.split(',').map(b => b.trim()).filter(b => b);
      if (bowlers.length === 0) {
        resultBox.innerHTML = `<p style="color:red;">Please enter at least one bowler.</p>`;
        return;
      }
  
      fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          batsman,
          bowlers,
          venue,
          pitch,
          innings
        })
      })
        .then(res => res.json())
        .then(data => {
          if (data.error) {
            resultBox.innerHTML = `<p style="color:red;">${data.error}</p>`;
          } else if (data.status === 'Not out') {
            resultBox.innerHTML = `<p><strong>${data.batsman}</strong> is predicted to remain <strong>Not Out</strong>.</p>`;
          } else {
            resultBox.innerHTML = `
              <h3>Simulation Result</h3>
              <p><strong>${data.batsman}</strong> is predicted to be dismissed in <strong>over ${data.over}</strong> by <strong>${data.dismissed_by}</strong>.</p>
              <p>Venue: ${venue}, Pitch: ${pitch}, Innings: ${innings}</p>
            `;
          }
        })
        .catch(err => {
          console.error(err);
          resultBox.innerHTML = `<p style="color:red;">Server error occurred. Please try again.</p>`;
        });
    });
  })(); 