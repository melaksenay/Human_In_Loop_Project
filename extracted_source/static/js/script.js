// Add hover effect to choice cards
document.addEventListener('DOMContentLoaded', function() {
    console.log('Script loaded, initializing event handlers');
    
    // For the questionnaire page choice cards
    const choiceCards = document.querySelectorAll('.choice-card');
    console.log('Found ' + choiceCards.length + ' choice cards');
    
    choiceCards.forEach(card => {
        card.addEventListener('click', function() {
            console.log('Card clicked');
            // Find the button within this card and click it
            const button = this.querySelector('button');
            if (button) {
                console.log('Clicking button: ' + button.value);
                button.click();
            }
        });
    });

    // Animate progress bar on questionnaire page
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        console.log('Animating progress bar');
        const currentWidth = progressBar.style.width;
        progressBar.style.width = '0%';
        setTimeout(() => {
            progressBar.style.transition = 'width 0.8s ease-in-out';
            progressBar.style.width = currentWidth;
        }, 100);
    }
});