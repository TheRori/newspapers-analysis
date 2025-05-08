// Script pour appliquer un style rétro pixel à la timeline
document.addEventListener('DOMContentLoaded', function() {
    console.log('Application des styles rétro à la timeline');
    
    // Ajouter la police rétro pixel
    if (!document.getElementById('pixel-font-link')) {
        const fontLink = document.createElement('link');
        fontLink.id = 'pixel-font-link';
        fontLink.rel = 'stylesheet';
        fontLink.href = 'https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap';
        document.head.appendChild(fontLink);
    }
    
    // Fonction pour appliquer les styles après le chargement de la visualisation
    function applyRetroStyles() {
        // Styles pour les textes des axes
        const axisTexts = document.querySelectorAll('.x-axis text, .y-axis text');
        axisTexts.forEach(text => {
            text.style.fontFamily = '"Press Start 2P", monospace';
            text.style.fill = '#ffffff';
            text.style.fontSize = '8px';
        });
        
        // Styles pour les titres des axes
        const axisTitles = document.querySelectorAll('svg text:not(.x-axis text):not(.y-axis text):not(.event-year-label):not(.legend text)');
        axisTitles.forEach(title => {
            if (title.textContent === 'Année' || title.textContent === 'Fréquence d\'apparition') {
                title.style.fontFamily = '"Press Start 2P", monospace';
                title.style.fill = '#ffffff';
                title.style.fontSize = '10px';
            }
        });
        
        // Styles pour les labels d'année des événements
        const yearLabels = document.querySelectorAll('.event-year-label');
        yearLabels.forEach(label => {
            label.style.fontFamily = '"Press Start 2P", monospace';
            label.style.fill = '#ffffff';
            label.style.fontSize = '10px';
        });
        
        // Masquer les cercles de halo derrière les années
        const haloCircles = document.querySelectorAll('.event-marker-halo');
        haloCircles.forEach(circle => {
            circle.style.display = 'none';
        });
        
        // Styles pour la légende
        const legendTexts = document.querySelectorAll('.legend text');
        legendTexts.forEach(text => {
            text.style.fontFamily = '"Press Start 2P", monospace';
            text.style.fill = '#ffffff';
            text.style.fontSize = '10px';
        });
    }
    
    // Observer pour détecter quand la visualisation est chargée
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length) {
                // Vérifier si des éléments SVG ont été ajoutés
                if (document.querySelector('svg .x-axis, svg .y-axis')) {
                    applyRetroStyles();
                    
                    // Réappliquer les styles après un court délai pour s'assurer que tout est chargé
                    setTimeout(applyRetroStyles, 1000);
                }
            }
        });
    });
    
    // Observer les changements dans le document
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Appliquer les styles immédiatement si la visualisation est déjà chargée
    if (document.querySelector('svg .x-axis, svg .y-axis')) {
        applyRetroStyles();
    }
});
