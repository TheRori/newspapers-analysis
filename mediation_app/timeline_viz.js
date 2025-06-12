// Visualisation de la timeline historique
// Ce fichier contient les fonctions pour visualiser l'évolution des termes avec des annotations historiques

// État local pour la timeline
const timelineState = {
    initialized: false,
    vizType: 'line',
    selectedTerms: [],
    startYear: 1950,
    endYear: 2010,
    data: {},
    events: []
};

// Créer le tooltip DOM (hors SVG) s'il n'existe pas déjà
const createTooltip = () => {
    let tooltip = document.querySelector('.timeline-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'timeline-tooltip';
        document.body.appendChild(tooltip);
    }
    return tooltip;
};

// Fonctions pour gérer les tooltips
const showTimelineTooltip = (event, html) => {
    const tooltip = createTooltip();
    tooltip.innerHTML = html;
    tooltip.classList.add('visible');
    tooltip.style.left = (event.pageX + 12) + 'px';
    tooltip.style.top = (event.pageY - 32) + 'px';
};

const hideTimelineTooltip = () => {
    const tooltip = document.querySelector('.timeline-tooltip');
    if (tooltip) {
        tooltip.classList.remove('visible');
    }
};

// Fonction d'initialisation de la timeline
window.initTimelineVisualization = function() {
    console.log('Initialisation de la timeline historique');
    
    // Afficher l'indicateur de chargement
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-indicator';
    loadingDiv.innerHTML = `
        <div class="loading-spinner">
            <div class="spinner-circle"></div>
            <div class="spinner-circle"></div>
            <div class="spinner-circle"></div>
            <div class="spinner-circle"></div>
        </div>
        <p>Chargement des données...</p>
    `;
    const container = document.getElementById('timeline-visualization');
    if (container) {
        container.innerHTML = '';
        container.appendChild(loadingDiv);
    }
    
    // Vérifier que le chargeur de données est disponible
    if (typeof window.dataLoader === 'undefined') {
        console.error('Le chargeur de données global n\'est pas défini');
        displayError('Le chargeur de données n\'est pas disponible');
        return;
    }
    
    // Obtenir les données depuis le chargeur global
    window.dataLoader.getData().then(dataState => {
        if (!dataState || !dataState.isLoaded) {
            console.log('Données pas encore chargées, nouvelle tentative dans 1s');
            setTimeout(window.initTimelineVisualization, 1000);
            return;
        }
        
        // Supprimer l'indicateur de chargement
        if (loadingDiv && loadingDiv.parentNode) {
            loadingDiv.parentNode.removeChild(loadingDiv);
        }
        
        // Initialiser l'état local
        initializeTimelineState(dataState);
        
        // Initialiser l'interface utilisateur
        createTimelineUI(dataState);
        
        // Préparer les données et créer la visualisation
        prepareTimelineData(dataState);
        updateTimelineVisualization(dataState);
        
        // Afficher les événements historiques
        displayTimelineEvents(dataState.timelineEvents);
    }).catch(error => {
        console.error('Erreur lors de l\'initialisation:', error);
        displayError(error.message);
    });
};

// Fonction pour afficher une erreur
function displayError(message) {
    const container = document.getElementById('timeline-visualization');
    if (container) {
        container.innerHTML = `<div class="error-message">Erreur: ${message}</div>`;
    }
}

// Fonction pour initialiser l'état local de la timeline
function initializeTimelineState(dataState) {
    timelineState.initialized = true;
    timelineState.events = dataState.timelineEvents || [];
    
    // Ne pas sélectionner de termes par défaut
    // Laisser l'utilisateur choisir les événements qu'il souhaite afficher
    timelineState.selectedTerms = [];
    
    console.log('Termes sélectionnés pour la timeline:', timelineState.selectedTerms);
}

// Fonction pour créer l'interface utilisateur
function createTimelineUI(dataState) {
    createTimelineTermCheckboxes(dataState.terms);
    createTimelineYearSlider();
    
    // Ajouter l'événement pour le changement de type de visualisation
    const vizTypeSelect = document.getElementById('timeline-viz-type');
    if (vizTypeSelect) {
        vizTypeSelect.addEventListener('change', (e) => {
            timelineState.vizType = e.target.value;
            updateTimelineVisualization(dataState);
        });
    }
}

// Fonction pour mettre en évidence les événements correspondant aux termes sélectionnés
function highlightSelectedEvents() {
    console.log('Mise en évidence des événements pour les termes:', timelineState.selectedTerms);
    
    // D'abord, cacher toutes les infobulles
    d3.selectAll('.event-tooltip')
        .transition()
        .duration(200)
        .attr('opacity', 0);
    
    // Réinitialiser tous les marqueurs
    d3.selectAll('.event-marker')
        .transition()
        .duration(200)
        .attr('r', 6)
        .attr('stroke-width', 2);
    
    d3.selectAll('.event-marker-halo')
        .transition()
        .duration(200)
        .attr('r', 12)
        .attr('opacity', 0.3);
    
    // Si aucun terme n'est sélectionné, ne rien mettre en évidence
    if (!timelineState.selectedTerms || timelineState.selectedTerms.length === 0) {
        return;
    }
    
    // Parcourir tous les événements
    Object.values(window.timelineEventMap || {}).forEach(event => {
        if (!event || !event.keywords) return;
        
        // Vérifier si au moins un des termes de l'événement est sélectionné
        const isHighlighted = event.keywords.some(keyword => 
            timelineState.selectedTerms.includes(keyword)
        );
        
        if (isHighlighted) {
            console.log('Mise en évidence de l\'\u00e9vénement:', event.title);
            
            const markerId = `event-marker-${event.title.replace(/[\s'\-\+\&\:\[\]\(\)\.\,]/g, '_')}`;
            const markerGroup = d3.select(`#${markerId}`);
            
            if (!markerGroup.empty()) {
                // Mettre en évidence l'événement
                markerGroup.select('.event-marker')
                    .transition()
                    .duration(500)
                    .attr('r', 8)
                    .attr('stroke-width', 3);
                
                markerGroup.select('.event-marker-halo')
                    .transition()
                    .duration(500)
                    .attr('r', 16)
                    .attr('opacity', 0.5);
                
                // Afficher l'infobulle avec un délai pour éviter les chevauchements
                markerGroup.select('.event-tooltip')
                    .transition()
                    .delay(300) // Délai pour que l'animation soit plus fluide
                    .duration(500)
                    .attr('opacity', 1);
            }
        }
    });
}

// Fonction pour créer les checkboxes des termes
function createTimelineTermCheckboxes(terms) {
    const container = document.getElementById('timeline-term-checkboxes');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Déterminer quels termes utiliser
    let availableTerms = terms;
    
    // Si les données spécifiques aux événements historiques sont disponibles, utiliser ces termes
    if (window.dataLoader.state.timelineData && window.dataLoader.state.timelineData.length > 0) {
        availableTerms = Object.keys(window.dataLoader.state.timelineData[0]).filter(key => key !== 'key');
        console.log('Utilisation des termes des événements historiques:', availableTerms);
    }
    
    // Organiser les termes par événement
    const eventGroups = {};
    const ungroupedTerms = [];
    
    if (window.dataLoader.state.timelineEvents) {
        // Créer un groupe pour chaque événement
        window.dataLoader.state.timelineEvents.forEach(event => {
            if (event.keywords && event.keywords.length > 0) {
                const eventTerms = event.keywords.filter(keyword => availableTerms.includes(keyword));
                
                if (eventTerms.length > 0) {
                    if (!eventGroups[event.title]) {
                        eventGroups[event.title] = {
                            title: event.title,
                            year: event.year,
                            category: event.category,
                            terms: [],
                            description: event.description
                        };
                    }
                    eventGroups[event.title].terms = eventGroups[event.title].terms.concat(eventTerms);
                }
            }
        });
        
        // Identifier les termes qui ne sont pas associés à un événement
        availableTerms.forEach(term => {
            const isInEvent = Object.values(eventGroups).some(group => 
                group.terms.includes(term)
            );
            
            if (!isInEvent) {
                ungroupedTerms.push(term);
            }
        });
    } else {
        // Si pas d'événements, tous les termes sont non groupés
        ungroupedTerms.push(...availableTerms);
    }
    
    // Trier les événements par année
    const sortedEvents = Object.values(eventGroups).sort((a, b) => a.year - b.year);
    
    // Créer les checkboxes pour les événements
    sortedEvents.forEach(event => {
        const eventDiv = document.createElement('div');
        eventDiv.className = 'event-checkbox-group';
        
        const eventTitle = document.createElement('div');
        eventTitle.className = 'event-title';
        eventTitle.innerHTML = `<strong>${event.title}</strong> (${event.year})`;
        eventTitle.title = event.description;
        eventDiv.appendChild(eventTitle);
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `timeline-event-${event.title.replace(/\s+/g, '-')}`;
        checkbox.className = 'event-checkbox';
        
        // Vérifier si tous les termes de l'événement sont sélectionnés
        const allTermsSelected = event.terms.every(term => 
            timelineState.selectedTerms.includes(term)
        );
        checkbox.checked = allTermsSelected && event.terms.length > 0;
        
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                // Ajouter tous les termes de l'événement
                event.terms.forEach(term => {
                    if (!timelineState.selectedTerms.includes(term)) {
                        timelineState.selectedTerms.push(term);
                    }
                });
            } else {
                // Retirer tous les termes de l'événement
                timelineState.selectedTerms = timelineState.selectedTerms.filter(term => 
                    !event.terms.includes(term)
                );
            }
            
            // Limiter le nombre de termes sélectionnés
            const maxTerms = window.dataLoader.config.maxTermsToShow || 8;
            if (timelineState.selectedTerms.length > maxTerms) {
                timelineState.selectedTerms = timelineState.selectedTerms.slice(0, maxTerms);
                alert(`Maximum ${maxTerms} termes peuvent être sélectionnés simultanément.`);
            }
            
            // Mettre à jour la visualisation
            prepareTimelineData(window.dataLoader.state);
            updateTimelineVisualization(window.dataLoader.state);
            
            // Mettre en évidence les événements correspondant aux termes sélectionnés
            setTimeout(highlightSelectedEvents, 1000); // Délai pour permettre au graphique de se dessiner d'abord
        });
        
        eventTitle.insertBefore(checkbox, eventTitle.firstChild);
        container.appendChild(eventDiv);
    });
    
    // Ajouter une section pour les termes non groupés (si nécessaire)
    if (ungroupedTerms.length > 0) {
        const ungroupedDiv = document.createElement('div');
        ungroupedDiv.className = 'ungrouped-terms';
        ungroupedDiv.innerHTML = '<h4>Autres termes</h4>';
        
        ungroupedTerms.forEach(term => {
            const isChecked = timelineState.selectedTerms.includes(term);
            
            const checkboxDiv = document.createElement('div');
            checkboxDiv.className = 'term-checkbox';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `timeline-term-${term}`;
            checkbox.value = term;
            checkbox.checked = isChecked;
            
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    if (!timelineState.selectedTerms.includes(term)) {
                        timelineState.selectedTerms.push(term);
                    }
                } else {
                    timelineState.selectedTerms = timelineState.selectedTerms.filter(t => t !== term);
                }
                
                // Limiter le nombre de termes sélectionnés
                const maxTerms = window.dataLoader.config.maxTermsToShow || 8;
                if (timelineState.selectedTerms.length > maxTerms) {
                    timelineState.selectedTerms = timelineState.selectedTerms.slice(0, maxTerms);
                    document.getElementById(`timeline-term-${term}`).checked = false;
                    alert(`Maximum ${maxTerms} termes peuvent être sélectionnés simultanément.`);
                }
                
                // Mettre à jour la visualisation
                prepareTimelineData(window.dataLoader.state);
                updateTimelineVisualization(window.dataLoader.state);
            });
            
            const label = document.createElement('label');
            label.htmlFor = `timeline-term-${term}`;
            label.textContent = term;
            
            checkboxDiv.appendChild(checkbox);
            checkboxDiv.appendChild(label);
            ungroupedDiv.appendChild(checkboxDiv);
        });
        
        container.appendChild(ungroupedDiv);
    }
}

// Fonction pour créer le slider des années
function createTimelineYearSlider() {
    const container = document.getElementById('timeline-year-slider');
    if (!container) return;
    
    container.innerHTML = '';
    
    const sliderContainer = document.createElement('div');
    sliderContainer.className = 'year-slider-container';
    
    const slider = document.createElement('div');
    slider.id = 'timeline-year-range';
    sliderContainer.appendChild(slider);
    
    const valueDisplay = document.createElement('div');
    valueDisplay.className = 'year-range-values';
    valueDisplay.innerHTML = `<span id="timeline-year-min">${timelineState.startYear}</span> - <span id="timeline-year-max">${timelineState.endYear}</span>`;
    sliderContainer.appendChild(valueDisplay);
    
    container.appendChild(sliderContainer);
    
    // Créer le slider avec noUiSlider
    if (typeof noUiSlider !== 'undefined') {
        noUiSlider.create(slider, {
            start: [timelineState.startYear, timelineState.endYear],
            connect: true,
            step: 1,
            range: {
                'min': 1950,
                'max': 2010
            },
            format: {
                to: value => Math.round(value),
                from: value => Math.round(value)
            }
        });
        
        // Mettre à jour l'affichage et les données lors du changement
        slider.noUiSlider.on('update', (values, handle) => {
            const [startYear, endYear] = values;
            document.getElementById('timeline-year-min').textContent = startYear;
            document.getElementById('timeline-year-max').textContent = endYear;
            
            timelineState.startYear = parseInt(startYear);
            timelineState.endYear = parseInt(endYear);
        });
        
        // Mettre à jour la visualisation à la fin du glissement
        slider.noUiSlider.on('change', () => {
            updateTimelineVisualization(window.dataLoader.state);
            displayTimelineEvents(window.dataLoader.state.timelineEvents);
        });
    } else {
        console.error('noUiSlider n\'est pas défini. Impossible de créer le slider.');
    }
}

// Fonction pour préparer les données de la timeline
function prepareTimelineData(dataState) {
    console.log('Préparation des données de la timeline');
    console.log('État de dataState:', dataState);
    
    // Vérifier si nous avons les données spécifiques aux événements historiques
    if (dataState.timelineData && dataState.timelineData.length > 0) {
        console.log('Utilisation des données spécifiques aux événements historiques');
        prepareHistoricalEventsData(dataState);
        return;
    }
    
    // Si les données spécifiques ne sont pas disponibles, utiliser les données générales
    if (!dataState.data || dataState.data.length === 0) {
        console.error('Aucune donnée disponible pour la visualisation de la timeline');
        return;
    }
    
    console.log('Données disponibles:', dataState.data);
    
    // Filtrer les données par année
    const filteredData = dataState.data.filter(d => {
        const year = parseInt(d.key);
        return !isNaN(year) && year >= 1950 && year <= 2010;
    });
    
    console.log('Données filtrées par année:', filteredData);
    console.log('Termes sélectionnés:', timelineState.selectedTerms);
    
    // Créer un objet avec les données pour chaque terme
    const timelineData = {};
    
    // Pour chaque terme sélectionné
    timelineState.selectedTerms.forEach(term => {
        console.log('Traitement du terme:', term);
        
        // Créer un tableau pour stocker les données de ce terme
        const termData = [];
        
        // Pour chaque année, ajouter les données
        filteredData.forEach(d => {
            const year = parseInt(d.key);
            const value = parseFloat(d[term]) || 0;
            
            termData.push({
                year: year,
                value: value
            });
        });
        
        console.log(`Données pour le terme "${term}":`, termData);
        
        // Trier les données par année
        termData.sort((a, b) => a.year - b.year);
        
        // Stocker les données du terme
        timelineData[term] = termData;
    });
    
    // Mettre à jour l'état
    timelineState.data = timelineData;
    console.log('État final des données de la timeline:', timelineState.data);
}

// Fonction pour préparer les données spécifiques aux événements historiques
function prepareHistoricalEventsData(dataState) {
    console.log('Préparation des données spécifiques aux événements historiques');
    console.log('Données de timeline disponibles:', dataState.timelineData);
    
    // Extraire les termes disponibles dans les données de timeline
    const availableTerms = Object.keys(dataState.timelineData[0]).filter(key => key !== 'key');
    console.log('Termes disponibles dans les données de timeline:', availableTerms);
    
    // Mettre à jour les termes disponibles dans l'état
    dataState.terms = availableTerms;
    
    // Conserver uniquement les termes valides parmi ceux sélectionnés
    const validSelectedTerms = timelineState.selectedTerms.filter(term => availableTerms.includes(term));
    timelineState.selectedTerms = validSelectedTerms;
    
    // Afficher un message dans la console pour le débogage
    console.log('Termes sélectionnés valides:', timelineState.selectedTerms);
    
    // Filtrer les données par année
    const filteredData = dataState.timelineData.filter(d => {
        const year = parseInt(d.key);
        return !isNaN(year) && year >= timelineState.startYear && year <= timelineState.endYear;
    });
    
    console.log('Données filtrées par année:', filteredData);
    
    // Créer un objet avec les données pour chaque terme sélectionné
    const timelineData = {};
    
    // Pour chaque terme sélectionné
    timelineState.selectedTerms.forEach(term => {
        console.log('Traitement du terme:', term);
        
        // Créer un tableau pour stocker les données de ce terme
        const termData = [];
        
        // Pour chaque année, ajouter les données
        filteredData.forEach(d => {
            const year = parseInt(d.key);
            // S'assurer que la valeur est un nombre valide
            let value = 0;
            if (d[term] !== undefined && d[term] !== null) {
                value = parseFloat(d[term]);
                if (isNaN(value)) value = 0;
            }
            
            termData.push({
                year: year,
                value: value
            });
            
            console.log(`Année ${year}, terme "${term}", valeur: ${value}`);
        });
        
        // Trier les données par année
        termData.sort((a, b) => a.year - b.year);
        
        // Vérifier si nous avons des données valides
        const hasValidData = termData.some(d => d.value > 0);
        console.log(`Le terme "${term}" a-t-il des données valides? ${hasValidData}`);
        
        // Stocker les données du terme
        timelineData[term] = termData;
    });
    
    // Mettre à jour l'état
    timelineState.data = timelineData;
    console.log('État final des données de la timeline avec les événements historiques:', timelineState.data);
}

// Fonction pour mettre à jour la visualisation de la timeline
function updateTimelineVisualization(dataState) {
    const container = document.getElementById('timeline-visualization');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Vérifier qu'il y a des termes sélectionnés
    if (timelineState.selectedTerms.length === 0) {
        container.innerHTML = '<div class="info-message">Veuillez sélectionner au moins un terme à visualiser.</div>';
        return;
    }
    
    // Créer la visualisation en fonction du type sélectionné
    if (timelineState.vizType === 'line') {
        createTimelineLineChart(container, dataState);
    } else if (timelineState.vizType === 'stacked') {
        createTimelineStackedAreaChart(container, dataState);
    }
}

// Fonction pour créer un graphique en ligne pour la timeline
function createTimelineLineChart(container, dataState) {
    console.log('Création du graphique en ligne pour la timeline');
    
    // Dimensions et marges
    const width = container.clientWidth;
    const height = 700; // Hauteur significativement augmentée
    const margin = { top: 80, right: 120, bottom: 120, left: 60 }; // Marge inférieure augmentée aussi
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Nettoyer le conteneur avant de créer un nouveau graphique
    container.innerHTML = '';
    
    // Créer une div conteneur pour appliquer les styles
    const timelineCard = document.createElement('div');
    timelineCard.className = 'timeline-card';
    container.appendChild(timelineCard);
    
    // Créer le SVG avec un fond légèrement transparent et hauteur forcée
    const svg = d3.select(timelineCard)
        .append('svg')
        .attr('class', 'timeline-svg')
        .attr('width', width)
        .attr('height', 900) // Hauteur forcée à 900px
        .style('min-height', '900px'); // Assure une hauteur minimale
    
    // Ajouter un fond pour le graphique
    svg.append('rect')
        .attr('width', width)
        .attr('height', 900) // Hauteur forcée à 900px comme le SVG
        .attr('fill', '#1a1a1a')
        .attr('rx', 8) // Coins arrondis
        .attr('ry', 8);
    
    // Créer le groupe principal
    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Échelle X (années) - Définir x avant de l'utiliser
    const x = d3.scaleLinear()
        .domain([timelineState.startYear, timelineState.endYear])
        .range([0, innerWidth]);
    
    // Ajouter une grille de fond
    const gridColor = 'rgba(255, 255, 255, 0.1)';
    
    // Grille horizontale
    g.append('g')
        .attr('class', 'grid-lines horizontal-grid')
        .selectAll('line')
        .data(d3.range(0, innerHeight + 1, innerHeight / 5))
        .enter()
        .append('line')
        .attr('x1', 0)
        .attr('y1', d => d)
        .attr('x2', innerWidth)
        .attr('y2', d => d)
        .attr('stroke', gridColor)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');
    
    // Grille verticale (pour les décennies)
    const decades = [];
    for (let year = Math.ceil(timelineState.startYear / 10) * 10; year <= timelineState.endYear; year += 10) {
        decades.push(year);
    }
    
    g.append('g')
        .attr('class', 'grid-lines vertical-grid')
        .selectAll('line')
        .data(decades)
        .enter()
        .append('line')
        .attr('x1', d => x(d))
        .attr('y1', 0)
        .attr('x2', d => x(d))
        .attr('y2', innerHeight)
        .attr('stroke', gridColor)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');
    
    // Trouver la valeur maximale pour l'échelle Y
    let maxValue = 0;
    timelineState.selectedTerms.forEach(term => {
        const termData = timelineState.data[term];
        if (termData) {
            const filteredData = termData.filter(d => d.year >= timelineState.startYear && d.year <= timelineState.endYear);
            const termMax = d3.max(filteredData, d => d.value);
            if (termMax > maxValue) maxValue = termMax;
        }
    });
    
    // Échelle Y (valeurs)
    const y = d3.scaleLinear()
        .domain([0, maxValue * 1.1]) // Ajouter 10% d'espace en haut
        .range([innerHeight, 0]);
    
    // Échelle de couleur avec des couleurs plus vives
    const color = d3.scaleOrdinal()
        .domain(timelineState.selectedTerms)
        .range(['#00bfff', '#ff7f50', '#9370db', '#32cd32', '#ff69b4', '#ffd700', '#ff6347', '#7fffd4']);
    
    // Créer les axes avec un style amélioré
    const xAxis = d3.axisBottom(x)
        .tickFormat(d => d)
        .ticks(10);
    
    const yAxis = d3.axisLeft(y)
        .ticks(5)
        .tickFormat(d => d.toFixed(1));
    
    // Ajouter les axes avec un style amélioré
    g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(xAxis)
        .selectAll('text')
        .attr('y', 10)
        .attr('x', 0)
        .attr('dy', '.35em')
        .attr('fill', '#ffffff')
        .attr('font-size', '12px')
        .style('text-anchor', 'middle');
    
    g.append('g')
        .attr('class', 'y-axis')
        .call(yAxis)
        .selectAll('text')
        .attr('fill', '#ffffff')
        .attr('font-size', '12px');
    
    // Styliser les lignes des axes
    g.selectAll('.domain, .tick line')
        .attr('stroke', '#ffffff')
        .attr('stroke-opacity', 0.5);
    
    // Ajouter le titre de l'axe Y avec un style amélioré
    g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -45)
        .attr('x', -innerHeight / 2)
        .attr('dy', '1em')
        .attr('fill', '#ffffff')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .style('text-anchor', 'middle')
        .text('Fréquence d\'apparition');
    
    // Ajouter le titre de l'axe X avec un style amélioré
    g.append('text')
        .attr('y', innerHeight + 45)
        .attr('x', innerWidth / 2)
        .attr('dy', '1em')
        .attr('fill', '#ffffff')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .style('text-anchor', 'middle')
        .text('Année');
    
    // Créer la fonction de ligne avec une courbe plus lisse
    const line = d3.line()
        .x(d => x(d.year))
        .y(d => y(d.value))
        .curve(d3.curveCatmullRom.alpha(0.5)); // Courbe plus lisse
    
    // Créer le tooltip DOM (hors SVG) s'il n'existe pas déjà
    let tooltip = document.querySelector('.timeline-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'timeline-tooltip';
        document.body.appendChild(tooltip);
    }
    
    // Fonctions pour gérer les tooltips
    function showTimelineTooltip(event, html) {
        tooltip.innerHTML = html;
        tooltip.classList.add('visible');
        tooltip.style.left = (event.pageX + 12) + 'px';
        tooltip.style.top = (event.pageY - 32) + 'px';
    }
    
    function hideTimelineTooltip() {
        tooltip.classList.remove('visible');
    }

    // Ajouter les lignes pour chaque terme avec animation et effets
    timelineState.selectedTerms.forEach(term => {
        const termData = timelineState.data[term];
        if (termData) {
            // Filtrer les données pour la plage d'années sélectionnée
            const filteredData = termData.filter(d => d.year >= timelineState.startYear && d.year <= timelineState.endYear);
            
            console.log(`Données filtrées pour le terme "${term}":`, filteredData);
            
            // Vérifier si nous avons des données valides
            if (filteredData.length === 0) {
                console.warn(`Aucune donnée valide pour le terme "${term}"`);
                return;
            }
            
            // Vérifier si nous avons au moins une valeur non nulle
            const hasNonZeroValues = filteredData.some(d => d.value > 0);
            if (!hasNonZeroValues) {
                console.warn(`Toutes les valeurs sont nulles pour le terme "${term}"`);
                // Ajouter quand même une petite valeur pour rendre la ligne visible
                filteredData.forEach(d => {
                    if (d.year % 5 === 0) { // Ajouter une valeur tous les 5 ans
                        d.value = 0.1;
                    }
                });
            }
            
            // Créer un gradient pour la ligne
            const gradientId = `line-gradient-${term.replace(/\s+/g, '-')}`;
            const gradient = svg.append('linearGradient')
                .attr('id', gradientId)
                .attr('gradientUnits', 'userSpaceOnUse')
                .attr('x1', 0)
                .attr('y1', 0)
                .attr('x2', innerWidth)
                .attr('y2', 0);
                
            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', color(term))
                .attr('stop-opacity', 0.8);
                
            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', color(term))
                .attr('stop-opacity', 1);
            
            // Créer un effet de lueur (glow)
            const glowId = `glow-${term.replace(/\s+/g, '-')}`;
            const filter = svg.append('defs')
                .append('filter')
                .attr('id', glowId)
                .attr('x', '-50%')
                .attr('y', '-50%')
                .attr('width', '200%')
                .attr('height', '200%');
                
            filter.append('feGaussianBlur')
                .attr('stdDeviation', '2.5')
                .attr('result', 'coloredBlur');
                
            const feMerge = filter.append('feMerge');
            feMerge.append('feMergeNode')
                .attr('in', 'coloredBlur');
            feMerge.append('feMergeNode')
                .attr('in', 'SourceGraphic');
            
            // Ajouter la ligne au graphique
            const path = g.append('path')
                .datum(filteredData)
                .attr('class', `line line-${term.replace(/\s+/g, '-')}`)
                .attr('d', line)
                .attr('fill', 'none')
                .attr('stroke', `url(#${gradientId})`)
                .attr('stroke-width', 3.5)
                .attr('filter', `url(#${glowId})`)
                .attr('opacity', 0.9);
                
            // Ajouter des points interactifs pour ce terme
            g.selectAll(`.timeline-point-${term.replace(/\s+/g, '-')}`)
                .data(filteredData)
                .enter()
                .append('circle')
                .attr('class', `timeline-point timeline-point-${term.replace(/\s+/g, '-')}`)
                .attr('cx', d => x(d.year))
                .attr('cy', d => y(d.value))
                .attr('r', 7)
                .attr('fill', color(term))
                .attr('stroke', '#fff')
                .attr('stroke-width', 2)
                .style('cursor', 'pointer')
                .on('mouseover', function(event, d) {
                    d3.select(this).classed('active', true);
                    showTimelineTooltip(event, `<b>${term}</b><br>Année : <b>${d.year}</b><br>Valeur : <b>${d.value}</b>`);
                })
                .on('mouseout', function() {
                    d3.select(this).classed('active', false);
                    hideTimelineTooltip();
                })
                .on('click', function(event, d) {
                    d3.selectAll('.timeline-point').classed('active', false);
                    d3.select(this).classed('active', true);
                    showTimelineTooltip(event, `<b>${term}</b><br>Année : <b>${d.year}</b><br>Valeur : <b>${d.value}</b>`);
                });
            
            // Animation de la ligne
            const pathLength = path.node().getTotalLength();
            path.attr('stroke-dasharray', pathLength)
                .attr('stroke-dashoffset', pathLength)
                .transition()
                .duration(1500)
                .attr('stroke-dashoffset', 0);
            
            // Ajouter des points pour chaque donnée
            // Créer un identifiant CSS valide en remplaçant tous les caractères spéciaux
            const safeTermClass = `point-${term.replace(/[\s'\-\+\&\:\[\]\(\)\.\,]/g, '_')}`;
            const points = g.selectAll(`.${safeTermClass}`)
                .data(filteredData)
                .enter()
                .append('circle')
                .attr('class', safeTermClass)
                .attr('cx', d => x(d.year))
                .attr('cy', d => y(d.value))
                .attr('r', 0) // Commencer avec un rayon de 0 pour l'animation
                .attr('fill', color(term))
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 1)
                .style('cursor', 'pointer') // Changer le curseur pour indiquer que c'est cliquable
                .transition()
                .delay((d, i) => i * 50) // Délai progressif pour chaque point
                .duration(500)
                .attr('r', 4); // Taille finale du point
            
            // Ajouter l'interactivité aux points (après la transition)
            g.selectAll(`.${safeTermClass}`)
                .on('click', function(event, d) {
                    console.log(`Point cliqué: terme "${term}", année ${d.year}`);
                    // Appeler la fonction pour afficher les articles correspondants
                    if (window.showArticlesForTermAndYear) {
                        window.showArticlesForTermAndYear(term, d.year);
                    }
                })
                .on('mouseover', function() {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .attr('r', 6)
                        .attr('stroke-width', 2);
                })
                .on('mouseout', function() {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .attr('r', 4)
                        .attr('stroke-width', 1);
                });
        }
    });
    
    // Ajouter la légende avec un style amélioré
    const legendBackground = svg.append('rect')
        .attr('x', margin.left + innerWidth - 120)
        .attr('y', margin.top - 10)
        .attr('width', 110)
        .attr('height', timelineState.selectedTerms.length * 25 + 10)
        .attr('fill', 'rgba(0, 0, 0, 0.7)')
        .attr('rx', 5)
        .attr('ry', 5);
    
    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${margin.left + innerWidth - 110}, ${margin.top})`);
    
    timelineState.selectedTerms.forEach((term, i) => {
        const legendRow = legend.append('g')
            .attr('transform', `translate(0, ${i * 25})`);
        
        legendRow.append('rect')
            .attr('width', 12)
            .attr('height', 12)
            .attr('rx', 2)
            .attr('ry', 2)
            .attr('fill', color(term));
        
        legendRow.append('text')
            .attr('x', 20)
            .attr('y', 10)
            .attr('fill', '#ffffff')
            .attr('font-size', '12px')
            .text(term);
    });
    
    // Ajouter les annotations pour les événements historiques
    addTimelineAnnotations(svg, x, y, margin, dataState.timelineEvents);
}

// Fonction pour créer un graphique en aires empilées pour la timeline
function createTimelineStackedAreaChart(container, dataState) {
    // Dimensions et marges
    const width = container.clientWidth;
    const height = 800;
    const margin = { top: 60, right: 80, bottom: 80, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Créer le SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Créer le groupe principal
    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Préparer les données pour le graphique empilé
    const years = [];
    for (let year = timelineState.startYear; year <= timelineState.endYear; year++) {
        years.push(year);
    }
    
    const stackData = years.map(year => {
        const yearData = { year };
        timelineState.selectedTerms.forEach(term => {
            const termData = timelineState.data[term];
            if (termData) {
                const yearEntry = termData.find(d => d.year === year);
                yearData[term] = yearEntry ? yearEntry.value : 0;
            } else {
                yearData[term] = 0;
            }
        });
        return yearData;
    });
    
    // Échelle X (années)
    const x = d3.scaleLinear()
        .domain([timelineState.startYear, timelineState.endYear])
        .range([0, innerWidth]);
    
    // Créer le stack
    const stack = d3.stack()
        .keys(timelineState.selectedTerms)
        .order(d3.stackOrderNone)
        .offset(d3.stackOffsetNone);
    
    const stackedData = stack(stackData);
    
    // Trouver la valeur maximale pour l'échelle Y
    const yMax = d3.max(stackedData, layer => d3.max(layer, d => d[1]));
    
    // Échelle Y (valeurs)
    const y = d3.scaleLinear()
        .domain([0, yMax * 1.1]) // Ajouter 10% d'espace en haut
        .range([innerHeight, 0]);
    
    // Échelle de couleur
    const color = d3.scaleOrdinal()
        .domain(timelineState.selectedTerms)
        .range(window.dataLoader.config.colors || d3.schemeCategory10);
    
    // Créer les axes
    const xAxis = d3.axisBottom(x)
        .tickFormat(d => d)
        .ticks(10);
    
    const yAxis = d3.axisLeft(y);
    
    // Ajouter les axes
    g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(xAxis)
        .selectAll('text')
        .attr('y', 10)
        .attr('x', 0)
        .attr('dy', '.35em')
        .attr('transform', 'rotate(0)')
        .style('text-anchor', 'middle');
    
    g.append('g')
        .attr('class', 'y-axis')
        .call(yAxis);
    
    // Ajouter le titre de l'axe Y
    g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -40)
        .attr('x', -innerHeight / 2)
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .text('Fréquence d\'apparition');
    
    // Ajouter le titre de l'axe X
    g.append('text')
        .attr('y', innerHeight + 40)
        .attr('x', innerWidth / 2)
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .text('Année');
    
    // Créer la fonction d'aire
    const area = d3.area()
        .x(d => x(d.data.year))
        .y0(d => y(d[0]))
        .y1(d => y(d[1]))
        .curve(d3.curveMonotoneX);
    
    // Ajouter les aires pour chaque terme
    g.selectAll('.area')
        .data(stackedData)
        .enter()
        .append('path')
        .attr('class', 'area')
        .attr('fill', (d, i) => color(timelineState.selectedTerms[i]))
        .attr('d', area)
        .attr('opacity', 0.8);
    
    // Ajouter la légende
    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${margin.left + innerWidth - 90}, ${margin.top})`);
    
    timelineState.selectedTerms.forEach((term, i) => {
        const legendRow = legend.append('g')
            .attr('transform', `translate(0, ${i * 20})`);
        
        legendRow.append('rect')
            .attr('width', 10)
            .attr('height', 10)
            .attr('fill', color(term));
        
        legendRow.append('text')
            .attr('x', 15)
            .attr('y', 10)
            .text(term);
    });
    
    // Ajouter les annotations pour les événements historiques
    addTimelineAnnotations(svg, x, y, margin, dataState.timelineEvents);
}

// Fonction pour créer un graphique en ligne pour la timeline
function createTimelineLineChart(container, dataState) {
    // Dimensions et marges
    const width = container.clientWidth;
    const height = 500;
    const margin = { top: 60, right: 80, bottom: 180, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Créer le SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Créer le groupe principal
    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Échelle X (années)
    const x = d3.scaleLinear()
        .domain([timelineState.startYear, timelineState.endYear])
        .range([0, innerWidth]);
    
    // Échelle Y (valeurs)
    const y = d3.scaleLinear()
        .domain([0, 100]) // Valeurs par défaut pour l'axe Y
        .range([innerHeight, 0]);
    
    // Échelle de couleur
    const color = d3.scaleOrdinal()
        .domain(timelineState.selectedTerms)
        .range(window.dataLoader.config.colors || d3.schemeCategory10);
    
    // Créer les axes
    const xAxis = d3.axisBottom(x)
        .tickFormat(d => d)
        .ticks(10);
    
    const yAxis = d3.axisLeft(y);
    
    // Ajouter les axes
    g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(xAxis)
        .selectAll('text')
        .attr('y', 10)
        .attr('x', 0)
        .attr('dy', '.35em')
        .attr('transform', 'rotate(0)')
        .style('text-anchor', 'middle');
    
    g.append('g')
        .attr('class', 'y-axis')
        .call(yAxis);
    
    // Ajouter le titre de l'axe Y
    g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -40)
        .attr('x', -innerHeight / 2)
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .text('Fréquence d\'apparition');
    
    // Ajouter le titre de l'axe X
    g.append('text')
        .attr('y', innerHeight + 40)
        .attr('x', innerWidth / 2)
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .text('Année');
    
    // Créer la fonction de ligne
    const line = d3.line()
        .x(d => x(d.year))
        .y(d => y(d.value));
    
    // Ajouter les lignes pour chaque terme
    timelineState.selectedTerms.forEach(term => {
        const termData = timelineState.data[term];
        if (termData) {
            g.append('path')
                .datum(termData)
                .attr('class', 'line')
                .attr('fill', 'none')
                .attr('stroke', color(term))
                .attr('stroke-width', 1.5)
                .attr('d', line);
        }
    });
    
    // Ajouter les points pour chaque terme
    timelineState.selectedTerms.forEach(term => {
        const termData = timelineState.data[term];
        if (termData) {
            // Créer un identifiant CSS valide en échappant les caractères spéciaux
            const safeTermClass = `term-${term.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase()}`;
            
            g.selectAll(`.${safeTermClass}`)
                .data(termData)
                .enter()
                .append('circle')
                .attr('class', safeTermClass)
                .attr('cx', d => x(d.year))
                .attr('cy', d => y(d.value))
                .attr('r', 0) // Commencer avec un rayon de 0 pour l'animation
                .attr('fill', color(term))
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 1)
                .style('cursor', 'pointer') // Changer le curseur pour indiquer que c'est cliquable
                .transition()
                .delay((d, i) => i * 50) // Délai progressif pour chaque point
                .duration(500)
                .attr('r', 4); // Taille finale du point
            
            // Ajouter l'interactivité aux points (après la transition)
            g.selectAll(`.${safeTermClass}`)
                .on('click', function(event, d) {
                    console.log(`Point cliqué: terme "${term}", année ${d.year}`);
                    // Appeler la fonction pour afficher les articles correspondants
                    if (window.showArticlesForTermAndYear) {
                        window.showArticlesForTermAndYear(term, d.year);
                    }
                })
                .on('mouseover', function() {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .attr('r', 6)
                        .attr('stroke-width', 2);
                })
                .on('mouseout', function() {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .attr('r', 4)
                        .attr('stroke-width', 1);
                });
        }
    });
    
    // Ajouter la légende
    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${margin.left + innerWidth - 150}, ${margin.top})`);
    
    timelineState.selectedTerms.forEach((term, i) => {
        const legendItem = legend.append('g')
            .attr('transform', `translate(0, ${i * 25})`);
            
        legendItem.append('rect')
            .attr('width', 15)
            .attr('height', 15)
            .attr('fill', color(term));
            
        legendItem.append('text')
            .attr('x', 20)
            .attr('y', 12)
            .text(term)
            .attr('fill', '#ffffff')
            .attr('font-size', '10px')
            .attr('font-family', '"Press Start 2P", monospace');
    });
    
    // Créer un groupe invisible pour les événements historiques (sans marqueurs visibles)
    if (dataState.timelineEvents && Array.isArray(dataState.timelineEvents)) {
        const eventGroups = g.selectAll('.timeline-event-group')
            .data(dataState.timelineEvents)
            .enter()
            .append('g')
            .attr('class', 'timeline-event-group')
            .attr('transform', d => `translate(${x(d.year)}, -18)`) // Positionner au-dessus du graphique
            .on('mouseover', function(event, d) {
                d3.select(this).classed('active', true);
                showTimelineTooltip(event, `<b>${d.label || d.title}</b><br>${d.description || ''}<br><span style='color:#ffd700;font-size:0.9em;'>Événement historique</span>`);
            })
            .on('mouseout', function() {
                d3.select(this).classed('active', false);
                hideTimelineTooltip();
            })
            .on('click', function(event, d) {
                d3.selectAll('.timeline-event-group').classed('active', false);
                d3.select(this).classed('active', true);
                showTimelineTooltip(event, `<b>${d.label || d.title}</b><br>${d.description || ''}<br><span style='color:#ffd700;font-size:0.9em;'>Événement historique</span>`);
            });
            
        // Ajouter une zone de détection invisible pour les interactions
        eventGroups.append('rect')
            .attr('x', -15)
            .attr('y', -15)
            .attr('width', 30)
            .attr('height', 30)
            .attr('fill', 'transparent');
    }
    
    // Ajouter les annotations pour les événements historiques
    addTimelineAnnotations(svg, x, y, margin, dataState.timelineEvents);
}

// Fonction pour ajouter les annotations des événements historiques au graphique
function addTimelineAnnotations(svg, x, y, margin, events) {
    // Filtrer les événements dans la plage d'années sélectionnée
    const filteredEvents = events.filter(event => 
        event.year >= timelineState.startYear && event.year <= timelineState.endYear);
        
    // Créer un mapping des événements par titre pour faciliter la mise en évidence
    window.timelineEventMap = {};
    filteredEvents.forEach(event => {
        window.timelineEventMap[event.title] = event;
    });
    
    // Fonction pour vérifier si un événement doit être mis en évidence
    window.checkIfEventIsHighlighted = function(event) {
        if (!event || !event.keywords || !timelineState.selectedTerms) return false;
        
        // Vérifier si au moins un des termes de l'événement est sélectionné
        return event.keywords.some(keyword => timelineState.selectedTerms.includes(keyword));
    };
    
    // Ajouter un groupe pour les annotations
    const annotationGroup = svg.append('g')
        .attr('class', 'annotations')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
    
    // Créer un effet de lueur pour les événements
    const eventGlow = svg.append('defs')
        .append('filter')
        .attr('id', 'event-glow')
        .attr('x', '-50%')
        .attr('y', '-50%')
        .attr('width', '200%')
        .attr('height', '200%');
        
    eventGlow.append('feGaussianBlur')
        .attr('stdDeviation', '2')
        .attr('result', 'coloredBlur');
        
    const feMerge = eventGlow.append('feMerge');
    feMerge.append('feMergeNode')
        .attr('in', 'coloredBlur');
    feMerge.append('feMergeNode')
        .attr('in', 'SourceGraphic');
    
    // Ajouter une police rétro pixel si elle n'est pas déjà présente
    if (!document.getElementById('pixel-font-link')) {
        const fontLink = document.createElement('link');
        fontLink.id = 'pixel-font-link';
        fontLink.rel = 'stylesheet';
        fontLink.href = 'https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap';
        document.head.appendChild(fontLink);
    }
    
    // Définir les couleurs par catégorie d'événement
    const categoryColors = {
        'Technologie': '#4CAF50',  // Vert
        'Politique': '#2196F3',    // Bleu
        'Science': '#9C27B0',      // Violet
        'Économie': '#FF9800',     // Orange
        'Culture': '#E91E63',      // Rose
        'Société': '#00BCD4',      // Cyan
        'default': '#FFC107'       // Jaune (couleur par défaut)
    };
    
    // Ajouter les lignes verticales et les marqueurs pour chaque événement
    filteredEvents.forEach(event => {
        // Déterminer la couleur en fonction de la catégorie
        const eventColor = categoryColors[event.category] || categoryColors.default;
        
        // Ajouter une ligne verticale avec animation
        const line = annotationGroup.append('line')
            .attr('class', 'event-line')
            .attr('x1', x(event.year))
            .attr('y1', 0)
            .attr('x2', x(event.year))
            .attr('y2', 0) // Commencer avec une hauteur de 0 pour l'animation
            .attr('stroke', eventColor)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '3,3')
            .attr('opacity', 0.7);
        
        // Animation de la ligne
        line.transition()
            .duration(1000)
            .attr('y2', y.range()[0]);
        
        // Créer un groupe pour le marqueur et le tooltip
        const markerGroup = annotationGroup.append('g')
            .attr('class', 'event-marker-group')
            .attr('transform', `translate(${x(event.year)}, 0)`);
        
        // Pas de cercle de halo pour éviter les points orange derrière les années
        
        // Ajouter un cercle pour le marqueur principal
        markerGroup.append('circle')
            .attr('class', 'event-marker')
            .attr('cx', 0)
            .attr('cy', 0)
            .attr('r', 0) // Commencer avec un rayon de 0 pour l'animation
            .attr('fill', eventColor)
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 2)
            .attr('filter', 'url(#event-glow)')
            .transition()
            .duration(1000)
            .attr('r', 6);
        
        // Ajouter un label pour l'année avec un style rétro pixel
        markerGroup.append('text')
            .attr('class', 'event-year-label')
            .attr('x', 0)
            .attr('y', -15)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('font-family', '"Press Start 2P", monospace')
            .attr('font-weight', 'normal')
            .attr('fill', '#ffffff') // Blanc pour meilleure visibilité
            .attr('stroke', 'none') // Pas de contour
            .attr('opacity', 0) // Commencer invisible pour l'animation
            .text(event.year)
            .transition()
            .duration(1000)
            .attr('opacity', 1);
        
        // Créer une infobulle interactive en dessous de l'axe X
        const tooltip = markerGroup.append('g')
            .attr('class', 'event-tooltip')
            .attr('opacity', 0) // Caché par défaut
            .attr('transform', `translate(0, ${y.range()[0] + 50})`) // Positionner plus bas en dessous de l'axe X
            .style('pointer-events', 'none'); // Permet les clics à travers pour tout le groupe
        
        // Définir les variables pour la description avant de les utiliser
        const description = event.description;
        const maxCharsPerLine = 40;
        const bubbleHeight = description.length <= maxCharsPerLine ? 100 : 110;
        
        // Fond de l'infobulle - hauteur ajustée en fonction de la longueur de la description
        tooltip.append('rect')
            .attr('x', -120)
            .attr('y', 10)
            .attr('width', 240)
            .attr('height', bubbleHeight) // Hauteur ajustée selon la longueur
            .attr('stroke', eventColor);
        
        // Petit triangle pointant vers le haut (vers le point d'événement)
        tooltip.append('path')
            .attr('d', 'M-10,0 L10,0 L0,-10 Z') // Modifié pour pointer vers le haut
            .attr('stroke', eventColor);
        
        // Titre de l'événement
        tooltip.append('text')
            .attr('class', 'event-title')
            .attr('x', 0)
            .attr('y', 35)
            .text(event.title);
        
        // Description de l'événement - gérer les textes longs avec des sauts de ligne
        if (description.length <= maxCharsPerLine) {
            // Si la description est courte, l'afficher sur une seule ligne
            tooltip.append('text')
                .attr('class', 'event-description')
                .attr('x', 0)
                .attr('y', 60)
                .text(description);
        } else {
            // Si la description est longue, la diviser en deux lignes
            const firstLine = description.substring(0, maxCharsPerLine);
            let secondLine = description.substring(maxCharsPerLine);
            
            // Limiter la seconde ligne si nécessaire
            if (secondLine.length > maxCharsPerLine - 3) {
                secondLine = secondLine.substring(0, maxCharsPerLine - 3) + '...';
            }
            
            tooltip.append('text')
                .attr('class', 'event-description')
                .attr('x', 0)
                .attr('y', 55)
                .text(firstLine);
                
            tooltip.append('text')
                .attr('class', 'event-description')
                .attr('x', 0)
                .attr('y', 70)
                .text(secondLine);
        }
        
        // Catégorie de l'événement - position ajustée en fonction de la longueur de la description
        tooltip.append('text')
            .attr('class', 'event-category')
            .attr('x', 0)
            .attr('y', description.length <= maxCharsPerLine ? 85 : 95) // Ajuster la position selon la longueur
            .attr('fill', eventColor)
            .text(event.category);
        
        // Stocker une référence à l'événement dans le groupe du marqueur pour un accès facile
        markerGroup.datum(event);
        
        // Ajouter un identifiant unique au groupe du marqueur pour pouvoir le retrouver facilement
        // Utiliser le même format de remplacement pour tous les caractères spéciaux
        markerGroup.attr('id', `event-marker-${event.title.replace(/[\s'\-\+\&\:\[\]\(\)\.\,]/g, '_')}`);
        
        // Ajouter des interactions
        markerGroup
            .on('mouseover', function() {
                // Afficher l'infobulle
                tooltip.transition()
                    .duration(200)
                    .attr('opacity', 1);
                
                // Agrandir le marqueur
                d3.select(this).select('.event-marker')
                    .transition()
                    .duration(200)
                    .attr('r', 8);
                
                // Agrandir le halo
                d3.select(this).select('.event-marker-halo')
                    .transition()
                    .duration(200)
                    .attr('r', 16);
            })
            .on('mouseout', function() {
                // Cacher l'infobulle seulement si l'événement n'est pas sélectionné
                const eventData = d3.select(this).datum();
                const isHighlighted = checkIfEventIsHighlighted(eventData);
                
                if (!isHighlighted) {
                    tooltip.transition()
                        .duration(200)
                        .attr('opacity', 0);
                    
                    // Réduire le marqueur
                    d3.select(this).select('.event-marker')
                        .transition()
                        .duration(200)
                        .attr('r', 6);
                    
                    // Réduire le halo
                    d3.select(this).select('.event-marker-halo')
                        .transition()
                        .duration(200)
                        .attr('r', 12);
                }
            });
    });
}

// Fonction pour afficher les événements historiques dans la section dédiée
function displayTimelineEvents(events) {
    const container = document.getElementById('timeline-events-content');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Filtrer les événements dans la plage d'années sélectionnée
    const filteredEvents = events.filter(event => 
        event.year >= timelineState.startYear && event.year <= timelineState.endYear);
    
    // Trier les événements par année
    filteredEvents.sort((a, b) => a.year - b.year);
    
    // Regrouper les événements par catégorie
    const eventsByCategory = {};
    filteredEvents.forEach(event => {
        if (!eventsByCategory[event.category]) {
            eventsByCategory[event.category] = [];
        }
        eventsByCategory[event.category].push(event);
    });
    
    // Créer un élément pour chaque catégorie
    Object.keys(eventsByCategory).forEach(category => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'event-category';
        
        const categoryTitle = document.createElement('h4');
        categoryTitle.textContent = category;
        categoryDiv.appendChild(categoryTitle);
        
        // Créer un élément pour chaque événement dans cette catégorie
        eventsByCategory[category].forEach(event => {
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event-item';
            eventDiv.dataset.eventYear = event.year;
            if (event.month) eventDiv.dataset.eventMonth = event.month;
            if (event.day) eventDiv.dataset.eventDay = event.day;
            
            const eventDate = document.createElement('div');
            eventDate.className = 'event-date';
            
            // Formater la date
            let dateText = event.year.toString();
            if (event.month && event.day) {
                const date = new Date(event.year, event.month - 1, event.day);
                dateText = date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' });
            }
            
            eventDate.textContent = dateText;
            eventDiv.appendChild(eventDate);
            
            const eventContent = document.createElement('div');
            eventContent.className = 'event-content';
            
            const eventTitle = document.createElement('h5');
            eventTitle.innerHTML = event.title;
            eventContent.appendChild(eventTitle);
            
            const eventDescription = document.createElement('p');
            eventDescription.textContent = event.description;
            eventContent.appendChild(eventDescription);
            
            const eventImpact = document.createElement('p');
            eventImpact.className = 'event-impact';
            eventImpact.textContent = event.impact;
            eventContent.appendChild(eventImpact);
            
            // Ajouter les mots-clés s'ils existent
            if (event.keywords && event.keywords.length > 0) {
                const keywordsContainer = document.createElement('div');
                keywordsContainer.className = 'event-keywords';
                
                const keywordsTitle = document.createElement('p');
                keywordsTitle.className = 'keywords-title';
                keywordsTitle.textContent = 'Mots-clés:';
                keywordsContainer.appendChild(keywordsTitle);
                
                const keywordsList = document.createElement('div');
                keywordsList.className = 'keywords-list';
                
                event.keywords.forEach(keyword => {
                    const keywordTag = document.createElement('span');
                    keywordTag.className = 'keyword-tag';
                    keywordTag.textContent = keyword;
                    keywordTag.addEventListener('click', () => {
                        // Mettre en évidence cet événement
                        highlightEventWithKeyword(keyword, event);
                    });
                    keywordsList.appendChild(keywordTag);
                });
                
                keywordsContainer.appendChild(keywordsList);
                eventContent.appendChild(keywordsContainer);
            }
            
            eventDiv.appendChild(eventContent);
            categoryDiv.appendChild(eventDiv);
        });
        
        container.appendChild(categoryDiv);
    });
    
    // Ajouter une section pour afficher les résultats de recherche par mots-clés
    addKeywordSearchInterface(container, filteredEvents);
}

// Fonction pour mettre en évidence un événement basé sur un mot-clé
function highlightEventWithKeyword(keyword, event) {
    // Réinitialiser tous les événements
    const allEvents = document.querySelectorAll('.event-item');
    allEvents.forEach(eventElement => {
        eventElement.classList.remove('highlighted');
    });
    
    // Mettre en évidence l'événement correspondant
    const eventElements = document.querySelectorAll(`.event-item[data-event-year="${event.year}"]`);
    eventElements.forEach(eventElement => {
        // Vérifier le mois et le jour si présents
        let isMatch = true;
        if (event.month && eventElement.dataset.eventMonth) {
            isMatch = parseInt(eventElement.dataset.eventMonth) === event.month;
        }
        if (isMatch && event.day && eventElement.dataset.eventDay) {
            isMatch = parseInt(eventElement.dataset.eventDay) === event.day;
        }
        
        if (isMatch) {
            eventElement.classList.add('highlighted');
            // Faire défiler jusqu'à l'événement
            eventElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
    
    // Mettre à jour l'interface de recherche par mots-clés
    updateKeywordSearchResults(keyword, event);
}

// Fonction pour ajouter l'interface de recherche par mots-clés
function addKeywordSearchInterface(container, events) {
    // Créer une section pour la recherche par mots-clés
    const searchSection = document.createElement('div');
    searchSection.className = 'keyword-search-section';
    searchSection.innerHTML = `
        <h3>Recherche par mots-clés</h3>
        <div class="keyword-search-container">
            <input type="text" id="keyword-search-input" placeholder="Rechercher un mot-clé..." />
            <button id="keyword-search-button">Rechercher</button>
        </div>
        <div id="keyword-search-results" class="keyword-search-results"></div>
    `;
    
    // Ajouter la section au conteneur principal
    container.appendChild(searchSection);
    
    // Extraire tous les mots-clés uniques de tous les événements
    const allKeywords = new Set();
    events.forEach(event => {
        if (event.keywords && event.keywords.length > 0) {
            event.keywords.forEach(keyword => allKeywords.add(keyword));
        }
    });
    
    // Créer des tags pour tous les mots-clés disponibles
    const keywordsCloud = document.createElement('div');
    keywordsCloud.className = 'keywords-cloud';
    
    Array.from(allKeywords).sort().forEach(keyword => {
        const keywordTag = document.createElement('span');
        keywordTag.className = 'keyword-cloud-tag';
        keywordTag.textContent = keyword;
        keywordTag.addEventListener('click', () => {
            document.getElementById('keyword-search-input').value = keyword;
            searchKeyword(keyword, events);
        });
        keywordsCloud.appendChild(keywordTag);
    });
    
    // Insérer le nuage de mots-clés après le champ de recherche
    const searchContainer = searchSection.querySelector('.keyword-search-container');
    searchContainer.parentNode.insertBefore(keywordsCloud, searchContainer.nextSibling);
    
    // Ajouter l'événement de recherche
    const searchButton = document.getElementById('keyword-search-button');
    const searchInput = document.getElementById('keyword-search-input');
    
    searchButton.addEventListener('click', () => {
        const keyword = searchInput.value.trim().toLowerCase();
        if (keyword) {
            searchKeyword(keyword, events);
        }
    });
    
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const keyword = searchInput.value.trim().toLowerCase();
            if (keyword) {
                searchKeyword(keyword, events);
            }
        }
    });
}

// Fonction pour rechercher un mot-clé parmi les événements
function searchKeyword(keyword, events) {
    const resultsContainer = document.getElementById('keyword-search-results');
    resultsContainer.innerHTML = '';
    
    // Trouver les événements correspondant au mot-clé
    const matchingEvents = events.filter(event => 
        event.keywords && event.keywords.some(k => k.toLowerCase().includes(keyword.toLowerCase()))
    );
    
    if (matchingEvents.length === 0) {
        resultsContainer.innerHTML = `<p class="no-results">Aucun événement trouvé pour le mot-clé "${keyword}".</p>`;
        return;
    }
    
    // Afficher les résultats
    const resultsList = document.createElement('div');
    resultsList.className = 'search-results-list';
    
    matchingEvents.forEach(event => {
        const resultItem = document.createElement('div');
        resultItem.className = 'search-result-item';
        
        // Formater la date
        let dateText = event.year.toString();
        if (event.month && event.day) {
            const date = new Date(event.year, event.month - 1, event.day);
            dateText = date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' });
        }
        
        resultItem.innerHTML = `
            <div class="result-date">${dateText}</div>
            <div class="result-content">
                <h5>${event.title}</h5>
                <p>${event.description}</p>
            </div>
        `;
        
        resultItem.addEventListener('click', () => {
            highlightEventWithKeyword(keyword, event);
        });
        
        resultsList.appendChild(resultItem);
    });
    
    resultsContainer.appendChild(resultsList);
}

// Fonction pour mettre à jour les résultats de recherche par mots-clés
function updateKeywordSearchResults(keyword, event) {
    const searchInput = document.getElementById('keyword-search-input');
    if (searchInput) {
        searchInput.value = keyword;
    }
    
    const resultsContainer = document.getElementById('keyword-search-results');
    if (resultsContainer) {
        // Mettre en évidence le résultat correspondant
        const resultItems = resultsContainer.querySelectorAll('.search-result-item');
        resultItems.forEach(item => {
            item.classList.remove('highlighted');
            
            // Vérifier si cet élément correspond à l'événement
            const itemTitle = item.querySelector('h5').textContent;
            if (itemTitle === event.title) {
                item.classList.add('highlighted');
            }
        });
    }
}

// Attendre que le DOM soit chargé pour initialiser
document.addEventListener('DOMContentLoaded', () => {
    console.log('Script timeline_viz.js chargé');
});