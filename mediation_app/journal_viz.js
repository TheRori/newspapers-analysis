// Visualisation par journal
// Ce fichier contient les fonctions pour visualiser la répartition du vocabulaire entre différents journaux

// Initialisation de la visualisation par journal
function initJournalVisualization() {
    console.log('Initialisation de la visualisation par journal');
    
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
        <p>Chargement des données des journaux...</p>
    `;
    const container = document.getElementById('journal-visualization');
    if (container) {
        container.innerHTML = '';
        container.appendChild(loadingDiv);
    }
    
    // Vérifier que les données sont chargées
    if (!state.data || !state.terms || !state.years || !state.newspapers) {
        console.error('Les données nécessaires ne sont pas encore chargées');
        // Supprimer l'indicateur de chargement
        if (loadingDiv && loadingDiv.parentNode) {
            loadingDiv.parentNode.removeChild(loadingDiv);
        }
        // Réessayer dans 1 seconde
        setTimeout(initJournalVisualization, 1000);
        return;
    }
    
    // Marquer la visualisation comme initialisée
    state.journalVizInitialized = true;
    
    // Supprimer l'indicateur de chargement
    if (loadingDiv && loadingDiv.parentNode) {
        loadingDiv.parentNode.removeChild(loadingDiv);
    }
    
    // Marquer la visualisation comme initialisée
    state.journalVizInitialized = true;
    
    // Initialiser les valeurs par défaut
    if (!state.journalStartYear && state.years.length > 0) {
        state.journalStartYear = state.years[0];
        state.journalEndYear = state.years[state.years.length - 1];
    }
    
    // Débogage: Vérifier si 'ordinateur' est dans la liste des termes
    console.log('Le terme "ordinateur" est-il présent dans state.terms?', state.terms.includes('ordinateur'));
    console.log('Liste complète des termes:', state.terms);
    
    // Recherche de termes similaires qui pourraient être mal encodés
    const similarTerms = state.terms.filter(term => 
        term.toLowerCase().includes('ordin') || 
        term.toLowerCase().includes('comput'));
    console.log('Termes similaires à "ordinateur":', similarTerms);
    
    // Vérifier les données brutes pour le terme 'ordinateur'
    if (state.rawData && state.rawData.length > 0) {
        console.log('Exemple de données brutes (premier élément):', state.rawData[0]);
        const ordinateurValues = state.rawData
            .filter(row => row.ordinateur && parseFloat(row.ordinateur) > 0)
            .slice(0, 5);
        console.log('Exemples d\'articles avec le terme "ordinateur":', ordinateurValues);
    }
    
    // Utiliser exactement les mêmes termes que l'onglet principal
    // Ces termes sont déjà extraits du CSV dans la fonction loadData() de mediation_app.js
    console.log('Utilisation des termes déjà extraits du CSV:', state.terms);
    
    console.log('Liste complète des termes disponibles:', state.terms);
    
    // Initialiser les termes sélectionnés
    if (state.journalSelectedTerms.length === 0 && state.terms.length > 0) {
        // Sélectionner informatique, ordinateur et programme par défaut
        state.journalSelectedTerms = [];
        
        if (state.terms.includes('informatique')) {
            state.journalSelectedTerms.push('informatique');
        }
        
        if (state.terms.includes('ordinateur')) {
            state.journalSelectedTerms.push('ordinateur');
        }
        
        if (state.terms.includes('programme')) {
            state.journalSelectedTerms.push('programme');
        }
        
        // Si aucun terme n'a été sélectionné, prendre le premier disponible
        if (state.journalSelectedTerms.length === 0 && state.terms.length > 0) {
            state.journalSelectedTerms.push(state.terms[0]);
        }
    }
    
    console.log('Termes disponibles:', state.terms);
    console.log('Termes sélectionnés pour la visualisation par journal:', state.journalSelectedTerms);
    console.log('Années disponibles:', state.years);
    console.log('Journaux disponibles:', state.newspapers);
    
    // Créer les checkboxes pour les termes
    createJournalTermCheckboxes();
    
    // Créer le slider pour les années
    createJournalYearSlider();
    
    // Ajouter l'événement pour le changement de type de visualisation
    document.getElementById('journal-viz-type').addEventListener('change', (e) => {
        state.journalVizType = e.target.value;
        updateJournalVisualization();
    });
    
    // Préparer les données et créer la visualisation
    prepareJournalData();
    updateJournalVisualization();
}

// Création des checkboxes pour les termes (visualisation par journal)
function createJournalTermCheckboxes() {
    const container = document.getElementById('journal-term-checkboxes');
    container.innerHTML = '';
    
    state.terms.forEach(term => {
        const isChecked = state.journalSelectedTerms.includes(term);
        
        const checkboxDiv = document.createElement('div');
        checkboxDiv.className = 'term-checkbox';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `journal-term-${term}`;
        checkbox.value = term;
        checkbox.checked = isChecked;
        
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                if (!state.journalSelectedTerms.includes(term)) {
                    state.journalSelectedTerms.push(term);
                }
            } else {
                state.journalSelectedTerms = state.journalSelectedTerms.filter(t => t !== term);
            }
            
            // Limiter le nombre de termes sélectionnés
            if (state.journalSelectedTerms.length > config.maxTermsToShow) {
                state.journalSelectedTerms = state.journalSelectedTerms.slice(0, config.maxTermsToShow);
                document.getElementById(`journal-term-${term}`).checked = false;
                alert(`Maximum ${config.maxTermsToShow} termes peuvent être sélectionnés simultanément.`);
            }
            
            // IMPORTANT: Recalculer les données avant de mettre à jour la visualisation
            // C'est ce qui manquait par rapport à l'onglet principal
            prepareJournalData();
            
            // Ensuite mettre à jour la visualisation avec les nouvelles données
            updateJournalVisualization();
        });
        
        const label = document.createElement('label');
        label.htmlFor = `journal-term-${term}`;
        label.textContent = term;
        
        checkboxDiv.appendChild(checkbox);
        checkboxDiv.appendChild(label);
        container.appendChild(checkboxDiv);
    });
}

// Création du slider pour les années (visualisation par journal)
function createJournalYearSlider() {
    const container = document.getElementById('journal-year-slider');
    container.innerHTML = '';
    
    if (state.years.length === 0) return;
    
    const sliderContainer = document.createElement('div');
    sliderContainer.className = 'year-slider-container';
    
    const slider = document.createElement('div');
    slider.id = 'journal-year-range';
    sliderContainer.appendChild(slider);
    
    const valueDisplay = document.createElement('div');
    valueDisplay.className = 'year-range-values';
    valueDisplay.innerHTML = `<span id="journal-year-min">${state.journalStartYear}</span> - <span id="journal-year-max">${state.journalEndYear}</span>`;
    sliderContainer.appendChild(valueDisplay);
    
    container.appendChild(sliderContainer);
    
    // Créer le slider avec noUiSlider
    const minYear = parseInt(state.years[0]);
    const maxYear = parseInt(state.years[state.years.length - 1]);
    
    noUiSlider.create(slider, {
        start: [state.journalStartYear || minYear, state.journalEndYear || maxYear],
        connect: true,
        step: 1,
        range: {
            'min': minYear,
            'max': maxYear
        },
        format: {
            to: value => Math.round(value),
            from: value => Math.round(value)
        }
    });
    
    // Mettre à jour l'affichage et les données lors du changement
    slider.noUiSlider.on('update', (values, handle) => {
        const [startYear, endYear] = values;
        document.getElementById('journal-year-min').textContent = startYear;
        document.getElementById('journal-year-max').textContent = endYear;
        
        state.journalStartYear = startYear;
        state.journalEndYear = endYear;
    });
    
    // Mettre à jour la visualisation à la fin du glissement
    slider.noUiSlider.on('change', () => {
        prepareJournalData();
        updateJournalVisualization();
    });
}

// Préparation des données pour la visualisation par journal
function prepareJournalData() {
    console.log('Préparation des données pour la visualisation par journal');

    if (!state.data || !state.newspapers || state.newspapers.length === 0 || !state.terms) {
        console.error('Données manquantes pour la visualisation par journal');
        return;
    }
    
    // Utiliser les noms de journaux tels qu'ils apparaissent dans les données, sans normalisation
    console.log('Journaux disponibles (sans normalisation):', state.newspapers);
    
    // Vérifier si le terme "ordinateur" est présent dans les données
    console.log('Le terme "ordinateur" est-il présent dans state.terms?', state.terms.includes('ordinateur'));
    
    // Initialiser une fois clairement les données agrégées
    state.journalYearlyData = {};

    state.journalSelectedTerms.forEach(term => {
        state.journalYearlyData[term] = {};

        state.newspapers.forEach(journal => {
            const articlesForJournal = state.data.filter(article => 
                article.journal === journal && 
                parseInt(article.year) >= parseInt(state.journalStartYear) &&
                parseInt(article.year) <= parseInt(state.journalEndYear)
            );

            const sum = articlesForJournal.reduce((total, article) => {
                return total + (article.values[term] || 0);
            }, 0);

            state.journalYearlyData[term][journal] = sum;
        });
    });

    console.log('Données agrégées par journal (corrigées):', state.journalYearlyData);
    
    // Limiter aux 10 journaux les plus fréquents
    if (state.journalSelectedTerms.length > 0) {
        const firstTerm = state.journalSelectedTerms[0];
        const journalCounts = {};
        
        // Calculer le nombre total d'occurrences pour chaque journal (tous termes confondus)
        state.newspapers.forEach(journal => {
            let totalCount = 0;
            state.journalSelectedTerms.forEach(term => {
                if (state.journalYearlyData[term] && state.journalYearlyData[term][journal]) {
                    totalCount += state.journalYearlyData[term][journal];
                }
            });
            journalCounts[journal] = totalCount;
        });
        
        // Trier les journaux par fréquence d'apparition
        const sortedJournals = Object.entries(journalCounts)
            .sort((a, b) => b[1] - a[1]) // Tri par ordre décroissant
            .slice(0, 10) // Prendre les 10 premiers
            .map(entry => entry[0]); // Extraire juste les noms de journaux
        
        console.log('Top 10 des journaux:', sortedJournals);
        
        // Mettre à jour la liste des journaux à afficher
        state.newspapers = sortedJournals;
    }
    
    // Vérifier si nous avons des données valides
    let hasData = false;
    for (const term in state.journalYearlyData) {
        for (const journal in state.journalYearlyData[term]) {
            if (state.journalYearlyData[term][journal] > 0) {
                hasData = true;
                break;
            }
        }
        if (hasData) break;
    }
    
    if (!hasData) {
        console.warn('Aucune donnée valide trouvée pour la visualisation par journal');
    }
}

// Mise à jour de la visualisation par journal
function updateJournalVisualization() {
    const container = document.getElementById('journal-visualization');
    container.innerHTML = '';
    
    if (state.journalSelectedTerms.length === 0) {
        container.innerHTML = '<div class="info-message">Veuillez sélectionner au moins un terme.</div>';
        return;
    }
    
    // Créer la visualisation en fonction du type sélectionné
    switch (state.journalVizType) {
        case 'bar':
            createJournalBarChart(container);
            break;
        case 'stacked':
            createJournalStackedBarChart(container);
            break;
        case 'pie':
            createJournalPieChart(container);
            break;
        default:
            createJournalBarChart(container);
    }
}

// Création d'un graphique en barres groupées pour la visualisation par journal
function createJournalBarChart(container) {
    console.log('Création du graphique en barres groupées');
    const width = container.clientWidth;
    const height = 500;
    const margin = config.margin;
    
    try {
        // Vérifier que les données sont prêtes
        if (!state.journalYearlyData) {
            console.error('Les données par journal ne sont pas disponibles');
            container.innerHTML = '<div class="error-message">Les données ne sont pas encore prêtes.</div>';
            return;
        }
        
        // Préparer les données pour le graphique
        const data = [];
        state.newspapers.forEach(journal => {
            const journalData = {
                journal: journal
            };
            
            state.journalSelectedTerms.forEach(term => {
                // Vérifier que le terme existe dans les données
                if (state.journalYearlyData[term]) {
                    journalData[term] = state.journalYearlyData[term][journal] || 0;
                } else {
                    console.warn(`Le terme '${term}' n'existe pas dans les données`);
                    journalData[term] = 0;
                }
            });
            
            data.push(journalData);
        });
        
        console.log('Données préparées pour le graphique en barres:', data);
        
        // Trier les données par ordre décroissant du premier terme
        if (state.journalSelectedTerms.length > 0) {
            const firstTerm = state.journalSelectedTerms[0];
            data.sort((a, b) => b[firstTerm] - a[firstTerm]);
        }
        
        // Créer le SVG
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet');
        
        // Créer un groupe pour le graphique
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Échelles
        const x0 = d3.scaleBand()
            .domain(data.map(d => d.journal))
            .rangeRound([0, width - margin.left - margin.right])
            .paddingInner(0.1);
        
        const x1 = d3.scaleBand()
            .domain(state.journalSelectedTerms)
            .rangeRound([0, x0.bandwidth()])
            .padding(0.05);
        
        const maxValue = d3.max(data, d => d3.max(state.journalSelectedTerms, term => d[term]));
        
        const y = d3.scaleLinear()
            .domain([0, maxValue * 1.1]) // Ajouter 10% pour la lisibilité
            .rangeRound([height - margin.top - margin.bottom, 0]);
        
        // Couleurs
        const color = d3.scaleOrdinal()
            .domain(state.journalSelectedTerms)
            .range(config.colors);
        
        // Barres
        g.append('g')
            .selectAll('g')
            .data(data)
            .join('g')
            .attr('transform', d => `translate(${x0(d.journal)},0)`)
            .selectAll('rect')
            .data(d => state.journalSelectedTerms.map(term => ({term, value: d[term], journal: d.journal})))
            .join('rect')
            .attr('x', d => x1(d.term))
            .attr('y', d => y(d.value))
            .attr('width', x1.bandwidth())
            .attr('height', d => height - margin.top - margin.bottom - y(d.value))
            .attr('fill', d => color(d.term))
            .on('mouseover', function(event, d) {
                d3.select(this).attr('opacity', 0.8);
                
                // Afficher le tooltip
                const tooltip = d3.select('#journal-tooltip');
                tooltip.style('display', 'block')
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 20) + 'px')
                    .html(`<strong>${d.term}</strong> dans <strong>${d.journal}</strong>: ${d.value.toFixed(0)} occurrences`);
            })
            .on('mouseout', function() {
                d3.select(this).attr('opacity', 1);
                d3.select('#journal-tooltip').style('display', 'none');
            })
            .on('click', function(event, d) {
                filterAndShowArticles({ term: d.term, journal: d.journal });
            });
            
        // Axe X
        g.append('g')
            .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
            .call(d3.axisBottom(x0))
            .selectAll('text')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end')
            .text(d => {
                const maxLength = 15;
                return d.length > maxLength ? d.substring(0, maxLength) + '...' : d;
            });
        
        // Axe Y
        g.append('g')
            .call(d3.axisLeft(y));
        
        // Titre
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('fill', 'white')
            .text(`Répartition par journal (${state.journalStartYear}-${state.journalEndYear})`);
        
        // Légende
        const legend = svg.append('g')
            .attr('font-family', 'sans-serif')
            .attr('font-size', 10)
            .attr('text-anchor', 'end')
            .selectAll('g')
            .data(state.journalSelectedTerms)
            .join('g')
            .attr('transform', (d, i) => `translate(0,${i * 20 + 20})`);
        
        legend.append('rect')
            .attr('x', width - 19)
            .attr('width', 19)
            .attr('height', 19)
            .attr('fill', color);
        
        legend.append('text')
            .attr('x', width - 24)
            .attr('y', 9.5)
            .attr('dy', '0.32em')
            .style('fill', 'white')
            .text(d => d);
    } catch (error) {
        console.error('Erreur lors de la création du graphique en barres groupées:', error);
        container.innerHTML = `<div class="error-message">Erreur lors de la création du graphique: ${error.message}</div>`;
    }
}

// Création d'un graphique en barres empilées pour la visualisation par journal
function createJournalStackedBarChart(container) {
    console.log('Création du graphique en barres empilées');
    const width = container.clientWidth;
    const height = 500;
    const margin = config.margin;
    
    // Vérifier qu'il y a des termes sélectionnés
    if (state.journalSelectedTerms.length === 0) {
        container.innerHTML = '<div class="info-message">Veuillez sélectionner au moins un terme.</div>';
        return;
    }
    
    console.log('Termes sélectionnés:', state.journalSelectedTerms);
    console.log('Journaux disponibles:', state.newspapers);
    
    // Préparer les données pour le graphique
    const data = [];
    state.newspapers.forEach(journal => {
        const journalData = {
            journal: journal
        };
        
        state.journalSelectedTerms.forEach(term => {
            // Vérifier que le terme existe dans les données
            if (state.journalYearlyData[term]) {
                journalData[term] = state.journalYearlyData[term][journal] || 0;
            } else {
                console.warn(`Le terme '${term}' n'existe pas dans les données`);
                journalData[term] = 0;
            }
        });
        
        data.push(journalData);
    });
    
    console.log('Données préparées pour le graphique:', data);
    
    // Trier les données par total décroissant
    data.sort((a, b) => {
        const totalA = state.journalSelectedTerms.reduce((sum, term) => sum + a[term], 0);
        const totalB = state.journalSelectedTerms.reduce((sum, term) => sum + b[term], 0);
        return totalB - totalA;
    });
    
    // Créer le SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');
    
    // Créer un groupe pour le graphique
    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Échelles
    const x0 = d3.scaleBand()
        .domain(data.map(d => d.journal))
        .rangeRound([0, width - margin.left - margin.right])
        .paddingInner(0.1);
    
    try {
        // S'assurer que tous les journaux ont des valeurs pour tous les termes
        data.forEach(d => {
            state.journalSelectedTerms.forEach(term => {
                if (d[term] === undefined) {
                    d[term] = 0;
                }
            });
        });
        
        // Créer la pile
        const stack = d3.stack()
            .keys(state.journalSelectedTerms);
        
        const stackedData = stack(data);
        console.log('Données empilées:', stackedData);
        
        // Vérifier si les données empilées sont valides
        if (!stackedData || stackedData.length === 0 || !stackedData[0] || stackedData[0].length === 0) {
            console.error('Données empilées invalides');
            container.innerHTML = '<div class="error-message">Erreur lors de la création du graphique en barres empilées.</div>';
            return;
        }
        
        // Calculer la valeur maximale pour l'échelle Y
        let maxValue = 0;
        stackedData.forEach(layer => {
            layer.forEach(d => {
                if (d[1] > maxValue) maxValue = d[1];
            });
        });
        
        const y = d3.scaleLinear()
            .domain([0, maxValue * 1.1]) // Ajouter 10% pour la lisibilité
            .rangeRound([height - margin.top - margin.bottom, 0]);
        
        // Couleurs
        const color = d3.scaleOrdinal()
            .domain(state.journalSelectedTerms)
            .range(config.colors);
        
        // Barres empilées
        g.append('g')
            .selectAll('g')
            .data(stackedData)
            .join('g')
            .attr('fill', (d, i) => color(state.journalSelectedTerms[i]))
            .selectAll('rect')
            .data(d => d)
            .join('rect')
            .attr('x', d => x0(d.data.journal))
            .attr('y', d => y(d[1]))
            .attr('height', d => y(d[0]) - y(d[1]))
            .attr('width', x0.bandwidth())
            .on('mouseover', function(event, d) {
                d3.select(this).attr('opacity', 0.8);
                
                // Récupérer le terme correspondant à cette couche
                const termIndex = stackedData.findIndex(layer => layer.includes(d));
                const term = state.journalSelectedTerms[termIndex];
                const value = d[1] - d[0];
                
                // Afficher le tooltip
                const tooltip = d3.select('#journal-tooltip');
                tooltip.style('display', 'block')
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 20) + 'px')
                    .html(`<strong>${term}</strong> dans <strong>${d.data.journal}</strong>: ${value.toFixed(0)} occurrences`);
            })
            .on('mouseout', function() {
                d3.select(this).attr('opacity', 1);
                d3.select('#journal-tooltip').style('display', 'none');
            })
            .on('click', function(event, d) {
                const termIndex = stackedData.findIndex(layer => layer.includes(d));
                const term = state.journalSelectedTerms[termIndex];
                const journal = d.data.journal;
                filterAndShowArticles({ term: term, journal: journal });
            });
            
        // Axe X
        g.append('g')
            .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
            .call(d3.axisBottom(x0))
            .selectAll('text')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end');
        
        // Axe Y
        g.append('g')
            .call(d3.axisLeft(y));
        
        // Titre
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('fill', 'white')
            .text(`Répartition par journal (${state.journalStartYear}-${state.journalEndYear})`);
        
        // Légende
        const legend = svg.append('g')
            .attr('font-family', 'sans-serif')
            .attr('font-size', 10)
            .attr('text-anchor', 'end')
            .selectAll('g')
            .data(state.journalSelectedTerms)
            .join('g')
            .attr('transform', (d, i) => `translate(0,${i * 20 + 20})`);
        
        legend.append('rect')
            .attr('x', width - 19)
            .attr('width', 19)
            .attr('height', 19)
            .attr('fill', color);
        
        legend.append('text')
            .attr('x', width - 24)
            .attr('y', 9.5)
            .attr('dy', '0.32em')
            .style('fill', 'white')
            .text(d => d);
    } catch (error) {
        console.error('Erreur lors de la création du graphique en barres empilées:', error);
        container.innerHTML = `<div class="error-message">Erreur lors de la création du graphique: ${error.message}</div>`;
    }
}

// Création d'un graphique en camembert pour la visualisation par journal
function createJournalPieChart(container) {
    if (state.journalSelectedTerms.length === 0) return;
    
    // On ne peut afficher qu'un terme à la fois en camembert
    const term = state.journalSelectedTerms[0];
    
    const width = container.clientWidth;
    const height = 500;
    const radius = Math.min(width, height) / 2 - 40;
    
    // Préparer les données pour le graphique
    const data = [];
    state.newspapers.forEach(journal => {
        const value = state.journalYearlyData[term][journal] || 0;
        if (value > 0) {
            data.push({
                journal: journal,
                value: value
            });
        }
    });
    
    // Trier les données par valeur décroissante
    data.sort((a, b) => b.value - a.value);
    
    // Limiter le nombre de segments pour la lisibilité
    const maxSegments = 10;
    let pieData = data;
    
    if (data.length > maxSegments) {
        const topData = data.slice(0, maxSegments - 1);
        const otherData = data.slice(maxSegments - 1);
        const otherValue = otherData.reduce((sum, d) => sum + d.value, 0);
        
        pieData = [
            ...topData,
            {
                journal: 'Autres',
                value: otherValue
            }
        ];
    }
    
    // Créer le SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');
    
    // Créer un groupe pour le graphique
    const g = svg.append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`);
    
    // Couleurs
    const color = d3.scaleOrdinal()
        .domain(pieData.map(d => d.journal))
        .range(d3.quantize(t => d3.interpolateSpectral(t * 0.8 + 0.1), pieData.length));
    
    // Créer le générateur de camembert
    const pie = d3.pie()
        .value(d => d.value)
        .sort(null);
    
    const arc = d3.arc()
        .innerRadius(0)
        .outerRadius(radius);
    
    // Créer les segments du camembert
    const arcs = g.selectAll('path')
        .data(pie(pieData))
        .join('path')
        .attr('d', arc)
        .attr('fill', d => color(d.data.journal))
        .attr('stroke', 'white')
        .style('stroke-width', '2px')
        .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 0.8);
            
            // Calculer le pourcentage
            const total = pieData.reduce((sum, item) => sum + item.value, 0);
            const percentage = (d.data.value / total * 100).toFixed(1);
            
            // Afficher le tooltip
            const tooltip = d3.select('#journal-tooltip');
            tooltip.style('display', 'block')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 20) + 'px')
                .html(`<strong>${d.data.journal}</strong>: ${d.data.value.toFixed(0)} occurrences (${percentage}%)`);
        })
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 1);
            d3.select('#journal-tooltip').style('display', 'none');
        })
        .on('click', function(event, d) {
            showArticlesForJournal(d.data.journal);
        });
    
    // Ajouter des étiquettes
    const arcLabel = d3.arc()
        .innerRadius(radius * 0.6)
        .outerRadius(radius * 0.6);
    
    const labels = g.selectAll('text')
        .data(pie(pieData))
        .join('text')
        .attr('transform', d => `translate(${arcLabel.centroid(d)})`)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', 'white')
        .style('pointer-events', 'none')
        .text(d => {
            const total = pieData.reduce((sum, item) => sum + item.value, 0);
            const percentage = (d.data.value / total * 100).toFixed(0);
            return percentage >= 5 ? `${percentage}%` : '';
        });
    
    // Titre
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .style('fill', 'white')
        .text(`Répartition de "${term}" par journal (${state.journalStartYear}-${state.journalEndYear})`);
    
    // Légende
    const legend = svg.append('g')
        .attr('font-family', 'sans-serif')
        .attr('font-size', 10)
        .attr('text-anchor', 'end')
        .selectAll('g')
        .data(pieData)
        .join('g')
        .attr('transform', (d, i) => `translate(${width - 120},${i * 20 + 50})`);
    
    legend.append('rect')
        .attr('x', 0)
        .attr('width', 19)
        .attr('height', 19)
        .attr('fill', d => color(d.journal));
    
    legend.append('text')
        .attr('x', -5)
        .attr('y', 9.5)
        .attr('dy', '0.32em')
        .style('text-anchor', 'end')
        .style('fill', 'white')
        .text(d => {
            // Tronquer les noms trop longs
            const maxLength = 15;
            return d.journal.length > maxLength ? d.journal.substring(0, maxLength) + '...' : d.journal;
        });
}
