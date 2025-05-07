// Configuration
const config = {
    dataPath: 'data/results/exports/collections/term_tracking_chrono/9f00067a-0044-4cc2-b6f4-3cdaa9404998/source_files/term_tracking_results.csv',
    articlesPath: 'data/results/exports/collections/term_tracking_chrono/9f00067a-0044-4cc2-b6f4-3cdaa9404998/source_files/articles/filtered_articles_20250507_205410.json',
    margin: { top: 40, right: 80, bottom: 60, left: 60 },
    transitionDuration: 800,
    colors: d3.schemeSet2,
    defaultTerms: ['informatique', 'ordinateur', 'programme', 'intelligence artificielle', 'logiciel'],
    maxTermsToShow: 8,
    maxArticlesToShow: 20,
    cantonColors: {
        'FR': '#c8102e', // Rouge
        'GE': '#e3000f', // Rouge
        'JU': '#cf142b', // Rouge
        'NE': '#008000', // Vert
        'VD': '#008000', // Vert
        'VS': '#ff0000', // Rouge
        'BE': '#ff0000', // Rouge
        'TI': '#ff0000', // Rouge
        'ZH': '#0066cc', // Bleu
        'other': '#666666' // Gris pour les autres
    }
};
// État de l'application
let state = {
    data: null,
    rawData: null,
    articles: null,
    terms: [],
    selectedTerms: [],
    years: [],
    startYear: null,
    endYear: null,
    vizType: 'line',
    yearlyData: {},
    journalData: {},
    cantonData: {},
    swiper: null,
    selectedArticles: [],
    cantons: [],
    newspapers: [],
    filteredData: [],
    selectedCantons: [],
    selectedJournals: []
};

// Initialisation de l'application
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    try {
        // Charger les données
        await loadData();
        
        // Initialiser les contrôles
        initControls();
        
        // Créer la visualisation initiale
        createVisualization();
        
        // Générer les insights
        generateInsights();
        
        // Créer les visualisations géographiques
        createGeoVisualizations();
        
        // Initialiser le modal d'articles
        initArticleModal();
        
    } catch (error) {
        console.error('Erreur lors de l\'initialisation de l\'application:', error);
        document.getElementById('chart-container').innerHTML = 
            `<div class="error-message">Erreur lors du chargement des données: ${error.message}</div>`;
    }
}

// Chargement et traitement des données
async function loadData() {
    try {
        // Charger les données de suivi de termes
        const data = await d3.csv(config.dataPath);
        
        // Essayer de charger les articles (avec gestion d'erreur)
        let articles = [];
        try {
            const articlesResponse = await fetch(config.articlesPath);
            if (articlesResponse.ok) {
                articles = await articlesResponse.json();
                console.log(`Chargé ${articles.length} articles`);
            } else {
                console.warn('Impossible de charger les articles:', articlesResponse.statusText);
            }
        } catch (err) {
            console.warn('Erreur lors du chargement des articles:', err);
        }
        
        // Extraire les termes (colonnes sauf 'key')
        const terms = Object.keys(data[0]).filter(key => key !== 'key');
        
        // Extraire les années, journaux et cantons à partir des clés d'articles
        const articleInfo = data.map(row => {
            const parts = row.key.split('_');
            const dateStr = parts[1];
            const journal = parts[3] || 'inconnu';
            
            // Trouver l'article correspondant dans le fichier JSON si disponible
            const articleDetails = articles.find(a => a.id === row.key) || {};
            const canton = articleDetails.canton || 'unknown';
            
            return {
                id: row.key,
                year: dateStr.substring(0, 4),
                date: dateStr,
                journal: journal,
                canton: canton,
                values: terms.reduce((acc, term) => {
                    acc[term] = row[term] ? parseFloat(row[term]) : 0;
                    return acc;
                }, {}),
                details: articleDetails
            };
        });
        
        // Obtenir la liste des années uniques et les trier
        const years = [...new Set(articleInfo.map(item => item.year))].sort();
        
        // Obtenir la liste des cantons et journaux uniques
        const cantons = [...new Set(articleInfo.map(item => item.canton).filter(c => c !== 'unknown'))];
        const journals = [...new Set(articleInfo.map(item => item.journal).filter(j => j !== 'inconnu'))];
        
        // Agréger les données par année
        const yearlyData = years.reduce((acc, year) => {
            const articlesThisYear = articleInfo.filter(a => a.year === year);
            
            acc[year] = terms.reduce((termAcc, term) => {
                termAcc[term] = articlesThisYear.reduce((sum, article) => sum + (article.values[term] || 0), 0);
                return termAcc;
            }, {});
            
            return acc;
        }, {});
        
        // Agréger les données par journal
        const journalData = journals.reduce((acc, journal) => {
            const articlesThisJournal = articleInfo.filter(a => a.journal === journal);
            
            acc[journal] = terms.reduce((termAcc, term) => {
                termAcc[term] = articlesThisJournal.reduce((sum, article) => sum + (article.values[term] || 0), 0);
                return termAcc;
            }, {});
            
            return acc;
        }, {});
        
        // Agréger les données par canton
        const cantonData = cantons.reduce((acc, canton) => {
            const articlesThisCanton = articleInfo.filter(a => a.canton === canton);
            
            acc[canton] = terms.reduce((termAcc, term) => {
                termAcc[term] = articlesThisCanton.reduce((sum, article) => sum + (article.values[term] || 0), 0);
                return termAcc;
            }, {});
            
            return acc;
        }, {});
        
        // Mettre à jour l'état
        state.rawData = data;
        state.data = articleInfo;
        state.articles = articles;
        state.terms = terms;
        state.selectedTerms = config.defaultTerms.filter(term => terms.includes(term));
        state.years = years;
        state.startYear = years[0];
        state.endYear = years[years.length - 1];
        state.yearlyData = yearlyData;
        state.journalData = journalData;
        state.cantonData = cantonData;
        state.cantons = cantons;
        state.newspapers = journals;
    } catch (error) {
        console.error('Erreur lors du chargement des données:', error);
        throw error;
    }
}

// Initialisation des contrôles
function initControls() {
    // Créer les cases à cocher pour les termes
    const termCheckboxes = document.getElementById('term-checkboxes');
    
    state.terms.forEach(term => {
        const checkboxDiv = document.createElement('div');
        checkboxDiv.className = 'term-checkbox';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `term-${term}`;
        checkbox.value = term;
        checkbox.checked = state.selectedTerms.includes(term);
        
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                if (state.selectedTerms.length < config.maxTermsToShow) {
                    state.selectedTerms.push(term);
                } else {
                    e.target.checked = false;
                    alert(`Vous ne pouvez sélectionner que ${config.maxTermsToShow} termes maximum.`);
                    return;
                }
            } else {
                state.selectedTerms = state.selectedTerms.filter(t => t !== term);
            }
            
            updateFilteredData();
        });
        
        const label = document.createElement('label');
        label.htmlFor = `term-${term}`;
        label.textContent = term;
        
        checkboxDiv.appendChild(checkbox);
        checkboxDiv.appendChild(label);
        termCheckboxes.appendChild(checkboxDiv);
    });
    
    // Configurer le sélecteur de type de visualisation
    const vizTypeSelect = document.getElementById('viz-type');
    vizTypeSelect.addEventListener('change', (e) => {
        state.vizType = e.target.value;
        updateFilteredData();
    });
    
    // Créer un slider pour les années (à implémenter avec D3)
    createYearSlider();
}

// Création du slider pour les années
function createYearSlider() {
    const yearSlider = document.getElementById('year-slider');
    yearSlider.innerHTML = '';
    
    const width = yearSlider.clientWidth || 300;
    const height = 50;
    
    const svg = d3.select('#year-slider')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    const margin = {left: 30, right: 30};
    const innerWidth = width - margin.left - margin.right;
    
    const x = d3.scaleLinear()
        .domain([parseInt(state.years[0]), parseInt(state.years[state.years.length - 1])])
        .range([0, innerWidth])
        .clamp(true);
    
    const slider = svg.append('g')
        .attr('class', 'slider')
        .attr('transform', `translate(${margin.left},${height/2})`);
    
    slider.append('line')
        .attr('class', 'track')
        .attr('x1', x.range()[0])
        .attr('x2', x.range()[1])
        .attr('stroke', '#ccc')
        .attr('stroke-width', 10)
        .attr('stroke-linecap', 'round');
    
    const startHandle = slider.append('circle')
        .attr('class', 'handle')
        .attr('r', 9)
        .attr('cx', x(parseInt(state.startYear)))
        .attr('fill', '#3498db')
        .call(d3.drag()
            .on('drag', function(event) {
                const newYear = Math.round(x.invert(event.x));
                if (newYear >= parseInt(state.years[0]) && newYear < parseInt(state.endYear)) {
                    state.startYear = newYear.toString();
                    d3.select(this).attr('cx', x(newYear));
                    updateYearLabels();
                    filterDataAndUpdateVisualization();
                }
            }));
    
    const endHandle = slider.append('circle')
        .attr('class', 'handle')
        .attr('r', 9)
        .attr('cx', x(parseInt(state.endYear)))
        .attr('fill', '#e74c3c')
        .call(d3.drag()
            .on('drag', function(event) {
                const newYear = Math.round(x.invert(event.x));
                if (newYear <= parseInt(state.years[state.years.length - 1]) && newYear > parseInt(state.startYear)) {
                    state.endYear = newYear.toString();
                    d3.select(this).attr('cx', x(newYear));
                    updateYearLabels();
                    filterDataAndUpdateVisualization();
                }
            }));
    
    // Ajouter des labels pour les années
    const startLabel = slider.append('text')
        .attr('class', 'year-label start-year')
        .attr('x', x(parseInt(state.startYear)))
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('fill', '#e0e0e0')
        .text(state.startYear);
    
    const endLabel = slider.append('text')
        .attr('class', 'year-label end-year')
        .attr('x', x(parseInt(state.endYear)))
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('fill', '#e0e0e0')
        .text(state.endYear);
    
    function updateYearLabels() {
        startLabel
            .attr('x', x(parseInt(state.startYear)))
            .text(state.startYear);
        
        endLabel
            .attr('x', x(parseInt(state.endYear)))
            .text(state.endYear);
    }
}

// Mise à jour des données filtrées
function updateFilteredData() {
    // Appeler directement la fonction de filtrage et de mise à jour
    filterDataAndUpdateVisualization();
}

// Afficher les articles pour un terme et une année spécifiques
function showArticlesForTermAndYear(term, year) {
    console.log(`Affichage des articles pour ${term} en ${year}`);
    
    // Vérifier si nous avons des articles pré-filtrés pour cette année et ce terme
    let articlesForTerm = [];
    
    if (state.availableArticles && state.availableArticles[year] && state.availableArticles[year][term]) {
        // Utiliser les articles pré-filtrés
        articlesForTerm = state.availableArticles[year][term];
    } else {
        // Rechercher manuellement les articles
        articlesForTerm = state.articles.filter(article => {
            // Extraire l'année de l'ID de l'article
            const match = article.id.match(/article_([0-9]{4})-([0-9]{2})-([0-9]{2})/);
            const articleYear = match ? match[1] : null;
            
            // Vérifier si l'article correspond à l'année et contient le terme
            const hasYear = articleYear === year;
            const hasTerm = article.content && article.content.toLowerCase().includes(term.toLowerCase());
            
            // Appliquer également les filtres de canton et journal si nécessaire
            const cantonMatch = state.selectedCantons.length === 0 || 
                              state.selectedCantons.includes(article.canton);
            
            let journalMatch = true;
            if (state.selectedJournals.length > 0) {
                const normalizedArticleJournal = article.newspaper ? 
                    article.newspaper.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "") : "";
                
                journalMatch = state.selectedJournals.some(journal => {
                    const normalizedSelectedJournal = journal.toLowerCase()
                        .normalize("NFD").replace(/[\u0300-\u036f]/g, "");
                    return normalizedArticleJournal.includes(normalizedSelectedJournal) || 
                           normalizedSelectedJournal.includes(normalizedArticleJournal);
                });
            }
            
            return hasYear && hasTerm && cantonMatch && journalMatch;
        });
    }
    
    // Limiter le nombre d'articles à afficher
    const limitedArticles = articlesForTerm.slice(0, config.maxArticlesToShow);
    
    // Mettre à jour l'état
    state.selectedArticles = limitedArticles;
    
    // Mettre à jour le titre du modal
    document.getElementById('modal-title').textContent = `Articles contenant "${term}" en ${year} (${limitedArticles.length}/${articlesForTerm.length})`;
    
    // Afficher les articles
    updateArticleDisplay(limitedArticles);
    
    // Afficher le modal
    document.getElementById('article-modal').style.display = 'block';
}

// Fonction pour filtrer les données et mettre à jour la visualisation
function filterDataAndUpdateVisualization() {
    console.log("Début du filtrage avec:", { 
        selectedCantons: state.selectedCantons, 
        selectedJournals: state.selectedJournals,
        selectedTerms: state.selectedTerms,
        startYear: state.startYear,
        endYear: state.endYear
    });
    
    // Si aucun filtre de canton ou journal n'est sélectionné, utiliser toutes les données
    if (state.selectedCantons.length === 0 && state.selectedJournals.length === 0) {
        console.log("Aucun filtre sélectionné, utilisation des données complètes");
        createVisualization();
        return;
    }
    
    // IMPORTANT: Utiliser directement les données annuelles précalculées
    // Au lieu d'essayer de compter les occurrences dans les articles
    const filteredData = [];
    const filteredYears = state.years.filter(year => 
        parseInt(year) >= state.startYear && 
        parseInt(year) <= state.endYear
    );
    
    console.log(`Années filtrées: ${filteredYears.join(', ')}`);
    
    // Fonction pour extraire l'année et le journal d'un ID d'article
    function extractInfoFromArticleId(id) {
        // Format: article_1969-01-04_la_sentinelle_01187e92_mistral
        const match = id.match(/article_([0-9]{4})-([0-9]{2})-([0-9]{2})_([^_]+)/);
        if (match) {
            return {
                year: match[1],
                journal: match[4].replace(/_/g, ' ')
            };
        }
        return { year: null, journal: null };
    }
    
    // Stocker les articles disponibles par année et terme pour une utilisation ultérieure
    state.availableArticles = {};
    
    // Filtrer les données originales en fonction des années et des filtres
    for (const year of filteredYears) {
        // Créer un point de données pour cette année
        const dataPoint = { year };
        state.availableArticles[year] = {};
        
        // Pour chaque terme sélectionné, récupérer la valeur originale
        for (const term of state.selectedTerms) {
            // Valeur par défaut si aucune donnée n'est disponible
            let termValue = 0;
            state.availableArticles[year][term] = [];
            
            // Si des données sont disponibles pour cette année et ce terme
            if (state.yearlyData[year] && state.yearlyData[year][term] !== undefined) {
                // Récupérer la valeur originale
                termValue = state.yearlyData[year][term];
                
                // Appliquer les filtres de canton et journal si nécessaire
                if (state.selectedCantons.length > 0 || state.selectedJournals.length > 0) {
                    // Trouver les articles de cette année qui contiennent ce terme
                    const articlesWithTerm = state.articles.filter(article => {
                        // Extraire l'année et le journal de l'ID
                        const info = extractInfoFromArticleId(article.id);
                        if (info.year !== year) return false;
                        
                        // Vérifier si l'article correspond aux filtres
                        const cantonMatch = state.selectedCantons.length === 0 || 
                                          state.selectedCantons.includes(article.canton);
                        
                        // Normaliser les noms de journaux pour la comparaison
                        let journalMatch = false;
                        if (state.selectedJournals.length === 0) {
                            journalMatch = true;
                        } else {
                            // Convertir le nom du journal de l'article en minuscules sans accents
                            const normalizedArticleJournal = article.newspaper ? 
                                article.newspaper.toLowerCase()
                                    .normalize("NFD").replace(/[\u0300-\u036f]/g, "") : "";
                            
                            // Vérifier si un des journaux sélectionnés correspond
                            journalMatch = state.selectedJournals.some(journal => {
                                const normalizedSelectedJournal = journal.toLowerCase()
                                    .normalize("NFD").replace(/[\u0300-\u036f]/g, "");
                                return normalizedArticleJournal.includes(normalizedSelectedJournal) || 
                                       normalizedSelectedJournal.includes(normalizedArticleJournal);
                            });
                        }
                        
                        // Vérifier si l'article contient le terme
                        const hasTerm = article.content && 
                                      article.content.toLowerCase().includes(term.toLowerCase());
                        
                        return cantonMatch && journalMatch && hasTerm;
                    });
                    
                    // Stocker les articles pour cette année et ce terme
                    state.availableArticles[year][term] = articlesWithTerm;
                    
                    // Compter le nombre d'articles filtrés pour cette année et ce terme
                    // Si aucun article ne correspond aux filtres, mettre la valeur à 0
                    termValue = articlesWithTerm.length;
                }
            }
            
            // Ajouter la valeur au point de données seulement si des articles sont disponibles
            if (termValue > 0) {
                dataPoint[term] = termValue;
            } else {
                // Mettre à null pour que D3.js ignore ce point dans le graphique
                dataPoint[term] = null;
            }
        }
        
        // Ajouter le point de données aux données filtrées
        filteredData.push(dataPoint);
    }
    
    console.log("Données filtrées générées:", filteredData);
    
    // Mettre à jour temporairement les données pour la visualisation
    const originalData = state.data;
    state.data = filteredData;
    
    // Créer la visualisation avec les données filtrées
    createVisualization();
    
    // Restaurer les données originales
    state.data = originalData;
}

// Création de la visualisation principale
function createVisualization() {
    const container = document.getElementById('chart-container');
    container.innerHTML = '';
    
    if (state.selectedTerms.length === 0) {
        container.innerHTML = '<div class="no-data-message">Veuillez sélectionner au moins un terme.</div>';
        return;
    }
    
    // Dimensions
    const width = container.clientWidth;
    const height = container.clientHeight || 500;
    const innerWidth = width - config.margin.left - config.margin.right;
    const innerHeight = height - config.margin.top - config.margin.bottom;
    
    // Filtrer les données selon la période sélectionnée
    const filteredYears = state.years.filter(year => 
        parseInt(year) >= state.startYear && 
        parseInt(year) <= state.endYear
    );
    
    // Préparer les données pour la visualisation
    const chartData = prepareChartData(filteredYears);
    
    // Créer le SVG
    const svg = d3.select('#chart-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Groupe principal avec marge
    const g = svg.append('g')
        .attr('transform', `translate(${config.margin.left},${config.margin.top})`);
    
    // Échelles
    const x = d3.scaleLinear()
        .domain([parseInt(state.startYear), parseInt(state.endYear)])
        .range([0, innerWidth]);
    
    const y = d3.scaleLinear()
        .domain([0, d3.max(chartData, d => d3.max(state.selectedTerms, term => d[term]))])
        .nice()
        .range([innerHeight, 0]);
    
    // Couleurs
    const color = d3.scaleOrdinal()
        .domain(state.selectedTerms)
        .range(config.colors);
    
    // Axes
    const xAxis = g.append('g')
        .attr('class', 'axis x-axis')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).tickFormat(d => d.toString()).ticks(Math.min(filteredYears.length, 10)));
    
    const yAxis = g.append('g')
        .attr('class', 'axis y-axis')
        .call(d3.axisLeft(y));
    
    // Titre des axes
    g.append('text')
        .attr('class', 'axis-label')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 40)
        .attr('text-anchor', 'middle')
        .attr('fill', '#e0e0e0')
        .text('Année');
        
    // Ajuster la marge supérieure pour la légende
    config.margin.top = Math.max(40, state.selectedTerms.length > 3 ? 60 : 40);
    
    g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .attr('fill', '#e0e0e0')
        .text('Nombre de mentions');
    
    // Créer la visualisation selon le type sélectionné
    switch(state.vizType) {
        case 'line':
            createLineChart(g, chartData, x, y, color);
            break;
        case 'stacked':
            createStackedAreaChart(g, chartData, x, y, color);
            break;
        case 'stream':
            createStreamgraph(g, chartData, x, y, color);
            break;
    }
    
    // Légende
    createLegend(svg, color);
    
    // Tooltip
    setupTooltip(chartData, x, y);
}

// Préparation des données pour le graphique
function prepareChartData(filteredYears) {
    return filteredYears.map(year => {
        const yearData = { year };
        
        // Pour chaque terme, vérifier si des articles existent pour cette année
        state.selectedTerms.forEach(term => {
            // Vérifier si nous avons des articles pré-filtrés pour cette année et ce terme
            if (state.availableArticles && 
                state.availableArticles[year] && 
                state.availableArticles[year][term] && 
                state.availableArticles[year][term].length > 0) {
                // Utiliser le nombre d'articles comme valeur
                yearData[term] = state.availableArticles[year][term].length;
            } else {
                // Aucun article trouvé, mettre à null pour que D3 ignore ce point
                yearData[term] = null;
            }
        });
        
        return yearData;
    });
}

// Création d'un graphique en ligne
function createLineChart(g, data, x, y, color) {
    const line = d3.line()
        .x(d => x(parseInt(d.year)))
        .y(d => y(d.value))
        .curve(d3.curveMonotoneX)
        .defined(d => d.value !== null); // Ignorer les points sans valeur
    
    state.selectedTerms.forEach(term => {
        const termData = data.map(d => ({
            year: d.year,
            value: d[term]
        })).filter(d => d.value !== null); // Filtrer les points sans valeur
        
        g.append('path')
            .datum(termData)
            .attr('class', 'line')
            .attr('d', line)
            .attr('stroke', color(term))
            .attr('data-term', term)
            .attr('opacity', 0)
            .transition()
            .duration(config.transitionDuration)
            .attr('opacity', 1);
        
        // Ajouter des points pour chaque année
        const points = g.selectAll(`.point-${term}`)
            .data(termData)
            .enter()
            .append('circle')
            .attr('class', `point point-${term}`)
            .attr('cx', d => x(parseInt(d.year)))
            .attr('cy', d => y(d.value))
            .attr('r', 6) // Légèrement plus grand pour faciliter le clic
            .attr('fill', color(term))
            .attr('data-term', term)
            .attr('data-year', d => d.year)
            .attr('data-value', d => d.value)
            .attr('opacity', 0)
            .attr('cursor', 'pointer'); // Ajouter un curseur pointer pour indiquer que c'est cliquable
            
        // Ajouter l'événement de clic directement (pas dans la transition)
        points.on('click', function(event, d) {
            event.stopPropagation(); // Empêcher la propagation de l'événement
            showArticlesForTermAndYear(term, d.year);
        });
        
        // Animer l'apparition des points
        points.transition()
            .duration(config.transitionDuration)
            .attr('opacity', 1);
    });
}

// Création d'un graphique en aires empilées
function createStackedAreaChart(g, data, x, y, color) {
    const stack = d3.stack()
        .keys(state.selectedTerms)
        .order(d3.stackOrderNone)
        .offset(d3.stackOffsetNone);
    
    const stackedData = stack(data);
    
    const area = d3.area()
        .x(d => x(parseInt(d.data.year)))
        .y0(d => y(d[0]))
        .y1(d => y(d[1]))
        .curve(d3.curveMonotoneX);
    
    g.selectAll('.area')
        .data(stackedData)
        .enter()
        .append('path')
        .attr('class', 'area')
        .attr('d', area)
        .attr('fill', d => color(d.key))
        .attr('data-term', d => d.key)
        .attr('opacity', 0)
        .transition()
        .duration(config.transitionDuration)
        .attr('opacity', 0.7);
}

// Création d'un streamgraph
function createStreamgraph(g, data, x, y, color) {
    const stack = d3.stack()
        .keys(state.selectedTerms)
        .order(d3.stackOrderInsideOut)
        .offset(d3.stackOffsetWiggle);
    
    const stackedData = stack(data);
    
    // Recalculer l'échelle y pour le streamgraph
    const yStreamScale = d3.scaleLinear()
        .domain([
            d3.min(stackedData, layer => d3.min(layer, d => d[0])),
            d3.max(stackedData, layer => d3.max(layer, d => d[1]))
        ])
        .range([innerHeight, 0]);
    
    const area = d3.area()
        .x(d => x(parseInt(d.data.year)))
        .y0(d => yStreamScale(d[0]))
        .y1(d => yStreamScale(d[1]))
        .curve(d3.curveBasis);
    
    g.selectAll('.stream')
        .data(stackedData)
        .enter()
        .append('path')
        .attr('class', 'stream')
        .attr('d', area)
        .attr('fill', d => color(d.key))
        .attr('data-term', d => d.key)
        .attr('opacity', 0)
        .transition()
        .duration(config.transitionDuration)
        .attr('opacity', 0.8);
}

// Création de la légende
function createLegend(svg, color) {
    // Calculer la largeur disponible
    const availableWidth = svg.attr('width') - config.margin.left - config.margin.right;
    
    // Créer un fond pour la légende
    const legendBackground = svg.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', svg.attr('width'))
        .attr('height', state.selectedTerms.length > 3 ? 60 : 35)
        .attr('fill', '#1a2639')
        .attr('opacity', 0.8);
    
    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(20, 10)`);
    
    // Limiter à 3 éléments par ligne maximum
    const itemsPerRow = Math.min(3, Math.floor(availableWidth / 250));
    
    const legendItems = legend.selectAll('.legend-item')
        .data(state.selectedTerms)
        .enter()
        .append('g')
        .attr('class', 'legend-item')
        .attr('transform', (d, i) => {
            const row = Math.floor(i / itemsPerRow);
            const col = i % itemsPerRow;
            return `translate(${col * 250}, ${row * 25})`;
        });
    
    legendItems.append('rect')
        .attr('class', 'legend-color')
        .attr('width', 15)
        .attr('height', 15)
        .attr('fill', d => color(d));
    
    legendItems.append('text')
        .attr('x', 20)
        .attr('y', 12)
        .attr('fill', '#e0e0e0')
        .text(d => d);
}

// Configuration du tooltip
function setupTooltip(data, x, y) {
    const tooltip = d3.select('#tooltip');
    
    // Pour les lignes et points
    d3.selectAll('.line, .point, .area, .stream')
        .on('mouseover', function(event) {
            const term = d3.select(this).attr('data-term');
            const year = d3.select(this).attr('data-year');
            const value = d3.select(this).attr('data-value');
            
            let tooltipContent = '';
            
            if (year && value) {
                // Point spécifique
                tooltipContent = `<strong style="color:#fff">${term}</strong><br>Année: ${year}<br>Mentions: ${value}<br><span style="font-size:0.8em;color:#aaa">(Cliquez pour voir les articles)</span>`;
            } else {
                // Ligne ou aire
                tooltipContent = `<strong style="color:#fff">${term}</strong>`;
            }
            
            tooltip
                .style('display', 'block')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px')
                .html(tooltipContent);
            
            // Mettre en évidence l'élément
            d3.selectAll(`[data-term="${term}"]`)
                .style('stroke-width', '5px')
                .style('opacity', 1);
        })
        .on('mouseout', function() {
            tooltip.style('display', 'none');
            
            // Restaurer le style
            d3.selectAll('.line, .area, .stream')
                .style('stroke-width', '3px')
                .style('opacity', state.vizType === 'line' ? 1 : 0.7);
        })
        .on('click', function(event) {
            const term = d3.select(this).attr('data-term');
            const year = d3.select(this).attr('data-year');
            
            if (term && year) {
                // Afficher les articles pour ce terme et cette année
                showArticlesForTermAndYear(term, year);
            }
        });
}

// Génération d'insights sur les données
function generateInsights() {
    const insightsContent = document.getElementById('insights-content');
    
    if (state.selectedTerms.length === 0) {
        insightsContent.innerHTML = '<p>Sélectionnez des termes pour voir les tendances.</p>';
        return;
    }
    
    // Filtrer les années selon la période sélectionnée
    const filteredYears = state.years.filter(year => 
        parseInt(year) >= parseInt(state.startYear) && 
        parseInt(year) <= parseInt(state.endYear)
    );
    
    // Calculer les tendances pour chaque terme sélectionné
    const insights = state.selectedTerms.map(term => {
        const yearValues = filteredYears.map(year => ({
            year,
            value: state.yearlyData[year][term] || 0
        }));
        
        // Trouver l'année avec le plus de mentions
        const peakYear = yearValues.reduce((max, current) => 
            current.value > max.value ? current : max, { value: 0 }
        );
        
        // Calculer la tendance (croissance/décroissance)
        const firstValue = yearValues[0].value;
        const lastValue = yearValues[yearValues.length - 1].value;
        const trend = lastValue > firstValue ? 'croissante' : 
                     lastValue < firstValue ? 'décroissante' : 'stable';
        
        // Calculer le taux de croissance
        const growthRate = firstValue === 0 ? 'N/A' : 
            Math.round((lastValue - firstValue) / firstValue * 100);
        
        return {
            term,
            peakYear,
            trend,
            growthRate,
            totalMentions: yearValues.reduce((sum, item) => sum + item.value, 0)
        };
    });
    
    // Trier les insights par nombre total de mentions
    insights.sort((a, b) => b.totalMentions - a.totalMentions);
    
    // Générer le HTML
    let html = '<div class="insights-grid">';
    
    insights.forEach(insight => {
        html += `
            <div class="insight-card">
                <h4>${insight.term}</h4>
                <p>Mentions totales: <strong>${insight.totalMentions}</strong></p>
                <p>Pic en: <strong>${insight.peakYear.year}</strong> (${insight.peakYear.value} mentions)</p>
                <p>Tendance: <strong>${insight.trend}</strong>`;
        
        if (insight.growthRate !== 'N/A') {
            html += ` (${insight.growthRate > 0 ? '+' : ''}${insight.growthRate}%)`;
        }
        
        html += '</p></div>';
    });
    
    html += '</div>';
    
    // Ajouter une analyse globale
    const topTerm = insights[0];
    const totalMentionsAll = insights.reduce((sum, insight) => sum + insight.totalMentions, 0);
    
    html += `
        <div class="global-insights">
            <h4>Analyse globale</h4>
            <p>Sur la période ${state.startYear}-${state.endYear}, on compte un total de <strong>${totalMentionsAll}</strong> mentions des termes sélectionnés.</p>
            <p>Le terme le plus mentionné est <strong>${topTerm.term}</strong> avec ${topTerm.totalMentions} mentions (${Math.round(topTerm.totalMentions/totalMentionsAll*100)}% du total).</p>
            <p><button id="show-articles-btn" class="btn-primary">Voir des exemples d'articles</button></p>
        </div>
    `;
    
    insightsContent.innerHTML = html;
    
    // Ajouter un écouteur d'événement pour le bouton
    document.getElementById('show-articles-btn').addEventListener('click', () => {
        showArticlesForTerm(topTerm.term);
    });
}

// Création des sélecteurs géographiques
function createGeoVisualizations() {
    createCantonSelector();
    createNewspaperSelector();
}

// Création du sélecteur par canton
function createCantonSelector() {
    const container = document.getElementById('canton-selector');
    
    // Préparer les données
    const cantonData = [];
    
    for (const canton in state.cantonData) {
        let total = 0;
        for (const term of state.selectedTerms) {
            total += state.cantonData[canton][term] || 0;
        }
        
        if (total > 0) {
            cantonData.push({
                canton: canton,
                count: total
            });
        }
    }
    
    // Trier par nombre de mentions
    cantonData.sort((a, b) => b.count - a.count);
    
    // Vider le conteneur
    container.innerHTML = '';
    
    // Créer les éléments de sélection
    cantonData.forEach(d => {
        const item = document.createElement('div');
        item.className = 'selector-item';
        item.setAttribute('data-canton', d.canton);
        item.innerHTML = `${d.canton} <span class="count">${d.count}</span>`;
        
        // Ajouter l'événement de clic
        item.addEventListener('click', () => {
            // Basculer la classe active
            item.classList.toggle('active');
            
            // Mettre à jour la liste des cantons sélectionnés
            if (item.classList.contains('active')) {
                if (!state.selectedCantons.includes(d.canton)) {
                    state.selectedCantons.push(d.canton);
                }
                // Suppression de l'ouverture du modal d'articles ici
            } else {
                state.selectedCantons = state.selectedCantons.filter(canton => canton !== d.canton);
            }
            
            // Mettre à jour le graphique
            filterDataAndUpdateVisualization();
        });
        
        container.appendChild(item);
    });
}

// Création du sélecteur par journal
function createNewspaperSelector() {
    const container = document.getElementById('newspaper-selector');
    
    // Préparer les données
    const newspaperData = [];
    
    for (const journal in state.journalData) {
        let total = 0;
        for (const term of state.selectedTerms) {
            total += state.journalData[journal][term] || 0;
        }
        
        if (total > 0) {
            newspaperData.push({
                journal: journal,
                count: total
            });
        }
    }
    
    // Trier par nombre de mentions
    newspaperData.sort((a, b) => b.count - a.count);
    
    // Limiter à 15 journaux pour la lisibilité
    const topNewspapers = newspaperData.slice(0, 15);
    
    // Vider le conteneur
    container.innerHTML = '';
    
    // Créer les éléments de sélection
    topNewspapers.forEach(d => {
        const item = document.createElement('div');
        item.className = 'selector-item';
        item.setAttribute('data-journal', d.journal);
        item.innerHTML = `${d.journal} <span class="count">${d.count}</span>`;
        
        // Ajouter l'événement de clic
        item.addEventListener('click', () => {
            // Basculer la classe active
            item.classList.toggle('active');
            
            // Mettre à jour la liste des journaux sélectionnés
            if (item.classList.contains('active')) {
                if (!state.selectedJournals.includes(d.journal)) {
                    state.selectedJournals.push(d.journal);
                }
                // Suppression de l'ouverture du modal d'articles ici
            } else {
                state.selectedJournals = state.selectedJournals.filter(journal => journal !== d.journal);
            }
            
            // Mettre à jour le graphique
            filterDataAndUpdateVisualization();
        });
        
        container.appendChild(item);
    });
}

// Initialisation du modal d'articles
function initArticleModal() {
    const modal = document.getElementById('article-modal');
    const closeBtn = modal.querySelector('.close');
    
    // Fermer le modal quand on clique sur la croix
    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });
    
    // Fermer le modal quand on clique en dehors du contenu
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Initialiser les filtres
    initArticleFilters();
}

// Initialisation des filtres d'articles
function initArticleFilters() {
    const cantonFilter = document.getElementById('canton-filter');
    const newspaperFilter = document.getElementById('newspaper-filter');
    
    // Remplir les options de cantons
    state.cantons.forEach(canton => {
        const option = document.createElement('option');
        option.value = canton;
        option.textContent = canton;
        cantonFilter.appendChild(option);
    });
    
    // Remplir les options de journaux
    state.newspapers.forEach(newspaper => {
        const option = document.createElement('option');
        option.value = newspaper;
        option.textContent = newspaper;
        newspaperFilter.appendChild(option);
    });
    
    // Écouteurs d'événements pour les filtres
    cantonFilter.addEventListener('change', filterArticles);
    newspaperFilter.addEventListener('change', filterArticles);
}

// Filtrer les articles selon les sélections
function filterArticles() {
    const cantonFilter = document.getElementById('canton-filter').value;
    const newspaperFilter = document.getElementById('newspaper-filter').value;
    
    let filteredArticles = state.selectedArticles;
    
    if (cantonFilter) {
        filteredArticles = filteredArticles.filter(article => 
            article.details && article.details.canton === cantonFilter);
    }
    
    if (newspaperFilter) {
        filteredArticles = filteredArticles.filter(article => 
            article.journal === newspaperFilter);
    }
    
    // Mettre à jour l'affichage
    updateArticleDisplay(filteredArticles);
}

// Afficher les articles pour un terme spécifique
function showArticlesForTerm(term) {
    console.log(`Affichage des articles pour ${term}`);
    
    // Trouver tous les articles qui contiennent ce terme, en respectant les filtres actuels
    let articlesWithTerm = [];
    
    // Si des filtres sont appliqués (cantons ou journaux)
    if (state.selectedCantons.length > 0 || state.selectedJournals.length > 0) {
        // Parcourir toutes les années disponibles pour ce terme
        for (const year in state.availableArticles) {
            if (state.availableArticles[year][term] && state.availableArticles[year][term].length > 0) {
                articlesWithTerm = articlesWithTerm.concat(state.availableArticles[year][term]);
            }
        }
    } else {
        // Sans filtre, utiliser tous les articles qui contiennent ce terme
        articlesWithTerm = state.articles.filter(article => {
            return article.content && article.content.toLowerCase().includes(term.toLowerCase());
        });
    }
    
    // Limiter le nombre d'articles à afficher
    const limitedArticles = articlesWithTerm.slice(0, config.maxArticlesToShow);
    
    // Mettre à jour l'état
    state.selectedArticles = limitedArticles;
    
    // Mettre à jour le titre du modal
    document.getElementById('modal-title').textContent = `Articles contenant le terme "${term}" (${limitedArticles.length}/${articlesWithTerm.length})`;
    
    // Afficher les articles
    updateArticleDisplay(limitedArticles);
    
    // Afficher le modal
    document.getElementById('article-modal').style.display = 'block';
}

// Afficher les articles pour un canton spécifique
function showArticlesForCanton(canton) {
    // Trouver tous les articles de ce canton qui contiennent un des termes sélectionnés
    const articlesFromCanton = state.data.filter(article => 
        article.canton === canton && 
        state.selectedTerms.some(term => article.values[term] > 0)
    );
    
    // Limiter le nombre d'articles à afficher
    const limitedArticles = articlesFromCanton.slice(0, config.maxArticlesToShow);
    
    // Mettre à jour l'état
    state.selectedArticles = limitedArticles;
    
    // Mettre à jour le titre du modal
    document.getElementById('modal-title').textContent = `Articles du canton ${canton}`;
    
    // Afficher les articles
    updateArticleDisplay(limitedArticles);
    
    // Afficher le modal
    document.getElementById('article-modal').style.display = 'block';
}

// Afficher les articles pour un journal spécifique
function showArticlesForJournal(journal) {
    // Trouver tous les articles de ce journal qui contiennent un des termes sélectionnés
    const articlesFromJournal = state.data.filter(article => 
        article.journal === journal && 
        state.selectedTerms.some(term => article.values[term] > 0)
    );
    
    // Limiter le nombre d'articles à afficher
    const limitedArticles = articlesFromJournal.slice(0, config.maxArticlesToShow);
    
    // Mettre à jour l'état
    state.selectedArticles = limitedArticles;
    
    // Mettre à jour le titre du modal
    document.getElementById('modal-title').textContent = `Articles du journal ${journal}`;
    
    // Afficher les articles
    updateArticleDisplay(limitedArticles);
    
    // Afficher le modal
    document.getElementById('article-modal').style.display = 'block';
}

// Mettre à jour l'affichage des articles
function updateArticleDisplay(articles) {
    // Mettre à jour le compteur d'articles
    document.getElementById('article-count').textContent = `${articles.length} article${articles.length > 1 ? 's' : ''} trouvé${articles.length > 1 ? 's' : ''}`;
    
    // Détruire le swiper existant s'il existe
    if (state.swiper) {
        state.swiper.destroy();
    }
    
    // Vider le conteneur d'articles
    const articlesContainer = document.getElementById('articles-container');
    articlesContainer.innerHTML = '';
    
    // Créer les slides pour chaque article
    articles.forEach(article => {
        const slide = document.createElement('div');
        slide.className = 'swiper-slide';
        
        // Récupérer les détails de l'article
        // Accéder directement aux propriétés de l'article
        const content = article.content || 'Contenu non disponible';
        const title = article.title || 'Titre non disponible';
        const date = article.date || 'Date inconnue';
        const journal = article.newspaper || article.journal || 'Journal inconnu';
        const canton = article.canton || 'Canton inconnu';
        
        // Mettre en évidence les termes sélectionnés dans le contenu
        let highlightedContent = content;
        state.selectedTerms.forEach(term => {
            // Vérifier si le terme est présent dans le contenu de l'article
            if (content && content.toLowerCase().includes(term.toLowerCase())) {
                const regex = new RegExp(`(${term})`, 'gi');
                highlightedContent = highlightedContent.replace(regex, '<span class="highlight">$1</span>');
            }
        });
        
        // Créer le HTML de l'article
        slide.innerHTML = `
            <div class="article-header">
                <div class="article-title">${title}</div>
                <div class="article-meta">
                    <span>${date}</span>
                    <span>${journal} (${canton})</span>
                </div>
            </div>
            <div class="article-content">
                <p>${highlightedContent}</p>
            </div>
            <div class="article-footer">
                <div class="article-tags">
                    ${state.selectedTerms.filter(term => content && content.toLowerCase().includes(term.toLowerCase()))
                        .map(term => `<span class="article-tag">${term}</span>`)
                        .join('')}
                </div>
                <div class="article-id">${article.id}</div>
            </div>
        `;
        
        articlesContainer.appendChild(slide);
    });
    
    // Initialiser le nouveau swiper
    setTimeout(() => {
        state.swiper = new Swiper('.swiper-container', {
            slidesPerView: 1,
            spaceBetween: 30,
            pagination: {
                el: '.swiper-pagination',
                clickable: true,
            },
            navigation: {
                nextEl: '.swiper-button-next',
                prevEl: '.swiper-button-prev',
            },
        });
        console.log('Swiper initialisé avec', articles.length, 'articles');
    }, 100); // Petit délai pour s'assurer que le DOM est prêt
}
