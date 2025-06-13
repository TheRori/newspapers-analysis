// Module de chargement de données global pour l'application de médiation
// Ce module centralise le chargement des données et pilote la barre de progression de l'interface.

// Configuration globale
const dataConfig = {
    dataPath: 'data/source/collections/term_tracking_chrono/0cc5204f-5f9b-464e-89aa-880a31b514d1/source_files/term_tracking_results.csv',
    articlesPath: 'data/source/articles_v1_filtered.json',
    timelineEventsPath: 'config/timeline_events.json',
    timelineDataPath: 'data/source/collections/term_tracking_chrono/ab7dd30b-8f09-46d0-ae5e-08aabddd1a90/source_files/term_tracking_results.csv',
    maxTermsToShow: 8,
    maxArticlesToShow: 20,
    colors: d3.schemeSet2,
    cantonColors: {
        'FR': '#c8102e', 'GE': '#e3000f', 'JU': '#cf142b', 'NE': '#008000', 
        'VD': '#008000', 'VS': '#ff0000', 'BE': '#ff0000', 'TI': '#ff0000', 
        'ZH': '#0066cc', 'other': '#666666'
    }
};

// État global des données
const dataState = {
    isLoaded: false, isLoading: false, error: null, data: null, rawData: null,
    articles: null, terms: [], years: [], cantons: [], newspapers: [],
    timelineEvents: [], articleInfo: null, journalData: {}, cantonData: {},
    availableArticles: {}, timelineData: null, callbacks: []
};

// Fonction pour charger toutes les données nécessaires
async function loadAllData() {
    if (dataState.isLoading) {
        return new Promise(resolve => dataState.callbacks.push(resolve));
    }
    if (dataState.isLoaded) {
        return Promise.resolve(dataState);
    }

    console.log('Début du chargement des données...');
    dataState.isLoading = true;
    
    // Affiche l'overlay de chargement
    updateLoadingProgress(0, 'Initialisation...');

    try {
        // --- ÉTAPE 1: Téléchargement des fichiers principaux (Progression 0% -> 30%) ---
        updateLoadingProgress(5, 'Chargement des données principales (CSV)...');
        const dataResponse = await fetch(dataConfig.dataPath);
        if (!dataResponse.ok) throw new Error(`Erreur HTTP: ${dataResponse.status}`);
        const dataCsvText = await dataResponse.text();

        updateLoadingProgress(15, 'Chargement de la bibliothèque d\'articles (JSON)...');
        const articlesResponse = await fetch(dataConfig.articlesPath);
        if (!articlesResponse.ok) console.warn('Impossible de charger les articles:', articlesResponse.statusText);
        dataState.articles = articlesResponse.ok ? await articlesResponse.json() : [];

        updateLoadingProgress(25, 'Chargement des données pour la timeline...');
        const timelineResponse = await fetch(dataConfig.timelineDataPath);
        if (!timelineResponse.ok) console.warn('Impossible de charger les données de timeline:', timelineResponse.statusText);
        const timelineCsvText = timelineResponse.ok ? await timelineResponse.text() : null;
        
        const data = d3.csvParse(dataCsvText);
        dataState.rawData = data;

        // --- ÉTAPE 2: Premier traitement lourd - Matching Timeline (Progression 30% -> 60%) ---
        if (timelineCsvText) {
            let timelineData = d3.csvParse(timelineCsvText);
            if (timelineData.length > 0 && timelineData[0].key) {
                timelineData = timelineData.map(d => {
                    const newRow = { key: d.key };
                    Object.keys(d).forEach(key => { if (key !== 'key') newRow[key] = parseFloat(d[key]) || 0; });
                    return newRow;
                });

                const yearlyData = {};
                const batchSize = 200; // Traiter 200 articles à la fois
                for (let i = 0; i < timelineData.length; i++) {
                    const article = timelineData[i];
                    const match = article.key.match(/article_(\d{4})-/);
                    if (match) {
                        const year = match[1];
                        if (!yearlyData[year]) {
                            yearlyData[year] = { key: year };
                            Object.keys(article).forEach(key => { if (key !== 'key') yearlyData[year][key] = 0; });
                        }
                        Object.keys(article).forEach(key => {
                            if (key !== 'key' && article[key]) yearlyData[year][key] += (parseFloat(article[key]) || 0);
                        });
                    }

                    if ((i + 1) % batchSize === 0 || i === timelineData.length - 1) {
                        const progress = 30 + Math.round(((i + 1) / timelineData.length) * 30);
                        updateLoadingProgress(progress, `Association des articles... ${i + 1}/${timelineData.length}`);
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                }
                dataState.timelineData = Object.values(yearlyData);
            }
        }
        
        // --- ÉTAPE 3: Deuxième traitement lourd - Préparation des données de visualisation (Progression 60% -> 90%) ---
        const articlesMap = new Map(dataState.articles.map(a => [a.id, a]));
        dataState.terms = Object.keys(data[0]).filter(key => key !== 'key');
        const articleInfo = [];
        const cleanJournalName = name => name ? name.replace(/\s+\d+\.?\s*$/, '').replace(/\.\s*$/, '').trim() || 'inconnu' : 'inconnu';
        
        const processingBatchSize = 500;
        for (let i = 0; i < data.length; i++) {
            const row = data[i];
            const parts = row.key.split('_');
            const dateStr = parts[1];
            const articleDetails = articlesMap.get(row.key) || {};
            const rawJournalName = articleDetails.newspaper || parts[3] || 'inconnu';
            
            articleInfo.push({
                id: row.key,
                year: dateStr.substring(0, 4),
                date: dateStr,
                journal: cleanJournalName(rawJournalName),
                canton: articleDetails.canton || 'unknown',
                values: dataState.terms.reduce((acc, term) => { acc[term] = parseFloat(row[term]) || 0; return acc; }, {}),
                details: articleDetails
            });

            if ((i + 1) % processingBatchSize === 0 || i === data.length - 1) {
                const progress = 60 + Math.round(((i + 1) / data.length) * 30);
                updateLoadingProgress(progress, `Analyse des données... ${i + 1}/${data.length}`);
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        dataState.articleInfo = articleInfo;

        // --- ÉTAPE 4: Finalisation et agrégations (Progression 90% -> 100%) ---
        updateLoadingProgress(95, 'Finalisation...');
        dataState.years = [...new Set(articleInfo.map(item => item.year))].sort();
        dataState.cantons = [...new Set(articleInfo.map(item => item.canton).filter(c => c !== 'unknown'))];
        dataState.newspapers = [...new Set(articleInfo.map(item => item.journal).filter(j => j !== 'inconnu'))];

        // Agrégation par année pour la visualisation principale
        const aggregatedData = [];
        dataState.years.forEach(year => {
            const yearArticles = articleInfo.filter(item => item.year === year);
            const yearData = { key: year };
            dataState.terms.forEach(term => {
                const values = yearArticles.map(article => article.values[term]).filter(v => !isNaN(v));
                yearData[term] = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
            });
            aggregatedData.push(yearData);
        });
        dataState.data = aggregatedData.sort((a, b) => parseInt(a.key) - parseInt(b.key));
        
        // Charger les événements optionnels
        try {
            const timelineEventsResponse = await fetch(dataConfig.timelineEventsPath);
            if (timelineEventsResponse.ok) {
                dataState.timelineEvents = (await timelineEventsResponse.json()).events || [];
            }
        } catch (err) { console.warn('Erreur au chargement des événements timeline:', err); }

        updateLoadingProgress(100, 'Terminé !');
        dataState.isLoaded = true;
        dataState.isLoading = false;
        
        // Cacher l'overlay et notifier les autres scripts
        setTimeout(() => {
            hideLoadingOverlay();
            dataState.callbacks.forEach(callback => callback(dataState));
            dataState.callbacks = [];
        }, 500); // Petit délai pour que l'utilisateur voie "Terminé !"

        return dataState;

    } catch (error) {
        console.error('Erreur majeure lors du chargement des données:', error);
        dataState.isLoading = false;
        dataState.error = error;
        hideLoadingOverlay(); // Cacher l'overlay en cas d'erreur
        dataState.callbacks.forEach(callback => callback(null, error));
        dataState.callbacks = [];
        throw error;
    }
}

function getData() {
    if (dataState.isLoaded) {
        return Promise.resolve(dataState);
    } else {
        return loadAllData();
    }
}

// Exposer les fonctions et l'état pour l'accès externe
window.dataLoader = {
    config: dataConfig,
    state: dataState,
    loadAllData,
    getData
};

// Charger les données automatiquement au chargement de la page
document.addEventListener('DOMContentLoaded', () => {
    console.log('Chargement automatique des données via data_loader.js...');
    loadAllData().catch(error => {
        console.error('Erreur interceptée lors du chargement automatique:', error);
        // On pourrait afficher un message d'erreur à l'utilisateur ici
    });
});