// Entity Cards JavaScript (version finale avec chargement fluide en superposition)

// --- CONFIGURATION ---
const entityConfig = {
    pathTemplate: {
        summary: 'data/source/collections/entities/065dd3cd-0462-419e-b2ef-d8fde883eb71/source_files/entity_summary_{source_id}.json',
        articles: 'data/source/collections/entities/065dd3cd-0462-419e-b2ef-d8fde883eb71/source_files/articles_with_entities_{source_id}.json'
    },
    maxArticlesToShow: 50,
    entityColors: { 'ORG': '#00ff9d', 'LOC': '#00a0ff', 'PERSON': '#ff00a0' },
    defaultEntityType: 'ORG'
};

// --- APPLICATION STATE ---
let entityState = {
    summary: null, articles: null,
    selectedEntityType: entityConfig.defaultEntityType,
    selectedEntity: null, swiper: null, entityArticlesCache: {}, isLoading: false
};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    initEntityTypeButtons();
    initSwiper(); // Swiper n'est initialisé qu'une seule fois ici !
    initModal();
    initDataSourceSelector();
    const initialSourceId = document.getElementById('date-range-select').value;
    await reloadData(initialSourceId);
}

// --- DATA LOADING & UI MANAGEMENT (SECTION CORRIGÉE) ---

function initDataSourceSelector() {
    const selector = document.getElementById('date-range-select');
    selector.addEventListener('change', (event) => {
        if (entityState.isLoading) return;
        reloadData(event.target.value);
    });
}

async function reloadData(sourceId) {
    console.log(`Chargement des données pour la source : ${sourceId}`);
    entityState.isLoading = true;

    const container = document.querySelector('.container');
    if (!container) return;

    // 1. Créer et ajouter l'overlay de chargement SANS rien détruire
    const loaderOverlay = document.createElement('div');
    loaderOverlay.id = 'loader-overlay';
    loaderOverlay.innerHTML = `
        <div class="loading-spinner">
            <div class="spinner-circle"></div>
            <div class="spinner-circle"></div>
            <div class="spinner-circle"></div>
            <div class="spinner-circle"></div>
        </div>
        <p>Chargement des entités...</p>
    `;
    container.appendChild(loaderOverlay);

    try {
        const [summary, articles] = await loadInitialData(sourceId);
        entityState.summary = summary;
        entityState.articles = articles;
        entityState.entityArticlesCache = {};

        // 2. Mettre à jour le contenu "en arrière-plan" pendant que le loader est visible
        loadEntityType(entityState.selectedEntityType);

    } catch (error) {
        console.error('Erreur critique lors du chargement des données:', error);
        // En cas d'erreur, on remplace tout le contenu par un message d'erreur
        container.innerHTML = `
            <div class="error-message" style="height: 400px; display: flex; align-items: center; justify-content: center;">
                <p><strong>Erreur :</strong> Impossible de charger les données pour la source ${sourceId}.</p>
            </div>`;
    } finally {
        // 3. Quoi qu'il arrive, on retire l'overlay pour révéler le contenu mis à jour (ou le message d'erreur)
        const overlayToRemove = document.getElementById('loader-overlay');
        if (overlayToRemove) {
            overlayToRemove.remove();
        }
        entityState.isLoading = false;
    }
}

async function loadInitialData(sourceId) {
    const summaryPath = entityConfig.pathTemplate.summary.replace('{source_id}', sourceId);
    const articlesPath = entityConfig.pathTemplate.articles.replace('{source_id}', sourceId);

    const [summaryRes, articlesRes] = await Promise.all([ fetch(summaryPath), fetch(articlesPath) ]);

    if (!summaryRes.ok) throw new Error(`Échec du chargement du résumé (${summaryPath})`);
    if (!articlesRes.ok) throw new Error(`Échec du chargement des articles (${articlesPath})`);

    return [await summaryRes.json(), await articlesRes.json()];
}

// --- UI & LOGIC ---
function loadEntityType(entityType) {
    entityState.selectedEntityType = entityType;
    entityState.selectedEntity = null;
    resetEntityDetails();
    
    if (!entityState.summary) return;

    const topEntities = entityState.summary.entity_frequencies[entityType];
    const cardsContainer = document.getElementById('entity-cards');
    
    if (!topEntities || Object.keys(topEntities).length === 0) {
        cardsContainer.innerHTML = `<div class="swiper-slide"><p>Aucune entité de type ${entityType} trouvée.</p></div>`;
    } else {
        cardsContainer.innerHTML = createEntityCards(topEntities, entityType);
    }

    // On met simplement à jour le Swiper, on ne le détruit/recrée plus
    if (entityState.swiper) {
        entityState.swiper.update();
        entityState.swiper.slideTo(0, 0); // Aller à la première slide sans animation
    }
    
    document.querySelectorAll('.entity-card').forEach(card => {
        card.addEventListener('click', () => {
            if (entityState.isLoading) return;
            const entityName = card.getAttribute('data-entity');
            const entityCount = parseInt(card.getAttribute('data-count'));
            selectEntity(entityName, entityCount, entityType);
        });
    });
}

// Fichier : entity_cards.js

async function selectEntity(entityName, entityCount, entityType) {
    if (entityState.isLoading || entityState.selectedEntity === entityName) return;

    entityState.isLoading = true; // Empêche les clics multiples
    entityState.selectedEntity = entityName;

    document.querySelectorAll('.entity-card').forEach(card => {
        card.classList.toggle('active', card.getAttribute('data-entity') === entityName);
    });

    document.getElementById('selected-entity-name').textContent = entityName;
    document.getElementById('entity-count').textContent = entityCount.toLocaleString();
    
    // Affiche un message de chargement qui, cette fois, ne sera pas bloqué
    const articlesListContainer = document.getElementById('entity-articles-list');
    articlesListContainer.innerHTML = '<p class="loading">Filtrage des articles en cours...</p>';

    // --- DÉBUT DE LA LOGIQUE ASYNCHRONE ---

    // Fonction qui exécute le filtrage par lots pour ne pas geler l'interface
    const filterArticlesInBackground = () => {
        return new Promise(resolve => {
            // Vérifie si les articles sont déjà en cache
            if (entityState.entityArticlesCache[entityName]) {
                resolve(entityState.entityArticlesCache[entityName]);
                return;
            }

            const allArticles = entityState.articles;
            const result = [];
            let currentIndex = 0;
            const batchSize = 250; // Traiter 250 articles à la fois

            function processBatch() {
                const batchEnd = Math.min(currentIndex + batchSize, allArticles.length);

                for (let i = currentIndex; i < batchEnd; i++) {
                    if (allArticles[i].entities?.some(e => e.text === entityName && e.label === entityType)) {
                        result.push(allArticles[i]);
                    }
                }
                currentIndex = batchEnd;

                if (currentIndex < allArticles.length) {
                    // Si ce n'est pas fini, on planifie le prochain lot
                    setTimeout(processBatch, 0);
                } else {
                    // C'est fini, on met en cache et on renvoie le résultat
                    entityState.entityArticlesCache[entityName] = result;
                    resolve(result);
                }
            }
            processBatch(); // On lance le premier lot
        });
    };

    // On appelle notre fonction asynchrone et on attend le résultat sans freezer
    const articlesForEntity = await filterArticlesInBackground();

    // --- FIN DE LA LOGIQUE ASYNCHRONE ---

    // Une fois le filtrage terminé, on met à jour l'interface
    document.getElementById('article-count').textContent = articlesForEntity.length.toLocaleString();
    updateYearRange(articlesForEntity);
    displayArticlesForEntity(articlesForEntity, entityName);
    
    entityState.isLoading = false; // On autorise à nouveau les clics
}

function createEntityCards(entities, entityType) {
    return Object.entries(entities)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 20)
        .map((entity, index) => 
            `<div class="swiper-slide">
                <div class="entity-card" data-entity="${entity.name}" data-count="${entity.count}">
                    <div class="entity-rank">${index + 1}</div>
                    <h3 class="entity-name">${entity.name}</h3>
                    <div class="entity-count">${entity.count.toLocaleString()}</div>
                    <div class="entity-visual"><svg viewBox="0 0 100 100">${generateTrendGraph(entity.name, entityType)}</svg></div>
                </div>
            </div>`
        ).join('');
}

function displayArticlesForEntity(articles, entityName) {
    const container = document.getElementById('entity-articles-list');
    if (!articles || articles.length === 0) {
        container.innerHTML = '<p class="no-articles">Aucun article trouvé pour cette entité.</p>';
        return;
    }
    container.innerHTML = articles
        .sort((a, b) => (b.date || '').localeCompare(a.date || ''))
        .slice(0, entityConfig.maxArticlesToShow)
        .map(article => 
            `<div class="article-item" data-article-id="${article.id}">
                <div class="article-title">${article.title || 'Sans titre'}</div>
                <div class="article-meta">
                    <span class="article-date">${formatDate(article.date)}</span>
                    <span class="article-newspaper">${article.newspaper || 'Journal inconnu'}</span>
                </div>
            </div>`
        ).join('');

    container.querySelectorAll('.article-item').forEach(item => {
        item.addEventListener('click', () => {
            const articleId = item.getAttribute('data-article-id');
            const article = entityState.articles.find(a => a.id === articleId);
            if (article) showArticleModal(article, entityName);
        });
    });
}


// --- FONCTIONS UTILITAIRES (inchangées) ---
function initEntityTypeButtons() {
    document.querySelectorAll('.entity-type-button').forEach(button => {
        button.addEventListener('click', () => {
            if (entityState.isLoading) return;
            document.querySelector('.entity-type-button.active').classList.remove('active');
            button.classList.add('active');
            loadEntityType(button.dataset.type);
        });
    });
}

function initSwiper() { 
    entityState.swiper = new Swiper('.entity-swiper', { 
        slidesPerView: 1, spaceBetween: 20, centeredSlides: true, grabCursor: true, keyboard: { enabled: true }, 
        pagination: { el: '.swiper-pagination', clickable: true }, 
        navigation: { nextEl: '.swiper-button-next', prevEl: '.swiper-button-prev' },
        breakpoints: { 640: { slidesPerView: 2 }, 1024: { slidesPerView: 3 }, 1200: { slidesPerView: 4 } } 
    });
}

function initModal() {
    const modal = document.getElementById('article-modal');
    const closeBtn = modal.querySelector('.close');
    closeBtn.onclick = () => { modal.style.display = 'none'; };
    window.onclick = (event) => { if (event.target == modal) modal.style.display = 'none'; };
}

function showArticleModal(article, entityName) {
    const modal = document.getElementById('article-modal');
    document.getElementById('modal-title').textContent = article.title || 'Sans titre';
    document.getElementById('article-date').textContent = formatDate(article.date);
    document.getElementById('article-newspaper').textContent = article.newspaper || 'Journal inconnu';
    let content = article.content || 'Contenu non disponible';
    const regex = new RegExp(`\\b(${entityName})\\b`, 'gi');
    content = content.replace(regex, '<span class="highlighted-entity">$1</span>');
    document.getElementById('article-content').innerHTML = content.replace(/\n/g, '<br>');
    modal.style.display = 'block';
}

function updateYearRange(articles) {
    const yearRangeEl = document.getElementById('year-range');
    if (!articles || articles.length === 0) { yearRangeEl.textContent = '-'; return; }
    const years = articles.map(a => a.date ? parseInt(a.date.substring(0, 4)) : null).filter(Boolean);
    if (years.length === 0) { yearRangeEl.textContent = '-'; return; }
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    yearRangeEl.textContent = minYear === maxYear ? minYear : `${minYear}-${maxYear}`;
}

function resetEntityDetails() {
    document.getElementById('selected-entity-name').textContent = 'Sélectionnez une entité';
    document.getElementById('entity-count').textContent = '0';
    document.getElementById('article-count').textContent = '0';
    document.getElementById('year-range').textContent = '-';
    document.getElementById('entity-articles-list').innerHTML = '<p class="no-selection">Sélectionnez une entité dans le carrousel pour voir les articles associés</p>';
}

function formatDate(dateString) {
    if (!dateString) return 'Date inconnue';
    try { 
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return dateString; // Retourne la chaine originale si la date est invalide
        const day = String(date.getDate()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const year = date.getFullYear();
        return `${day}/${month}/${year}`;
    } catch (e) { return dateString; }
}

function generateTrendGraph(entityName, entityType) {
    if (!entityState.articles) return `<text x="50" y="50" fill="#fff" font-size="10">Chargement...</text>`;
    const entityArticles = entityState.articles.filter(article => article.entities?.some(e => e.text === entityName && e.label === entityType));
    if (entityArticles.length === 0) return `<text x="50" y="50" fill="#fff" font-size="10">Pas de données</text>`;
    const yearCounts = {};
    entityArticles.forEach(article => {
        if (article.date) {
            const year = parseInt(article.date.substring(0, 4), 10);
            if (!isNaN(year)) yearCounts[year] = (yearCounts[year] || 0) + 1;
        }
    });
    const dataPoints = Object.entries(yearCounts).map(([year, value]) => ({ year: parseInt(year), value }));
    if (dataPoints.length < 2) return `<text x="50" y="50" fill="#fff" font-size="10">Données insuffisantes</text>`;
    dataPoints.sort((a, b) => a.year - b.year);
    const minYear = dataPoints[0].year;
    const maxYear = dataPoints[dataPoints.length - 1].year;
    const yearRange = maxYear - minYear || 1;
    const maxValue = Math.max(...dataPoints.map(d => d.value));
    const color = entityConfig.entityColors[entityType] || '#fff';
    const normalizedPoints = dataPoints.map(d => ({ x: 10 + ((d.year - minYear) / yearRange) * 80, y: 90 - ((d.value / maxValue) * 70) }));
    const pathData = "M " + normalizedPoints.map(p => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" L ");
    return `
        <line x1="10" y1="90" x2="90" y2="90" stroke="${color}" stroke-width="1" opacity="0.3" />
        <text x="10" y="98" text-anchor="start" font-size="8" fill="${color}" opacity="0.7">${minYear}</text>
        <text x="90" y="98" text-anchor="end" font-size="8" fill="${color}" opacity="0.7">${maxYear}</text>
        <path d="${pathData}" fill="none" stroke="${color}" stroke-width="2" />
        ${normalizedPoints.map(p => `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="2" fill="${color}" />`).join('')}
    `;
}