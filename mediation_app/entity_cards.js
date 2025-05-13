// Entity Cards JavaScript

// Configuration
const entityConfig = {
    dataPath: 'data/source/collections/entities/e201554b-f053-4d05-8572-649c5cc294c5/source_files/entity_summary_8941d60d.json',
    articlesPath: 'data/source/collections/entities/e201554b-f053-4d05-8572-649c5cc294c5/source_files/articles_with_entities_8941d60d.json',
    maxArticlesToShow: 20,
    entityColors: {
        'ORG': '#00ff9d',
        'LOC': '#00a0ff',
        'PERSON': '#ff00a0'
    },
    defaultEntityType: 'ORG'
};

// Generate a trend graph for entity mentions over time based on real data
function generateTrendGraph(entityName, entityType) {
    // If we don't have articles data yet, show a loading graph
    if (!entityState.articles) {
        // If articles aren't loaded yet, load them
        if (!entityState.loadingArticles) {
            entityState.loadingArticles = true;
            loadAllArticles().then(() => {
                // Refresh all entity cards once articles are loaded
                loadEntityType(entityState.selectedEntityType);
                entityState.loadingArticles = false;
            }).catch(error => {
                console.error('Error loading articles:', error);
                entityState.loadingArticles = false;
            });
        }
        
        // Return a placeholder loading graph
        return `
            <!-- Loading indicator -->
            <text x="50" y="50" text-anchor="middle" font-size="10" fill="${entityConfig.entityColors[entityType]}">Loading...</text>
            <circle cx="50" cy="60" r="5" fill="none" stroke="${entityConfig.entityColors[entityType]}" stroke-width="1">
                <animate attributeName="r" from="5" to="10" dur="1s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="1" to="0" dur="1s" repeatCount="indefinite" />
            </circle>
        `;
    }
    
    // Find articles that mention this entity
    const entityArticles = entityState.articles.filter(article => {
        return article.entities && article.entities.some(entity => 
            entity.text === entityName && entity.label === entityType);
    });
    
    // If no articles found for this entity, show empty graph
    if (entityArticles.length === 0) {
        return `
            <!-- Empty data indicator -->
            <text x="50" y="50" text-anchor="middle" font-size="10" fill="${entityConfig.entityColors[entityType]}">No data</text>
        `;
    }
    
    // Extract years from article dates
    const articleYears = entityArticles.map(article => {
        const date = article.date;
        return date ? parseInt(date.substring(0, 4)) : null;
    }).filter(year => year !== null);
    
    // Count occurrences by year
    const yearCounts = {};
    articleYears.forEach(year => {
        yearCounts[year] = (yearCounts[year] || 0) + 1;
    });
    
    // Get min and max years
    const years = Object.keys(yearCounts).map(y => parseInt(y));
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const yearRange = maxYear - minYear;
    
    // Create data points for each year
    const dataPoints = [];
    for (let year = minYear; year <= maxYear; year++) {
        dataPoints.push({
            year: year,
            value: yearCounts[year] || 0
        });
    }
    
    // Sort data points by year
    dataPoints.sort((a, b) => a.year - b.year);
    
    // Normalize values to fit in the SVG viewBox
    const maxValue = Math.max(...dataPoints.map(d => d.value));
    const normalizedPoints = dataPoints.map(d => {
        return {
            x: 10 + ((d.year - minYear) / (yearRange || 1)) * 80, // Map years to x coordinates (10-90)
            y: 90 - ((d.value / (maxValue || 1)) * 70)           // Map values to y coordinates (20-90)
        };
    });
    
    // Create the path data
    let pathData = '';
    if (normalizedPoints.length > 0) {
        pathData = `M ${normalizedPoints[0].x},${normalizedPoints[0].y}`;
        for (let i = 1; i < normalizedPoints.length; i++) {
            pathData += ` L ${normalizedPoints[i].x},${normalizedPoints[i].y}`;
        }
    }
    
    // Create the SVG elements
    return `
        <!-- Axis lines -->
        <line x1="10" y1="90" x2="90" y2="90" stroke="${entityConfig.entityColors[entityType]}" stroke-width="1" opacity="0.5" />
        <line x1="10" y1="20" x2="10" y2="90" stroke="${entityConfig.entityColors[entityType]}" stroke-width="1" opacity="0.5" />
        
        <!-- Year labels -->
        <text x="10" y="98" text-anchor="middle" font-size="8" fill="${entityConfig.entityColors[entityType]}" opacity="0.7">${minYear}</text>
        <text x="90" y="98" text-anchor="middle" font-size="8" fill="${entityConfig.entityColors[entityType]}" opacity="0.7">${maxYear}</text>
        
        <!-- Trend line -->
        <path d="${pathData}" fill="none" stroke="${entityConfig.entityColors[entityType]}" stroke-width="2" />
        
        <!-- Data points -->
        ${normalizedPoints.map(p => `<circle cx="${p.x}" cy="${p.y}" r="1.5" fill="${entityConfig.entityColors[entityType]}" />`).join('')}
    `;
}

// Application state
let entityState = {
    summary: null,
    entityData: null,
    articles: null,
    selectedEntityType: entityConfig.defaultEntityType,
    selectedEntity: null,
    swiper: null,
    entityArticles: {},  // Cache for articles by entity
    loadingArticles: false
};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initEntityApp();
});

// Initialize the entity application
async function initEntityApp() {
    try {
        // Load entity summary data
        entityState.summary = await loadEntitySummary();
        
        // Start loading articles in the background
        loadAllArticles().catch(error => {
            console.error('Error loading articles:', error);
        });
        
        // Initialize UI components
        initEntityTypeButtons();
        initSwiper();
        initModal();
        
        // Load initial entity type (ORG by default)
        loadEntityType(entityState.selectedEntityType);
        
    } catch (error) {
        console.error('Error initializing entity app:', error);
        document.querySelector('.entity-cards-container').innerHTML = 
            `<div class="error-message">Error loading entity data: ${error.message}</div>`;
    }
}

// Load all articles with entities
async function loadAllArticles() {
    if (entityState.articles) {
        return entityState.articles; // Return cached articles if already loaded
    }
    
    try {
        // Fetch the articles with entities file
        const response = await fetch(entityConfig.articlesPath);
        if (!response.ok) {
            throw new Error(`Failed to load articles: ${response.status} ${response.statusText}`);
        }
        
        // Parse the JSON response
        const articles = await response.json();
        entityState.articles = articles;
        
        console.log(`Loaded ${articles.length} articles with entities`);
        return articles;
    } catch (error) {
        console.error('Error loading articles:', error);
        throw error;
    }
}

// Load entity summary data
async function loadEntitySummary() {
    const response = await fetch(entityConfig.dataPath);
    if (!response.ok) {
        throw new Error(`Failed to load entity summary: ${response.status} ${response.statusText}`);
    }
    return await response.json();
}

// Initialize entity type selection buttons
function initEntityTypeButtons() {
    const buttons = document.querySelectorAll('.entity-type-button');
    buttons.forEach(button => {
        button.addEventListener('click', () => {
            // Update active button
            buttons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Load selected entity type
            const entityType = button.getAttribute('data-type');
            loadEntityType(entityType);
        });
    });
}

// Load entities for the selected type
function loadEntityType(entityType) {
    entityState.selectedEntityType = entityType;
    entityState.selectedEntity = null;
    
    // Clear entity details
    document.getElementById('selected-entity-name').textContent = 'Sélectionnez une organisation';
    document.getElementById('entity-count').textContent = '0';
    document.getElementById('article-count').textContent = '0';
    document.getElementById('year-range').textContent = '-';
    document.getElementById('entity-articles-list').innerHTML = 
        '<p class="no-selection">Sélectionnez une organisation pour voir les articles associés</p>';
    
    // Get top entities for the selected type
    const topEntities = entityState.summary.entity_frequencies[entityType];
    if (!topEntities || Object.keys(topEntities).length === 0) {
        document.getElementById('entity-cards').innerHTML = 
            `<div class="swiper-slide"><div class="entity-card">
                <h3 class="entity-name">Aucune entité</h3>
                <p>Aucune entité de type ${entityType} n'a été trouvée.</p>
            </div></div>`;
        if (entityState.swiper) {
            entityState.swiper.update();
        }
        return;
    }
    
    // Create entity cards for top 10 entities
    const entityCards = createEntityCards(topEntities, entityType);
    document.getElementById('entity-cards').innerHTML = entityCards;
    
    // Update swiper
    if (entityState.swiper) {
        entityState.swiper.update();
    }
    
    // Add click event to entity cards
    document.querySelectorAll('.entity-card').forEach(card => {
        card.addEventListener('click', () => {
            const entityName = card.getAttribute('data-entity');
            const entityCount = parseInt(card.getAttribute('data-count'));
            selectEntity(entityName, entityCount, entityType);
        });
    });
}

// Create entity cards HTML
function createEntityCards(entities, entityType) {
    // Convert to array and sort by count (descending)
    const entitiesArray = Object.entries(entities)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10); // Get top 10
    
    return entitiesArray.map((entity, index) => {
        return `
        <div class="swiper-slide">
            <div class="entity-card" data-entity="${entity.name}" data-count="${entity.count}">
                <div class="entity-rank">${index + 1}</div>
                <h3 class="entity-name">${entity.name}</h3>
                <div class="entity-count">${entity.count}</div>
                <div class="entity-visual">
                    <svg viewBox="0 0 100 100">
                        ${generateTrendGraph(entity.name, entityType)}
                    </svg>
                </div>
            </div>
        </div>`;
    }).join('');
}

// Initialize Swiper
function initSwiper() {
    entityState.swiper = new Swiper('.entity-swiper', {
        slidesPerView: 1,
        spaceBetween: 20,
        centeredSlides: true,
        grabCursor: true,
        keyboard: {
            enabled: true,
        },
        pagination: {
            el: '.swiper-pagination',
            clickable: true,
        },
        navigation: {
            nextEl: '.swiper-button-next',
            prevEl: '.swiper-button-prev',
        },
        breakpoints: {
            640: {
                slidesPerView: 2,
            },
            1024: {
                slidesPerView: 3,
            },
        }
    });
    
    // Add event listener for slide change
    entityState.swiper.on('slideChange', function () {
        const activeSlide = document.querySelector('.swiper-slide-active');
        if (activeSlide) {
            const card = activeSlide.querySelector('.entity-card');
            if (card) {
                const entityName = card.getAttribute('data-entity');
                const entityCount = parseInt(card.getAttribute('data-count'));
                selectEntity(entityName, entityCount, entityState.selectedEntityType);
            }
        }
    });
}

// Select an entity and load its details
async function selectEntity(entityName, entityCount, entityType) {
    entityState.selectedEntity = entityName;
    
    // Update UI to show the entity is selected
    document.querySelectorAll('.entity-card').forEach(card => {
        if (card.getAttribute('data-entity') === entityName) {
            card.classList.add('active');
        } else {
            card.classList.remove('active');
        }
    });
    
    // Update entity details
    document.getElementById('selected-entity-name').textContent = entityName;
    document.getElementById('entity-count').textContent = entityCount.toLocaleString();
    
    // Show loading state
    document.getElementById('entity-articles-list').innerHTML = 
        '<p class="loading">Chargement des articles...</p>';
    
    try {
        // Load articles for this entity if not already cached
        if (!entityState.entityArticles[entityName]) {
            entityState.loadingArticles = true;
            await loadArticlesForEntity(entityName, entityType);
            entityState.loadingArticles = false;
        }
        
        // Get articles for this entity
        const articles = entityState.entityArticles[entityName] || [];
        
        // Update article count
        document.getElementById('article-count').textContent = articles.length.toLocaleString();
        
        // Determine year range
        if (articles.length > 0) {
            const years = articles.map(article => {
                const date = article.date;
                return date ? parseInt(date.substring(0, 4)) : null;
            }).filter(year => year !== null);
            
            if (years.length > 0) {
                const minYear = Math.min(...years);
                const maxYear = Math.max(...years);
                document.getElementById('year-range').textContent = 
                    minYear === maxYear ? `${minYear}` : `${minYear}-${maxYear}`;
            } else {
                document.getElementById('year-range').textContent = 'N/A';
            }
        } else {
            document.getElementById('year-range').textContent = 'N/A';
        }
        
        // Display articles
        displayArticlesForEntity(articles, entityName);
        
    } catch (error) {
        console.error('Error selecting entity:', error);
        document.getElementById('entity-articles-list').innerHTML = 
            `<p class="error-message">Error loading articles: ${error.message}</p>`;
    }
}

// Load articles for a specific entity
async function loadArticlesForEntity(entityName, entityType) {
    // This would normally load from the full articles file, but for performance,
    // we'll simulate by creating a fetch request that would be intercepted by a server
    // In a real implementation, you'd need a server endpoint that filters the articles
    
    // For demo purposes, we'll create some sample articles
    const sampleArticles = [];
    
    // Try to fetch a small sample of articles (this would be replaced with real API call)
    try {
        const response = await fetch(`${entityConfig.articlesPath}?entity=${encodeURIComponent(entityName)}&type=${entityType}&limit=20`);
        
        // If the fetch fails (which it likely will without a proper backend), generate sample data
        if (!response.ok) {
            throw new Error('No direct API access');
        }
        
        const data = await response.json();
        entityState.entityArticles[entityName] = data;
        
    } catch (error) {
        console.log('Using sample data instead of API:', error.message);
        
        // Generate sample articles
        const years = ['1960', '1970', '1980', '1990', '2000'];
        const newspapers = ['Le Journal de Genève', 'La Tribune de Lausanne', 'Le Temps', 'Le Matin', 'La Liberté'];
        
        for (let i = 0; i < Math.min(entityConfig.maxArticlesToShow, 10 + Math.floor(Math.random() * 10)); i++) {
            const year = years[Math.floor(Math.random() * years.length)];
            const month = String(Math.floor(Math.random() * 12) + 1).padStart(2, '0');
            const day = String(Math.floor(Math.random() * 28) + 1).padStart(2, '0');
            const newspaper = newspapers[Math.floor(Math.random() * newspapers.length)];
            
            sampleArticles.push({
                id: `article_${year}-${month}-${day}_${newspaper.toLowerCase().replace(/\s+/g, '_')}_${Math.random().toString(36).substring(2, 10)}`,
                title: `Article mentionnant ${entityName} (${i + 1})`,
                date: `${year}-${month}-${day}`,
                newspaper: newspaper,
                content: `Ceci est un exemple d'article mentionnant ${entityName} plusieurs fois. ${entityName} est une organisation importante dans cet article. Les mentions de ${entityName} sont mises en évidence.`,
                entities: [
                    {
                        text: entityName,
                        label: entityType,
                        start: 30,
                        end: 30 + entityName.length
                    },
                    {
                        text: entityName,
                        label: entityType,
                        start: 30 + entityName.length + 40,
                        end: 30 + entityName.length + 40 + entityName.length
                    }
                ]
            });
        }
        
        // Cache the sample articles
        entityState.entityArticles[entityName] = sampleArticles;
    }
}

// Display articles for the selected entity
function displayArticlesForEntity(articles, entityName) {
    const articlesContainer = document.getElementById('entity-articles-list');
    
    if (articles.length === 0) {
        articlesContainer.innerHTML = '<p class="no-articles">Aucun article trouvé pour cette entité.</p>';
        return;
    }
    
    // Sort articles by date (newest first)
    articles.sort((a, b) => {
        if (!a.date) return 1;
        if (!b.date) return -1;
        return b.date.localeCompare(a.date);
    });
    
    // Create article list
    const articlesHTML = articles.slice(0, entityConfig.maxArticlesToShow).map((article, index) => {
        return `
        <div class="article-item" data-article-id="${article.id}" data-index="${index}">
            <div class="article-title">${article.title || 'Sans titre'}</div>
            <div class="article-meta">
                <span class="article-date">${formatDate(article.date)}</span>
                <span class="article-newspaper">${article.newspaper || 'Journal inconnu'}</span>
            </div>
        </div>`;
    }).join('');
    
    articlesContainer.innerHTML = articlesHTML;
    
    // Add click event to article items
    document.querySelectorAll('.article-item').forEach(item => {
        item.addEventListener('click', () => {
            const articleId = item.getAttribute('data-article-id');
            const index = parseInt(item.getAttribute('data-index'));
            showArticleModal(articles[index], entityName);
        });
    });
}

// Format date for display
function formatDate(dateString) {
    if (!dateString) return 'Date inconnue';
    
    try {
        const [year, month, day] = dateString.split('-');
        return `${day}/${month}/${year}`;
    } catch (e) {
        return dateString;
    }
}

// Initialize the article modal
function initModal() {
    const modal = document.getElementById('article-modal');
    const closeBtn = modal.querySelector('.close');
    
    // Close modal when clicking the X button
    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });
    
    // Close modal when clicking outside the content
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// Show article in modal
function showArticleModal(article, entityName) {
    const modal = document.getElementById('article-modal');
    const modalTitle = document.getElementById('modal-title');
    const articleContent = document.getElementById('article-content');
    const articleDate = document.getElementById('article-date');
    const articleNewspaper = document.getElementById('article-newspaper');
    
    // Set modal content
    modalTitle.textContent = article.title || 'Article sans titre';
    articleDate.textContent = formatDate(article.date);
    articleNewspaper.textContent = article.newspaper || 'Journal inconnu';
    
    // Highlight entity mentions in content
    let content = article.content || 'Contenu non disponible';
    
    // If we have entity positions, use them to highlight
    if (article.entities && article.entities.length > 0) {
        // Sort entities by start position (descending to avoid position shifts)
        const relevantEntities = article.entities
            .filter(entity => entity.text === entityName)
            .sort((a, b) => b.start - a.start);
        
        // Replace each occurrence with highlighted version
        relevantEntities.forEach(entity => {
            const before = content.substring(0, entity.start);
            const entityText = content.substring(entity.start, entity.end);
            const after = content.substring(entity.end);
            content = before + `<span class="highlighted-entity">${entityText}</span>` + after;
        });
    } else {
        // Simple text replacement (less accurate)
        const regex = new RegExp(`(${entityName})`, 'gi');
        content = content.replace(regex, '<span class="highlighted-entity">$1</span>');
    }
    
    // Add paragraphs
    content = content.split('\n').map(para => `<p>${para}</p>`).join('');
    
    articleContent.innerHTML = content;
    
    // Show modal
    modal.style.display = 'block';
}
