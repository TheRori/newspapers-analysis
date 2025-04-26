// Event listener for article links
document.addEventListener('click', function(event) {
    // Check if the clicked element is an article link
    const target = event.target;
    
    if (target.matches('a[data-article-id]')) {
        event.preventDefault();
        
        // Get the article ID from the data attribute
        const articleId = target.getAttribute('data-article-id');
        
        // Create a custom event that Dash can listen for
        const customEvent = new CustomEvent('article-selected', {
            detail: {
                articleId: articleId
            }
        });
        
        // Dispatch the event on the document
        document.dispatchEvent(customEvent);
        
        console.log('Article selected:', articleId);
    }
});

// Listen for the custom event in case we need to do additional processing
document.addEventListener('article-selected', function(e) {
    console.log('Article selection event received:', e.detail.articleId);
});
