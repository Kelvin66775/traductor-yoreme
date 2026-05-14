let currentDirection = 'es2yor';
let searchHistory = [];   // Almacenará { query, direction, resultData }

// --- Configuración de botones de dirección ---
document.querySelectorAll('.direction-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.direction-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentDirection = btn.dataset.direction;

        const placeholder = currentDirection === 'es2yor'
            ? 'Ej: buenos días, casa, agua...'
            : 'Ej: Kittianake, bwapo, bakte...';
        document.getElementById('searchInput').placeholder = placeholder;

        // Limpiar búsqueda y resultados al cambiar de dirección
        document.getElementById('searchInput').value = '';
        const resultsContainer = document.getElementById('resultsContainer');
        resultsContainer.innerHTML = '';
        resultsContainer.classList.remove('show');
    });
});

// --- Función principal de búsqueda ---
async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) {
        alert('Por favor ingresa una palabra o frase');
        return;
    }

    document.getElementById('loadingContainer').style.display = 'block';
    document.getElementById('resultsContainer').classList.remove('show');

    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, direction: currentDirection })
        });
        const data = await response.json();
        displayResults(data);

        // Solo agregar al historial si hay algún resultado
        const hasResults = data.exact_matches.length > 0 ||
            data.compositional ||
            data.morphological ||
            data.alternatives.length > 0;
        if (hasResults) {
            addToHistory(query, currentDirection, data);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error al procesar la traducción');
    } finally {
        document.getElementById('loadingContainer').style.display = 'none';
    }
}

// Eventos del input y botón
document.getElementById('searchBtn').addEventListener('click', performSearch);
document.getElementById('searchInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

// --- Mostrar resultados ---
function displayResults(data) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';

    const hasResults = data.exact_matches.length > 0 ||
        data.compositional ||
        data.morphological ||
        data.alternatives.length > 0;

    if (!hasResults) {
        container.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">🦌🔍</div>
                <h3>No se encontraron resultados</h3>
                <p>Intenta con:</p>
                <ul style="list-style: none; margin-top: 15px;">
                    <li>✓ Verificar ortografía</li>
                    <li>✓ Buscar palabras individuales</li>
                    <li>✓ Usar sinónimos</li>
                </ul>
            </div>
        `;
        container.classList.add('show');
        return;
    }

    // Coincidencias exactas
    if (data.exact_matches.length > 0) {
        const exactSection = document.createElement('div');
        exactSection.className = 'result-section';
        exactSection.innerHTML = `
            <div class="section-title">
                ✔ Coincidencia Exacta
                <span class="badge badge-exact">Precisión 100%</span>
            </div>
        `;

        data.exact_matches.forEach(match => {
            const card = document.createElement('div');
            card.className = 'translation-card';

            let matchTypeLabel = '';
            if (match.match_type === 'original_acento') {
                matchTypeLabel = '<span class="match-type green">✓ Coincidencia con acentos</span>';
            } else if (match.match_type === 'normalized') {
                matchTypeLabel = '<span class="match-type amber">✓ Coincidencia sin acentos</span>';
            } else if (match.match_type === 'typo') {
                matchTypeLabel = '<span class="match-type indigo">✓ Corrección ortográfica</span>';
            }

            card.innerHTML = `
                <div class="translation-main">${match.target}</div>
                <div class="translation-source">Fuente: ${match.source}</div>
                ${matchTypeLabel}
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: 100%"></div>
                </div>
            `;
            exactSection.appendChild(card);
        });
        container.appendChild(exactSection);
    }

    // Segmentación morfológica (simplificada, sin caja de análisis)
    if (data.morphological) {
        const morphSection = document.createElement('div');
        morphSection.className = 'result-section';
        morphSection.innerHTML = `
            <div class="section-title">
                🔬 Traducción Morfológica
                <span class="badge badge-compositional">Raíz + sufijo</span>
            </div>
        `;

        const card = document.createElement('div');
        card.className = 'translation-card';
        card.innerHTML = `
            <div class="translation-main">${data.morphological.translation}</div>
            <div class="translation-source">Análisis: ${data.morphological.part1} + ${data.morphological.part2}</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${(data.morphological.score * 100).toFixed(0)}%"></div>
            </div>
        `;
        morphSection.appendChild(card);
        container.appendChild(morphSection);
    }

    // Traducción composicional (con palabra por palabra y ocultando comillas)
    if (data.compositional) {
        const compSection = document.createElement('div');
        compSection.className = 'result-section';
        compSection.innerHTML = `
        <div class="section-title">
            🔗 Traducción Composicional
            <span class="badge badge-compositional">Ensamblada</span>
        </div>
    `;

        const card = document.createElement('div');
        card.className = 'translation-card';

        // Función para saber si una traducción debe ocultarse (solo comilla simple)
        const shouldHideTranslation = (translation) => translation === "'";

        // Generar HTML para cada chunk, mostrando "palabra original → traducción"
        const chunksHtml = data.compositional.chunks.map(chunk => {
            let className = '';
            let tooltip = '';
            let displayText = '';

            // Determinar clase y tooltip según el tipo de coincidencia
            if (chunk.match_type === 'unknown') {
                className = 'chunk-unknown';
                tooltip = 'Sin traducción';
                displayText = chunk.source;  // solo la palabra original
            } else {
                if (chunk.match_type === 'morphological') {
                    className = 'chunk-known';
                    tooltip = `Segmentado: ${chunk.morphological_detail.part1} + ${chunk.morphological_detail.part2}`;
                } else if (chunk.match_type === 'original_acento') {
                    className = 'chunk-known';
                    tooltip = 'Coincidencia con acentos';
                } else if (chunk.match_type === 'normalized') {
                    className = 'chunk-known';
                    tooltip = 'Coincidencia sin acentos';
                } else {
                    className = 'chunk-known';
                    tooltip = 'Coincidencia exacta';
                }

                // Si la traducción es comilla, no la mostramos
                if (shouldHideTranslation(chunk.translation)) {
                    displayText = chunk.source;  // solo la palabra original
                    tooltip += ' (traducción omitida)';
                } else {
                    displayText = `${chunk.source} → ${chunk.translation}`;
                }
            }

            return `<span class="chunk ${className}" title="${tooltip}">${displayText}</span>`;
        }).join('');

        card.innerHTML = `
        <div class="translation-main">${data.compositional.translation}</div>
        ${data.compositional.translation_corrected ? `
            <div class="ai-correction">
                <div class="ai-header"><img src="/static/gpt.png" alt="IA" class="ai-icon-img"> Corrección con IA (Azure gpt-4.1-mini):</div>
                <div class="ai-text">${data.compositional.translation_corrected}</div>
            </div>
        ` : ''}
        ${data.compositional.azure_error ? `
            <div class="ai-error">⚠ Error de IA: ${data.compositional.azure_error}</div>
        ` : ''}
        <div class="chunks-container">${chunksHtml}</div>
        ${data.compositional.has_unknowns ? '<p class="unknown-warning">⚠ Contiene palabras sin traducción conocida</p>' : ''}
    `;
        compSection.appendChild(card);
        container.appendChild(compSection);
    }

    // Alternativas (sin cambios)
    if (data.alternatives.length > 0) {
        const altSection = document.createElement('div');
        altSection.className = 'result-section';
        altSection.innerHTML = `
            <div class="section-title">
                🔎 Sugerencias Relacionadas
                <span class="badge badge-fuzzy">Similitud</span>
            </div>
            <div class="alternatives-grid" id="alternativesGrid"></div>
        `;
        const grid = altSection.querySelector('#alternativesGrid');
        data.alternatives.forEach(alt => {
            const card = document.createElement('div');
            card.className = 'alternative-card';
            const scorePercent = (alt.score * 100).toFixed(0);
            card.innerHTML = `
                <div class="alternative-main">${alt.target}</div>
                <div class="alternative-source">${alt.source}</div>
                <span class="score-badge">${scorePercent}% similar</span>
            `;
            grid.appendChild(card);
        });
        container.appendChild(altSection);
    }

    container.classList.add('show');
}

// --- Historial de búsqueda (ahora almacena resultados) ---
function addToHistory(query, direction, resultData) {
    // Evitar duplicados inmediatos (misma query, dirección y mismo resultado básico)
    if (searchHistory.length > 0 &&
        searchHistory[0].query === query &&
        searchHistory[0].direction === direction) {
        return;
    }
    searchHistory.unshift({ query, direction, resultData });
    // Mantener solo 3
    if (searchHistory.length > 3) {
        searchHistory = searchHistory.slice(0, 3);
    }
    renderHistory();
}

function renderHistory() {
    const historyList = document.getElementById('historyList');
    if (searchHistory.length === 0) {
        historyList.innerHTML = '<p class="history-empty">Aún no has hecho ninguna búsqueda.</p>';
        return;
    }

    let html = '';
    searchHistory.forEach(item => {
        const dirLabel = item.direction === 'es2yor' ? 'ES→YOR' : 'YOR→ES';
        html += `
            <div class="history-item" data-query="${item.query}" data-direction="${item.direction}">
                <span class="history-query">${item.query}</span>
                <span class="history-direction">${dirLabel}</span>
            </div>
        `;
    });
    historyList.innerHTML = html;

    // Asignar eventos de clic: mostrar resultado almacenado sin buscar de nuevo
    document.querySelectorAll('.history-item').forEach((itemElement, index) => {
        itemElement.addEventListener('click', () => {
            const historyEntry = searchHistory[index];
            if (!historyEntry || !historyEntry.resultData) return;

            // Actualizar la interfaz de dirección
            document.querySelectorAll('.direction-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.direction === historyEntry.direction) {
                    btn.classList.add('active');
                }
            });
            currentDirection = historyEntry.direction;

            // Poner el texto en el input (opcional, pero no ejecutará búsqueda)
            document.getElementById('searchInput').value = historyEntry.query;

            // Mostrar los resultados almacenados directamente
            displayResults(historyEntry.resultData);
        });
    });
}

// Enfoque inicial
document.getElementById('searchInput').focus();