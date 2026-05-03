// (El script es exactamente el mismo que tenías, sin cambios)
let currentDirection = 'es2yor';

document.querySelectorAll('.direction-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.direction-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentDirection = btn.dataset.direction;

        const placeholder = currentDirection === 'es2yor'
            ? 'Ej: buenos días, casa, agua...'
            : 'Ej: Kittianake, bwapo, bakte...';
        document.getElementById('searchInput').placeholder = placeholder;
    });
});

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
    } catch (error) {
        console.error('Error:', error);
        alert('Error al procesar la traducción');
    } finally {
        document.getElementById('loadingContainer').style.display = 'none';
    }
}

document.getElementById('searchBtn').addEventListener('click', performSearch);
document.getElementById('searchInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

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

    // Segmentación morfológica
    if (data.morphological) {
        const morphSection = document.createElement('div');
        morphSection.className = 'result-section';
        morphSection.innerHTML = `
                    <div class="section-title">
                        ✔ Segmentación Morfológica
                        <span class="badge badge-compositional">Análisis de raíz + sufijo</span>
                    </div>
                `;

        const card = document.createElement('div');
        card.className = 'translation-card';
        const part1Type = data.morphological.part1_exact ? 'exacta' : `fuzzy (${(data.morphological.part1_similarity * 100).toFixed(1)}%)`;

        card.innerHTML = `
                    <div class="translation-main">${data.morphological.translation}</div>
                    <div class="morph-detail-box">
                        <div class="morph-label">Análisis de segmentación:</div>
                        <div class="morph-segments">
                            <span class="chunk chunk-known">${data.morphological.part1}</span>
                            <span class="plus-sign">+</span>
                            <span class="chunk chunk-known">${data.morphological.part2}</span>
                            <span class="arrow">→</span>
                            <span class="chunk chunk-translation">${data.morphological.part1_translation}</span>
                            <span class="plus-sign">+</span>
                            <span class="chunk chunk-translation">${data.morphological.part2_translation}</span>
                        </div>
                        <div class="morph-meta">Raíz: ${part1Type} | Sufijo: exacta | Score: ${(data.morphological.score * 100).toFixed(0)}%</div>
                    </div>
                `;
        morphSection.appendChild(card);
        container.appendChild(morphSection);
    }

    // Traducción composicional
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

        const chunksHtml = data.compositional.chunks.map(chunk => {
            let className = '';
            let tooltip = '';
            if (chunk.match_type === 'unknown') {
                className = 'chunk-unknown';
                tooltip = 'Sin traducción';
            } else if (chunk.match_type === 'morphological') {
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
            return `<span class="chunk ${className}" title="${tooltip}">${chunk.translation}</span>`;
        }).join('');

        const hasMorph = data.compositional.chunks.some(c => c.match_type === 'morphological');
        let morphDetails = '';
        if (hasMorph) {
            const morphChunks = data.compositional.chunks.filter(c => c.match_type === 'morphological');
            morphDetails = '<div class="morph-detail-box"><strong>Análisis morfológico:</strong>';
            morphChunks.forEach(chunk => {
                const d = chunk.morphological_detail;
                morphDetails += `
                            <div class="morph-sub">
                                "<span>${chunk.source}</span>" → 
                                <span class="chunk chunk-known">${d.part1}</span> + <span class="chunk chunk-known">${d.part2}</span> → 
                                <span class="chunk-translation">${d.part1_translation} ${d.part2_translation}</span>
                            </div>
                        `;
            });
            morphDetails += '</div>';
        }

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
                    ${morphDetails}
                    ${data.compositional.has_unknowns ? '<p class="unknown-warning">⚠ Contiene palabras sin traducción conocida</p>' : ''}
                `;
        compSection.appendChild(card);
        container.appendChild(compSection);
    }

    // Alternativas
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

document.getElementById('searchInput').focus();