const form = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const submitButton = document.getElementById('submit-button');
const healthPill = document.getElementById('health-pill');
const banner = document.getElementById('message-banner');
const indexStatus = document.getElementById('index-status');
const documentCount = document.getElementById('document-count');
const embeddingDimension = document.getElementById('embedding-dimension');
const resultsSummary = document.getElementById('results-summary');
const resultsState = document.getElementById('results-state');
const resultsList = document.getElementById('results-list');
const answerState = document.getElementById('answer-state');
const answerText = document.getElementById('answer-text');

function setHealth(text, className) {
  healthPill.textContent = text;
  healthPill.className = `pill ${className}`;
}

function showBanner(message, kind) {
  banner.textContent = message;
  banner.className = `message-banner message-${kind}`;
}

function hideBanner() {
  banner.textContent = '';
  banner.className = 'message-banner message-hidden';
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? 'Retrieving...' : 'Retrieve';
}

function clearResults(message = 'Retrieval results will appear here after a successful query.') {
  resultsList.innerHTML = '';
  resultsState.textContent = message;
  resultsState.style.display = 'block';
  resultsSummary.textContent = 'Submit a query to inspect the generated answer and supporting documents.';
  embeddingDimension.textContent = '-';
  answerText.textContent = '';
  answerText.style.display = 'none';
  answerState.textContent = 'The generated answer will appear here after a successful query.';
  answerState.style.display = 'block';
}

function renderResults(payload) {
  const documents = payload.documents || [];
  embeddingDimension.textContent = String(payload.query_embedding_dimension ?? '-');
  resultsSummary.textContent = `Showing ${documents.length} retrieved document${documents.length === 1 ? '' : 's'} for "${payload.query}".`;
  answerState.style.display = 'none';
  answerText.textContent = payload.answer || 'No generated answer was returned.';
  answerText.style.display = 'block';

  if (!documents.length) {
    clearResults('The backend returned no retrieved documents for this query.');
    return;
  }

  resultsState.style.display = 'none';
  resultsList.innerHTML = documents.map((document, index) => {
    const score = typeof document.score === 'number' ? document.score.toFixed(3) : 'n/a';
    const metadata = JSON.stringify(document.metadata || {}, null, 2);
    const escapedText = escapeHtml(document.text || '');
    const escapedMetadata = escapeHtml(metadata);

    return `
      <article class="result-card">
        <header>
          <p class="result-rank">Result ${index + 1}</p>
          <p class="result-score">Score ${score}</p>
        </header>
        <p class="result-text">${escapedText}</p>
        <pre class="result-meta">${escapedMetadata}</pre>
      </article>
    `;
  }).join('');
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

async function parseResponse(response) {
  const text = await response.text();

  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function refreshBackendStatus() {
  try {
    const [healthResponse, statsResponse] = await Promise.all([
      fetch('/health'),
      fetch('/vector-db/stats'),
    ]);

    if (!healthResponse.ok) {
      throw new Error(`Health check failed with status ${healthResponse.status}.`);
    }

    const statsPayload = await parseResponse(statsResponse);
    setHealth('API Ready', 'pill-success');

    if (statsResponse.ok && statsPayload) {
      indexStatus.textContent = statsPayload.index_ready ? 'Ready' : 'Not initialized';
      documentCount.textContent = String(statsPayload.document_count ?? 0);
    } else {
      indexStatus.textContent = 'Unknown';
      documentCount.textContent = '-';
    }
  } catch (error) {
    setHealth('API Error', 'pill-error');
    indexStatus.textContent = 'Unavailable';
    documentCount.textContent = '-';
    showBanner(
      'The API server is not responding correctly. Start or restart it with: uv run uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 --reload',
      'error',
    );
  }
}

async function ensureIndexReady() {
  const statsResponse = await fetch('/vector-db/stats');
  const statsPayload = await parseResponse(statsResponse);

  if (!statsResponse.ok) {
    throw new Error('Unable to read vector DB status from the backend.');
  }

  indexStatus.textContent = statsPayload.index_ready ? 'Ready' : 'Not initialized';
  documentCount.textContent = String(statsPayload.document_count ?? 0);

  if (statsPayload.index_ready) {
    return;
  }

  showBanner('Vector index not initialized. Attempting to ingest sample documents now.', 'info');

  const ingestResponse = await fetch('/ingest?document_count=10', { method: 'POST' });
  const ingestPayload = await parseResponse(ingestResponse);

  if (!ingestResponse.ok) {
    const details = typeof ingestPayload === 'string' ? ingestPayload : JSON.stringify(ingestPayload);
    throw new Error(
      `The backend could not initialize the vector index. Azure authentication may be missing. Run: azd auth login --scope api://ailab/Model.Access. Details: ${details.slice(0, 220)}`,
    );
  }

  indexStatus.textContent = ingestPayload.index_ready ? 'Ready' : 'Not initialized';
  documentCount.textContent = String(ingestPayload.documents_ingested ?? '-');
  showBanner('Vector index initialized successfully.', 'success');
}

async function runQuery(query) {
  const response = await fetch('/rag/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: 3 }),
  });

  const payload = await parseResponse(response);

  if (response.status === 404) {
    throw new Error('The /rag/answer endpoint is missing on the running server. Restart FastAPI so it picks up the Sprint 4 backend changes.');
  }

  if (response.status === 400 && payload && typeof payload === 'object' && String(payload.detail || '').includes('Call /ingest first')) {
    await ensureIndexReady();
    return runQuery(query);
  }

  if (!response.ok) {
    const details = typeof payload === 'string' ? payload : JSON.stringify(payload);
    throw new Error(`Query request failed with status ${response.status}. ${details}`);
  }

  return payload;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();

  if (!query) {
    showBanner('Enter a query before submitting.', 'error');
    clearResults('A query is required before retrieval can run.');
    return;
  }

  hideBanner();
  setLoading(true);
  clearResults('Running retrieval...');

  try {
    await ensureIndexReady();
    const payload = await runQuery(query);
    renderResults(payload);
    showBanner('Retrieved documents successfully.', 'success');
  } catch (error) {
    clearResults('No results were rendered because the request did not complete successfully.');
    showBanner(error.message, 'error');
  } finally {
    setLoading(false);
  }
});

clearResults();
refreshBackendStatus();
