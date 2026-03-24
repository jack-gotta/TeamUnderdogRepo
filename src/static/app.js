const form = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const submitButton = document.getElementById('submit-button');
const checkHealthButton = document.getElementById('check-health-button');
const rebuildIndexButton = document.getElementById('rebuild-index-button');
const sourceModeSelect = document.getElementById('source-mode');
const documentCountInput = document.getElementById('document-count-input');
const healthPill = document.getElementById('health-pill');
const healthStatusText = document.getElementById('health-status-text');
const banner = document.getElementById('message-banner');
const indexStatus = document.getElementById('index-status');
const documentCount = document.getElementById('document-count');
const embeddingDimension = document.getElementById('embedding-dimension');
const qualityTopScore = document.getElementById('quality-top-score');
const qualityAvgScore = document.getElementById('quality-avg-score');
const qualityConfidence = document.getElementById('quality-confidence');
const resultsSummary = document.getElementById('results-summary');
const resultsState = document.getElementById('results-state');
const resultsList = document.getElementById('results-list');
const answerState = document.getElementById('answer-state');
const answerText = document.getElementById('answer-text');
const processSummary = document.getElementById('process-summary');
const processSteps = {
  health: document.getElementById('process-step-health'),
  stats: document.getElementById('process-step-stats'),
  ingest: document.getElementById('process-step-ingest'),
  answer: document.getElementById('process-step-answer'),
  llm: document.getElementById('process-step-llm'),
  render: document.getElementById('process-step-render'),
};

let apiReady = false;
let indexReady = false;

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
  rebuildIndexButton.disabled = isLoading;
  checkHealthButton.disabled = isLoading;
  submitButton.textContent = isLoading ? 'Retrieving...' : 'Retrieve';
}

function setStepAvailability() {
  sourceModeSelect.disabled = !apiReady;
  documentCountInput.disabled = !apiReady;
  rebuildIndexButton.disabled = !apiReady;
  queryInput.disabled = !apiReady || !indexReady;
  submitButton.disabled = !apiReady || !indexReady;
}

function setProcessStep(stepKey, state) {
  const node = processSteps[stepKey];
  if (!node) {
    return;
  }
  node.className = `process-step process-${state}`;
}

function resetProcessFlow() {
  Object.keys(processSteps).forEach((stepKey) => setProcessStep(stepKey, 'idle'));
  processSummary.textContent = 'Idle';
}

function markProcessError(stepKey, message) {
  setProcessStep(stepKey, 'error');
  processSummary.textContent = message;
}

function setQuality(topScore, avgScore, label, className) {
  qualityTopScore.textContent = typeof topScore === 'number' ? topScore.toFixed(3) : '-';
  qualityAvgScore.textContent = typeof avgScore === 'number' ? avgScore.toFixed(3) : '-';
  qualityConfidence.textContent = label;
  qualityConfidence.className = `quality-pill ${className}`;
}

function updateQualityFromDocuments(documents) {
  const scores = documents
    .map((document) => document.score)
    .filter((score) => typeof score === 'number');

  if (!scores.length) {
    setQuality(null, null, 'Unknown', 'quality-unknown');
    return;
  }

  const topScore = Math.max(...scores);
  const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;

  if (topScore >= 0.75) {
    setQuality(topScore, avgScore, 'High', 'quality-high');
  } else if (topScore >= 0.45) {
    setQuality(topScore, avgScore, 'Medium', 'quality-medium');
  } else {
    setQuality(topScore, avgScore, 'Low', 'quality-low');
  }
}

function getIngestConfig() {
  const sourceMode = sourceModeSelect?.value === 'huggingface' ? 'huggingface' : 'sample';
  const useSample = sourceMode === 'sample';
  const requestedCount = Number.parseInt(documentCountInput?.value ?? '10', 10);
  const safeCount = Number.isFinite(requestedCount) ? Math.min(Math.max(requestedCount, 1), 500) : 10;

  return {
    sourceMode,
    useSample,
    safeCount,
  };
}

function clearResults(message = 'Retrieval results will appear here after a successful query.') {
  resultsList.innerHTML = '';
  resultsState.textContent = message;
  resultsState.style.display = 'block';
  resultsSummary.textContent = 'Submit a query to inspect the generated answer and supporting documents.';
  embeddingDimension.textContent = '-';
  setQuality(null, null, 'Unknown', 'quality-unknown');
  answerText.textContent = '';
  answerText.style.display = 'none';
  answerState.textContent = 'The generated answer will appear here after a successful query.';
  answerState.style.display = 'block';
}

function renderResults(payload) {
  const documents = payload.documents || [];
  embeddingDimension.textContent = String(payload.query_embedding_dimension ?? '-');
  resultsSummary.textContent = `Showing ${documents.length} retrieved document${documents.length === 1 ? '' : 's'} for "${payload.query}".`;
  updateQualityFromDocuments(documents);
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
    const textId = `result-text-${index}`;
    const metaId = `result-meta-${index}`;

    return `
      <article class="result-card">
        <header>
          <p class="result-rank">Result ${index + 1}</p>
          <p class="result-score">Score ${score}</p>
        </header>
        <p id="${textId}" class="result-text result-text-collapsed">${escapedText}</p>
        <div class="result-controls">
          <button type="button" class="link-toggle text-toggle" data-target="${textId}" aria-expanded="false">Show more</button>
          <button type="button" class="link-toggle meta-toggle" data-target="${metaId}" aria-expanded="false">View metadata</button>
        </div>
        <pre id="${metaId}" class="result-meta result-meta-collapsed">${escapedMetadata}</pre>
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
    setProcessStep('health', 'running');
    processSummary.textContent = 'Checking API health...';
    const [healthResponse, statsResponse] = await Promise.all([
      fetch('/health'),
      fetch('/vector-db/stats'),
    ]);

    if (!healthResponse.ok) {
      throw new Error(`Health check failed with status ${healthResponse.status}.`);
    }

    setProcessStep('health', 'done');
    setProcessStep('stats', 'done');
    setProcessStep('ingest', 'skipped');
    setProcessStep('answer', 'skipped');
    setProcessStep('llm', 'skipped');
    setProcessStep('render', 'skipped');
    processSummary.textContent = 'API ready. Waiting for query.';

    const statsPayload = await parseResponse(statsResponse);
    setHealth('API Ready', 'pill-success');
    apiReady = true;

    if (statsResponse.ok && statsPayload) {
      indexStatus.textContent = statsPayload.index_ready ? 'Ready' : 'Not initialized';
      documentCount.textContent = String(statsPayload.document_count ?? 0);
      indexReady = Boolean(statsPayload.index_ready);
    } else {
      indexStatus.textContent = 'Unknown';
      documentCount.textContent = '-';
      indexReady = false;
    }

    healthStatusText.textContent = 'Backend reachable. Continue to Step 2 to build or verify the index.';
    setStepAvailability();
  } catch (error) {
    markProcessError('health', 'Health check failed');
    setHealth('API Error', 'pill-error');
    apiReady = false;
    indexReady = false;
    indexStatus.textContent = 'Unavailable';
    documentCount.textContent = '-';
    healthStatusText.textContent = 'Backend unreachable. Start the API, then click Check API health.';
    setStepAvailability();
    showBanner(
      'The API server is not responding correctly. Start or restart it with: uv run uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 --reload',
      'error',
    );
  }
}

async function ensureIndexReady(forceRebuild = false) {
  setProcessStep('stats', 'running');
  processSummary.textContent = 'Checking index status...';
  const statsResponse = await fetch('/vector-db/stats');
  const statsPayload = await parseResponse(statsResponse);

  if (!statsResponse.ok) {
    markProcessError('stats', 'Index status check failed');
    throw new Error('Unable to read vector DB status from the backend.');
  }

  setProcessStep('stats', 'done');

  indexStatus.textContent = statsPayload.index_ready ? 'Ready' : 'Not initialized';
  documentCount.textContent = String(statsPayload.document_count ?? 0);
  indexReady = Boolean(statsPayload.index_ready);
  setStepAvailability();

  if (statsPayload.index_ready && !forceRebuild) {
    setProcessStep('ingest', 'skipped');
    return;
  }

  const { sourceMode, useSample, safeCount } = getIngestConfig();

  const actionText = forceRebuild ? 'Rebuilding' : 'Attempting to ingest';
  showBanner(`Vector index ${statsPayload.index_ready ? 'already exists' : 'not initialized'}. ${actionText} ${sourceMode} documents now.`, 'info');
  setProcessStep('ingest', 'running');
  processSummary.textContent = 'Ingesting documents...';

  const ingestResponse = await fetch(`/ingest?document_count=${safeCount}&use_sample=${useSample}`, { method: 'POST' });
  const ingestPayload = await parseResponse(ingestResponse);

  if (!ingestResponse.ok) {
    markProcessError('ingest', 'Ingestion failed');
    const details = typeof ingestPayload === 'string' ? ingestPayload : JSON.stringify(ingestPayload);
    throw new Error(
      `The backend could not initialize the vector index. Azure authentication may be missing. Run: azd auth login --scope api://ailab/Model.Access. Details: ${details.slice(0, 220)}`,
    );
  }

  setProcessStep('ingest', 'done');

  indexStatus.textContent = ingestPayload.index_ready ? 'Ready' : 'Not initialized';
  documentCount.textContent = String(ingestPayload.documents_ingested ?? '-');
  indexReady = Boolean(ingestPayload.index_ready);
  setStepAvailability();
  const loadedSource = ingestPayload?.source || sourceMode;
  showBanner(`Vector index initialized successfully using ${loadedSource} documents.`, 'success');
}

async function runQuery(query) {
  setProcessStep('answer', 'running');
  setProcessStep('llm', 'running');
  processSummary.textContent = 'Calling /rag/answer...';
  const response = await fetch('/rag/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: 3 }),
  });

  const payload = await parseResponse(response);

  if (response.status === 404) {
    markProcessError('answer', 'RAG endpoint missing');
    setProcessStep('llm', 'error');
    throw new Error('The /rag/answer endpoint is missing on the running server. Restart FastAPI so it picks up the Sprint 4 backend changes.');
  }

  if (response.status === 400 && payload && typeof payload === 'object' && String(payload.detail || '').includes('Call /ingest first')) {
    setProcessStep('answer', 'skipped');
    setProcessStep('llm', 'skipped');
    await ensureIndexReady();
    return runQuery(query);
  }

  if (!response.ok) {
    markProcessError('answer', 'Query request failed');
    setProcessStep('llm', 'error');
    const details = typeof payload === 'string' ? payload : JSON.stringify(payload);
    throw new Error(`Query request failed with status ${response.status}. ${details}`);
  }

  setProcessStep('answer', 'done');
  setProcessStep('llm', 'done');
  processSummary.textContent = 'Response received. Rendering results...';

  return payload;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();

  resetProcessFlow();

  if (!query) {
    showBanner('Enter a query before submitting.', 'error');
    clearResults('A query is required before retrieval can run.');
    return;
  }

  hideBanner();
  setLoading(true);
  clearResults('Running retrieval...');

  try {
    await ensureIndexReady(false);
    const payload = await runQuery(query);
    setProcessStep('render', 'running');
    renderResults(payload);
    setProcessStep('render', 'done');
    processSummary.textContent = 'Completed';
    showBanner('Retrieved documents successfully.', 'success');
  } catch (error) {
    if (processSteps.render.className.includes('process-running')) {
      markProcessError('render', 'Render failed');
    }
    clearResults('No results were rendered because the request did not complete successfully.');
    showBanner(error.message, 'error');
  } finally {
    setLoading(false);
  }
});

rebuildIndexButton.addEventListener('click', async () => {
  resetProcessFlow();
  hideBanner();
  setLoading(true);
  clearResults('Rebuilding vector index with your selected source...');

  try {
    await ensureIndexReady(true);
    await refreshBackendStatus();
    processSummary.textContent = 'Index rebuilt and API ready.';
  } catch (error) {
    showBanner(error.message, 'error');
  } finally {
    setLoading(false);
  }
});

checkHealthButton.addEventListener('click', async () => {
  hideBanner();
  setLoading(true);
  resetProcessFlow();

  try {
    await refreshBackendStatus();
  } finally {
    setLoading(false);
  }
});

resultsList.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  const isTextToggle = target.classList.contains('text-toggle');
  const isMetaToggle = target.classList.contains('meta-toggle');
  if (!isTextToggle && !isMetaToggle) {
    return;
  }

  const targetId = target.getAttribute('data-target');
  if (!targetId) {
    return;
  }

  const contentNode = document.getElementById(targetId);
  if (!contentNode) {
    return;
  }

  if (isTextToggle) {
    const collapsed = contentNode.classList.toggle('result-text-collapsed');
    target.textContent = collapsed ? 'Show more' : 'Show less';
    target.setAttribute('aria-expanded', String(!collapsed));
  }

  if (isMetaToggle) {
    const collapsed = contentNode.classList.toggle('result-meta-collapsed');
    target.textContent = collapsed ? 'View metadata' : 'Hide metadata';
    target.setAttribute('aria-expanded', String(!collapsed));
  }
});

clearResults();
resetProcessFlow();
setStepAvailability();
refreshBackendStatus();
