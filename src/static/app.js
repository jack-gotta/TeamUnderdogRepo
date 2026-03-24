const form = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const submitButton = document.getElementById('submit-button');
const checkHealthButton = document.getElementById('check-health-button');
const rebuildIndexButton = document.getElementById('rebuild-index-button');
const toggleIndexCardButton = document.getElementById('toggle-index-card-button');
const stepIndexCard = document.getElementById('step-index-card');
const setupSummary = document.getElementById('setup-summary');
const sourceModeSelect = document.getElementById('source-mode');
const documentCountInput = document.getElementById('document-count-input');
const topKInput = document.getElementById('top-k-input');
const answerThresholdSelect = document.getElementById('answer-threshold-select');
const sourceGuidance = document.getElementById('source-guidance');
const banner = document.getElementById('message-banner');
const indexStatus = document.getElementById('index-status');
const documentCount = document.getElementById('document-count');
const embeddingDimension = document.getElementById('embedding-dimension');
const qualityTopScore = document.getElementById('quality-top-score');
const qualityAvgScore = document.getElementById('quality-avg-score');
const qualityConfidence = document.getElementById('quality-confidence');
const groundingAlert = document.getElementById('grounding-alert');
const resultsSummary = document.getElementById('results-summary');
const resultsState = document.getElementById('results-state');
const resultsList = document.getElementById('results-list');
const answerState = document.getElementById('answer-state');
const answerText = document.getElementById('answer-text');
const promptState = document.getElementById('prompt-state');
const promptText = document.getElementById('prompt-text');
const processSummary = document.getElementById('process-summary');
const processRuntimeLabel = document.getElementById('process-runtime-label');
const processTotalTime = document.getElementById('process-total-time');
const resultTabs = Array.from(document.querySelectorAll('.result-tab'));
const tabPanels = {
  answer: document.getElementById('tab-answer'),
  evidence: document.getElementById('tab-evidence'),
  prompt: document.getElementById('tab-prompt'),
};
const processSteps = {
  stats: document.getElementById('process-step-stats'),
  ingest: document.getElementById('process-step-ingest'),
  answer: document.getElementById('process-step-answer'),
  llm: document.getElementById('process-step-llm'),
  render: document.getElementById('process-step-render'),
};

const QUALITY_BANDS = {
  high: 0.75,
  medium: 0.45,
  weak: 0.22,
};

let apiReady = false;
let indexReady = false;
let setupCollapsed = false;
let lastIndexedConfigSignature = null;
let lastIndexedConfigDetails = null;
let lastRetrievedQuery = '';
let isLoading = false;
let activeRunStartedAt = null;

function setHealth(text, className) {
  checkHealthButton.textContent = text;
  checkHealthButton.className = `pill ${className}`;
}

function showBanner(message, kind) {
  banner.textContent = message;
  banner.className = `message-banner message-${kind}`;
}

function hideBanner() {
  banner.textContent = '';
  banner.className = 'message-banner message-hidden';
}

function setLoading(loadingState) {
  isLoading = Boolean(loadingState);
  checkHealthButton.disabled = isLoading;
  submitButton.textContent = isLoading ? 'Retrieving...' : 'Retrieve';
  setStepAvailability();
}

function setStepAvailability() {
  sourceModeSelect.disabled = !apiReady || isLoading;
  documentCountInput.disabled = !apiReady || isLoading;
  topKInput.disabled = !apiReady || !indexReady || isLoading;
  answerThresholdSelect.disabled = !apiReady || isLoading;
  rebuildIndexButton.disabled = !apiReady || isLoading;
  queryInput.disabled = !apiReady || !indexReady || isLoading;
  toggleIndexCardButton.disabled = isLoading;
  updateRebuildButtonState();
  updateRetrieveButtonState();
  updateSourceGuidance();
  updateProcessEstimates();
  syncSetupCardState();
}

function getCurrentConfigSignature() {
  const { sourceMode, safeCount } = getIngestConfig();
  return `${sourceMode}:${safeCount}`;
}

function getIngestConfig() {
  const sourceMode = sourceModeSelect.value === 'huggingface' ? 'huggingface' : 'sample';
  const useSample = sourceMode === 'sample';
  const requestedCount = Number.parseInt(documentCountInput.value || '10', 10);
  const safeCount = Number.isFinite(requestedCount) ? Math.min(Math.max(requestedCount, 1), 3200) : 10;

  return {
    sourceMode,
    useSample,
    safeCount,
  };
}

function getQueryConfig() {
  const requestedTopK = Number.parseInt(topKInput.value || '3', 10);
  const safeTopK = Number.isFinite(requestedTopK) ? Math.min(Math.max(requestedTopK, 1), 8) : 3;
  const answerThreshold = Number.parseFloat(answerThresholdSelect.value || '0.22');

  return {
    safeTopK,
    answerThreshold: Number.isFinite(answerThreshold) ? answerThreshold : 0.22,
  };
}

function getIndexedSummaryText() {
  if (!indexReady || !lastIndexedConfigDetails) {
    return 'Configure source and size before building the index.';
  }

  const sourceText = lastIndexedConfigDetails.sourceMode === 'huggingface' ? 'HuggingFace passages' : 'sample documents';
  const hasPendingChanges = getCurrentConfigSignature() !== lastIndexedConfigSignature;
  const baseText = `Index ready: ${sourceText}, ${lastIndexedConfigDetails.safeCount} docs.`;

  return hasPendingChanges ? `${baseText} Current controls changed, so rebuild to apply them.` : baseText;
}

function syncSetupCardState() {
  const canCollapse = indexReady && Boolean(lastIndexedConfigDetails);
  const needsRebuild = getCurrentConfigSignature() !== lastIndexedConfigSignature;

  if (!canCollapse || needsRebuild) {
    setupCollapsed = false;
  }

  setupSummary.textContent = getIndexedSummaryText();
  toggleIndexCardButton.hidden = !canCollapse;
  toggleIndexCardButton.textContent = setupCollapsed ? 'Edit setup' : 'Collapse setup';
  stepIndexCard.classList.toggle('step-card-collapsed', canCollapse && setupCollapsed);
}

function updateRetrieveButtonState() {
  const query = queryInput.value.trim();
  const currentSignature = getCurrentConfigSignature();
  const needsRebuild = !indexReady || currentSignature !== lastIndexedConfigSignature;
  const alreadyRetrieved = query.length > 0 && query === lastRetrievedQuery;
  const canRetrieve = apiReady && indexReady && !needsRebuild && query.length > 0 && !alreadyRetrieved && !isLoading;

  submitButton.disabled = !canRetrieve;
  submitButton.classList.toggle('button-attention', canRetrieve);
  submitButton.classList.toggle('button-muted', !canRetrieve);
}

function updateProcessEstimates() {
  if (activeRunStartedAt !== null) {
    return;
  }

  const { sourceMode, safeCount } = getIngestConfig();
  processRuntimeLabel.textContent = 'Estimated total time';

  if (sourceMode === 'huggingface') {
    // ~0.35-0.6s per document for embedding API calls
    const ingestMin = Math.round(safeCount * 0.35);
    const ingestMax = Math.round(safeCount * 0.6);
    processTotalTime.textContent = `~${ingestMin + 3}-${ingestMax + 13}s`;
    return;
  }

  processTotalTime.textContent = '~9-27s';
}

function startRunTimer() {
  activeRunStartedAt = performance.now();
  processRuntimeLabel.textContent = 'Run in progress';
  processTotalTime.textContent = '...';
}

function finishRunTimer(statusLabel) {
  if (activeRunStartedAt === null) {
    updateProcessEstimates();
    return;
  }

  const elapsedSeconds = (performance.now() - activeRunStartedAt) / 1000;
  processRuntimeLabel.textContent = statusLabel;
  processTotalTime.textContent = formatDuration(elapsedSeconds);
  activeRunStartedAt = null;
}

function formatDuration(totalSeconds) {
  if (totalSeconds < 10) {
    return `${totalSeconds.toFixed(1)}s`;
  }

  return `${Math.round(totalSeconds)}s`;
}

function updateSourceGuidance() {
  const { sourceMode, safeCount } = getIngestConfig();

  if (sourceMode === 'huggingface') {
    if (safeCount < 300) {
      sourceGuidance.textContent = `Using HuggingFace source: the corpus has 3,200 passages total. Use 3200 for best accuracy — fewer documents miss topics in the 918 test questions. At least 300 recommended.`;
      sourceGuidance.className = 'source-guidance source-guidance-warning';
      return;
    }

    if (safeCount < 3200) {
      sourceGuidance.textContent = `Using HuggingFace source: ${safeCount} of 3,200 passages loaded. Set to 3200 for complete coverage and best evaluation accuracy.`;
      sourceGuidance.className = 'source-guidance';
      return;
    }

    sourceGuidance.textContent = 'Using HuggingFace source: all 3,200 passages loaded — best possible retrieval accuracy against the 918 test questions.';
    sourceGuidance.className = 'source-guidance source-guidance-success';
    return;
  }

  sourceGuidance.textContent = 'Using Sample source: 10 hand-picked documents for quick smoke-testing. Switch to HuggingFace with 3200 documents for real evaluation.';
  sourceGuidance.className = 'source-guidance';
}

function updateRebuildButtonState() {
  if (!apiReady || isLoading) {
    rebuildIndexButton.classList.remove('secondary-button-attention');
    return;
  }

  const needsRebuild = !indexReady || getCurrentConfigSignature() !== lastIndexedConfigSignature;
  rebuildIndexButton.classList.toggle('secondary-button-attention', needsRebuild);
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

function assessRetrievalQuality(documents) {
  const scores = documents
    .map((document) => document.score)
    .filter((score) => typeof score === 'number');

  if (!scores.length) {
    return {
      state: 'unknown',
      topScore: null,
      avgScore: null,
      label: 'Unknown',
      className: 'quality-unknown',
    };
  }

  const topScore = Math.max(...scores);
  const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;

  if (topScore >= QUALITY_BANDS.high) {
    return { state: 'high', topScore, avgScore, label: 'High', className: 'quality-high' };
  }

  if (topScore >= QUALITY_BANDS.medium) {
    return { state: 'medium', topScore, avgScore, label: 'Medium', className: 'quality-medium' };
  }

  if (topScore >= QUALITY_BANDS.weak) {
    return { state: 'weak', topScore, avgScore, label: 'Weak', className: 'quality-weak' };
  }

  return { state: 'ungrounded', topScore, avgScore, label: 'Ungrounded', className: 'quality-low' };
}

function updateQualityFromDocuments(documents) {
  const quality = assessRetrievalQuality(documents);
  setQuality(quality.topScore, quality.avgScore, quality.label, quality.className);
  return quality;
}

function setGroundingAlert(message, level) {
  groundingAlert.innerHTML = message;
  groundingAlert.className = `grounding-alert grounding-alert-${level}`;
}

function hideGroundingAlert() {
  groundingAlert.innerHTML = '';
  groundingAlert.className = 'grounding-alert grounding-alert-hidden';
}

function clearResults(message = 'Retrieval results will appear here after a successful query.') {
  resultsList.innerHTML = '';
  resultsState.textContent = message;
  resultsState.style.display = 'block';
  resultsSummary.textContent = 'Submit a query to inspect the generated answer and supporting documents.';
  embeddingDimension.textContent = '-';
  setQuality(null, null, 'Unknown', 'quality-unknown');
  hideGroundingAlert();
  answerText.textContent = '';
  answerText.style.display = 'none';
  answerState.textContent = 'The generated answer will appear here after a successful query.';
  answerState.style.display = 'block';
  promptText.textContent = '';
  promptText.style.display = 'none';
  promptState.textContent = 'The final prompt will appear here after a successful query.';
  promptState.style.display = 'block';
  setActiveTab('answer');
}

function setActiveTab(tabName) {
  resultTabs.forEach((tabButton) => {
    const isActive = tabButton.dataset.tab === tabName;
    tabButton.classList.toggle('result-tab-active', isActive);
    tabButton.setAttribute('aria-selected', String(isActive));
  });

  Object.entries(tabPanels).forEach(([name, panel]) => {
    const isActive = name === tabName;
    panel.classList.toggle('tab-panel-active', isActive);
    panel.hidden = !isActive;
  });
}

function renderAnswer(payload, quality) {
  const { answerThreshold } = getQueryConfig();
  const query = repairTextEncoding(payload.query || '');
  const topScore = typeof quality.topScore === 'number' ? quality.topScore : null;
  const shouldHideAnswer = topScore !== null && topScore < answerThreshold;
  const hasDocuments = (payload.documents || []).length > 0;

  if (!hasDocuments) {
    answerState.textContent = 'No documents were retrieved, so there is no grounded answer to show.';
    answerState.style.display = 'block';
    answerText.style.display = 'none';
    setGroundingAlert(
      'No supporting passages were retrieved. Rebuild the index, increase the document count, or try a more specific question.',
      'error',
    );
    return { answerVisible: false, recommendedTab: 'evidence' };
  }

  if (quality.state === 'ungrounded') {
    setGroundingAlert(
      `No grounded answer found. The top retrieval score was ${topScore.toFixed(3)}, which is too low to treat the model output as evidence-backed. Try a different phrasing, raise the document count, or inspect the Evidence tab.`,
      'error',
    );
  } else if (quality.state === 'weak') {
    setGroundingAlert(
      `Weak grounding only. The top retrieval score was ${topScore.toFixed(3)}, so treat the answer as tentative and verify it against the evidence below.`,
      'warning',
    );
  } else {
    setGroundingAlert(
      `Evidence looks usable for "${escapeHtml(query)}". Review the top documents and prompt if you want to verify how the answer was grounded.`,
      'success',
    );
  }

  if (shouldHideAnswer) {
    answerState.innerHTML = `The model answer is hidden because retrieval confidence was below your policy threshold (${answerThreshold.toFixed(2)}).<br>Switch the answer policy to <strong>Always show model answer</strong> if you want to inspect the raw output anyway.`;
    answerState.style.display = 'block';
    answerText.style.display = 'none';
    return { answerVisible: false, recommendedTab: 'evidence' };
  }

  answerState.style.display = 'none';
  answerText.textContent = repairTextEncoding(payload.answer || 'No generated answer was returned.');
  answerText.style.display = 'block';
  return { answerVisible: true, recommendedTab: 'answer' };
}

function renderPrompt(payload) {
  const prompt = repairTextEncoding(payload.prompt || '');

  if (!prompt) {
    promptState.textContent = 'The backend did not return an augmented prompt for this query.';
    promptState.style.display = 'block';
    promptText.style.display = 'none';
    return;
  }

  promptState.style.display = 'none';
  promptText.textContent = prompt;
  promptText.style.display = 'block';
}

function getDocumentHeading(metadata, index) {
  if (typeof metadata.title === 'string' && metadata.title.trim()) {
    return repairTextEncoding(metadata.title.trim());
  }

  if (typeof metadata.id === 'string' || typeof metadata.id === 'number') {
    return `Document ${metadata.id}`;
  }

  return `Result ${index + 1}`;
}

function getDocumentMetadataChips(metadata) {
  const chips = [];

  if (metadata.source) {
    chips.push(repairTextEncoding(String(metadata.source)));
  }

  if (metadata.dataset) {
    chips.push(repairTextEncoding(String(metadata.dataset)));
  }

  if (metadata.dataset_index !== undefined && metadata.dataset_index !== null) {
    chips.push(`row ${metadata.dataset_index}`);
  }

  return chips;
}

function renderResults(payload) {
  const documents = Array.isArray(payload.documents) ? payload.documents : [];
  const repairedQuery = repairTextEncoding(payload.query || '');
  const { safeTopK } = getQueryConfig();

  embeddingDimension.textContent = String(payload.query_embedding_dimension ?? '-');
  resultsSummary.textContent = `Showing ${documents.length} retrieved document${documents.length === 1 ? '' : 's'} for "${repairedQuery}" with top K set to ${safeTopK}.`;

  const quality = updateQualityFromDocuments(documents);
  const answerResult = renderAnswer(payload, quality);
  renderPrompt(payload);

  if (!documents.length) {
    resultsList.innerHTML = '';
    resultsState.textContent = 'The backend returned no retrieved documents for this query.';
    resultsState.style.display = 'block';
    setActiveTab(answerResult.recommendedTab);
    return;
  }

  resultsState.style.display = 'none';
  resultsList.innerHTML = documents.map((document, index) => {
    const metadata = sanitizeValue(document.metadata || {});
    const heading = getDocumentHeading(metadata, index);
    const chips = getDocumentMetadataChips(metadata);
    const score = typeof document.score === 'number' ? document.score.toFixed(3) : 'n/a';
    const metadataJson = JSON.stringify(metadata, null, 2);
    const textId = `result-text-${index}`;
    const metaId = `result-meta-${index}`;
    const escapedText = escapeHtml(repairTextEncoding(document.text || ''));
    const escapedHeading = escapeHtml(heading);
    const escapedMetadata = escapeHtml(metadataJson);
    const chipMarkup = chips
      .map((chip) => `<span class="result-chip">${escapeHtml(chip)}</span>`)
      .join('');

    return `
      <article class="result-card">
        <header class="result-header">
          <div>
            <p class="result-rank">Result ${index + 1}</p>
            <h3 class="result-title">${escapedHeading}</h3>
            <div class="result-chip-row">${chipMarkup}</div>
          </div>
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

  setActiveTab(answerResult.recommendedTab);
}

function sanitizeValue(value) {
  if (typeof value === 'string') {
    return repairTextEncoding(value);
  }

  if (Array.isArray(value)) {
    return value.map((item) => sanitizeValue(item));
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, nestedValue]) => [key, sanitizeValue(nestedValue)]),
    );
  }

  return value;
}

function repairTextEncoding(value) {
  if (typeof value !== 'string') {
    return '';
  }

  const looksBroken = /[Ãâð�]/.test(value);
  if (!looksBroken) {
    return value;
  }

  try {
    const repaired = decodeURIComponent(escape(value));
    if (countEncodingArtifacts(repaired) < countEncodingArtifacts(value)) {
      return repaired;
    }
  } catch (error) {
    // Fall back to the original value if legacy decoding fails.
  }

  return value;
}

function countEncodingArtifacts(value) {
  const matches = value.match(/[Ãâð�]/g);
  return matches ? matches.length : 0;
}

function escapeHtml(value) {
  return String(value)
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
  } catch (error) {
    return text;
  }
}

function rememberIndexedConfig() {
  const currentConfig = getIngestConfig();
  lastIndexedConfigSignature = getCurrentConfigSignature();
  lastIndexedConfigDetails = currentConfig;
}

async function refreshBackendStatus() {
  try {
    processSummary.textContent = 'Checking API health...';
    const [healthResponse, statsResponse] = await Promise.all([
      fetch('/health'),
      fetch('/vector-db/stats'),
    ]);

    if (!healthResponse.ok) {
      throw new Error(`Health check failed with status ${healthResponse.status}.`);
    }

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

      if (indexReady && lastIndexedConfigSignature === null) {
        rememberIndexedConfig();
        setupCollapsed = true;
      }
    } else {
      indexStatus.textContent = 'Unknown';
      documentCount.textContent = '-';
      indexReady = false;
      lastIndexedConfigSignature = null;
      lastIndexedConfigDetails = null;
    }

    setStepAvailability();
  } catch (error) {
    processSummary.textContent = 'Health check failed';
    setHealth('API Error', 'pill-error');
    apiReady = false;
    indexReady = false;
    lastIndexedConfigSignature = null;
    lastIndexedConfigDetails = null;
    indexStatus.textContent = 'Unavailable';
    documentCount.textContent = '-';
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

  if (indexReady && lastIndexedConfigSignature === null) {
    rememberIndexedConfig();
  }

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

  if (indexReady) {
    rememberIndexedConfig();
    lastRetrievedQuery = '';
  }

  setStepAvailability();
  const loadedSource = ingestPayload && ingestPayload.source ? ingestPayload.source : sourceMode;
  showBanner(`Vector index initialized successfully using ${loadedSource} documents.`, 'success');
}

async function runQuery(query) {
  const { safeTopK } = getQueryConfig();

  setProcessStep('answer', 'running');
  setProcessStep('llm', 'running');
  processSummary.textContent = 'Calling /rag/answer...';

  const response = await fetch('/rag/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: safeTopK }),
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
  startRunTimer();
  clearResults('Running retrieval...');

  try {
    await ensureIndexReady(false);
    const payload = await runQuery(query);
    setProcessStep('render', 'running');
    renderResults(payload);
    setProcessStep('render', 'done');
    processSummary.textContent = 'Completed';
    showBanner('Retrieved documents successfully.', 'success');
    lastRetrievedQuery = query;
    updateRetrieveButtonState();
    finishRunTimer('Actual run time');
  } catch (error) {
    if (processSteps.render.className.includes('process-running')) {
      markProcessError('render', 'Render failed');
    }

    clearResults('No results were rendered because the request did not complete successfully.');
    showBanner(error.message, 'error');
    finishRunTimer('Elapsed before failure');
  } finally {
    setLoading(false);
  }
});

rebuildIndexButton.addEventListener('click', async () => {
  resetProcessFlow();
  hideBanner();
  setLoading(true);
  startRunTimer();
  clearResults('Rebuilding vector index with your selected source...');

  try {
    await ensureIndexReady(true);
    await refreshBackendStatus();
    lastRetrievedQuery = '';
    updateRetrieveButtonState();
    setupCollapsed = true;
    syncSetupCardState();
    processSummary.textContent = 'Index rebuilt and API ready.';
    finishRunTimer('Index build time');
  } catch (error) {
    showBanner(error.message, 'error');
    finishRunTimer('Elapsed before failure');
  } finally {
    setLoading(false);
  }
});

toggleIndexCardButton.addEventListener('click', () => {
  setupCollapsed = !setupCollapsed;
  syncSetupCardState();
});

sourceModeSelect.addEventListener('change', () => {
  lastRetrievedQuery = '';
  updateRebuildButtonState();
  updateRetrieveButtonState();
  updateSourceGuidance();
  updateProcessEstimates();
  syncSetupCardState();
});

documentCountInput.addEventListener('input', () => {
  lastRetrievedQuery = '';
  updateRebuildButtonState();
  updateRetrieveButtonState();
  updateSourceGuidance();
  updateProcessEstimates();
  syncSetupCardState();
});

topKInput.addEventListener('input', () => {
  lastRetrievedQuery = '';
  updateRetrieveButtonState();
});

answerThresholdSelect.addEventListener('change', () => {
  lastRetrievedQuery = '';
});

queryInput.addEventListener('input', () => {
  updateRetrieveButtonState();
});

checkHealthButton.addEventListener('click', async () => {
  hideBanner();
  setLoading(true);
  startRunTimer();
  resetProcessFlow();

  try {
    await refreshBackendStatus();
    finishRunTimer('Health check time');
  } finally {
    setLoading(false);
  }
});

document.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  if (target.classList.contains('sample-question-chip')) {
    queryInput.value = target.dataset.question || '';
    lastRetrievedQuery = '';
    updateRetrieveButtonState();
    queryInput.focus();
    return;
  }

  if (target.classList.contains('result-tab')) {
    setActiveTab(target.dataset.tab);
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
updateSourceGuidance();
updateProcessEstimates();
syncSetupCardState();
refreshBackendStatus();
