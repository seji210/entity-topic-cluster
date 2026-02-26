import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2/dist/transformers.min.js';
import { SAMPLE_DATA } from './sample-data.js';

// Configuration
env.allowLocalModels = false; // We use Hugging Face Hub directly via CDN

// UI Elements
const els = {
  btn: document.getElementById('generate-btn'),
  loadSampleBtn: document.getElementById('load-sample-btn'),
  input: document.getElementById('input-text'),
  inputContainer: document.getElementById('input-container'),
  tableContainer: document.getElementById('table-container'),
  tableBody: document.getElementById('table-body'),
  clearBtn: document.getElementById('clear-btn'),
  exportBtn: document.getElementById('export-btn'),
  threshold: document.getElementById('similarity-threshold'),
  thresholdDisplay: document.getElementById('threshold-display'),
  showPages: document.getElementById('show-pages'),
  viewMode: document.getElementById('view-mode'),
  statusLog: document.getElementById('status-log'),
  graphContainer: document.getElementById('graph-container'),
  placeholder: document.getElementById('graph-placeholder')
};

// --- Realtime Table UI Trick ---
function parseInputToTable() {
  const rawInput = els.input.value;
  if (!rawInput.trim()) return;

  // Smart parsing: if it detects 'title:' format
  if (/^title:/im.test(rawInput)) {
    const blocks = rawInput.split(/(?=\btitle:)/i).filter(b => b.trim().length > 0);
    
    let html = '';
    blocks.forEach(block => {
      const titleMatch = block.match(/title:\s*(.*)/i);
      const descMatch = block.match(/desc:\s*(.*)/i);
      
      const title = titleMatch ? titleMatch[1].trim() : '';
      const desc = descMatch ? descMatch[1].trim() : block.replace(/title:\s*.*\n?/i, '').trim();
      
      html += `
        <tr class="hover:bg-gray-50">
          <td class="px-3 py-2 text-gray-900 font-medium align-top">${title}</td>
          <td class="px-3 py-2 text-gray-500 align-top">${desc}</td>
        </tr>
      `;
    });
    
    if (html) {
      els.tableBody.innerHTML = html;
      els.input.classList.add('hidden');
      els.tableContainer.classList.remove('hidden');
      els.clearBtn.classList.remove('hidden');
    }
  }
}

// Listen for paste or input
els.input.addEventListener('input', () => {
  // Give it a tiny delay to allow paste to complete
  setTimeout(parseInputToTable, 50);
});

els.clearBtn.addEventListener('click', () => {
  els.input.classList.remove('hidden');
  els.tableContainer.classList.add('hidden');
  els.clearBtn.classList.add('hidden');
  els.input.focus();
});

// State
let ml = { embedder: null, ner: null };
let lastExportData = null;

// --- Logging Helper ---
function log(msg, type = 'info') {
  const d = document.createElement('div');
  d.textContent = `> ${msg}`;
  if (type === 'error') d.classList.add('text-red-500');
  if (type === 'success') d.classList.add('text-green-600', 'font-bold');
  els.statusLog.appendChild(d);
  els.statusLog.scrollTop = els.statusLog.scrollHeight;
  console.log(`[${type}] ${msg}`);
}

// --- Export Logic ---
els.exportBtn.addEventListener('click', () => {
  if (!lastExportData) return;
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastExportData, null, 2));
  const dlAnchorElem = document.createElement('a');
  dlAnchorElem.setAttribute("href", dataStr);
  dlAnchorElem.setAttribute("download", "knowledge-graph-export.json");
  dlAnchorElem.click();
});

// --- Initialize ML Models ---
async function loadModels() {
  if (ml.embedder && ml.ner) return; // Already loaded

  log("Downloading ML models... (this may take a few seconds on first run, cached after)");
  
  try {
    // 1. Text Embedder
    log("Loading all-MiniLM-L6-v2...");
    ml.embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    
    // 2. NER Model (Multilingual DistilBERT)
    log("Loading distilbert-base-multilingual-cased-ner-hrl...");
    ml.ner = await pipeline('token-classification', 'Xenova/distilbert-base-multilingual-cased-ner-hrl');
    
    log("ML Models loaded successfully.", "success");
  } catch (err) {
    log(`Error loading models: ${err.message}`, "error");
    throw err;
  }
}

// --- Process Input Data ---
async function processInput(texts) {
  const results = [];
  
  for (let i = 0; i < texts.length; i++) {
    const text = texts[i].trim();
    if (!text) continue;
    
    log(`Processing [${i+1}/${texts.length}]: "${text.substring(0, 30)}..."`);
    
    // Generate Embedding
    const embedResult = await ml.embedder(text, { pooling: 'mean', normalize: true });
    // embedResult.data is a Float32Array of length 384
    const embedding = Array.from(embedResult.data);
    
    // Help the 'cased' multilingual NER model by capitalizing first letters
    // if the user typed everything in lowercase.
    const nerText = text.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    
    // Extract Entities
    const nerResult = await ml.ner(nerText);
    let entities = processNER(nerResult);
    
    // Fallback: if NER found nothing, extract capitalized proper nouns via regex
    if (entities.length === 0) {
      entities = extractProperNouns(nerText);
    } else {
      // Supplement: merge in any proper nouns that NER missed
      const nerTexts = new Set(entities.map(e => e.text.toLowerCase()));
      const fallback = extractProperNouns(nerText);
      fallback.forEach(f => {
        if (!nerTexts.has(f.text.toLowerCase())) {
          entities.push(f);
        }
      });
    }
    
    results.push({
      id: `page_${i}`,
      text: text,
      embedding: embedding,
      entities: entities
    });
  }
  
  return results;
}

// --- Utility: Merge NER tokens ---
function processNER(rawEntities) {
  if (!rawEntities || rawEntities.length === 0) return [];
  
  const merged = [];
  let current = null;
  
  for (const token of rawEntities) {
    const isBeginning = token.entity.startsWith('B-');
    const isInside = token.entity.startsWith('I-');
    const label = token.entity.replace(/^[BI]-/, '');
    
    const word = token.word.replace(/^##/, ''); 
    const isSubword = token.word.startsWith('##');
    const isPunct = /^[-\/&']$/.test(word);
    
    if (isBeginning) {
      if (current) merged.push(current);
      current = { text: token.word, label: label, score: token.score };
    } else if (isInside && current && current.label === label) {
      if (isSubword) {
        current.text += word;
      } else if (isPunct) {
        current.text += word; // glue hyphens/punctuation directly (T-Mobile, AT&T)
      } else {
        current.text += ' ' + word;
      }
      current.score = Math.min(current.score, token.score);
    } else {
      if (current) merged.push(current);
      current = { text: token.word, label: label, score: token.score };
    }
  }
  if (current) merged.push(current);
  
  const uniqueEntities = {};
  for (const e of merged) {
    if (e.score < 0.3) continue;
    
    const cleanText = e.text.replace(/ ([.,?!])/g, '$1').trim();
    if (cleanText.length < 2) continue;
    
    if (!uniqueEntities[cleanText]) {
      uniqueEntities[cleanText] = { text: cleanText, label: e.label };
    }
  }
  
  return Object.values(uniqueEntities);
}

// --- Fallback: Regex-based proper noun extraction ---
// Catches capitalized multi-word names that the NER model misses
function extractProperNouns(text) {
  // Match sequences of 1-4 capitalized words, including hyphens (T-Mobile, AT&T)
  const pattern = /(?<!\.\s)(?:^|\s)((?:[A-Z][A-Za-z&'-]*(?:\s+(?:of|the|and|for|in|de|von)\s+)?)+[A-Z][A-Za-z&'-]*)/g;
  const matches = [];
  let m;
  
  while ((m = pattern.exec(text)) !== null) {
    const candidate = m[1].trim();
    // Filter out very short or very long matches, and sentence starters
    if (candidate.length >= 3 && candidate.length <= 40 && candidate.split(' ').length <= 5) {
      matches.push({ text: candidate, label: 'MISC' });
    }
  }
  
  return matches;
}

// --- Wikidata Resolution ---
const WIKIDATA_API = "https://www.wikidata.org/w/api.php";

async function resolveWikidata(entityName) {
  const params = new URLSearchParams({
    action: "wbsearchentities",
    search: entityName,
    language: "en",
    uselang: "en",
    limit: 1,
    format: "json",
    origin: "*"
  });

  try {
    const res = await fetch(`${WIKIDATA_API}?${params.toString()}`);
    const data = await res.json();
    
    if (data.search && data.search.length > 0) {
      const match = data.search[0];
      return {
        id: match.id,
        label: match.label || match.title,
        description: match.description,
        url: `https://www.wikidata.org/wiki/${match.id}`
      };
    }
  } catch (err) {
    console.warn(`Wikidata resolution failed for ${entityName}:`, err);
  }
  return null;
}

// --- Graph Construction Logic ---
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function buildGraphData(processedData, threshold, showPages) {
  const nodes = [];
  const edges = [];
  
  const entityCounts = new Map();
  const entityDetails = new Map();
  const entityCoOccurrences = new Map();

  // 1. Gather entity stats (counts and co-occurrences)
  processedData.forEach(item => {
    if (!item.resolvedEntities) return;
    
    const uniqueItemEntities = Array.from(new Set(item.resolvedEntities.map(e => e.id)))
      .map(id => item.resolvedEntities.find(e => e.id === id));

    uniqueItemEntities.forEach(wdEntity => {
      entityCounts.set(wdEntity.id, (entityCounts.get(wdEntity.id) || 0) + 1);
      if (!entityDetails.has(wdEntity.id)) {
        entityDetails.set(wdEntity.id, wdEntity);
      }
    });

    // Count co-occurrences within the same page
    for (let i = 0; i < uniqueItemEntities.length; i++) {
      for (let j = i + 1; j < uniqueItemEntities.length; j++) {
        const id1 = uniqueItemEntities[i].id;
        const id2 = uniqueItemEntities[j].id;
        const pairKey = [id1, id2].sort().join('--');
        entityCoOccurrences.set(pairKey, (entityCoOccurrences.get(pairKey) || 0) + 1);
      }
    }
  });

  // 2. Create Entity Nodes
  entityDetails.forEach((wdEntity, id) => {
    const count = entityCounts.get(id);
    nodes.push({
      id: wdEntity.id,
      label: wdEntity.label + (count > 1 ? ` (${count})` : ''),
      title: wdEntity.description || 'Wikidata Entity',
      group: 'entity',
      url: wdEntity.url,
      value: count * 5 // size based on frequency
    });
  });

  if (showPages) {
    // Show Page Nodes and Page->Entity Edges, plus Page->Page Similarity Edges
    processedData.forEach(item => {
      nodes.push({
        id: item.id,
        label: item.text.substring(0, 30) + (item.text.length > 30 ? '...' : ''),
        title: item.text,
        group: 'page',
        value: 5
      });
      
      if (item.resolvedEntities) {
        item.resolvedEntities.forEach(wdEntity => {
          edges.push({
            from: item.id,
            to: wdEntity.id,
            label: 'mentions',
            color: { color: '#d1d5db' },
            dashes: true
          });
        });
      }
    });
    
    for (let i = 0; i < processedData.length; i++) {
      for (let j = i + 1; j < processedData.length; j++) {
        const sim = cosineSimilarity(processedData[i].embedding, processedData[j].embedding);
        if (sim >= threshold) {
          edges.push({
            from: processedData[i].id,
            to: processedData[j].id,
            label: sim.toFixed(2),
            value: sim,
            color: { color: '#93c5fd' }
          });
        }
      }
    }
  } else {
    // Show Only Entity Nodes and Entity<->Entity Co-occurrence Edges
    entityCoOccurrences.forEach((count, pairKey) => {
      const [id1, id2] = pairKey.split('--');
      edges.push({
        from: id1,
        to: id2,
        label: `${count} shared pages`,
        value: count, // thickness based on co-occurrence
        color: { color: '#fcd34d' }
      });
    });
  }
  
  return { nodes, edges };
}

// --- 2D Graph Construction ---
function render2DGraph(processedData, threshold, showPages) {
  const graphData = buildGraphData(processedData, threshold, showPages);
  
  els.placeholder.style.display = 'none'; // Hide placeholder
  els.graphContainer.innerHTML = ''; // clear previous plotly/vis
  
  // Initialize vis-network
  const container = els.graphContainer;
  const data = {
    nodes: new vis.DataSet(graphData.nodes),
    edges: new vis.DataSet(graphData.edges)
  };
  
  const options = {
    groups: {
      page: {
        shape: 'box',
        color: { background: '#3b82f6', border: '#2563eb' },
        font: { color: 'white' },
        margin: 10
      },
      entity: {
        shape: 'ellipse',
        color: { background: '#f59e0b', border: '#d97706' },
        font: { color: 'white' }
      }
    },
    edges: {
      font: { size: 10, align: 'middle' },
      smooth: { type: 'continuous' }
    },
    physics: {
      stabilization: { iterations: 100 },
      barnesHut: { springLength: 200 }
    },
    interaction: {
      hover: true,
      tooltipDelay: 200
    }
  };
  
  const network = new vis.Network(container, data, options);
  
  // Handle clicks (e.g. open Wikidata URL)
  network.on("doubleClick", function (params) {
    if (params.nodes.length > 0) {
      const nodeId = params.nodes[0];
      const node = data.nodes.get(nodeId);
      if (node && node.url) {
        window.open(node.url, '_blank');
      }
    }
  });

  log(`Graph generated with ${graphData.nodes.length} nodes and ${graphData.edges.length} edges.`, "success");
}

// --- 3D t-SNE Map Construction ---
async function render3DMap(processedData, showPages) {
  els.placeholder.style.display = 'none';
  els.graphContainer.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-blue-600 font-medium">Computing 3D layout (t-SNE). This may take a few seconds...</div>';
  
  // Yield to browser to render the loading message
  await new Promise(r => setTimeout(r, 100));

  const items = [];
  
  // 1. Gather Pages
  if (showPages) {
    processedData.forEach(p => {
      items.push({
        id: p.id,
        label: p.text.substring(0, 40) + '...',
        hoverText: p.text,
        embedding: p.embedding,
        type: 'page',
        color: '#3b82f6',
        size: 5
      });
    });
  }

  // 2. Gather Entities (compute average embedding of mentioning pages)
  const entityDict = new Map();
  processedData.forEach(p => {
    if (!p.resolvedEntities) return;
    p.resolvedEntities.forEach(wd => {
      if (!entityDict.has(wd.id)) {
        entityDict.set(wd.id, {
          id: wd.id,
          label: wd.label,
          hoverText: wd.description || 'Wikidata Entity',
          embeddings: [],
          url: wd.url
        });
      }
      entityDict.get(wd.id).embeddings.push(p.embedding);
    });
  });

  entityDict.forEach(ent => {
    // Average embedding
    const avgEmbedding = new Array(384).fill(0);
    ent.embeddings.forEach(emb => {
      for (let i = 0; i < 384; i++) avgEmbedding[i] += emb[i];
    });
    for (let i = 0; i < 384; i++) avgEmbedding[i] /= ent.embeddings.length;
    
    items.push({
      id: ent.id,
      label: ent.label,
      hoverText: ent.label + ' - ' + ent.hoverText,
      embedding: avgEmbedding,
      type: 'entity',
      color: '#f59e0b',
      size: Math.min(15, 5 + (ent.embeddings.length * 2)), // Size by frequency
      url: ent.url
    });
  });

  if (items.length === 0) {
    els.graphContainer.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-gray-500">Not enough data to plot.</div>';
    return;
  }

  // 3. Run t-SNE
  const X = items.map(item => item.embedding);
  const opt = {
    epsilon: 10, 
    perplexity: Math.min(30, Math.max(5, Math.floor(X.length / 3))),
    dim: 3
  };
  
  const tsne = new tsnejs.tSNE(opt);
  tsne.initDataRaw(X);
  
  // Run iterations (block UI for a short moment)
  for (let k = 0; k < 500; k++) {
    tsne.step();
  }
  
  const Y = tsne.getSolution(); // Array of [x, y, z]

  // 4. Plotly 3D Scatter
  const tracePage = {
    x: [], y: [], z: [],
    mode: 'markers',
    type: 'scatter3d',
    name: 'Pages',
    text: [],
    hoverinfo: 'text',
    marker: { color: '#3b82f6', size: 5, opacity: 0.8 }
  };

  const traceEntity = {
    x: [], y: [], z: [],
    mode: 'markers',
    type: 'scatter3d',
    name: 'Entities',
    text: [],
    hoverinfo: 'text',
    marker: { color: '#f59e0b', size: [], opacity: 1.0 }
  };
  
  const urls = [];

  items.forEach((item, i) => {
    const coords = Y[i];
    const trace = item.type === 'page' ? tracePage : traceEntity;
    trace.x.push(coords[0]);
    trace.y.push(coords[1]);
    trace.z.push(coords[2]);
    trace.text.push(item.hoverText);
    if (item.type === 'entity') {
      trace.marker.size.push(item.size);
    }
  });

  const plotData = [];
  if (tracePage.x.length > 0) plotData.push(tracePage);
  if (traceEntity.x.length > 0) plotData.push(traceEntity);

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 0 },
    scene: {
      xaxis: { showticklabels: false, title: '' },
      yaxis: { showticklabels: false, title: '' },
      zaxis: { showticklabels: false, title: '' }
    },
    legend: { x: 0, y: 1 }
  };

  els.graphContainer.innerHTML = ''; // clear loading text
  Plotly.newPlot(els.graphContainer, plotData, layout, { responsive: true });
  log(`3D t-SNE Map generated with ${items.length} points.`, "success");
}

// --- Treemap Construction ---
function renderTreemap(processedData) {
  els.placeholder.style.display = 'none';
  
  const entityMap = new Map();
  let totalPageMentions = 0;

  processedData.forEach(p => {
    if (!p.resolvedEntities) return;
    
    // Deduplicate entities for this page
    const uniqueEntities = Array.from(new Set(p.resolvedEntities.map(e => e.id)))
      .map(id => p.resolvedEntities.find(e => e.id === id));
      
    uniqueEntities.forEach(wd => {
      if (!entityMap.has(wd.id)) {
        entityMap.set(wd.id, { id: wd.id, label: wd.label, desc: wd.description, pages: [] });
      }
      entityMap.get(wd.id).pages.push({ 
        id: p.id + '_' + wd.id, 
        title: p.text.substring(0, 50) + (p.text.length > 50 ? '...' : ''), 
        fullText: p.text 
      });
      totalPageMentions++;
    });
  });

  if (totalPageMentions === 0) {
    els.graphContainer.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-gray-500">Not enough data to plot.</div>';
    return;
  }

  const ids = ["Root"];
  const labels = ["All Topics"];
  const parents = [""];
  const values = [totalPageMentions];
  const hovertext = ["Knowledge Graph Overview"];

  entityMap.forEach(ent => {
    ids.push(ent.id);
    labels.push(`${ent.label} (${ent.pages.length})`);
    parents.push("Root");
    values.push(ent.pages.length);
    hovertext.push(`${ent.desc || 'Wikidata Entity'} — ${ent.pages.length} pages`);
    
    ent.pages.forEach(page => {
      ids.push(page.id);
      labels.push(page.title);
      parents.push(ent.id);
      values.push(1);
      hovertext.push(page.fullText);
    });
  });

  const trace = [{
    type: "treemap",
    ids: ids,
    labels: labels,
    parents: parents,
    values: values,
    hovertext: hovertext,
    hoverinfo: "label+hovertext",
    textinfo: "label",
    branchvalues: "total",
    pathbar: { visible: true },
    tiling: { packing: "squarify" }
  }];

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 30 },
    treemapcolorway: ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]
  };

  els.graphContainer.innerHTML = ''; 
  Plotly.newPlot(els.graphContainer, trace, layout, { responsive: true });
  log(`Treemap generated with ${entityMap.size} topic clusters.`, "success");
}

// --- Main Execution Loop ---
els.btn.addEventListener('click', async () => {
  const rawInput = els.input.value;
  let texts = [];
  
  // Smart parsing: if it detects 'title:' format, chunk them together
  if (/^title:/im.test(rawInput)) {
    const blocks = rawInput.split(/(?=\btitle:)/i);
    texts = blocks.map(b => {
      // Strip out the "title:" and "desc:" labels so they don't affect ML embeddings/NER
      return b.replace(/\btitle:\s*/i, '')
              .replace(/\bdesc:\s*/i, ' ')
              .replace(/\n/g, ' ')
              .trim();
    }).filter(t => t.length > 0);
  } else {
    // Fallback: one page per line
    texts = rawInput.split('\n').map(t => t.trim()).filter(t => t.length > 0);
  }
  
  if (texts.length === 0) {
    alert("Please enter at least one line of text.");
    return;
  }
  
  els.btn.disabled = true;
  els.statusLog.innerHTML = '';
  
  try {
    await loadModels();
    
    log(`Starting processing for ${texts.length} inputs...`);
    const processedData = await processInput(texts);
    
    log("Resolving entities on Wikidata...");
    
    // Resolve all unique entities
    const allEntities = new Set();
    processedData.forEach(item => {
      item.entities.forEach(e => allEntities.add(e.text));
    });
    
    const resolvedEntities = {};
    for (const entityName of allEntities) {
      log(`Resolving Wikidata for: "${entityName}"...`);
      const wdData = await resolveWikidata(entityName);
      if (wdData) {
        resolvedEntities[entityName] = wdData;
      }
    }
    
    // Attach resolved Wikidata to processed items
    processedData.forEach(item => {
      item.resolvedEntities = item.entities.map(e => resolvedEntities[e.text]).filter(Boolean);
    });
    
    log("Wikidata resolution complete.", "success");
    
    log("Building visualization...", "info");
    
    const threshold = parseFloat(els.threshold.value);
    const showPages = els.showPages.checked;
    
    if (els.viewMode.value === '3d') {
      await render3DMap(processedData, showPages);
    } else if (els.viewMode.value === 'treemap') {
      renderTreemap(processedData);
    } else {
      render2DGraph(processedData, threshold, showPages);
    }
    
    // Setup Export Data
    lastExportData = {
      type: els.viewMode.value,
      threshold: threshold,
      showPages: showPages,
      data: processedData.map(item => ({
        id: item.id,
        text: item.text,
        entities: item.entities,
        wikidata: item.resolvedEntities || []
      }))
    };
    els.exportBtn.classList.remove('hidden');
    
  } catch (err) {
    log(`Process failed: ${err.message}`, "error");
  } finally {
    els.btn.disabled = false;
  }
});

// Sync threshold display
els.threshold.addEventListener('input', (e) => {
  els.thresholdDisplay.textContent = parseFloat(e.target.value).toFixed(2);
});

// Load Sample Data
els.loadSampleBtn.addEventListener('click', () => {
  els.input.value = SAMPLE_DATA;
  parseInputToTable();
});
