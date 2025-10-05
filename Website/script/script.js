// ====== Configuration ======
// Change this to your backend endpoint
const PREDICT_ENDPOINT = '/api/predict';

// ====== DOM refs ======
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const form = document.getElementById('featureForm');
const btnPredict = document.getElementById('btnPredict');
const btnClear = document.getElementById('btnClear');
const errorBox = document.getElementById('error');
const statusMessage = document.getElementById('statusMessage');
const probabilityBox = document.getElementById('probabilityBox');
const probText = document.getElementById('probText');
const progBar = document.getElementById('progBar');
const metaInfo = document.getElementById('metaInfo');

// list of feature inputs in form order - used to map CSV rows and JSON keys
const FEATURE_KEYS = ['period','depth','duration','snr'];

// ====== Helpers ======
function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove('hidden');
}
function hideError() {
    errorBox.classList.add('hidden');
    errorBox.textContent = '';
}
function setStatus(msg) {
    statusMessage.textContent = msg || '';
}
function setProbability(prob, details = '') {
    // prob is 0..1 or 0..100
    let p = Number(prob);
    if (p <= 1.01) p = p * 100; // assume 0..1 -> convert to %
    p = Math.max(0, Math.min(100, p));
    probText.textContent = p.toFixed(2) + '%';
    progBar.style.width = p.toFixed(2) + '%';
    probabilityBox.classList.remove('hidden');
    metaInfo.textContent = details;
}

function clearProbability() {
    probabilityBox.classList.add('hidden');
    progBar.style.width = '0%';
    probText.textContent = '0%';
    metaInfo.textContent = '';
}

function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onerror = () => reject(new Error('Failed to read file'));
        r.onload = () => resolve(r.result);
        r.readAsText(file);
    });
}

function parseCSV(text) {
    // Minimal CSV parsing: expects a single row or a header + single data row
    // Comma separated; not handling escaped commas/newlines.
    const lines = text.trim().split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    if (lines.length === 0) throw new Error('CSV is empty');
    const header = lines[0].split(',').map(h => h.trim());
    let values;
    if (lines.length >= 2 && header.some(h => isNaN(parseFloat(h)) && h.length>0)) {
        // header row present
        values = lines[1].split(',').map(v => v.trim());
        const obj = {};
        header.forEach((h, i) => {
            obj[h] = values[i] !== undefined ? values[i] : '';
        });
        return obj;
    } else {
        // assume single row of values in fixed order
        const row = header;
        const obj = {};
        FEATURE_KEYS.forEach((k, i) => {
            obj[k] = row[i] !== undefined ? row[i] : '';
        });
        return obj;
    }
}

function parseJSONorCSV(text, filename) {
    // Try JSON first, then CSV
    try {
        const j = JSON.parse(text);
        if (typeof j === 'object' && j !== null) return j;
    } catch (e) {
        // not JSON
    }
    // fallback CSV
    if (filename && filename.toLowerCase().endsWith('.csv')) {
        return parseCSV(text);
    }
    // try to parse CSV anyway
    try {
        return parseCSV(text);
    } catch (e) {
        throw new Error('File is not valid JSON or CSV: ' + e.message);
    }
}

function fillFormFromObject(obj) {
    hideError();
    FEATURE_KEYS.forEach(key => {
        const el = document.querySelector(`[name="${key}"]`);
        if (!el) return;
        if (obj[key] !== undefined && obj[key] !== null) {
            el.value = String(obj[key]);
        }
    });
    setStatus('Form populated from file.');
}

// ====== Drag & Drop handlers ======
['dragenter','dragover'].forEach(ev => {
    dropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
    });
});
['dragleave','drop','dragend'].forEach(ev => {
    dropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
    });
});

dropzone.addEventListener('drop', async (e) => {
    const file = (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) || null;
    if (!file) {
        showError('No file dropped.');
        return;
    }
    try {
        setStatus('Reading file...');
        const txt = await readFileAsText(file);
        const parsed = parseJSONorCSV(txt, file.name);
        fillFormFromObject(parsed);
        setStatus('Ready. You may edit fields, then press Predict.');
    } catch (err) {
        console.error(err);
        showError(err.message || 'Failed to parse file.');
    } finally {
        setTimeout(() => setStatus(''), 3000);
    }
});

// click opens file picker
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { fileInput.click(); e.preventDefault(); }
});

fileInput.addEventListener('change', async (e) => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    try {
        setStatus('Reading file...');
        const txt = await readFileAsText(file);
        const parsed = parseJSONorCSV(txt, file.name);
        fillFormFromObject(parsed);
        setStatus('Ready. You may edit fields, then press Predict.');
    } catch (err) {
        showError(err.message || 'Failed to parse file.');
    } finally {
        setTimeout(() => setStatus(''), 2500);
    }
});

// ====== Form -> payload ======
function formToPayload() {
    const payload = {};
    FEATURE_KEYS.forEach(k => {
        const el = document.querySelector(`[name="${k}"]`);
        if (!el) return;
        const val = el.value.trim();
        // convert to number when possible
        const asNum = Number(val);
        payload[k] = val === '' ? null : (isNaN(asNum) ? val : asNum);
    });
    return payload;
}

// ====== Predict action ======
btnPredict.addEventListener('click', async () => {
    hideError();
    clearProbability();
    const payload = formToPayload();
    // basic validation: require at least one non-null
    if (Object.values(payload).every(v => v === null)) {
        showError('Please fill at least one field (or drop a file).');
        return;
    }

    try {
        btnPredict.disabled = true;
        setStatus('Sending to server...');
        const spinner = document.createElement('span');
        spinner.className = 'spinner';
        btnPredict.appendChild(spinner);

        // POST JSON payload
        const res = await fetch(PREDICT_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: payload })
        });

        if (!res.ok) {
            const txt = await res.text();
            throw new Error('Server error: ' + (txt || res.statusText || res.status));
        }

        const json = await res.json();
        // expected shape: { probability: 0.87, label: "exoplanet", details: {...} }
        if (typeof json !== 'object') throw new Error('Invalid server response');

        if (json.probability === undefined && json.proba === undefined) {
            // try common keys
            const p = json.probability ?? json.proba ?? json.score;
            if (p === undefined) throw new Error('Server did not return probability');
            setProbability(p, json.details ? JSON.stringify(json.details) : '');
        } else {
            setProbability(json.probability ?? json.proba, json.details ? JSON.stringify(json.details) : (json.label ? ('Label: ' + json.label) : ''));
        }

        setStatus('Result received');
    } catch (err) {
        console.error(err);
        showError(err.message || 'Prediction failed');
    } finally {
        setStatus('');
        btnPredict.disabled = false;
        // remove spinner if still present
        const sp = btnPredict.querySelector('.spinner');
        if (sp) sp.remove();
    }
});

btnClear.addEventListener('click', () => {
    hideError();
    setStatus('');
    FEATURE_KEYS.forEach(k => {
        const el = document.querySelector(`[name="${k}"]`);
        if (el) el.value = '';
    });
    clearProbability();
});

// On page load: small demo population (optional)
(function demoFill() {
    // leave empty; you can uncomment for demo values
    // document.querySelector('[name="period"]').value = '365.24';
})();