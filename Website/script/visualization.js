/***********************************************
 * >>> TOGGLE MODES HERE <<<
 ***********************************************/
const FADE_MODE = false;     // ← Fade-out ON
// const FADE_MODE = false;  // ← Fade-out OFF

const DIMMING_MODE = false;  // ← Star brightness decreases
// const DIMMING_MODE = false; // ← Star stays constant
/***********************************************/

const starCanvas = document.getElementById('starCanvas');
const sctx = starCanvas.getContext('2d');
const lightCanvas = document.getElementById('lightCanvas');
const lctx = lightCanvas.getContext('2d');

const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const fluxVal = document.getElementById('fluxVal');

const W = starCanvas.width, H = starCanvas.height;
const starX = W/2, starY = H/2;

const starRadiusPx = Math.min(W, H) * 0.08;
const planetRadiusPx = Math.round(starRadiusPx * 0.55);
const pathMargin = Math.max(W, H) * 0.35;

let t = 0;
const period = 6.0;
let running = true;
let last = null;

const data = [];
const maxPoints = 400;

const starfield = [];
for (let i=0;i<160;i++){
    starfield.push({
        x: Math.random()*W,
        y: Math.random()*H,
        r: Math.random()*1.4+0.2,
        a: Math.random()*0.8+0.2
    });
}

function drawSky(){
    const g = sctx.createLinearGradient(0,0,0,H);
    g.addColorStop(0, '#021226');
    g.addColorStop(1, '#06112a');
    sctx.fillStyle = g;
    sctx.fillRect(0,0,W,H);

    const ng = sctx.createRadialGradient(starX, starY, starRadiusPx*0.4, starX, starY, starRadiusPx*6);
    ng.addColorStop(0, 'rgba(180,200,255,0.14)');
    ng.addColorStop(1, 'rgba(0,0,0,0)');
    sctx.fillStyle = ng;
    sctx.beginPath(); sctx.arc(starX, starY, starRadiusPx*6, 0, Math.PI*2); sctx.fill();

    for (const s of starfield){
        sctx.beginPath();
        sctx.globalAlpha = s.a;
        sctx.fillStyle = '#ffffff';
        sctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
        sctx.fill();
    }
    sctx.globalAlpha = 1;
}

function drawStar(brightness=1){
    sctx.save();
    const intensity = Math.max(0, Math.min(1, brightness));
    const coreColor = `rgba(255,255,255,${intensity})`;
    const haloColor = `rgba(200,220,255,${0.22 * intensity})`;

    sctx.beginPath();
    sctx.fillStyle = coreColor;
    sctx.arc(starX, starY, starRadiusPx, 0, Math.PI*2);
    sctx.fill();

    const hg = sctx.createRadialGradient(starX, starY, starRadiusPx*0.2, starX, starY, starRadiusPx*3);
    hg.addColorStop(0, `rgba(255,255,255,${0.9*intensity})`);
    hg.addColorStop(0.4, haloColor);
    hg.addColorStop(1, 'rgba(0,0,0,0)');
    sctx.fillStyle = hg;
    sctx.beginPath(); sctx.arc(starX, starY, starRadiusPx*3, 0, Math.PI*2); sctx.fill();
    sctx.restore();
}

function circleOverlapArea(r, R, d){
    if (d >= r + R) return 0;
    if (d <= Math.abs(R - r)) return Math.PI * Math.min(r,R)**2;
    const r2=r*r, R2=R*R;
    const alpha = Math.acos((d*d + r2 - R2) / (2*d*r)) * 2;
    const beta  = Math.acos((d*d + R2 - r2) / (2*d*R)) * 2;
    return 0.5 * (r2*(alpha - Math.sin(alpha)) + R2*(beta - Math.sin(beta)));
}

function drawPlanet(px, py, opacity){
    sctx.save();
    sctx.globalAlpha = opacity;
    sctx.beginPath();
    sctx.fillStyle = '#000000';
    sctx.arc(px, py, planetRadiusPx, 0, Math.PI*2);
    sctx.fill();
    sctx.restore();
}

function computeFlux(px, py){
    const d = Math.hypot(px - starX, py - starY);
    const overlap = circleOverlapArea(planetRadiusPx, starRadiusPx, d);
    const starArea = Math.PI * starRadiusPx**2;
    return 1 - overlap/starArea;
}

function drawLight(){
    lctx.clearRect(0,0,lightCanvas.width, lightCanvas.height);
    const margin = 48;
    const top = margin;
    const bottom = lightCanvas.height - margin;

    // grid + axes
    lctx.strokeStyle = 'rgba(200,220,255,0.08)';
    lctx.lineWidth = 1;
    lctx.beginPath();
    lctx.moveTo(margin, top);
    lctx.lineTo(margin, bottom);
    lctx.lineTo(lightCanvas.width - margin, bottom);
    lctx.stroke();

    if (data.length < 2) return;
    const times = data.map(p=>p.t);
    const fluxes = data.map(p=>p.flux);
    const tmin = times[0], tmax = times.at(-1);
    const fmin = Math.min(...fluxes), fmax = Math.max(...fluxes);
    const displayFmin = Math.max(0, fmin - 0.02);
    const displayFmax = Math.min(1.0, fmax + 0.02);

    const tx = x => margin + (x-tmin)/(tmax-tmin)*(lightCanvas.width-margin*2);
    const fy = y => bottom - (y-displayFmin)/(displayFmax-displayFmin)*(bottom-top);

    // light curve
    lctx.lineWidth = 2; lctx.strokeStyle = '#ffffff';
    lctx.beginPath();
    data.forEach((p,i)=>{
        const x = tx(p.t), y = fy(p.flux);
        if(i===0) lctx.moveTo(x,y); else lctx.lineTo(x,y);
    });
    lctx.stroke();

    // last point marker
    const lastP = data.at(-1);
    lctx.fillStyle = '#ffdd57';
    lctx.beginPath();
    lctx.arc(tx(lastP.t), fy(lastP.flux), 3, 0, Math.PI*2);
    lctx.fill();

    // Y-axis labels + title
    lctx.fillStyle = '#bcd';
    lctx.font = '13px system-ui';
    lctx.textAlign = 'right';
    lctx.textBaseline = 'middle';
    const tickCount = 5;
    for (let i=0;i<=tickCount;i++){
        const val = displayFmax - i*(displayFmax - displayFmin)/tickCount;
        const y = fy(val);
        lctx.fillText(val.toFixed(2), margin-6, y);
        lctx.beginPath();
        lctx.strokeStyle = 'rgba(255,255,255,0.05)';
        lctx.moveTo(margin, y);
        lctx.lineTo(lightCanvas.width - margin, y);
        lctx.stroke();
    }

    // Axis titles
    lctx.save();
    lctx.fillStyle = '#eaf2ff';
    lctx.font = '15px system-ui';
// Y title (rotated)
    lctx.translate(6, lightCanvas.height/2);
    lctx.rotate(-Math.PI/2);
    lctx.textAlign = 'center';
    lctx.fillText('Relative Flux', 0, 0);
    lctx.restore();


    // X title (time)
    lctx.textAlign = 'center';
    lctx.fillText('Time (s)', lightCanvas.width/2, lightCanvas.height - 12);
}

function step(now){
    if (!last) last = now;
    const dt = (now - last)/1000; last = now;
    if (running){
        t += dt;
        const cycle = (t % period)/period;
        const phase = cycle < 0.5 ? cycle*2 : 1 - (cycle-0.5)*2;
        const startX = starX - pathMargin;
        const endX = starX + pathMargin;
        const px = startX + phase*(endX - startX);
        const py = starY;

        const flux = computeFlux(px, py);
        const timeNow = performance.now()/1000;
        data.push({t: timeNow, flux});
        if (data.length > maxPoints) data.shift();

        fluxVal.textContent = flux.toFixed(4);

        // Planet fade logic
        let opacity = 1.0;
        if (FADE_MODE) {
            const d = Math.abs(px - starX);
            opacity = Math.min(1, d/(starRadiusPx*2));
        }

        // Star dimming logic
        const brightness = DIMMING_MODE ? flux : 1.0;

        drawSky();
        drawStar(brightness);
        drawPlanet(px, py, opacity);
        drawLight();
    }
    requestAnimationFrame(step);
}

pauseBtn.addEventListener('click', ()=>{ running=!running; pauseBtn.textContent=running?'Pause':'Resume'; });
resetBtn.addEventListener('click', ()=>{ t=0; data.length=0; fluxVal.textContent='1.000'; drawSky(); drawStar(); drawLight(); });

drawSky(); drawStar(); drawLight();
requestAnimationFrame(step);