/* Model performance page. Data comes only from /assets/data/model_performance.json (exported from results/). */
const pct=v=>v==null?'n/a':(v*100).toFixed(1)+'%';
const num=v=>v==null?'n/a':Number(v).toFixed(3);
function metricTile(label,value,note){return `<div class="metric"><span>${esc(label)}</span><span class="metric-value">${value}</span>${note?`<small>${esc(note)}</small>`:''}</div>`;}
function confusionTable(c){return `<table class="confusion" aria-label="Confusion matrix"><thead><tr><th></th><th>Predicted Low</th><th>Predicted Elevated</th></tr></thead><tbody><tr><th>Actual Low</th><td class="tn">${c.tn}</td><td class="fp">${c.fp}</td></tr><tr><th>Actual Elevated</th><td class="fn">${c.fn}</td><td class="tp">${c.tp}</td></tr></tbody></table>`;}
function prChart(curves){
  if(!curves||!curves.length)return '';
  const W=360,H=300,P=38,x=r=>P+r*(W-P-10),y=p=>H-P-p*(H-P-10);
  const colours=['#0E8388','#E25822','#0C1E36'];
  const lines=curves.map((c,i)=>`<polyline fill="none" stroke="${colours[i%3]}" stroke-width="2.4" points="${c.points.map(([r,p])=>x(r).toFixed(1)+','+y(p).toFixed(1)).join(' ')}"/>`).join('');
  const base=curves[0].prevalence!=null?`<line x1="${x(0)}" x2="${x(1)}" y1="${y(curves[0].prevalence)}" y2="${y(curves[0].prevalence)}" stroke="#7a8b99" stroke-dasharray="4 4"/>`:'';
  const legend=curves.map((c,i)=>`<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;font-size:13px"><i style="width:14px;height:3px;background:${colours[i%3]};display:inline-block"></i>${esc(c.label)}</span>`).join('');
  return `<figure style="margin:0"><svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision-recall curves"><rect x="${P}" y="10" width="${W-P-10}" height="${H-P-10}" fill="none" stroke="#dbe5e8"/>${base}${lines}<text x="${W/2}" y="${H-6}" text-anchor="middle" font-size="12" fill="#4a5f70">Recall</text><text x="12" y="${H/2}" transform="rotate(-90 12 ${H/2})" text-anchor="middle" font-size="12" fill="#4a5f70">Precision</text></svg><figcaption>${legend}<small class="hint"> Dashed line = prevalence (chance level).</small></figcaption></figure>`;
}
function modelView(){return `${heading('Model performance','Measured results for the Wound Trace screening classifier (Low vs Elevated).')}<section id="model-root" class="section"><p class="loading">Loading results...</p></section>`;}
async function loadModelPage(){
  const root=document.getElementById('model-root');if(!root)return;
  try{
    const d=await fetch('/assets/data/model_performance.json').then(r=>{if(!r.ok)throw Error('Results file unavailable');return r.json();});
    const t=d.active.test,c=t.confusion;
    root.innerHTML=`<div class="notice warning"><p><strong>Research prototype.</strong> ${esc(d.caveat)}</p></div>
    <h2>Active model on the held-out test split <span class="muted">(${t.n} images, ${t.positives} Elevated)</span></h2>
    <p class="hint">${esc(d.active.name)}. Threshold chosen on development data only.</p>
    <div class="perf-grid">${metricTile('Accuracy',pct(t.accuracy),'Always predicting Low scores '+pct(t.always_low_accuracy))}${metricTile('Precision',pct(t.precision),'Of flagged Elevated, how many were')}${metricTile('Recall',pct(t.recall),'Of true Elevated, how many were caught')}${metricTile('F1',pct(t.f1))}${metricTile('Balanced accuracy',pct(t.balanced_accuracy))}${metricTile('AUROC',num(t.auroc),t.auroc_ci?('95% CI '+t.auroc_ci.map(num).join(' to ')):'')}${metricTile('Average precision',num(t.ap),'Chance = '+num(t.prevalence))}</div>
    <div class="grid"><section class="card"><h3>Confusion matrix</h3>${confusionTable(c)}<p class="hint">With only ${t.positives} Elevated test images, one image moves recall by ${(100/t.positives).toFixed(0)} points.</p></section><section class="card"><h3>Precision-recall</h3>${prChart(d.pr_curves)}</section></div>
    <section class="card"><h3>Baseline vs candidates (development cross-validation, ${d.experiments.dev_n} images)</h3><table class="perf-table"><thead><tr><th>Configuration</th><th>AP</th><th>AUROC</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${d.experiments.rows.map(r=>`<tr class="${r.selected?'best':''}"><td>${esc(r.name)}${r.selected?' (selected)':''}</td><td class="num">${num(r.ap)}</td><td class="num">${num(r.auroc)}</td><td class="num">${pct(r.precision)}</td><td class="num">${pct(r.recall)}</td><td class="num">${pct(r.f1)}</td></tr>`).join('')}</tbody></table><p class="hint">${esc(d.experiments.note)}</p></section>
    <section class="card"><h3>Limits</h3><ul>${d.limits.map(l=>`<li>${esc(l)}</li>`).join('')}</ul></section>`;
  }catch(e){root.innerHTML=`<p class="error">${esc(e.message)}</p>`;}
}
