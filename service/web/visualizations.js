'use strict';
// This module uses the shared app state after the main page has rendered.
function sameWound(a,b){return a.patient_id===b.patient_id && a.surgery_date===b.surgery_date && a.wound_location.toLowerCase()===b.wound_location.toLowerCase();}
function earlierScan(a,b){return a.scan_date<b.scan_date || (a.scan_date===b.scan_date && a.created_at<b.created_at);}
function compatibleColour(a,b){return a.features && b.features && (a.colour_method||'legacy')===(b.colour_method||'legacy') && a.colour_segmenter_sha256===b.colour_segmenter_sha256;}
function visualPanel(s,kind){const colour=kind==='colour';return `<section class="card"><span class="eyebrow">${colour?'MEASURED IMAGE COLOUR':'MODEL EXPLANATION'}</span><h2 style="margin-top:12px">${colour?'Colour gradient · CIELAB a*':'Grad-CAM · Elevated class'}</h2><p class="muted">${colour?'Full-photograph red–green appearance on a fixed scale. This is not an infection map or a between-photo pixel comparison.':'Coarse regions supporting the classifier’s raw Elevated logit. This is not wound segmentation, certainty, or a causal explanation.'}</p><button class="button secondary" data-visual="${kind}" data-id="${s.id}">Generate ${colour?'colour heatmap':'Grad-CAM'}</button><div class="visual-output" role="status"></div><div class="heat-legend"><span>${colour?'−20 a* · greener':'Lower relative activation'}</span><span>${colour?'+60 a* · redder':'Higher relative activation'}</span></div>${s.result.status==='retake'?'<div class="notice warning">The quality gate requested a retake. Raw colour may be unreliable; Grad-CAM is unavailable because classification did not run.</div>':''}</section>`;}
function enhanceVisualizations(){
    const s=chosen();
    if(state.page==='colour' && s){
        const options=state.scans.filter(p=>sameWound(s,p)&&earlierScan(p,s));
        const box=document.createElement('section');box.className='card no-print';box.style.marginBottom='22px';
        box.innerHTML=`<h2>Set up a colour comparison</h2><p class="muted">A single photograph has colour measurements, but drift requires two scans of the same wound. Recalculate older scans to use the same measurement method.</p><div class="toolbar"><select id="colour-previous" aria-label="Previous scan for colour comparison"><option value="">No previous scan</option>${options.map(p=>`<option value="${p.id}" ${s.previous_id===p.id?'selected':''}>${p.scan_date} · ${p.id.slice(0,6)}</option>`).join('')}</select><button class="button" id="save-comparison">Save comparison</button><button class="button secondary" id="recalculate-colour">Recalculate colour</button></div><p id="colour-action-status" role="status" class="muted">${esc(s.comparison_error||(!s.previous_id?'Upload an earlier photograph of this wound, then select it here.':''))}</p>${s.colour_warning?`<div class="notice warning">${esc(s.colour_warning)}</div>`:''}<small class="muted">Method: ${esc(s.colour_method||'legacy')} · colour values are independent of the screening probability.</small>`;
        const selector=$('#select-scan').closest('.toolbar');selector.after(box);
        $('#save-comparison').onclick=async()=>{const button=$('#save-comparison');button.disabled=true;try{await api(`/api/scans/${s.id}/comparison`,json('PATCH',{previous_id:$('#colour-previous').value||null}));await refresh();render();toast('Comparison updated.');}catch(e){$('#colour-action-status').textContent=e.message;button.disabled=false;}};
        $('#recalculate-colour').onclick=async()=>{const button=$('#recalculate-colour');button.disabled=true;$('#colour-action-status').innerHTML='<span class="spinner"></span> Extracting colour measurements from the stored photograph…';try{await api(`/api/scans/${s.id}/colour`,{method:'POST'});await refresh();render();toast('Colour measurements recalculated. Screening result preserved.');}catch(e){$('#colour-action-status').textContent=e.message;button.disabled=false;}};
    }
    if(s && ['colour','analysis'].includes(state.page)){
        const section=document.createElement('div');section.className='grid equal no-print';
        section.innerHTML=visualPanel(s,'colour')+visualPanel(s,'gradcam');$('#main').append(section);
        section.querySelectorAll('[data-visual]').forEach(button=>button.onclick=async()=>{
            const output=button.parentElement.querySelector('.visual-output');button.disabled=true;
            output.innerHTML='<p class="muted"><span class="spinner"></span> Generating from the stored image…</p>';
            try{
                const response=await fetch(`/api/scans/${button.dataset.id}/visualization/${button.dataset.visual}`);
                if(!response.ok){const body=await response.json();throw Error(body.detail||'Visualization failed');}
                const url=URL.createObjectURL(await response.blob());
                const image=document.createElement('img');image.className='scan-photo';image.alt=button.dataset.visual==='colour'?'CIELAB a-star colour heatmap':'Elevated-class Grad-CAM overlay';
                image.onload=()=>URL.revokeObjectURL(url);image.src=url;output.replaceChildren(image);
            }catch(e){output.innerHTML=`<p class="error">${esc(e.message)}</p>`;}finally{button.disabled=false;}
        });
    }
}
