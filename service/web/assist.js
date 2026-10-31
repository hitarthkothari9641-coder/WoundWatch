// Help panel: plain-language steps in English / Hindi / Gujarati with read-aloud. Content: assist.json.
(()=>{
  let data=null,lang='en';
  try{lang=localStorage.getItem('wt-lang')||(navigator.language||'en').slice(0,2);}catch(e){}
  const speak=(text,code)=>{
    if(!('speechSynthesis' in window))return;
    speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang=code;u.rate=.9;speechSynthesis.speak(u);
  };
  const el=(tag,props={},...kids)=>{const n=Object.assign(document.createElement(tag),props);kids.forEach(k=>n.append(k));return n;};
  function render(dlg){
    if(!data.langs[lang])lang='en';
    const ui=data.ui[lang],code=data.langs[lang][1];dlg.replaceChildren();
    const chips=el('div',{className:'assist-langs',role:'group',ariaLabel:ui.lang});
    Object.entries(data.langs).forEach(([k,[name]])=>{
      const b=el('button',{type:'button',className:'assist-chip'+(k===lang?' on':''),textContent:name,lang:k});
      b.setAttribute('aria-pressed',k===lang);
      b.onclick=()=>{lang=k;try{localStorage.setItem('wt-lang',k);}catch(e){}speechSynthesis&&speechSynthesis.cancel();render(dlg);};
      chips.append(b);
    });
    const close=el('button',{type:'button',className:'assist-close',textContent:ui.close});close.onclick=()=>{speechSynthesis&&speechSynthesis.cancel();dlg.close();};
    dlg.append(el('div',{className:'assist-head'},el('h2',{textContent:ui.title,lang}),close),chips);
    data.sections.forEach((s,i)=>{
      const steps=s.steps[lang];
      const d=el('details',{className:'assist-sec',open:i===0});
      d.append(el('summary',{textContent:s.title[lang],lang}));
      const ol=el('ol',{lang});steps.forEach(t=>ol.append(el('li',{textContent:t})));
      const say=el('button',{type:'button',className:'assist-say',textContent:ui.read});
      say.onclick=()=>speak(s.title[lang]+'. '+steps.join('. '),code);
      const stop=el('button',{type:'button',className:'assist-say ghost',textContent:ui.stop});stop.onclick=()=>speechSynthesis.cancel();
      if(s.img)d.append(el('img',{src:s.img,alt:'QR code: open Wound Trace on your phone',className:'assist-qr',width:220,height:220}));
      d.append(ol,el('div',{className:'assist-actions'},say,stop));dlg.append(d);
    });
  }
  async function open(){
    const dlg=document.getElementById('assist-dlg');
    if(!data){try{data=await (await fetch('/assets/assist.json')).json();}catch(e){return;}}
    render(dlg);dlg.showModal();
  }
  function mount(){
    if(document.getElementById('assist-btn'))return;
    const b=el('button',{id:'assist-btn',type:'button',textContent:'?',title:'Help'});
    b.setAttribute('aria-label','Help');b.onclick=open;
    const dlg=el('dialog',{id:'assist-dlg'});dlg.setAttribute('aria-label','Help');
    dlg.addEventListener('click',e=>{if(e.target===dlg)dlg.close();});
    document.body.append(b,dlg);
  }
  document.readyState==='loading'?document.addEventListener('DOMContentLoaded',mount):mount();
})();
