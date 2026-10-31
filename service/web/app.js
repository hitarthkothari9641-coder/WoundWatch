'use strict';
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state={user:null,csrf:'',patients:[],scans:[],wounds:[],notifications:[],settings:{},patient:'',selected:'',wound:'',page:'home',tab:'summary',query:'',status:'',risk:'',sort:'latest',loginRole:'patient'};
let busy=false,previewURL=null,uploadFile=null;
const iconPaths={home:'M3 10l9-7 9 7v11h-7v-7h-4v7H3z',profile:'M12 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M4 21v-2a8 8 0 0 1 16 0v2',scans:'M4 7h4l2-3h4l2 3h4v13H4z M12 10a4 4 0 1 0 0 8 4 4 0 0 0 0-8',analysis:'M5 3h14v18H5z M8 8h8 M8 12h5 M8 16h8',risk:'M12 3l8 4v6c0 5-8 9-8 9S4 18 4 13V7z M8 12l3 3 5-6',notifications:'M5 17h14l-2-3V9a5 5 0 0 0-10 0v5z M10 21h4',settings:'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1z',arrow:'M5 12h14 M13 6l6 6-6 6',plus:'M12 5v14 M5 12h14',logout:'M9 3H4v18h5 M12 12h9 M17 8l4 4-4 4',check:'M5 12l4 4L19 6'};
const icon=k=>`<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${iconPaths[k]||iconPaths.profile}"/></svg>`;
const brand=()=>'<a class="brand" href="#home" aria-label="Wound Trace home"><span>WOUND TRACE</span><small>CONTINUOUS POST-SURGICAL VISUAL MONITORING</small></a>';
const isDoctor=()=>state.user?.role==='clinician';
const person=id=>state.patients.find(p=>p.id===id);
const chosen=()=>state.scans.find(s=>s.id===state.selected);
const currentPatient=()=>person(state.patient)||state.patients[0];
const imageURL=s=>`/api/scans/${encodeURIComponent(s.id)}/image`;
const fmt=v=>v==null?'Not available':Number(v).toFixed(2);
const localDate=()=>new Date().toLocaleDateString('en-CA');
const niceDate=d=>d?new Date(d.length===10?d+'T12:00:00':d).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'}):'Not recorded';
const niceTime=d=>d?new Date(d).toLocaleString(undefined,{dateStyle:'medium',timeStyle:'short'}):'Not recorded';
const ordered=list=>[...list].sort((a,b)=>b.scan_date.localeCompare(a.scan_date)||b.created_at.localeCompare(a.created_at));
const patientScans=()=>ordered(state.scans.filter(s=>s.patient_id===currentPatient()?.id&&(!state.wound||s.wound_id===state.wound)));
const resultLabel=s=>({retake:'Retake required',defer:'Needs human review',elevated:'Elevated risk',low:'Low risk',unknown:'Not available'}[s.result.status==='retake'?'retake':s.result.status==='defer'?'defer':s.result.risk_band]||'Not available');
const badge=s=>`<span class="badge ${esc(s.result.status==='retake'?'retake':s.result.status==='defer'?'defer':s.result.risk_band)}">${esc(resultLabel(s))}</span>`;
const heading=(title,sub,action='')=>`<div class="page-heading"><div><span class="eyebrow">YOUR CARE, CONNECTED</span><h1>${esc(title)}</h1><p>${esc(sub)}</p></div>${action}</div>`;
const button=(label,hash='scans')=>`<a class="button" href="#${hash}">${icon(hash==='scans'?'plus':'arrow')}${label}</a>`;
const empty=(title,text,action='')=>`<div class="empty">${icon('scans')}<h3>${title}</h3><p>${text}</p>${action}</div>`;
function toast(text){$('#toast').textContent=text;$('#toast').classList.add('visible');setTimeout(()=>$('#toast').classList.remove('visible'),4500);}
async function api(path,options={}){
    const headers=new Headers(options.headers||{});
    if(options.method&&!['GET','HEAD'].includes(options.method))headers.set('X-CSRF-Token',state.csrf);
    const r=await fetch(path,{...options,headers});
    if(!r.ok){let body={};try{body=await r.json();}catch{}const error=Error(Array.isArray(body.detail)?body.detail.map(d=>d.msg).join('; '):body.detail||`Request failed (${r.status})`);error.status=r.status;
        if(r.status===401&&state.user){state.user=null;state.csrf='';clearRecords();authScreen();}
        throw error;
    }
    return r.json();
}
const json=(method,body)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function clearRecords(){if(previewURL)URL.revokeObjectURL(previewURL);previewURL=null;uploadFile=null;state.patients=[];state.scans=[];state.wounds=[];state.notifications=[];state.patient='';state.selected='';state.wound='';$('#dialog').close();}
async function refresh(){
    [state.patients,state.scans,state.notifications,state.settings]=await Promise.all(['/api/patients','/api/scans','/api/notifications','/api/settings'].map(url=>api(url)));
    if(!person(state.patient))state.patient=state.user.patient_id||state.patients[0]?.id||'';
    state.wounds=state.patient?await api(`/api/patients/${state.patient}/wounds`):[];
    if(!chosen())state.selected=ordered(state.scans)[0]?.id||'';
}
async function choosePatient(id){state.patient=id;state.wound='';state.selected='';state.wounds=await api(`/api/patients/${id}/wounds`);navigate(isDoctor()?'patient':'profile');}
function navigate(page){if(location.hash==='#'+page)render();else location.hash=page;}
function authScreen(register=false,message=''){
    const role=state.loginRole,doctor=role==='clinician';
    $('#app').innerHTML=`<div class="auth-layout"><section class="auth-story">${brand()}<div><span class="eyebrow">RECOVERY IS A JOURNEY</span><h1>A little care.<br>A clearer picture.</h1><p>Keep your wound photographs in one place.<br>See changes over time. Share your journey with your healthcare worker.</p><div class="care-art" aria-hidden="true"><div class="care-circle">${icon('risk')}</div><span class="care-note">${icon('check')} One step at a time</span></div></div><p class="auth-foot">Your photographs stay with your account.</p></section><main id="main" class="auth-main"><div class="auth-box"><span class="eyebrow">WELCOME TO WOUND TRACE</span><h2>${register?'Let’s start your journey':doctor?'Doctor login':'Patient login'}</h2><p>${register?'Create your patient account to begin.':doctor?'Sign in with your healthcare-worker account.':'Sign in to continue your wound journey.'}</p><div class="auth-tabs" role="tablist" aria-label="Account type"><button id="patient-login-tab" role="tab" aria-selected="${!doctor}" class="${!doctor?'active':''}">Patient Login</button><button id="doctor-login-tab" role="tab" aria-selected="${doctor}" class="${doctor?'active':''}">Doctor Login</button></div><button id="sign-up-tab" class="auth-register ${register?'active':''}">${register?'Create patient account':'New patient? Create an account'}</button><form id="auth-form">${register?'<label>Your name<input name="name" autocomplete="name" maxlength="100" required></label>':''}<label>Email address<input name="email" type="email" autocomplete="username" maxlength="254" required></label><label>Password<input name="password" type="password" autocomplete="${register?'new-password':'current-password'}" minlength="12" maxlength="128" required></label><p class="hint">${register?'Use at least 12 characters. A few memorable words work well.':doctor?'Doctor accounts are created by the local administrator.':'Healthcare workers should choose Doctor Login.'}</p><p class="error" id="auth-error" role="alert">${esc(message)}</p><button class="button full" type="submit">${register?'Create my account':'Sign in securely'} ${icon('arrow')}</button></form><p class="help-text">Need help signing in? Contact your Wound Trace administrator. Email password recovery is not available.</p><div class="research-note">A research screening and monitoring tool.<br>Not a replacement for professional medical evaluation.</div></div></main></div>`;
    $('#patient-login-tab').onclick=()=>{state.loginRole='patient';authScreen(false);};
    $('#doctor-login-tab').onclick=()=>{state.loginRole='clinician';authScreen(false);};
    $('#sign-up-tab').onclick=()=>{state.loginRole='patient';authScreen(!register);};
    $('#auth-form').onsubmit=async e=>{e.preventDefault();const b=e.target.querySelector('[type=submit]');b.disabled=true;$('#auth-error').textContent='';try{const credentials=Object.fromEntries(new FormData(e.target));if(!register)credentials.account_type=state.loginRole;const result=await api('/api/auth/'+(register?'register':'login'),json('POST',credentials));if(!register&&result.user.role!==state.loginRole)throw Error('The account type did not match the selected login.');state.user=result.user;state.csrf=result.csrf_token;await refresh();location.hash='home';render();}catch(err){$('#auth-error').textContent=err.message;b.disabled=false;}};
}
function render(){
    if(!state.user){authScreen();return;}
    const aliases={overview:'home',patients:'home',history:isDoctor()?'patient':'profile',colour:isDoctor()?'patient':'profile',reports:isDoctor()?'patient':'profile',analysis:isDoctor()?'patient':'profile',risk:isDoctor()?'patient':'profile',notifications:'home'};
    let page=location.hash.slice(1)||'home';page=aliases[page]||page;
    if(!['home','profile','patient','scans','settings'].includes(page))page='home';
    if(page==='patient'&&!isDoctor())page='profile';state.page=page;
    const nav=[['home','Home'],['profile','My Profile'],['settings','Settings']];
    const active=page==='patient'?'home':page==='scans'?'profile':page;
    const unread=state.notifications.filter(n=>!n.read).length;
    const showMobileScan=!isDoctor()&&page!=='home'&&page!=='scans';
    $('#app').innerHTML=`<div class="shell"><header class="app-header">${brand()}<nav class="nav-capsule" aria-label="Main navigation">${nav.map(([k,label])=>`<a href="#${k}" class="${k===active?'active':''}" ${k===active?'aria-current="page"':''}>${icon(k)}<span>${label}</span></a>`).join('')}</nav><div class="account-capsule"><div class="notification-control"><button id="notification-bell" class="icon-button" aria-label="Notifications${unread?`, ${unread} unread`:''}" aria-expanded="false" aria-controls="notification-popover">${icon('notifications')}${unread?`<span class="notification-count">${unread>99?'99+':unread}</span>`:''}</button><section id="notification-popover" class="notification-popover" aria-label="Notifications" hidden>${notificationContent()}</section></div><a class="account-link" href="#profile"><span class="account-label">${isDoctor()?'Doctor Account':'Patient Account'}</span><span class="avatar" aria-hidden="true">${esc(state.user.name.slice(0,2).toUpperCase())}</span></a><button id="logout" class="icon-button sign-out" aria-label="Sign out" title="Sign out">${icon('logout')}<span>Sign Out</span></button></div></header><main id="main" tabindex="-1">${({home:homeView,profile:()=>isDoctor()?accountProfileView():profileView(),patient:profileView,scans:uploadView,settings:settingsView}[page])()}</main><footer>This application provides AI-assisted screening information and does not replace clinical examination or professional medical diagnosis.</footer>${showMobileScan?'<a class="mobile-scan" href="#scans">'+icon('plus')+' Scan My Wound</a>':''}</div>`;
    $('#notification-bell').onclick=()=>toggleNotifications();
    $('#close-notifications').onclick=()=>toggleNotifications(false,true);
    document.querySelectorAll('[data-sign-out],#logout').forEach(b=>b.onclick=signOut);
    bindViews();enhanceVisualizations();
}
function toggleNotifications(open,focusBell=false){
    const panel=$('#notification-popover'),bell=$('#notification-bell');if(!panel)return;
    const show=open??panel.hidden;panel.hidden=!show;bell.setAttribute('aria-expanded',String(show));
    if(show)$('#close-notifications').focus();else if(focusBell)bell.focus();
}
async function signOut(){if(busy){toast('Please wait for your analysis to finish.');return;}try{await api('/api/auth/logout',{method:'POST'});state.user=null;state.csrf='';clearRecords();authScreen();}catch(e){toast(e.message);}}
document.addEventListener('click',e=>{if(!e.target.closest('.notification-control'))toggleNotifications(false);});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('#notification-popover')?.hidden)toggleNotifications(false,true);});
function openDialog(title,content,onSubmit){
    $('#dialog-content').innerHTML=`<form id="dialog-form"><div class="section-heading"><h2>${title}</h2><button type="button" id="close-dialog" class="icon-button" aria-label="Close dialog">×</button></div>${content}<p class="error" id="dialog-error" role="alert"></p><button class="button" type="submit">Save details</button></form>`;
    $('#close-dialog').onclick=()=>$('#dialog').close();$('#dialog').showModal();
    $('#dialog-form').onsubmit=async e=>{e.preventDefault();const b=e.target.querySelector('[type=submit]');b.disabled=true;try{await onSubmit(Object.fromEntries(new FormData(e.target)));if(!state.user){$('#dialog').close();return;}await refresh();$('#dialog').close();render();toast('Details saved.');}catch(err){$('#dialog-error').textContent=err.message;b.disabled=false;}};
}
window.addEventListener('hashchange',()=>{if(busy){location.hash=state.page;toast('Please wait for your analysis to finish.');return;}render();window.scrollTo(0,0);});
window.addEventListener('pageshow',e=>{if(e.persisted)location.reload();});
async function start(){try{const result=await api('/api/auth/me');state.user=result.user;state.csrf=result.csrf_token;await refresh();render();}catch(e){if(e.status===401)authScreen();else if(e.status===404)authScreen(false,'The old server is still running. Press Ctrl+C in its terminal, run .\\start.ps1, and refresh this page.');else authScreen(false,'Cannot connect to Wound Trace. '+e.message);}}
start();
