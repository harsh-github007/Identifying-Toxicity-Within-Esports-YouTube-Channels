const $ = id => document.getElementById(id);
const pct = value => (value * 100).toFixed(1) + '%';
const safe = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data, estimate = 'prevalence_classifier';
function channels() {
  const rows = data[estimate];
  const max = Math.max(...rows.map(r=>r.high)) * 1.12;
  $('channelRows').innerHTML = rows.map(r=>{
    const context = data.prevalence_classifier.find(c=>c.channel===r.channel);
    return `<article class="channel-row"><div><h3>${safe(r.channel)}</h3><small>${context.comments.toLocaleString()} comments · ${context.videos} videos${r.labelled ? ` · ${r.labelled} labels` : ''}</small></div><div class="range-wrap" aria-label="${safe(r.channel)} 95% interval ${pct(r.low)} to ${pct(r.high)}"><div class="range"><span class="interval" style="left:${r.low/max*100}%;width:${(r.high-r.low)/max*100}%"></span><span class="point" style="left:${r.estimate/max*100}%"></span></div><div class="range-label"><span>0%</span><span>${pct(max)}</span></div></div><div class="channel-number"><strong>${pct(r.estimate)}</strong><small>95% interval ${pct(r.low)}–${pct(r.high)}</small></div></article>`;
  }).join('');
  $('estimateNote').textContent = estimate === 'prevalence_classifier' ? 'All-comment estimates use the word list and correct for measured sensitivity and specificity. Bootstrap intervals resample labels and whole videos. A point is the estimate; the shaded band shows its uncertainty.' : 'Human-labelled estimates use sampling weights and stratified Jeffreys intervals. They are based on only 18–22 labelled comments per channel. The chart scale expands to fit these wider intervals.';
}
const explanations = {f1:'F1 balances precision and recall. Higher is better, but the wide 95% intervals reflect just five toxic labels.',precision:'Precision: among comments flagged as toxic, how many were human-labelled toxic? Higher means fewer false alarms.',recall:'Recall: among human-labelled toxic comments, how many did the method catch? Higher means fewer missed toxic comments.',specificity:'Specificity: among human-labelled non-toxic comments, how many were correctly left unflagged? High specificity alone does not mean a method detects toxicity well.'};
function models() {
  const key = $('metric').value;
  $('modelRows').innerHTML = data.methods.map(m=>`<div class="model-row"><h3>${safe(m.name.replace(' (trained on your labels)',''))}</h3><div class="bar" aria-hidden="true"><span style="width:${m[key]*100}%"></span></div><div class="model-value"><strong>${pct(m[key])}</strong><small>95% interval ${pct(m.ci[key][0])}–${pct(m.ci[key][1])}</small></div></div>`).join('');
  $('metricNote').textContent=explanations[key];
}
function videos() {
  const rows=data.flagged_by_video.filter(r=>r.channel===$('channel').value);
  $('videoCount').textContent=`${rows.length} sampled videos`;
  $('videoRows').innerHTML=rows.map((r,i)=>`<tr><td>Video ${String(i+1).padStart(2,'0')} <small>· ${safe(r.video_id)}</small></td><td>${pct(r.rate)}</td><td><a target="_blank" rel="noreferrer" href="https://www.youtube.com/watch?v=${encodeURIComponent(r.video_id)}">YouTube ↗</a></td></tr>`).join('');
}
async function init(){try{const response=await fetch('results/metrics.json');if(!response.ok)throw Error('Saved results unavailable');data=await response.json();$('stats').innerHTML=[[data.comments.toLocaleString(),'Comments collected'],[data.prevalence_classifier.length,'Channels studied'],[data.labelled,'Human-labelled comments'],[data.labelled_toxic,'Toxic examples in labels']].map(([n,label])=>`<div class="stat"><b>${n}</b><span>${label}</span></div>`).join('');$('channel').innerHTML=data.prevalence_classifier.map(r=>`<option>${safe(r.channel)}</option>`).join('');channels();models();videos();}catch(e){$('error').hidden=false;$('error').textContent=`${e.message}. Serve this repository through a local web server and reload.`;$('stats').textContent='Saved results could not be loaded.'}}
document.querySelectorAll('[data-estimate]').forEach(b=>b.addEventListener('click',()=>{estimate=b.dataset.estimate;document.querySelectorAll('[data-estimate]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));if(data)channels();}));$('metric').addEventListener('change',()=>data&&models());$('channel').addEventListener('change',()=>data&&videos());init();
