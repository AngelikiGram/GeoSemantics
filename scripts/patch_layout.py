with open('static/morph.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = """    html = '<div style="display:flex;gap:8px;margin-bottom:8px">'
      + '<div style="flex:1;font-size:.62em;font-weight:700;color:var(--blue);text-transform:uppercase;letter-spacing:.07em">Before</div>'
      + '<div style="flex:1;font-size:.62em;font-weight:700;color:var(--amber);text-transform:uppercase;letter-spacing:.07em">After</div>'
      + '</div><div class="dim-bars">';
    keys.forEach(k => {
      const col=DIM_COLORS[k];
      const bPct=Math.round((baseline[k]||0)*100);
      const mPct=Math.round((modified[k]||0)*100);
      const d=delta ? (delta[k]||0) : 0;
      if (!bPct && !mPct) return;
      const arrowTxt = d>0.015 ? '+'+Math.round(d*100)+'pp' : d<-0.015 ? Math.round(d*100)+'pp' : '';
      const arrowCol = d>0.015 ? col : '#94a3b8';
      html += '<div class="dim-row" style="gap:5px">'
        + '<span class="dim-glyph">'+dimGlyph(k,col)+'</span>'
        + '<div class="dim-name" style="color:'+col+'">'+k+'</div>'
        + '<div style="flex:1;display:flex;flex-direction:column;gap:2px">'
        + '<div class="dim-bg" style="height:4px"><div class="dim-fill" style="width:'+bPct+'%;background:'+col+';opacity:.4"></div></div>'
        + '<div class="dim-bg" style="height:4px"><div class="dim-fill" style="width:'+mPct+'%;background:'+col+'"></div></div>'
        + '</div><div style="width:44px;text-align:right;font-size:.63em;color:'+arrowCol+'">'+( arrowTxt||mPct+'%' )+'</div>'
        + '</div>';
    });"""

replacement = """    html = '<div style="display:flex;gap:4px;margin-bottom:8px;padding-right:4px;">'
      + '<div style="width:85px;"></div>'
      + '<div style="width:45px;text-align:right;font-size:.62em;font-weight:700;color:var(--blue);text-transform:uppercase;letter-spacing:.07em">Before</div>'
      + '<div style="width:45px;text-align:right;font-size:.62em;font-weight:700;color:var(--amber);text-transform:uppercase;letter-spacing:.07em">After</div>'
      + '<div style="flex:1;"></div>' 
      + '</div><div class="dim-bars">';
    keys.forEach(k => {
      const col=DIM_COLORS[k];
      const bPct=Math.round((baseline[k]||0)*100);
      const mPct=Math.round((modified[k]||0)*100);
      const d=delta ? (delta[k]||0) : 0;
      if (!bPct && !mPct) return;
      const arrowTxt = d>0.015 ? '+'+Math.round(d*100)+'pp' : d<-0.015 ? Math.round(d*100)+'pp' : '';
      const arrowCol = d>0.015 ? col : (d<-0.015 ? '#f43f5e' : '#94a3b8');
      
      html += '<div class="dim-row" style="gap:4px">'
        + '<span class="dim-glyph" style="flex-shrink:0">'+dimGlyph(k,col)+'</span>'
        + '<div class="dim-name" style="width:65px;flex-shrink:0;color:'+col+'">'+k+'</div>'
        + '<div style="width:45px;text-align:right;font-size:.74em;opacity:0.6">'+bPct+'%</div>'
        + '<div style="width:45px;text-align:right;font-size:.74em;font-weight:700">'+mPct+'%</div>'
        + '<div style="flex:1;text-align:right;font-size:.63em;color:'+arrowCol+'">'+(arrowTxt||' ')+'</div>'
        + '</div>';
    });"""

text = text.replace(target, replacement)

with open('static/morph.html', 'w', encoding='utf-8') as f:
    f.write(text)