import re

with open('src/app/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix Plotly Error
content = content.replace(
    "fig5 = ff.create_distplot(hist_data, group_labels, bin_size=[0.05, 0.05], show_rug=False)",
    """fig5 = go.Figure()
            fig5.add_trace(go.Histogram(x=conv1_w, name='Conv1D Weights', marker_color='#854F6C', opacity=0.75, histnorm='probability density'))
            fig5.add_trace(go.Histogram(x=lstm_w, name='LSTM Weight (IH)', marker_color='#DFB6B2', opacity=0.75, histnorm='probability density'))
            fig5.update_layout(barmode='overlay', title_font=dict(color='#E0E0E0'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='#E0E0E0')))"""
)

# 2. Add Cyber Box CSS if not exists
cyber_css = """
    .cyber-box {
        background: #190019 !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(82, 43, 91, 0.3) !important;
        padding: 25px !important;
        margin-bottom: 30px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5), inset 0 0 15px rgba(82, 43, 91, 0.05) !important;
        position: relative;
        padding-top: 50px !important;
    }
    .cyber-box::before {
        content: '';
        position: absolute;
        top: 15px;
        left: 20px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #FF5F56;
        box-shadow: 0 0 8px rgba(255,95,86,0.6), 20px 0 0 #FFBD2E, 20px 0 8px rgba(255,189,46,0.6), 40px 0 0 #27C93F, 40px 0 8px rgba(39,201,63,0.6);
    }
    .cyber-box::after {
        content: attr(data-title);
        position: absolute;
        top: 13px;
        left: 45px;
        color: #854F6C;
        font-size: 13px;
        letter-spacing: 1.5px;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 0 5px rgba(133, 79, 108, 0.4);
    }
    .cyber-box-inner {
        border-top: 1px solid rgba(133, 79, 108, 0.3);
        padding-top: 15px;
        color: #FBE4D8;
    }
    .cyber-box-inner h3, .cyber-box-inner h4 { margin-top: 0; }
"""
if ".cyber-box {" not in content:
    content = content.replace("</style>", cyber_css + "\n</style>")

# 3. Replace gray boxes

# Replace in Model Architectures:
content = content.replace(
    '<div class="panel-card mr-accent" style="background: #190019; border: 1px solid rgba(82, 43, 91, 0.4); border-left: 6px solid #522B5B; padding: 40px; border-radius: 12px; margin-bottom: 40px;">',
    '<div class="cyber-box" data-title="MINIROCKET // DET_FEATURE_EXTRACT"><div class="cyber-box-inner">'
)
content = content.replace(
    '<div class="panel-card cl-accent" style="background: #190019; border: 1px solid rgba(133, 79, 108, 0.4); border-left: 6px solid #854F6C; padding: 40px; border-radius: 12px; margin-bottom: 40px;">',
    '<div class="cyber-box" data-title="HYBRID_CNN_LSTM // SPATIOTEMPORAL"><div class="cyber-box-inner">'
)
content = content.replace(
    '<div class="panel-card" style="background: #190019; border: 1px solid rgba(223, 182, 178, 0.4); border-left: 6px solid #DFB6B2; padding: 40px; border-radius: 12px; margin-bottom: 40px;">',
    '<div class="cyber-box" data-title="SYS_REGULARIZATION // VAL_STRAT"><div class="cyber-box-inner">'
)

# Need to replace the end divs for these 3 boxes which are at the end of the markdown string
content = content.replace(
    """<li><strong>Holdout Split:</strong> Models are trained on an $80\\%$ subset and evaluated strictly on a $20\\%$ unseen holdout set to accurately simulate real-world BCI generalization.</li>
</ul>
</div>
</div>""",
    """<li><strong>Holdout Split:</strong> Models are trained on an $80\\%$ subset and evaluated strictly on a $20\\%$ unseen holdout set to accurately simulate real-world BCI generalization.</li>
</ul>
</div>
</div></div>"""
)
content = content.replace(
    """<li>Solved analytically via Cholesky decomposition.</li>
</ul>
</div>
</div>""",
    """<li>Solved analytically via Cholesky decomposition.</li>
</ul>
</div>
</div></div>"""
)
content = content.replace(
    """<li><strong>Optimizer:</strong> Adam ($lr=0.001$, $\\beta_1=0.9$, $\\beta_2=0.999$).</li>
</ul>
</div>
</div>""",
    """<li><strong>Optimizer:</strong> Adam ($lr=0.001$, $\\beta_1=0.9$, $\\beta_2=0.999$).</li>
</ul>
</div>
</div></div>"""
)


# Replace in Live Training Console (712 and 957)
content = content.replace(
    '<div style="background: #190019; padding: 35px; border-radius: 12px; border: 1px solid rgba(133, 79, 108, 0.4); border-left: 6px solid #854F6C; margin-bottom: 20px; margin-top: 20px;">\n<h4 style="color: #FBE4D8; margin-top: 0; margin-bottom: 25px; font-family: \'Space Grotesk\', sans-serif; font-size: 22px;">Advanced Training Configuration</h4>',
    '<div class="cyber-box" data-title="MI-BCI // SYSTEM_CONFIG"><div class="cyber-box-inner">\n<h4 style="color: #FBE4D8; margin-top: 0; margin-bottom: 25px; font-family: \'Space Grotesk\', sans-serif; font-size: 22px;">Advanced Training Configuration</h4>'
)

# And fix the ending div for these two (there are 2 occurrences of this exact block)
content = content.replace(
    'st.markdown("</div>", unsafe_allow_html=True)',
    'st.markdown("</div></div>", unsafe_allow_html=True)'
)

# Also fix the inner gray boxes in Model Architectures:
# <div style="background: rgba(0,0,0,0.3); padding: 25px; border-radius: 8px; border: 1px solid #333;">
content = content.replace(
    '<div style="background: rgba(0,0,0,0.3); padding: 25px; border-radius: 8px; border: 1px solid #333;">',
    '<div style="background: rgba(10,0,10,0.4); padding: 25px; border-radius: 8px; border: 1px solid rgba(82, 43, 91, 0.5);">'
)

with open('src/app/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
