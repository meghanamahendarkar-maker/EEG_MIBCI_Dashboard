import re

with open('src/app/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Revert Model Architectures blocks
content = content.replace(
    '<div class="cyber-box" data-title="MINIROCKET // DET_FEATURE_EXTRACT"><div class="cyber-box-inner">',
    '<div class="panel-card mr-accent" style="background: #190019; border: 1px solid rgba(82, 43, 91, 0.4); border-left: 6px solid #522B5B; padding: 40px; border-radius: 12px; margin-bottom: 40px;">'
)

content = content.replace(
    '<div class="cyber-box" data-title="HYBRID_CNN_LSTM // SPATIOTEMPORAL"><div class="cyber-box-inner">',
    '<div class="panel-card cl-accent" style="background: #190019; border: 1px solid rgba(133, 79, 108, 0.4); border-left: 6px solid #854F6C; padding: 40px; border-radius: 12px; margin-bottom: 40px;">'
)

content = content.replace(
    '<div class="cyber-box" data-title="SYS_REGULARIZATION // VAL_STRAT"><div class="cyber-box-inner">',
    '<div class="panel-card" style="background: #190019; border: 1px solid rgba(223, 182, 178, 0.4); border-left: 6px solid #DFB6B2; padding: 40px; border-radius: 12px; margin-bottom: 40px;">'
)

# Revert the closing divs for Model Architectures
content = content.replace(
    """<li><strong>Holdout Split:</strong> Models are trained on an $80\\%$ subset and evaluated strictly on a $20\\%$ unseen holdout set to accurately simulate real-world BCI generalization.</li>
</ul>
</div>
</div></div>""",
    """<li><strong>Holdout Split:</strong> Models are trained on an $80\\%$ subset and evaluated strictly on a $20\\%$ unseen holdout set to accurately simulate real-world BCI generalization.</li>
</ul>
</div>
</div>"""
)
content = content.replace(
    """<li>Solved analytically via Cholesky decomposition.</li>
</ul>
</div>
</div></div>""",
    """<li>Solved analytically via Cholesky decomposition.</li>
</ul>
</div>
</div>"""
)
content = content.replace(
    """<li><strong>Optimizer:</strong> Adam ($lr=0.001$, $\\beta_1=0.9$, $\\beta_2=0.999$).</li>
</ul>
</div>
</div></div>""",
    """<li><strong>Optimizer:</strong> Adam ($lr=0.001$, $\\beta_1=0.9$, $\\beta_2=0.999$).</li>
</ul>
</div>
</div>"""
)

# Revert Live Training Console boxes
content = content.replace(
    '<div class="cyber-box" data-title="MI-BCI // SYSTEM_CONFIG"><div class="cyber-box-inner">\n<h4 style="color: #FBE4D8; margin-top: 0; margin-bottom: 25px; font-family: \'Space Grotesk\', sans-serif; font-size: 22px;">Advanced Training Configuration</h4>',
    '<div style="background: #190019; padding: 35px; border-radius: 12px; border: 1px solid rgba(133, 79, 108, 0.4); border-left: 6px solid #854F6C; margin-bottom: 20px; margin-top: 20px;">\n<h4 style="color: #FBE4D8; margin-top: 0; margin-bottom: 25px; font-family: \'Space Grotesk\', sans-serif; font-size: 22px;">Advanced Training Configuration</h4>'
)

content = content.replace(
    '<div class="cyber-box" data-title="MINIROCKET // DET_FEATURE_EXTRACT"><div class="cyber-box-inner">\n<h4 style="color: #FBE4D8; margin-top: 0; margin-bottom: 25px; font-family: \'Space Grotesk\', sans-serif; font-size: 22px;">Deterministic Feature Extraction Configuration</h4>',
    '<div style="background: #190019; padding: 35px; border-radius: 12px; border: 1px solid rgba(133, 79, 108, 0.4); border-left: 6px solid #854F6C; margin-bottom: 20px; margin-top: 20px;">\n<h4 style="color: #FBE4D8; margin-top: 0; margin-bottom: 25px; font-family: \'Space Grotesk\', sans-serif; font-size: 22px;">Deterministic Feature Extraction Configuration</h4>'
)

content = content.replace(
    'st.markdown("</div></div>", unsafe_allow_html=True)',
    'st.markdown("</div>", unsafe_allow_html=True)'
)

with open('src/app/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
