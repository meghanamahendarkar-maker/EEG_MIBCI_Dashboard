import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import scipy.signal
import scipy.stats
import torch
import mne
from sklearn.metrics import confusion_matrix

from src.utils.physionet_mapping import EEGTrial, parse_physionet_filename
from src.utils.trial_extractor import TrialExtractor
from src.data.preprocessor import EEGPreprocessor

def predict_trial(trial: EEGTrial, preproc: EEGPreprocessor, mr_pipe, cnn_lstm):
    trials_data = trial.signal[np.newaxis, ...]
    X_processed, _, _ = preproc.process_trials(
        trials_data, 
        np.array([trial.class_id]), 
        trial.channel_names
    )
    sample = X_processed[0:1]
    
    mr_probs = mr_pipe.predict_proba(sample)[0]
    mr_pred = int(np.argmax(mr_probs))
    
    sample_tensor = torch.tensor(sample, dtype=torch.float32)
    with torch.no_grad():
        logits = cnn_lstm(sample_tensor)
        # Temperature Scaling (T=2.0) to calibrate confidence probabilities
        T = 2.0
        dl_probs = torch.softmax(logits / T, dim=1).numpy()[0]
        dl_pred = int(np.argmax(dl_probs))
        
    return mr_probs, mr_pred, dl_probs, dl_pred, sample

def render_bars(title, probs, color, classes):
    html = f"<div style='font-family:\"Space Grotesk\"; font-size:14px; font-weight:600; color:{color}; margin-bottom:14px;'>{title}</div>"
    html += "<div style='display:flex; flex-direction:column; gap:12px; margin-bottom:25px;'>"
    winner = np.argmax(probs)
    for i, cls in enumerate(classes):
        w = probs[i] * 100
        tc = "#FBE4D8" if i == winner else "#DFB6B2"
        html += f'<div style="display:grid; grid-template-columns:100px 1fr 60px; align-items:center; gap:12px; font-size:13px;">'
        html += f"<span style='font-family:\"Space Grotesk\"; color:{tc};'>{cls}</span>"
        html += f'<div style="height:12px; background:#141C24; border-radius:6px; overflow:hidden;">'
        html += f'<div style="height:100%; width:{w}%; background:{color}; border-radius:6px;"></div>'
        html += f'</div>'
        html += f"<span style='font-family:\"Space Grotesk\"; color:{tc}; text-align:right;'>{w:.1f}%</span>"
        html += f'</div>'
    html += "</div>"
    return html

def render_inference_tab(mr_pipe, cnn_lstm, X_test, y_test, classes):
    if 'filename' not in st.session_state:
        st.session_state['filename'] = None
    if 'trials' not in st.session_state:
        st.session_state['trials'] = None
    if 'inference_run' not in st.session_state:
        st.session_state.inference_run = False

    # ── Header ──
    st.html("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 10px;">
    <div style="display:flex; align-items:center; gap: 10px;">
        <h2 style="margin:0; font-size: 28px; color: #FBE4D8; font-family: 'Rye', serif;">📉 Live Inference Arena</h2>
    </div>
    <div style="display:flex; align-items:center; gap: 20px;">
        <div style="font-size: 12px; color: #a196aa; display:flex; align-items:center; gap:6px;">
            <span style="color:#00ff00;">●</span> System Ready
        </div>
    </div>
</div>
<p style="margin: 0 0 30px 0; font-size: 14px; color: #DFB6B2;">Upload a PhysioNet EDF file, select a trial, and run real-time inference through both model architectures.</p>
""")
    
    # ── Section 1: Data Input ──
    st.html("<h3 style='font-size:18px; margin-bottom:15px; color:#DFB6B2; border-bottom: 1px solid #332041; padding-bottom: 8px;'>1. Data Input Configuration</h3>")
    
    inp_c1, inp_c2 = st.columns([1, 1])
    
    with inp_c1:
        uploaded_file = st.file_uploader("Upload PhysioNet EDF", type=["edf"], key="file_upload")
        if uploaded_file is not None and (st.session_state.get('filename') != uploaded_file.name):
            try:
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                extractor = TrialExtractor(tmin=0.0, tmax=4.0)
                st.session_state['trials'] = extractor.extract_trials_from_raw(raw, uploaded_file.name)
                st.session_state['filename'] = uploaded_file.name
                st.session_state.inference_run = False  # Reset on new file
                
                del raw
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                    
                st.success(f"Loaded {uploaded_file.name}. Found {len(st.session_state['trials'])} trials.")
            except Exception as e:
                st.error(f"Error parsing file: {e}")

    trials = st.session_state.get('trials')
    selected_trial = None
    
    with inp_c2:
        def reset_inference():
            st.session_state.inference_run = False
            
        if trials:
            trial_options = [f"Trial {t.trial_index} | {t.annotation} | {t.class_name}" for t in trials]
            trial_options.insert(0, "--- Select a Trial ---")
            selected_option = st.selectbox("Select a Trial", trial_options, key="trial_select", on_change=reset_inference)
            if selected_option != "--- Select a Trial ---":
                sel_idx = trial_options.index(selected_option) - 1
                selected_trial = trials[sel_idx]
                preproc = EEGPreprocessor(raw_fs=int(trials[0].sfreq), target_fs=160, use_ica=False)
        else:
            st.selectbox("Select a Trial", ["Please upload a file first."], disabled=True)

    # ── Compute predictions ──
    subject_str = f"{selected_trial.subject_id} / R{int(selected_trial.run):02d}" if selected_trial else "--"
    exp_str = "Both Fists / Both Feet" if selected_trial and selected_trial.run in [6,10,14] else ("Left / Right Fist" if selected_trial else "--")
    gt_str = selected_trial.class_name if selected_trial else "--"
    
    pred_str = "--"
    conf_str = "--"
    dl_correct = False
    mr_probs, mr_pred, dl_probs, dl_pred, sample = None, None, None, None, None

    if selected_trial:
        mr_probs, mr_pred, dl_probs, dl_pred, sample = predict_trial(selected_trial, preproc, mr_pipe, cnn_lstm)
        if st.session_state.inference_run:
            dl_correct = dl_pred == selected_trial.class_id
            pred_str = classes[dl_pred]
            conf_str = f"{(dl_probs[dl_pred]*100):.1f}%"

    # ── Section 2: Top Info Cards (3 per row) ──
    st.html("<div style='margin-top: 30px;'></div>")
    st.html("<h3 style='font-size:18px; margin-bottom:15px; color:#DFB6B2; border-bottom: 1px solid #332041; padding-bottom: 8px;'>2. Trial Overview</h3>")

    # Row 1: Dataset, Subject, Experiment
    st.html(f"""
<div style="display: flex; gap: 15px; margin-bottom: 15px;">
    <div style="background: #190019; border-radius: 10px; padding: 18px; flex: 1; border: 1px solid #332041; display: flex; align-items: center; gap: 14px;">
        <div style="background: #3c1e48; border-radius: 50%; width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink:0;">🗄️</div>
        <div>
            <div style="font-size: 11px; color: #a196aa;">Selected Dataset</div>
            <div style="font-size: 15px; color: #FBE4D8; font-weight: 600;">PhysioNet EEGMMIDB</div>
            <div style="font-size: 10px; color: #DFB6B2;">EEG Motor Movement/Imagery Dataset</div>
        </div>
    </div>
    <div style="background: #190019; border-radius: 10px; padding: 18px; flex: 1; border: 1px solid #332041; display: flex; align-items: center; gap: 14px;">
        <div style="background: #3c1e48; border-radius: 50%; width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink:0;">👤</div>
        <div>
            <div style="font-size: 11px; color: #a196aa;">Subject / Run</div>
            <div style="font-size: 15px; color: #FBE4D8; font-weight: 600;">{subject_str}</div>
        </div>
    </div>
    <div style="background: #190019; border-radius: 10px; padding: 18px; flex: 1; border: 1px solid #332041; display: flex; align-items: center; gap: 14px;">
        <div style="background: #3c1e48; border-radius: 50%; width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink:0;">🧪</div>
        <div>
            <div style="font-size: 11px; color: #a196aa;">Experiment Type</div>
            <div style="font-size: 15px; color: #FBE4D8; font-weight: 600;">Motor Imagery</div>
            <div style="font-size: 10px; color: #DFB6B2;">{exp_str}</div>
        </div>
    </div>
</div>
""")
    # Row 2: Ground Truth, Prediction, Confidence
    conf_bar_width = conf_str if conf_str != "--" else "0%"
    st.html(f"""
<div style="display: flex; gap: 15px; margin-bottom: 30px;">
    <div style="background: #190019; border-radius: 10px; padding: 18px; flex: 1; border: 1px solid #332041; display: flex; align-items: center; gap: 14px;">
        <div style="background: #3c1e48; border-radius: 50%; width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink:0;">📄</div>
        <div>
            <div style="font-size: 11px; color: #a196aa;">Ground Truth</div>
            <div style="font-size: 15px; color: #FBE4D8; font-weight: 600;">{gt_str}</div>
            <div style="font-size: 10px; color: #DFB6B2;">Imagery Label</div>
        </div>
    </div>
    <div style="background: #190019; border-radius: 10px; padding: 18px; flex: 1; border: 1px solid #332041; display: flex; align-items: center; gap: 14px;">
        <div style="background: #3c1e48; border-radius: 50%; width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink:0;">🧠</div>
        <div>
            <div style="font-size: 11px; color: #a196aa;">Current Prediction</div>
            <div style="font-size: 15px; color: #FBE4D8; font-weight: 600;">{pred_str}</div>
        </div>
    </div>
    <div style="background: #190019; border-radius: 10px; padding: 18px; flex: 1; border: 1px solid #332041; display: flex; align-items: center; gap: 14px;">
        <div style="background: #3c1e48; border-radius: 50%; width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink:0;">📊</div>
        <div style="width: 100%;">
            <div style="font-size: 11px; color: #a196aa;">Confidence</div>
            <div style="font-size: 15px; color: #FBE4D8; font-weight: 600; margin-bottom: 4px;">{conf_str}</div>
            <div style="height: 6px; background: #332041; border-radius: 3px; width: 100%;"><div style="height: 100%; width: {conf_bar_width}; background: linear-gradient(90deg, #854F6C, #F48FB1); border-radius: 3px;"></div></div>
        </div>
    </div>
</div>
""")

    # ── Section 3: Input Signal Data (full width) ──
    st.html("<h3 style='font-size:18px; margin-bottom:15px; color:#DFB6B2; border-bottom: 1px solid #332041; padding-bottom: 8px;'>3. Input Signal Data</h3>")
    
    if selected_trial and sample is not None:
        # File metadata card
        st.html(f"""
<div style="background:#190019; border: 1px solid #332041; border-radius:10px; padding:18px; display:flex; gap:40px; margin-bottom:25px;">
    <div style="display:flex; align-items:center; justify-content:center; font-size:28px; width:50px;">ℹ️</div>
    <div style="flex:1;">
        <div style="color:#DFB6B2; font-size:12px; margin-bottom:6px;">File Name <span style="color:#FBE4D8; float:right;">{st.session_state['filename']}</span></div>
        <div style="color:#DFB6B2; font-size:12px; margin-bottom:6px;">Subject <span style="color:#FBE4D8; float:right;">{selected_trial.subject_id}</span></div>
        <div style="color:#DFB6B2; font-size:12px; margin-bottom:6px;">Run <span style="color:#FBE4D8; float:right;">R{int(selected_trial.run):02d}</span></div>
        <div style="color:#DFB6B2; font-size:12px;">Channels <span style="color:#FBE4D8; float:right;">64</span></div>
    </div>
    <div style="flex:1;">
        <div style="color:#DFB6B2; font-size:12px; margin-bottom:6px;">Sampling Rate <span style="color:#FBE4D8; float:right;">160 Hz</span></div>
        <div style="color:#DFB6B2; font-size:12px; margin-bottom:6px;">Window <span style="color:#FBE4D8; float:right;">4.0 s</span></div>
        <div style="color:#DFB6B2; font-size:12px; margin-bottom:6px;">Preprocessing <span style="color:#FBE4D8; float:right;">Bandpass, ICA, Seg, Norm</span></div>
        <div style="color:#DFB6B2; font-size:12px;">Samples <span style="color:#FBE4D8; float:right;">{sample.shape[2]}</span></div>
    </div>
</div>
""")
        sig_1d = sample[0, 0, :]
        
        # Temporal Trace — full width, bigger
        st.html('<h4 style="color:#FBE4D8; font-size:16px; margin-top:20px; border-bottom:1px solid #332041; padding-bottom:5px;">📈 Temporal Trace (Selected Trial)</h4>')
        fig_1d = go.Figure()
        fig_1d.add_trace(go.Scatter(y=sig_1d, mode='lines', name='Amplitude', line=dict(color='#F48FB1', width=1.5)))
        fig_1d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FBE4D8', size=12), margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, color='#a196aa', title='Sample'),
            yaxis=dict(showgrid=True, gridcolor='rgba(50,32,65,0.5)', color='#a196aa', title='Amplitude'),
            height=350
        )
        st.plotly_chart(fig_1d, use_container_width=True, config={'displayModeBar': False})
        
        # PSD and Spectrogram — full width, bigger
        st.html('<h4 style="color:#FBE4D8; font-size:16px; margin-top:30px; border-bottom:1px solid #332041; padding-bottom:5px;">📊 Power Spectral Density</h4>')
        f_psd, pxx = scipy.signal.welch(sig_1d, fs=160, nperseg=256)
        mask_psd = f_psd <= 45
        fig_psd = go.Figure(data=go.Scatter(x=f_psd[mask_psd], y=pxx[mask_psd]*1e6, mode='lines', fill='tozeroy', line=dict(color='#854F6C', width=2)))
        fig_psd.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FBE4D8', size=12), margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title='Frequency (Hz)', showgrid=False, color='#a196aa'),
            yaxis=dict(title='Power (µV²/Hz)', showgrid=True, gridcolor='rgba(50,32,65,0.5)', color='#a196aa'),
            height=300
        )
        st.plotly_chart(fig_psd, use_container_width=True, config={'displayModeBar': False})
            
        st.html('<h4 style="color:#FBE4D8; font-size:16px; margin-top:30px; border-bottom:1px solid #332041; padding-bottom:5px;">🖼️ STFT Spectrogram</h4>')
        f_s, t_s, Sxx = scipy.signal.stft(sig_1d, fs=160, nperseg=64, noverlap=32)
        mask_s = f_s <= 45
        fig_spec = go.Figure(data=go.Heatmap(z=np.abs(Sxx[mask_s, :]), x=t_s, y=f_s[mask_s], colorscale='Magma', showscale=True, colorbar=dict(title='Magnitude', len=1)))
        fig_spec.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FBE4D8', size=12), margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title='Time (s)', showgrid=False, color='#a196aa'),
            yaxis=dict(title='Frequency (Hz)', showgrid=False, color='#a196aa'),
            height=350
        )
        st.plotly_chart(fig_spec, use_container_width=True, config={'displayModeBar': False})
    else:
        st.html("<p style='color:#DFB6B2; font-size:14px; padding:30px 0; text-align:center;'>Please upload a file and select a trial above to view signal data.</p>")
    
    # ── Section 4: Preprocessing Pipeline ──
    st.html("<div style='margin-top: 15px;'></div>")
    st.html("<h3 style='font-size:18px; margin-bottom:15px; color:#DFB6B2; border-bottom: 1px solid #332041; padding-bottom: 8px;'>4. Preprocessing Pipeline</h3>")

    st.html("""
<div style="background:#190019; padding:25px; border-radius:10px; border:1px solid #332041;">
    <div style="display:flex; justify-content:space-around; align-items:center; text-align:center; padding: 0 10px;">
        <div>
            <div style="background:#3c1e48; width:60px; height:60px; border-radius:12px; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto; font-size:24px;">🌪️</div>
            <div style="font-size:12px; color:#FBE4D8; font-weight:600;">Bandpass Filter</div>
            <div style="font-size:10px; color:#DFB6B2;">(0.5 – 40 Hz)</div>
        </div>
        <div style="color:#854F6C; font-size:24px;">→</div>
        <div>
            <div style="background:#3c1e48; width:60px; height:60px; border-radius:12px; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto; font-size:24px;">🧹</div>
            <div style="font-size:12px; color:#FBE4D8; font-weight:600;">ICA</div>
            <div style="font-size:10px; color:#DFB6B2;">Artifact Removal</div>
        </div>
        <div style="color:#854F6C; font-size:24px;">→</div>
        <div>
            <div style="background:#3c1e48; width:60px; height:60px; border-radius:12px; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto; font-size:24px;">✂️</div>
            <div style="font-size:12px; color:#FBE4D8; font-weight:600;">Segmentation</div>
            <div style="font-size:10px; color:#DFB6B2;">(4 s windows)</div>
        </div>
        <div style="color:#854F6C; font-size:24px;">→</div>
        <div>
            <div style="background:#3c1e48; width:60px; height:60px; border-radius:12px; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto; font-size:24px;">⚖️</div>
            <div style="font-size:12px; color:#FBE4D8; font-weight:600;">Normalization</div>
            <div style="font-size:10px; color:#DFB6B2;">(z-score)</div>
        </div>
    </div>
</div>
""")
    
    # ── Real Inference Button ──
    st.html("<div style='margin-top: 20px;'></div>")
    if selected_trial:
        if st.button("▶ Run Live Inference on Stream", key="run_inference_btn", type="primary", use_container_width=True):
            st.session_state.inference_run = True
            st.rerun()
    else:
        st.button("▶ Run Live Inference on Stream", key="run_inference_btn_disabled", type="primary", use_container_width=True, disabled=True)

    # ── Section 5: Model Outputs (full width, only after button click) ──
    st.html("<div style='margin-top: 30px;'></div>")
    st.html("<h3 style='font-size:18px; margin-bottom:15px; color:#DFB6B2; border-bottom: 1px solid #332041; padding-bottom: 8px;'>5. Model Outputs</h3>")
    
    if selected_trial and st.session_state.inference_run and mr_probs is not None and dl_probs is not None:
        # Model prediction bars — side by side, full width
        out_c1, out_c2 = st.columns([1, 1])
        with out_c1:
            st.html(render_bars("MiniRocket Prediction", mr_probs, "#854F6C", classes))
        with out_c2:
            st.html(render_bars("CNN-LSTM Prediction", dl_probs, "#F48FB1", classes))
        
        # Decision Summary + Uncertainty — side by side, full width
        mr_entropy = scipy.stats.entropy(mr_probs, base=2)
        dl_entropy = scipy.stats.entropy(dl_probs, base=2)
        
        status_color = "#00ff00" if dl_correct else "#ff4444"
        status_text = "CORRECT ✔" if dl_correct else "INCORRECT ✘"
        
        dec_c1, dec_c2 = st.columns([1, 1])
        with dec_c1:
            st.html(f"""
<div style="border:1px solid #332041; border-radius:10px; padding:20px; background:#190019;">
    <div style="color:#FBE4D8; font-size:14px; margin-bottom:15px; font-weight:bold;">✔️ Decision Summary</div>
    <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:8px;"><span style="color:#a196aa;">MiniRocket</span><span style="color:#FBE4D8;">{classes[mr_pred]}</span></div>
    <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:8px;"><span style="color:#a196aa;">CNN-LSTM</span><span style="color:#FBE4D8;">{classes[dl_pred]}</span></div>
    <hr style="border-color:#332041; margin:12px 0;"/>
    <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:8px;"><span style="color:#a196aa;">Final Decision</span><span style="color:#F48FB1; font-weight:bold;">{classes[dl_pred]}</span></div>
    <div style="display:flex; justify-content:space-between; font-size:13px;"><span style="color:#a196aa;">Status</span><span style="color:{status_color}; font-weight:bold;">{status_text}</span></div>
</div>
""")
        with dec_c2:
            st.html(f"""
<div style="border:1px solid #332041; border-radius:10px; padding:20px; background:#190019;">
    <div style="color:#FBE4D8; font-size:14px; margin-bottom:8px; font-weight:bold;">📉 Decision Uncertainty</div>
    <div style="color:#a196aa; font-size:11px; margin-bottom:15px;">(Shannon Entropy)</div>
    <div style="font-size:13px; color:#DFB6B2; margin-bottom:14px;">MiniRocket:<br/><b style="color:#FBE4D8; font-size:16px;">{mr_entropy:.2f} bits</b><br/><span style="font-size:10px;">(Lower = more confident)</span></div>
    <div style="font-size:13px; color:#DFB6B2;">CNN-LSTM:<br/><b style="color:#FBE4D8; font-size:16px;">{dl_entropy:.2f} bits</b></div>
</div>
""")
    elif selected_trial and not st.session_state.inference_run:
        st.html("<p style='color:#DFB6B2; font-size:14px; text-align:center; padding: 50px 0;'>Click <b>▶ Run Live Inference on Stream</b> above to classify this trial.</p>")
    else:
        st.html("<p style='color:#DFB6B2; font-size:14px; text-align:center; padding: 50px 0;'>Upload a file and select a trial first.</p>")
