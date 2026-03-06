import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import os

st.set_page_config(
    page_title="Kandilli Rasathanesi | 115 Yıllık İklim",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262b40);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
    .header-title { font-size: 2.2rem; font-weight: 800; color: #fff; }
    .header-sub { font-size: 1rem; color: #aaa; margin-top: -8px; }
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        margin: 10px 0;
    }
    .highlight-stat {
        background: linear-gradient(135deg, #2d1f3d, #1e3a5f);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────────
@st.cache_data
def load_all_data():
    def parse_matrix(df, year_row, date_col, data_start_row, data_start_col):
        years = df.iloc[year_row, data_start_col:].values
        dates = df.iloc[data_start_row:, date_col].values
        data  = df.iloc[data_start_row:, data_start_col:].values
        records = []
        for i, date in enumerate(dates):
            for j, year in enumerate(years):
                try:
                    val = float(data[i, j])
                    yr  = int(float(year))
                    d   = pd.Timestamp(date)
                    records.append({'date': pd.Timestamp(year=yr, month=d.month, day=d.day), 'value': val})
                except:
                    pass
        return pd.DataFrame(records).dropna()

    # Sıcaklık
    wb_temp = pd.read_excel("Veriler_H/Sıcaklık/Uzunyıl-Max-Min-Ort-orj.xlsx", sheet_name=None, header=None)
    df_max = parse_matrix(wb_temp['Max'], 0, 0, 2, 2)
    df_min = parse_matrix(wb_temp['Min'], 0, 0, 2, 2)
    df_ort = parse_matrix(wb_temp['Ort'], 0, 0, 2, 2)

    # Yağış
    wb_y = pd.read_excel("Veriler_H/Yağış/1911-2023.xlsx", sheet_name=None, header=None)
    df_yagis = parse_matrix(wb_y['Günlük'], 0, 0, 1, 1)

    # Nem
    wb_n = pd.read_excel("Veriler_H/Nem/1911-2022-Nem.xlsx", sheet_name=None, header=None)
    df_nem = parse_matrix(wb_n['Nem 1911-'], 1, 0, 3, 2)

    # Basınç
    try:
        wb_b = pd.read_excel("Veriler_H/Basınç/Basınç 1912-2018Kasim.xlsx", sheet_name=None, header=None)
        first_sheet = list(wb_b.keys())[0]
        df_basinc = parse_matrix(wb_b[first_sheet], 0, 0, 2, 2)
        df_basinc.columns = ['date', 'basinc']
    except:
        df_basinc = pd.DataFrame(columns=['date', 'basinc'])

    # Hepsini birleştir
    df_max.columns   = ['date', 'max_temp']
    df_min.columns   = ['date', 'min_temp']
    df_ort.columns   = ['date', 'ort_temp']
    df_yagis.columns = ['date', 'yagis']
    df_nem.columns   = ['date', 'nem']

    df = df_max.merge(df_min,   on='date', how='outer') \
               .merge(df_ort,   on='date', how='outer') \
               .merge(df_yagis, on='date', how='outer') \
               .merge(df_nem,   on='date', how='outer')
    
    if not df_basinc.empty:
        df = df.merge(df_basinc, on='date', how='outer')

    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['doy']   = df['date'].dt.dayofyear
    df = df.sort_values('date').reset_index(drop=True)
    return df

@st.cache_data
def load_monthly_rainfall():
    """Uzun Yıllar sheet'inden aylık yağış istatistiklerini yükle."""
    wb = pd.read_excel("Veriler_H/Yağış/1911-2023.xlsx", sheet_name='Uzun Yıllar', header=None)
    # Sütun yapısı: col0=yıl, sonra her ay için 8 sütun:
    # TOP.YAGIS, Y.GUN.SAY, KAR.GUN, MAX, TOP.YAGIS(2), Y.GUN.SAY(2), KAR.GUN(2), MAX(2)
    # İlk 4 sütun aylık veriler, sonraki 4 farklı bir istatistik olabilir
    # Gerçek yapı: her ay için 8 sütun -> 12 ay, sonra 12 sütun MAX günlük
    # Doğru parse: col 0=year, sonra 4 sütunluk grup × 12 ay
    
    records = []
    ay_map = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:12}
    
    for row_idx in range(1, wb.shape[0]):  # row 0 = header
        year = wb.iloc[row_idx, 0]
        if pd.isna(year):
            continue
        year = int(year)
        
        for month_idx in range(12):
            base_col = 1 + month_idx * 4  # her ay 4 sütun: TOP.YAGIS, Y.GUN.SAY, KAR.GUN, MAX
            if base_col + 3 >= wb.shape[1]:
                # Son sütunları kontrol et
                continue
            
            top_yagis = wb.iloc[row_idx, base_col]
            yagis_gun = wb.iloc[row_idx, base_col + 1]
            kar_gun   = wb.iloc[row_idx, base_col + 2]
            max_yagis = wb.iloc[row_idx, base_col + 3]
            
            records.append({
                'year': year,
                'month': month_idx + 1,
                'top_yagis': pd.to_numeric(top_yagis, errors='coerce'),
                'yagis_gun': pd.to_numeric(yagis_gun, errors='coerce'),
                'kar_gun': pd.to_numeric(kar_gun, errors='coerce'),
                'max_yagis': pd.to_numeric(max_yagis, errors='coerce'),
            })
    
    df_aylik = pd.DataFrame(records)
    df_aylik = df_aylik.sort_values(['year', 'month']).reset_index(drop=True)
    return df_aylik

with st.spinner("Veri yükleniyor..."):
    df = load_all_data()
    df_aylik = load_monthly_rainfall()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌡️ Kandilli Rasathanesi")
    st.markdown("**KRDAE Meteoroloji Lab.**  \n1911 – 2024  \n*WMO Asırlık İstasyon*")
    st.divider()

    page = st.radio("📊 Sayfa", [
        "🏠 Genel Bakış",
        "🌡️ Sıcaklık Analizi",
        "🌧️ Yağış Analizi",
        "💧 Nem Analizi",
        "🔥 İklim Değişikliği",
        "📅 Yıl Karşılaştırma",
        "🔮 Tahminleme",
        "📸 Sayısallaştırma"
    ])

    st.divider()
    year_min, year_max = int(df['year'].min()), int(df['year'].max())
    yil_aralik = st.slider("Yıl Aralığı", year_min, year_max, (1950, year_max))

# Filtrele
dff = df[(df['year'] >= yil_aralik[0]) & (df['year'] <= yil_aralik[1])]

AY_ISIM = {1:'Oca',2:'Şub',3:'Mar',4:'Nis',5:'May',6:'Haz',
           7:'Tem',8:'Ağu',9:'Eyl',10:'Eki',11:'Kas',12:'Ara'}

COLORS = {
    'max_temp': '#ff6b6b',
    'min_temp': '#74b9ff',
    'ort_temp': '#fdcb6e',
    'yagis':    '#00cec9',
    'nem':      '#a29bfe',
    'basinc':   '#fd79a8',
}

# ─────────────────────────────────────────────
# SAYFA: GENEL BAKIŞ
# ─────────────────────────────────────────────
if page == "🏠 Genel Bakış":
    st.markdown('<p class="header-title">🌤️ Kandilli Rasathanesi</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">115 Yıllık İklim Arşivi · Boğaziçi Üniversitesi KRDAE</p>', unsafe_allow_html=True)
    st.divider()

    yillik = dff.groupby('year').agg(
        max_temp=('max_temp','mean'),
        min_temp=('min_temp','mean'),
        ort_temp=('ort_temp','mean'),
        yagis=('yagis','sum'),
        nem=('nem','mean')
    ).reset_index()

    # NaN'lı yılları temizle
    yillik = yillik.dropna(subset=['max_temp','min_temp','ort_temp'])

    c1, c2, c3, c4, c5 = st.columns(5)
    son_yil = yillik.iloc[-1]
    ilk_yil = yillik.iloc[0]

    def metric_card(col, label, val, delta, unit, color):
        val_str = f"{val:.1f}" if not np.isnan(val) else "N/A"
        delta_str = f"{delta:+.1f}" if not np.isnan(delta) else "N/A"
        col.markdown(f"""
        <div class="metric-card" style="border-color:{color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{val_str}{unit}</div>
            <div style="font-size:12px;color:#aaa;margin-top:4px">{delta_str}{unit} (ilk yıla göre)</div>
        </div>""", unsafe_allow_html=True)

    metric_card(c1, "Ort. Max Sıcaklık", son_yil.max_temp, son_yil.max_temp - ilk_yil.max_temp, "°C", "#ff6b6b")
    metric_card(c2, "Ort. Min Sıcaklık", son_yil.min_temp, son_yil.min_temp - ilk_yil.min_temp, "°C", "#74b9ff")
    metric_card(c3, "Ort. Sıcaklık",     son_yil.ort_temp, son_yil.ort_temp - ilk_yil.ort_temp, "°C", "#fdcb6e")
    metric_card(c4, "Yıllık Yağış",      son_yil.get('yagis', 0) or 0,    (son_yil.get('yagis',0) or 0) - (ilk_yil.get('yagis',0) or 0),    "mm", "#00cec9")
    metric_card(c5, "Ort. Nem",          son_yil.get('nem', 0) or 0,      (son_yil.get('nem',0) or 0) - (ilk_yil.get('nem',0) or 0),       "%",  "#a29bfe")

    st.divider()

    # Sıcaklık trendi
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yillik['year'], y=yillik['max_temp'],
        name='Max', line=dict(color='#ff6b6b', width=1.5), fill='tonexty',
        fillcolor='rgba(255,107,107,0.1)'))
    fig.add_trace(go.Scatter(x=yillik['year'], y=yillik['ort_temp'],
        name='Ortalama', line=dict(color='#fdcb6e', width=2)))
    fig.add_trace(go.Scatter(x=yillik['year'], y=yillik['min_temp'],
        name='Min', line=dict(color='#74b9ff', width=1.5)))

    # Trend çizgisi
    z = np.polyfit(yillik['year'], yillik['ort_temp'].fillna(yillik['ort_temp'].mean()), 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=yillik['year'], y=p(yillik['year']),
        name=f'Trend ({z[0]*10:.2f}°C/10 yıl)',
        line=dict(color='white', dash='dash', width=2)))

    fig.update_layout(
        title='Yıllık Ortalama Sıcaklık (1911–2024)',
        template='plotly_dark', height=380,
        legend=dict(orientation='h', y=1.1),
        xaxis_title='Yıl', yaxis_title='°C',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig2 = go.Figure()
        fig2.add_bar(x=yillik['year'], y=yillik['yagis'],
                     marker_color='#00cec9', opacity=0.7, name='Yağış')
        z2 = np.polyfit(yillik['year'].dropna(), yillik['yagis'].fillna(yillik['yagis'].mean()), 1)
        p2 = np.poly1d(z2)
        fig2.add_trace(go.Scatter(x=yillik['year'], y=p2(yillik['year']),
            name='Trend', line=dict(color='white', dash='dash')))
        fig2.update_layout(title='Yıllık Toplam Yağış', template='plotly_dark',
                           height=300, xaxis_title='Yıl', yaxis_title='mm')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=yillik['year'], y=yillik['nem'],
            fill='tozeroy', fillcolor='rgba(162,155,254,0.2)',
            line=dict(color='#a29bfe'), name='Nem'))
        fig3.update_layout(title='Yıllık Ortalama Bağıl Nem', template='plotly_dark',
                           height=300, xaxis_title='Yıl', yaxis_title='%')
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# SAYFA: SICAKLIK ANALİZİ
# ─────────────────────────────────────────────
elif page == "🌡️ Sıcaklık Analizi":
    st.markdown("## 🌡️ Sıcaklık Analizi")

    tab1, tab2, tab3 = st.tabs(["📈 Uzun Dönem Trend", "📅 Mevsimsel", "🗓️ Isı Haritası"])

    with tab1:
        yillik = dff.groupby('year').agg(
            max_temp=('max_temp','mean'),
            min_temp=('min_temp','mean'),
            ort_temp=('ort_temp','mean'),
        ).reset_index()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('Yıllık Ortalama Sıcaklıklar', 'Isı Farkı (Max - Min)'),
                            row_heights=[0.7, 0.3])

        fig.add_trace(go.Scatter(x=yillik['year'], y=yillik['max_temp'],
            name='Max', line=dict(color='#ff6b6b')), row=1, col=1)
        fig.add_trace(go.Scatter(x=yillik['year'], y=yillik['ort_temp'],
            name='Ortalama', line=dict(color='#fdcb6e', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=yillik['year'], y=yillik['min_temp'],
            name='Min', line=dict(color='#74b9ff')), row=1, col=1)

        fark = yillik['max_temp'] - yillik['min_temp']
        fig.add_trace(go.Bar(x=yillik['year'], y=fark,
            marker_color='#6c5ce7', name='Max-Min Fark', opacity=0.7), row=2, col=1)

        fig.update_layout(template='plotly_dark', height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Onluk ortalamalar
        dff2 = dff.copy()
        dff2['decade'] = (dff2['year'] // 10) * 10
        decade_avg = dff2.groupby('decade')['ort_temp'].mean().reset_index()
        fig_d = px.bar(decade_avg, x='decade', y='ort_temp',
                       color='ort_temp', color_continuous_scale='RdYlBu_r',
                       title='On Yıllık Ortalama Sıcaklık',
                       labels={'decade': 'On Yıl', 'ort_temp': '°C'})
        fig_d.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig_d, use_container_width=True)

    with tab2:
        aylik = dff.groupby('month').agg(
            max_temp=('max_temp','mean'),
            min_temp=('min_temp','mean'),
            ort_temp=('ort_temp','mean'),
        ).reset_index()
        aylik['ay_isim'] = aylik['month'].map(AY_ISIM)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=aylik['ay_isim'], y=aylik['max_temp'],
            fill='tonexty', name='Max', line=dict(color='#ff6b6b'),
            fillcolor='rgba(255,107,107,0.15)'))
        fig.add_trace(go.Scatter(x=aylik['ay_isim'], y=aylik['ort_temp'],
            name='Ortalama', line=dict(color='#fdcb6e', width=2.5)))
        fig.add_trace(go.Scatter(x=aylik['ay_isim'], y=aylik['min_temp'],
            name='Min', line=dict(color='#74b9ff'),
            fill='tonexty', fillcolor='rgba(116,185,255,0.1)'))
        fig.update_layout(title='Aylık Ortalama Sıcaklık Profili',
                          template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Isı haritası: yıl × ay
        pivot = dff.groupby(['year','month'])['ort_temp'].mean().reset_index()
        pivot_tbl = pivot.pivot(index='year', columns='month', values='ort_temp')
        pivot_tbl.columns = [AY_ISIM[m] for m in pivot_tbl.columns]

        fig = px.imshow(pivot_tbl.T,
                        color_continuous_scale='RdBu_r',
                        title='Aylık Ortalama Sıcaklık Isı Haritası (Yıl × Ay)',
                        labels=dict(x='Yıl', y='Ay', color='°C'),
                        aspect='auto')
        fig.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# SAYFA: YAĞIŞ ANALİZİ (5 Tab)
# ─────────────────────────────────────────────
elif page == "🌧️ Yağış Analizi":
    st.markdown("## 🌧️ Yağış Analizi")

    # Aylık veriyi filtrele
    df_ay_filt = df_aylik[(df_aylik['year'] >= yil_aralik[0]) & (df_aylik['year'] <= yil_aralik[1])]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Genel & Trend", "🏜️ SPI Kuraklık", "❄️ Yağmur vs Kar",
        "⚡ Aşırı Yağış", "📊 Mevsimsel Kayma & Kapsama"
    ])

    # ─── TAB 1: GENEL & TREND ────────────────────
    with tab1:
        yillik_yagis = df_ay_filt.groupby('year')['top_yagis'].sum().reset_index()
        yillik_yagis = yillik_yagis[yillik_yagis['top_yagis'] > 0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Yıl Ortalaması", f"{yillik_yagis['top_yagis'].mean():.0f} mm")
        if len(yillik_yagis) > 0:
            en_yagisli = yillik_yagis.loc[yillik_yagis['top_yagis'].idxmax()]
            col2.metric("En Yağışlı Yıl", f"{int(en_yagisli['year'])} — {en_yagisli['top_yagis']:.0f} mm")
            en_kurak = yillik_yagis.loc[yillik_yagis['top_yagis'].idxmin()]
            col3.metric("En Kurak Yıl", f"{int(en_kurak['year'])} — {en_kurak['top_yagis']:.0f} mm")

        fig1 = go.Figure()
        fig1.add_bar(x=yillik_yagis['year'], y=yillik_yagis['top_yagis'],
                     marker_color='#00cec9', opacity=0.7, name='Yıllık Toplam')
        if len(yillik_yagis) > 3:
            z = np.polyfit(yillik_yagis['year'], yillik_yagis['top_yagis'], 1)
            p = np.poly1d(z)
            fig1.add_trace(go.Scatter(x=yillik_yagis['year'], y=p(yillik_yagis['year']),
                name=f'Trend ({z[0]*10:.1f} mm/10yıl)', line=dict(color='white', dash='dash', width=2)))
        fig1.update_layout(title='Yıllık Toplam Yağış (Aylık Veriden)',
                           template='plotly_dark', height=350, xaxis_title='Yıl', yaxis_title='mm')
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            aylik_ort = df_ay_filt.groupby('month')['top_yagis'].mean().reset_index()
            aylik_ort['ay_isim'] = aylik_ort['month'].map(AY_ISIM)
            fig2 = px.bar(aylik_ort, x='ay_isim', y='top_yagis',
                          color='top_yagis', color_continuous_scale='Blues',
                          title='Aylık Ortalama Yağış',
                          labels={'ay_isim': 'Ay', 'top_yagis': 'mm'})
            fig2.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            pivot_y = df_ay_filt.pivot_table(index='year', columns='month', values='top_yagis')
            if not pivot_y.empty:
                pivot_y.columns = [AY_ISIM.get(m, m) for m in pivot_y.columns]
                fig3 = px.imshow(pivot_y.T, color_continuous_scale='Blues',
                                 title='Yağış Isı Haritası (Yıl × Ay)',
                                 labels=dict(x='Yıl', y='Ay', color='mm'), aspect='auto')
                fig3.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig3, use_container_width=True)

    # ─── TAB 2: SPI KURAKLIK İNDEKSİ ────────────
    with tab2:
        st.markdown("### 🏜️ Standardized Precipitation Index (SPI)")
        st.markdown("""
        <div class="prediction-box">
            <p style="color:#aaa">SPI, belirli bir dönemdeki yağışın uzun dönem ortalamasından ne kadar saptığını 
            standart sapma cinsinden gösterir. Negatif değerler kuraklığı, pozitif değerler ıslak dönemleri ifade eder.</p>
        </div>
        """, unsafe_allow_html=True)

        spi_period = st.selectbox("SPI Periyodu", [3, 6, 12], index=2,
                                   format_func=lambda x: f"SPI-{x} ({x} Aylık)")

        # SPI hesaplama
        aylik_ts = df_aylik.sort_values(['year', 'month']).copy()
        aylik_ts['top_yagis'] = aylik_ts['top_yagis'].fillna(0)

        # Rolling sum
        aylik_ts['rolling_yagis'] = aylik_ts['top_yagis'].rolling(window=spi_period, min_periods=spi_period).sum()
        aylik_ts = aylik_ts.dropna(subset=['rolling_yagis'])

        if len(aylik_ts) > 30:
            # Basit z-skor SPI (Pearson Type III yerine Gaussian yaklaşımı)
            mean_val = aylik_ts['rolling_yagis'].mean()
            std_val  = aylik_ts['rolling_yagis'].std()
            if std_val > 0:
                aylik_ts['spi'] = (aylik_ts['rolling_yagis'] - mean_val) / std_val
            else:
                aylik_ts['spi'] = 0

            aylik_ts['date_label'] = aylik_ts.apply(
                lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)

            # Filtreleme
            spi_filt = aylik_ts[(aylik_ts['year'] >= yil_aralik[0]) & (aylik_ts['year'] <= yil_aralik[1])]

            fig_spi = go.Figure()
            pos = spi_filt[spi_filt['spi'] >= 0]
            neg = spi_filt[spi_filt['spi'] < 0]
            fig_spi.add_bar(x=pos['date_label'], y=pos['spi'], marker_color='#0984e3', name='Islak', opacity=0.8)
            fig_spi.add_bar(x=neg['date_label'], y=neg['spi'], marker_color='#d63031', name='Kurak', opacity=0.8)
            fig_spi.add_hline(y=-1, line_dash='dot', line_color='#e17055', opacity=0.7,
                              annotation_text='Orta Kuraklık')
            fig_spi.add_hline(y=-2, line_dash='dot', line_color='#d63031', opacity=0.7,
                              annotation_text='Şiddetli Kuraklık')
            fig_spi.add_hline(y=0, line_dash='solid', line_color='white', opacity=0.3)
            fig_spi.update_layout(
                title=f'SPI-{spi_period} Kuraklık İndeksi',
                template='plotly_dark', height=400,
                xaxis_title='Tarih', yaxis_title='SPI Değeri',
                barmode='relative', showlegend=True,
                xaxis=dict(dtick=max(1, len(spi_filt) // 30))
            )
            st.plotly_chart(fig_spi, use_container_width=True)

            # İstatistikler
            col1, col2, col3, col4 = st.columns(4)
            kurak_aylar = (spi_filt['spi'] < -1).sum()
            siddetli_kurak = (spi_filt['spi'] < -2).sum()
            islak_aylar = (spi_filt['spi'] > 1).sum()
            col1.metric("Kurak Ay (SPI<-1)", f"{kurak_aylar}")
            col2.metric("Şiddetli Kurak (SPI<-2)", f"{siddetli_kurak}")
            col3.metric("Islak Ay (SPI>1)", f"{islak_aylar}")
            col4.metric("En Kurak Dönem", f"SPI: {spi_filt['spi'].min():.2f}")

            # SPI sınıflandırma tablosu
            st.markdown("#### SPI Sınıflandırma")
            spi_cats = pd.DataFrame({
                'Kategori': ['Aşırı Islak', 'Çok Islak', 'Orta Islak', 'Normal', 'Orta Kurak', 'Şiddetli Kurak', 'Aşırı Kurak'],
                'SPI Aralığı': ['≥ 2.0', '1.5 – 1.99', '1.0 – 1.49', '-0.99 – 0.99', '-1.0 – -1.49', '-1.5 – -1.99', '≤ -2.0'],
                'Renk': ['🔵🔵🔵', '🔵🔵', '🔵', '⚪', '🟠', '🔴', '🔴🔴']
            })
            st.dataframe(spi_cats, use_container_width=True, hide_index=True)

    # ─── TAB 3: YAĞMUR VS KAR ───────────────────
    with tab3:
        st.markdown("### ❄️ Yağmur vs Kar Analizi")
        st.markdown("""
        <div class="prediction-box">
            <p style="color:#aaa">İklim değişikliğinin en net göstergelerinden biri: karlı gün sayısının azalması 
            ve kar olarak düşen yağışın yağmura dönüşmesi. Bu, şehir altyapısı ve su yönetimi için kritik bir değişimdir.</p>
        </div>
        """, unsafe_allow_html=True)

        # Yıllık kar günü toplamı
        yillik_kar = df_ay_filt.groupby('year').agg(
            toplam_kar_gun=('kar_gun', 'sum'),
            toplam_yagis_gun=('yagis_gun', 'sum'),
            toplam_yagis=('top_yagis', 'sum')
        ).reset_index()
        yillik_kar = yillik_kar[yillik_kar['toplam_yagis_gun'] > 0]
        yillik_kar['kar_orani'] = (yillik_kar['toplam_kar_gun'] / yillik_kar['toplam_yagis_gun'] * 100).round(1)

        col1, col2, col3 = st.columns(3)
        if len(yillik_kar) > 0:
            col1.metric("Ort. Yıllık Kar Günü", f"{yillik_kar['toplam_kar_gun'].mean():.0f} gün")
            ilk_10 = yillik_kar.head(10)['toplam_kar_gun'].mean()
            son_10 = yillik_kar.tail(10)['toplam_kar_gun'].mean()
            col2.metric("İlk 10 Yıl Ort.", f"{ilk_10:.0f} gün")
            col3.metric("Son 10 Yıl Ort.", f"{son_10:.0f} gün", f"{son_10 - ilk_10:+.0f} gün")

        fig_kar = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=('Yıllık Kar Günü Sayısı', 'Kar/Yağış Günü Oranı (%)'),
                                row_heights=[0.6, 0.4])

        fig_kar.add_bar(x=yillik_kar['year'], y=yillik_kar['toplam_kar_gun'],
                        marker_color='#74b9ff', opacity=0.8, name='Kar Günü', row=1, col=1)
        if len(yillik_kar) > 3:
            z_kar = np.polyfit(yillik_kar['year'], yillik_kar['toplam_kar_gun'], 1)
            p_kar = np.poly1d(z_kar)
            fig_kar.add_trace(go.Scatter(x=yillik_kar['year'], y=p_kar(yillik_kar['year']),
                name=f'Trend ({z_kar[0]*10:.1f} gün/10yıl)',
                line=dict(color='white', dash='dash', width=2)), row=1, col=1)

        fig_kar.add_trace(go.Scatter(x=yillik_kar['year'], y=yillik_kar['kar_orani'],
            fill='tozeroy', fillcolor='rgba(116,185,255,0.2)',
            line=dict(color='#a29bfe', width=2), name='Kar Oranı %'), row=2, col=1)
        if len(yillik_kar) > 3:
            z_oran = np.polyfit(yillik_kar['year'], yillik_kar['kar_orani'], 1)
            p_oran = np.poly1d(z_oran)
            fig_kar.add_trace(go.Scatter(x=yillik_kar['year'], y=p_oran(yillik_kar['year']),
                line=dict(color='white', dash='dash'), name='Oran Trend'), row=2, col=1)

        fig_kar.update_layout(template='plotly_dark', height=550, hovermode='x unified')
        st.plotly_chart(fig_kar, use_container_width=True)

        # Onluk dönem analizi
        st.markdown("#### 📊 Onluk Dönemlere Göre Kar Günü Dağılımı")
        df_kar_decade = df_ay_filt.copy()
        df_kar_decade['decade'] = (df_kar_decade['year'] // 10) * 10
        decade_kar = df_kar_decade.groupby(['decade', 'month']).agg(
            kar_gun=('kar_gun', 'mean')
        ).reset_index()
        decade_kar['ay_isim'] = decade_kar['month'].map(AY_ISIM)

        fig_kar_heat = px.imshow(
            decade_kar.pivot(index='decade', columns='month', values='kar_gun').rename(
                columns=AY_ISIM),
            color_continuous_scale='Blues',
            title='Ortalama Kar Günü (Onluk Dönem × Ay)',
            labels=dict(x='Ay', y='On Yıl', color='Kar Günü'),
            aspect='auto'
        )
        fig_kar_heat.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig_kar_heat, use_container_width=True)

    # ─── TAB 4: AŞIRI YAĞIŞ OLAYLARI ────────────
    with tab4:
        st.markdown("### ⚡ Aşırı Yağış Olayları")
        st.markdown("""
        <div class="prediction-box">
            <p style="color:#aaa">Günlük maksimum yağış değerleri ve yoğun yağış günlerinin sıklığı, 
            iklim değişikliğinin sel ve taşkın riski üzerindeki etkisini gösterir.</p>
        </div>
        """, unsafe_allow_html=True)

        # Aylık MAX değerleri + yağışlı gün sayıları yıllık bazda
        yillik_max = df_ay_filt.groupby('year').agg(
            max_gunluk=('max_yagis', 'max'),
            toplam_yagis_gun=('yagis_gun', 'sum'),
            toplam_yagis=('top_yagis', 'sum')
        ).reset_index()
        yillik_max = yillik_max.dropna(subset=['max_gunluk'])

        col1, col2, col3 = st.columns(3)
        if len(yillik_max) > 0:
            col1.metric("Ort. Yıllık Maks Günlük", f"{yillik_max['max_gunluk'].mean():.1f} mm")
            rekor = yillik_max.loc[yillik_max['max_gunluk'].idxmax()]
            col2.metric("Rekor Günlük Yağış", f"{rekor['max_gunluk']:.1f} mm ({int(rekor['year'])})")
            col3.metric("Ort. Yıllık Yağışlı Gün", f"{yillik_max['toplam_yagis_gun'].mean():.0f} gün")

        # Yıllık max günlük yağış
        fig_max = go.Figure()
        fig_max.add_bar(x=yillik_max['year'], y=yillik_max['max_gunluk'],
                        marker_color='#e17055', opacity=0.8, name='Max Günlük Yağış')
        if len(yillik_max) > 3:
            z_max = np.polyfit(yillik_max['year'], yillik_max['max_gunluk'], 1)
            p_max = np.poly1d(z_max)
            fig_max.add_trace(go.Scatter(x=yillik_max['year'], y=p_max(yillik_max['year']),
                name=f'Trend ({z_max[0]*10:.1f} mm/10yıl)',
                line=dict(color='white', dash='dash', width=2)))
        fig_max.add_hline(y=50, line_dash='dot', line_color='#d63031', opacity=0.5,
                          annotation_text='Şiddetli (>50mm)')
        fig_max.add_hline(y=30, line_dash='dot', line_color='#e17055', opacity=0.5,
                          annotation_text='Kuvvetli (>30mm)')
        fig_max.update_layout(title='Yıllık Maksimum Günlük Yağış',
                              template='plotly_dark', height=380,
                              xaxis_title='Yıl', yaxis_title='mm')
        st.plotly_chart(fig_max, use_container_width=True)

        # Aşırı yağış günleri trendi (günlük Excel verisinden)
        st.markdown("#### 📈 Günlük Veriden Aşırı Yağış Günleri")
        col1, col2 = st.columns(2)

        with col1:
            # >20mm gün sayıları
            yogun_20 = dff[dff['yagis'] > 20].groupby('year').size().reset_index(name='gun_sayisi')
            fig_20 = go.Figure()
            fig_20.add_bar(x=yogun_20['year'], y=yogun_20['gun_sayisi'],
                           marker_color='#0984e3', opacity=0.8, name='>20mm')
            if len(yogun_20) > 3:
                z20 = np.polyfit(yogun_20['year'], yogun_20['gun_sayisi'], 1)
                p20 = np.poly1d(z20)
                fig_20.add_trace(go.Scatter(x=yogun_20['year'], y=p20(yogun_20['year']),
                    line=dict(color='white', dash='dash'), name='Trend'))
            fig_20.update_layout(title='Günlük Yağış > 20mm (gün/yıl)',
                                 template='plotly_dark', height=300)
            st.plotly_chart(fig_20, use_container_width=True)

        with col2:
            # >50mm gün sayıları
            yogun_50 = dff[dff['yagis'] > 50].groupby('year').size().reset_index(name='gun_sayisi')
            fig_50 = go.Figure()
            fig_50.add_bar(x=yogun_50['year'], y=yogun_50['gun_sayisi'],
                           marker_color='#d63031', opacity=0.8, name='>50mm')
            if len(yogun_50) > 3:
                z50 = np.polyfit(yogun_50['year'], yogun_50['gun_sayisi'], 1)
                p50 = np.poly1d(z50)
                fig_50.add_trace(go.Scatter(x=yogun_50['year'], y=p50(yogun_50['year']),
                    line=dict(color='white', dash='dash'), name='Trend'))
            fig_50.update_layout(title='Günlük Yağış > 50mm (gün/yıl)',
                                 template='plotly_dark', height=300)
            st.plotly_chart(fig_50, use_container_width=True)

        # Yıllık yağışlı gün sayısı trendi
        fig_gun = go.Figure()
        fig_gun.add_trace(go.Scatter(x=yillik_max['year'], y=yillik_max['toplam_yagis_gun'],
            fill='tozeroy', fillcolor='rgba(0,206,201,0.2)',
            line=dict(color='#00cec9', width=2), name='Yağışlı Gün'))
        if len(yillik_max) > 3:
            z_gun = np.polyfit(yillik_max['year'], yillik_max['toplam_yagis_gun'].fillna(0), 1)
            p_gun = np.poly1d(z_gun)
            fig_gun.add_trace(go.Scatter(x=yillik_max['year'], y=p_gun(yillik_max['year']),
                name=f'Trend ({z_gun[0]*10:.1f} gün/10yıl)',
                line=dict(color='white', dash='dash')))
        fig_gun.update_layout(title='Yıllık Toplam Yağışlı Gün Sayısı',
                              template='plotly_dark', height=300,
                              xaxis_title='Yıl', yaxis_title='Gün')
        st.plotly_chart(fig_gun, use_container_width=True)

    # ─── TAB 5: MEVSİMSEL KAYMA & KAPSAMA ───────
    with tab5:
        st.markdown("### 📊 Mevsimsel Kayma Analizi")

        # Onyıllık dönemlere göre aylık yağış dağılımı
        df_shift = df_aylik.copy()
        df_shift['decade'] = (df_shift['year'] // 10) * 10

        # Son 4 onluk dönem karşılaştırması
        decades_to_show = sorted(df_shift['decade'].unique())
        if len(decades_to_show) > 6:
            selected_decades = [decades_to_show[0], decades_to_show[len(decades_to_show)//3],
                                decades_to_show[2*len(decades_to_show)//3], decades_to_show[-1]]
        else:
            selected_decades = decades_to_show

        decade_monthly = df_shift.groupby(['decade', 'month'])['top_yagis'].mean().reset_index()

        # Radar chart
        fig_radar = go.Figure()
        decade_colors = ['#74b9ff', '#a29bfe', '#fdcb6e', '#ff6b6b', '#00cec9', '#fd79a8']

        for i, dec in enumerate(selected_decades):
            dec_data = decade_monthly[decade_monthly['decade'] == dec].sort_values('month')
            if len(dec_data) == 12:
                values = dec_data['top_yagis'].tolist() + [dec_data['top_yagis'].iloc[0]]  # close the loop
                months = [AY_ISIM[m] for m in range(1, 13)] + [AY_ISIM[1]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=months,
                    name=f"{int(dec)}'ler",
                    line=dict(color=decade_colors[i % len(decade_colors)], width=2),
                    fill='toself', fillcolor=f'rgba(0,0,0,0.05)'
                ))

        fig_radar.update_layout(
            title='Mevsimsel Yağış Dağılımı (Onluk Dönemler)',
            template='plotly_dark', height=500,
            polar=dict(radialaxis=dict(visible=True, gridcolor='#333')),
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Mevsimsel oran değişimi
        st.markdown("#### 🔄 Mevsimsel Oran Değişimi")
        mevsim_map = {12: 'Kış', 1: 'Kış', 2: 'Kış', 3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
                      6: 'Yaz', 7: 'Yaz', 8: 'Yaz', 9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'}
        df_mevsim = df_ay_filt.copy()
        df_mevsim['mevsim'] = df_mevsim['month'].map(mevsim_map)
        mevsim_yillik = df_mevsim.groupby(['year', 'mevsim'])['top_yagis'].sum().reset_index()
        yillik_toplam = mevsim_yillik.groupby('year')['top_yagis'].sum().reset_index()
        yillik_toplam.columns = ['year', 'yillik_toplam']
        mevsim_yillik = mevsim_yillik.merge(yillik_toplam, on='year')
        mevsim_yillik['oran'] = (mevsim_yillik['top_yagis'] / mevsim_yillik['yillik_toplam'] * 100).round(1)
        mevsim_yillik = mevsim_yillik[mevsim_yillik['yillik_toplam'] > 0]

        mevsim_colors = {'Kış': '#74b9ff', 'İlkbahar': '#55efc4', 'Yaz': '#fdcb6e', 'Sonbahar': '#e17055'}
        fig_mevsim = go.Figure()
        for mevsim in ['Kış', 'İlkbahar', 'Yaz', 'Sonbahar']:
            mv = mevsim_yillik[mevsim_yillik['mevsim'] == mevsim]
            fig_mevsim.add_trace(go.Scatter(
                x=mv['year'], y=mv['oran'],
                name=mevsim, stackgroup='one',
                fillcolor=mevsim_colors.get(mevsim, '#888'),
                line=dict(width=0.5, color=mevsim_colors.get(mevsim, '#888'))
            ))
        fig_mevsim.update_layout(
            title='Mevsimsel Yağış Payı (%)',
            template='plotly_dark', height=380,
            xaxis_title='Yıl', yaxis_title='Pay (%)',
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_mevsim, use_container_width=True)

        # Veri kapsama
        st.markdown("---")
        st.markdown("### 📋 Veri Kapsama Haritası")
        st.markdown("""
        <div class="prediction-box">
            <p style="color:#aaa">Aşağıdaki harita, günlük yağış Excel verisindeki boşlukları gösterir. 
            Koyu renkler veri olan günleri, açık renkler eksik verileri temsil eder. 
            Bu boşlukların bir kısmı plüvyogram dijitalizasyonu ile doldurulabilir.</p>
        </div>
        """, unsafe_allow_html=True)

        # Aylık doluluğu hesapla
        kapsama = df_ay_filt.copy()
        # top_yagis dolu mu?
        kapsama['dolu'] = kapsama['top_yagis'].notna().astype(int)
        kapsama_pivot = kapsama.pivot_table(index='year', columns='month', values='dolu', fill_value=0)
        kapsama_pivot.columns = [AY_ISIM.get(m, m) for m in kapsama_pivot.columns]

        fig_kap = px.imshow(kapsama_pivot.T,
                            color_continuous_scale=['#1a1a2e', '#00cec9'],
                            title='Aylık Yağış Verisi Mevcudiyeti (Uzun Yıllar Tablosu)',
                            labels=dict(x='Yıl', y='Ay', color='Mevcut'),
                            aspect='auto')
        fig_kap.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig_kap, use_container_width=True)

        # Günlük veriden daha detaylı kapsama
        gunluk_kapsama = dff.groupby(['year', 'month'])['yagis'].apply(
            lambda x: x.notna().sum()).reset_index()
        gunluk_kapsama.columns = ['year', 'month', 'dolu_gun']

        gunluk_pivot = gunluk_kapsama.pivot_table(index='year', columns='month', values='dolu_gun', fill_value=0)
        if not gunluk_pivot.empty:
            gunluk_pivot.columns = [AY_ISIM.get(m, m) for m in gunluk_pivot.columns]
            fig_kap2 = px.imshow(gunluk_pivot.T,
                                 color_continuous_scale='Viridis',
                                 title='Günlük Yağış Verisi Doluluğu (gün/ay)',
                                 labels=dict(x='Yıl', y='Ay', color='Dolu Gün'),
                                 aspect='auto')
            fig_kap2.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig_kap2, use_container_width=True)

        # Yıllık kapsama yüzdesi
        yillik_kapsama = df_aylik.groupby('year').agg(
            dolu=('top_yagis', lambda x: x.notna().sum()),
            toplam=('top_yagis', 'size')
        ).reset_index()
        yillik_kapsama['oran'] = (yillik_kapsama['dolu'] / yillik_kapsama['toplam'] * 100).round(1)

        fig_oran = go.Figure()
        fig_oran.add_bar(x=yillik_kapsama['year'], y=yillik_kapsama['oran'],
                         marker_color=np.where(yillik_kapsama['oran'] >= 80, '#00cec9', '#e17055'),
                         opacity=0.8)
        fig_oran.add_hline(y=80, line_dash='dot', line_color='#00cec9', opacity=0.5,
                           annotation_text='%80 Eşik')
        fig_oran.update_layout(title='Aylık Veri Kapsama Oranı (Yıllık %)',
                               template='plotly_dark', height=250,
                               xaxis_title='Yıl', yaxis_title='%')
        st.plotly_chart(fig_oran, use_container_width=True)

# ─────────────────────────────────────────────
# SAYFA: NEM ANALİZİ
# ─────────────────────────────────────────────
elif page == "💧 Nem Analizi":
    st.markdown("## 💧 Nem Analizi")

    yillik_nem = dff.groupby('year')['nem'].mean().reset_index()
    aylik_nem  = dff.groupby('month')['nem'].mean().reset_index()
    aylik_nem['ay_isim'] = aylik_nem['month'].map(AY_ISIM)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yillik_nem['year'], y=yillik_nem['nem'],
        fill='tozeroy', fillcolor='rgba(162,155,254,0.2)',
        line=dict(color='#a29bfe', width=2), name='Ortalama Nem'))
    z = np.polyfit(yillik_nem['year'].dropna(), yillik_nem['nem'].fillna(yillik_nem['nem'].mean()), 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=yillik_nem['year'], y=p(yillik_nem['year']),
        name=f'Trend ({z[0]*10:.2f}%/10yıl)',
        line=dict(color='white', dash='dash', width=2)))
    fig.update_layout(title='Yıllık Ortalama Bağıl Nem',
                      template='plotly_dark', height=380)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(aylik_nem, x='ay_isim', y='nem',
                      color='nem', color_continuous_scale='Purples',
                      title='Aylık Ortalama Nem',
                      labels={'ay_isim':'Ay','nem':'%'})
        fig2.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        pivot_n = dff.groupby(['year','month'])['nem'].mean().reset_index()
        pivot_tbl = pivot_n.pivot(index='year', columns='month', values='nem')
        pivot_tbl.columns = [AY_ISIM[m] for m in pivot_tbl.columns]
        fig3 = px.imshow(pivot_tbl.T, color_continuous_scale='Purples',
                         title='Nem Isı Haritası',
                         labels=dict(x='Yıl',y='Ay',color='%'), aspect='auto')
        fig3.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# SAYFA: İKLİM DEĞİŞİKLİĞİ
# ─────────────────────────────────────────────
elif page == "🔥 İklim Değişikliği":
    st.markdown("## 🔥 İklim Değişikliği Göstergeleri")

    yillik = dff.groupby('year').agg(
        ort_temp=('ort_temp','mean'),
        yagis=('yagis','sum'),
        nem=('nem','mean')
    ).reset_index().dropna(subset=['ort_temp'])

    ref = yillik[(yillik['year']>=1961)&(yillik['year']<=1990)]['ort_temp'].mean()
    yillik['anomali'] = yillik['ort_temp'] - ref

    fig = go.Figure()
    pos = yillik[yillik['anomali'] >= 0]
    neg = yillik[yillik['anomali'] <  0]
    fig.add_bar(x=pos['year'], y=pos['anomali'], marker_color='#ff6b6b', name='Pozitif')
    fig.add_bar(x=neg['year'], y=neg['anomali'], marker_color='#74b9ff', name='Negatif')
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
    fig.update_layout(
        title=f'Sıcaklık Anomalisi (Referans: 1961-1990 ortalaması = {ref:.1f}°C)',
        template='plotly_dark', height=380,
        xaxis_title='Yıl', yaxis_title='Anomali (°C)', barmode='relative'
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    trend_z = np.polyfit(yillik['year'], yillik['ort_temp'], 1)
    col1.metric("Isınma Trendi", f"{trend_z[0]*10:.2f}°C / 10 yıl", "↑ İklim ısınıyor")
    hot_years = (yillik['anomali'] > 0).sum()
    col2.metric("Pozitif Anomali Yıl Sayısı", f"{hot_years} / {len(yillik)}")
    last20 = yillik[yillik['year']>=2000]['ort_temp'].mean()
    first20 = yillik[yillik['year']<1930]['ort_temp'].mean()
    col3.metric("2000'ler vs 1920'ler", f"+{last20-first20:.2f}°C")

    # Yaz ve Kış
    yaz = dff[dff['month'].isin([6,7,8])].groupby('year')['max_temp'].mean().reset_index()
    kis = dff[dff['month'].isin([12,1,2])].groupby('year')['min_temp'].mean().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=yaz['year'], y=yaz['max_temp'],
            fill='tozeroy', fillcolor='rgba(255,107,107,0.2)',
            line=dict(color='#ff6b6b'), name='Yaz Max Ort'))
        z2 = np.polyfit(yaz['year'], yaz['max_temp'], 1)
        p2 = np.poly1d(z2)
        fig2.add_trace(go.Scatter(x=yaz['year'], y=p2(yaz['year']),
            line=dict(color='white', dash='dash'), name='Trend'))
        fig2.update_layout(title='Yaz Dönemi Max Sıcaklıklar (Haz-Ağu)',
                           template='plotly_dark', height=320)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=kis['year'], y=kis['min_temp'],
            fill='tozeroy', fillcolor='rgba(116,185,255,0.2)',
            line=dict(color='#74b9ff'), name='Kış Min Ort'))
        z3 = np.polyfit(kis['year'], kis['min_temp'], 1)
        p3 = np.poly1d(z3)
        fig3.add_trace(go.Scatter(x=kis['year'], y=p3(kis['year']),
            line=dict(color='white', dash='dash'), name='Trend'))
        fig3.update_layout(title='Kış Dönemi Min Sıcaklıklar (Ara-Şub)',
                           template='plotly_dark', height=320)
        st.plotly_chart(fig3, use_container_width=True)

    # Aşırı hava olayları
    st.markdown("### 🌡️ Aşırı Hava Olayları Analizi")
    col1, col2 = st.columns(2)
    
    with col1:
        # Yıllık sıcak gün sayısı (max > 30°C)
        sicak_gunler = dff[dff['max_temp'] > 30].groupby('year').size().reset_index(name='sicak_gun')
        fig4 = go.Figure()
        fig4.add_bar(x=sicak_gunler['year'], y=sicak_gunler['sicak_gun'],
                     marker_color='#e17055', opacity=0.8)
        if len(sicak_gunler) > 3:
            z4 = np.polyfit(sicak_gunler['year'], sicak_gunler['sicak_gun'], 1)
            p4 = np.poly1d(z4)
            fig4.add_trace(go.Scatter(x=sicak_gunler['year'], y=p4(sicak_gunler['year']),
                line=dict(color='white', dash='dash'), name='Trend'))
        fig4.update_layout(title='Sıcak Gün Sayısı (Max > 30°C / yıl)',
                           template='plotly_dark', height=300)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        # Yoğun yağış günleri (>20mm/gün)
        yogun_yagis = dff[dff['yagis'] > 20].groupby('year').size().reset_index(name='yogun_gun')
        fig5 = go.Figure()
        fig5.add_bar(x=yogun_yagis['year'], y=yogun_yagis['yogun_gun'],
                     marker_color='#0984e3', opacity=0.8)
        if len(yogun_yagis) > 3:
            z5 = np.polyfit(yogun_yagis['year'], yogun_yagis['yogun_gun'], 1)
            p5 = np.poly1d(z5)
            fig5.add_trace(go.Scatter(x=yogun_yagis['year'], y=p5(yogun_yagis['year']),
                line=dict(color='white', dash='dash'), name='Trend'))
        fig5.update_layout(title='Yoğun Yağış Günleri (>20mm/gün/yıl)',
                           template='plotly_dark', height=300)
        st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────
# SAYFA: YIL KARŞILAŞTIRMA
# ─────────────────────────────────────────────
elif page == "📅 Yıl Karşılaştırma":
    st.markdown("## 📅 Yıl Karşılaştırma")

    available_years = sorted(df['year'].dropna().unique().astype(int).tolist())
    col1, col2 = st.columns(2)
    yil1 = col1.selectbox("1. Yıl", available_years, index=available_years.index(1950) if 1950 in available_years else 0)
    yil2 = col2.selectbox("2. Yıl", available_years, index=available_years.index(2020) if 2020 in available_years else -1)

    d1 = df[df['year']==yil1].groupby('month').agg(
        max_temp=('max_temp','mean'), min_temp=('min_temp','mean'),
        ort_temp=('ort_temp','mean'), yagis=('yagis','sum'),
        nem=('nem','mean')).reset_index()
    d2 = df[df['year']==yil2].groupby('month').agg(
        max_temp=('max_temp','mean'), min_temp=('min_temp','mean'),
        ort_temp=('ort_temp','mean'), yagis=('yagis','sum'),
        nem=('nem','mean')).reset_index()

    for df_x in [d1, d2]:
        df_x['ay_isim'] = df_x['month'].map(AY_ISIM)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d1['ay_isim'], y=d1['ort_temp'],
        name=str(yil1), line=dict(color='#fdcb6e', width=2.5)))
    fig.add_trace(go.Scatter(x=d2['ay_isim'], y=d2['ort_temp'],
        name=str(yil2), line=dict(color='#ff6b6b', width=2.5, dash='dash')))
    fig.update_layout(title='Aylık Ortalama Sıcaklık Karşılaştırması',
                      template='plotly_dark', height=350)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_bar(x=d1['ay_isim'], y=d1['yagis'],
                     name=str(yil1), marker_color='#00cec9', opacity=0.7)
        fig2.add_bar(x=d2['ay_isim'], y=d2['yagis'],
                     name=str(yil2), marker_color='#55efc4', opacity=0.7)
        fig2.update_layout(title='Aylık Yağış', template='plotly_dark',
                           height=320, barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=d1['ay_isim'], y=d1['nem'],
            name=str(yil1), fill='tozeroy',
            fillcolor='rgba(162,155,254,0.15)', line=dict(color='#a29bfe')))
        fig3.add_trace(go.Scatter(x=d2['ay_isim'], y=d2['nem'],
            name=str(yil2), line=dict(color='#fd79a8', dash='dash')))
        fig3.update_layout(title='Aylık Nem', template='plotly_dark', height=320)
        st.plotly_chart(fig3, use_container_width=True)

    # Delta tablosu
    st.markdown(f"### 📊 {yil1} vs {yil2} Fark Tablosu")
    merged = d1[['ay_isim','ort_temp','yagis','nem']].merge(
        d2[['ay_isim','ort_temp','yagis','nem']], on='ay_isim', suffixes=(f'_{yil1}',f'_{yil2}'))
    merged[f'Δ Sıcaklık'] = (merged[f'ort_temp_{yil2}'] - merged[f'ort_temp_{yil1}']).round(1)
    merged[f'Δ Yağış']    = (merged[f'yagis_{yil2}']    - merged[f'yagis_{yil1}']).round(1)
    merged[f'Δ Nem']      = (merged[f'nem_{yil2}']      - merged[f'nem_{yil1}']).round(1)
    st.dataframe(merged.rename(columns={'ay_isim':'Ay'}), use_container_width=True)

# ─────────────────────────────────────────────
# SAYFA: TAHMİNLEME
# ─────────────────────────────────────────────
elif page == "🔮 Tahminleme":
    st.markdown("## 🔮 İklim Tahminleme")
    st.markdown("""
    <div class="prediction-box">
        <h4>📈 115 Yıllık Veriden Gelecek Tahmini</h4>
        <p style="color:#aaa">KRDAE'nin 1911'den itibaren kesintisiz ölçüm verisi kullanılarak 
        istatistiksel ve makine öğrenimi modelleriyle İstanbul'un iklim geleceği tahmin edilmektedir.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Lineer Trend", "🧠 Polinom Regresyon", "📊 Korelasyon Matrisi", "🌧️ Yağış ARIMA"])

    with tab1:
        st.markdown("### Uzun Dönem Lineer Trend ve Projeksiyon")
        
        metric_choice = st.selectbox("Parametre", ["Ortalama Sıcaklık", "Yağış", "Nem"])
        
        if metric_choice == "Ortalama Sıcaklık":
            yillik = df.groupby('year')['ort_temp'].mean().dropna().reset_index()
            col_name = 'ort_temp'
            unit = '°C'
            color = '#fdcb6e'
        elif metric_choice == "Yağış":
            yillik = df.groupby('year')['yagis'].sum().dropna().reset_index()
            col_name = 'yagis'
            unit = 'mm'
            color = '#00cec9'
        else:
            yillik = df.groupby('year')['nem'].mean().dropna().reset_index()
            col_name = 'nem'
            unit = '%'
            color = '#a29bfe'
        
        # Son yıllar eksik veri içerebilir, filtrele
        yillik = yillik[yillik[col_name].notna() & (yillik[col_name] > 0)]

        X = yillik['year'].values
        y = yillik[col_name].values

        # Lineer regresyon
        z1 = np.polyfit(X, y, 1)
        p1 = np.poly1d(z1)

        # Gelecek tahmin
        future_years = np.arange(X.max() + 1, 2060)
        all_years = np.concatenate([X, future_years])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X, y=y, mode='lines', name='Gerçek Veri',
            line=dict(color=color, width=1.5)))
        fig.add_trace(go.Scatter(x=all_years, y=p1(all_years),
            name=f'Trend: {z1[0]*10:.3f}{unit}/10yıl',
            line=dict(color='white', dash='dash', width=2)))
        
        # Güven aralığı (basit)
        residuals = y - p1(X)
        std_residual = np.std(residuals)
        fig.add_trace(go.Scatter(x=np.concatenate([future_years, future_years[::-1]]),
            y=np.concatenate([p1(future_years) + 2*std_residual,
                              (p1(future_years) - 2*std_residual)[::-1]]),
            fill='toself', fillcolor='rgba(255,255,255,0.05)',
            line=dict(color='rgba(255,255,255,0)'), name='95% Güven Aralığı'))

        fig.add_vline(x=X.max(), line_dash='dot', line_color='gray', opacity=0.5)
        fig.add_annotation(x=X.max()+1, y=p1(X.max()), text="Projeksiyon →",
                          showarrow=False, font=dict(color='gray'))

        fig.update_layout(template='plotly_dark', height=450,
                          title=f'{metric_choice} — Lineer Trend ve Projeksiyon (2060)',
                          xaxis_title='Yıl', yaxis_title=f'{unit}',
                          hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Tahmin özet
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"""<div class="highlight-stat">
            <div style="color:#888;font-size:11px">2030 TAHMİNİ</div>
            <div style="font-size:22px;font-weight:700;color:{color}">{p1(2030):.1f}{unit}</div>
        </div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="highlight-stat">
            <div style="color:#888;font-size:11px">2040 TAHMİNİ</div>
            <div style="font-size:22px;font-weight:700;color:{color}">{p1(2040):.1f}{unit}</div>
        </div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="highlight-stat">
            <div style="color:#888;font-size:11px">2050 TAHMİNİ</div>
            <div style="font-size:22px;font-weight:700;color:{color}">{p1(2050):.1f}{unit}</div>
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Polinom Regresyon ile Tahmin")
        
        yillik_temp = df.groupby('year')['ort_temp'].mean().dropna().reset_index()
        yillik_temp = yillik_temp[yillik_temp['ort_temp'] > 0]  # sıfır/eksik filtrele
        X_t = yillik_temp['year'].values
        y_t = yillik_temp['ort_temp'].values

        degree = st.slider("Polinom Derecesi", 1, 5, 2)
        z_poly = np.polyfit(X_t, y_t, degree)
        p_poly = np.poly1d(z_poly)
        
        # Polinom tahminini makul aralıkta tut
        def clamped_poly(x):
            pred = p_poly(x)
            return np.clip(pred, 0, 30)  # 0-30°C arası mantıklı

        future = np.arange(X_t.max()+1, 2060)
        all_x = np.concatenate([X_t, future])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_t, y=y_t, mode='lines', name='Gerçek',
            line=dict(color='#fdcb6e', width=1.5)))
        fig.add_trace(go.Scatter(x=all_x, y=clamped_poly(all_x),
            name=f'Polinom (derece={degree})',
            line=dict(color='#ff6b6b', width=2.5)))
        fig.add_vline(x=X_t.max(), line_dash='dot', line_color='gray')
        fig.update_layout(template='plotly_dark', height=400,
                          title='Polinom Regresyon Tahmin',
                          xaxis_title='Yıl', yaxis_title='°C')
        st.plotly_chart(fig, use_container_width=True)

        # R² hesapla
        from sklearn.metrics import r2_score
        y_pred = clamped_poly(X_t)
        r2 = r2_score(y_t, y_pred)
        st.metric("R² Skoru", f"{r2:.4f}")

    with tab3:
        st.markdown("### Meteorolojik Değişkenler Arası Korelasyon")
        
        corr_df = dff[['max_temp','min_temp','ort_temp','yagis','nem']].dropna()
        if 'basinc' in dff.columns:
            corr_df = dff[['max_temp','min_temp','ort_temp','yagis','nem','basinc']].dropna()
        
        corr_matrix = corr_df.corr()
        labels = {
            'max_temp': 'Max Sıcaklık', 'min_temp': 'Min Sıcaklık',
            'ort_temp': 'Ort Sıcaklık', 'yagis': 'Yağış',
            'nem': 'Nem', 'basinc': 'Basınç'
        }
        corr_matrix.index = [labels.get(c, c) for c in corr_matrix.index]
        corr_matrix.columns = [labels.get(c, c) for c in corr_matrix.columns]

        fig = px.imshow(corr_matrix, 
                        color_continuous_scale='RdBu_r', 
                        zmin=-1, zmax=1,
                        title='Korelasyon Matrisi',
                        text_auto='.2f')
        fig.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### 🌧️ Aylık Yağış ARIMA Tahmini")
        st.markdown("""
        <div class="prediction-box">
            <p style="color:#aaa">ARIMA (AutoRegressive Integrated Moving Average) modeli, zaman serisi verilerinde 
            geçmiş değerlerin geleceği tahmin etmede kullanıldığı istatistiksel bir yöntemdir. 
            Burada aylık toplam yağış verisinden gelecek 24 ay tahmini yapılmaktadır.</p>
        </div>
        """, unsafe_allow_html=True)

        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            import warnings
            warnings.filterwarnings('ignore')

            # Aylık yağış zaman serisi hazırla
            aylik_arima = df_aylik[['year', 'month', 'top_yagis']].dropna().copy()
            aylik_arima = aylik_arima.sort_values(['year', 'month'])
            aylik_arima['date'] = pd.to_datetime(
                aylik_arima['year'].astype(int).astype(str) + '-' + aylik_arima['month'].astype(int).astype(str) + '-01'
            )
            aylik_arima = aylik_arima.set_index('date')['top_yagis']
            
            # Eksik ayları doldurmak gerekebilir
            full_idx = pd.date_range(start=aylik_arima.index.min(), end=aylik_arima.index.max(), freq='MS')
            aylik_arima = aylik_arima.reindex(full_idx)
            aylik_arima = aylik_arima.interpolate(method='linear')
            aylik_arima = aylik_arima.dropna()

            # Tahmin periyodu
            forecast_months = st.slider("Tahmin Periyodu (Ay)", 12, 60, 24)

            if len(aylik_arima) > 36:
                with st.spinner("ARIMA modeli eğitiliyor..."):
                    # SARIMAX ile mevsimsel model dene
                    try:
                        model = SARIMAX(aylik_arima, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                        enforce_stationarity=False, enforce_invertibility=False)
                        results = model.fit(disp=False, maxiter=200)
                        model_name = "SARIMAX(1,1,1)(1,1,1,12)"
                    except Exception:
                        # Basit ARIMA fallback
                        model = ARIMA(aylik_arima, order=(2, 1, 2))
                        results = model.fit()
                        model_name = "ARIMA(2,1,2)"

                    # Tahmin
                    forecast = results.get_forecast(steps=forecast_months)
                    forecast_mean = forecast.predicted_mean
                    forecast_ci = forecast.conf_int(alpha=0.05)
                    
                    # Negatif tahminleri sıfırla
                    forecast_mean = forecast_mean.clip(lower=0)
                    forecast_ci = forecast_ci.clip(lower=0)

                    # Grafik
                    fig_arima = go.Figure()

                    # Son 10 yıl gerçek veri
                    son_yillar = aylik_arima.tail(120)
                    fig_arima.add_trace(go.Scatter(
                        x=son_yillar.index, y=son_yillar.values,
                        name='Gerçek Veri', line=dict(color='#00cec9', width=1.5)
                    ))

                    # Tahmin
                    fig_arima.add_trace(go.Scatter(
                        x=forecast_mean.index, y=forecast_mean.values,
                        name=f'Tahmin ({model_name})',
                        line=dict(color='#ff6b6b', width=2.5)
                    ))

                    # Güven aralığı
                    ci_cols = forecast_ci.columns
                    fig_arima.add_trace(go.Scatter(
                        x=np.concatenate([forecast_ci.index, forecast_ci.index[::-1]]),
                        y=np.concatenate([forecast_ci.iloc[:, 1].values,
                                          forecast_ci.iloc[:, 0].values[::-1]]),
                        fill='toself', fillcolor='rgba(255,107,107,0.15)',
                        line=dict(color='rgba(255,107,107,0)'), name='95% Güven Aralığı'
                    ))

                    fig_arima.add_vline(x=aylik_arima.index[-1], line_dash='dot', line_color='gray', opacity=0.5)
                    fig_arima.add_annotation(x=forecast_mean.index[0], y=forecast_mean.max(),
                                            text="← Tahmin Başlangıcı", showarrow=False,
                                            font=dict(color='gray', size=11))

                    fig_arima.update_layout(
                        title=f'Aylık Yağış Tahmini — {model_name}',
                        template='plotly_dark', height=450,
                        xaxis_title='Tarih', yaxis_title='mm',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_arima, use_container_width=True)

                    # Model istatistikleri
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("AIC", f"{results.aic:.0f}")
                    col2.metric("BIC", f"{results.bic:.0f}")
                    col3.metric("Eğitim Veri Uzunluğu", f"{len(aylik_arima)} ay")
                    annual_forecast = forecast_mean.sum() / (forecast_months / 12)
                    col4.metric("Yıllık Tahmin", f"{annual_forecast:.0f} mm")

                    # Aylık tahmin tablosu
                    st.markdown("#### 📊 Aylık Tahmin Detayı")
                    forecast_df = pd.DataFrame({
                        'Tarih': forecast_mean.index.strftime('%Y-%m'),
                        'Tahmin (mm)': forecast_mean.values.round(1),
                        'Alt Sınır': forecast_ci.iloc[:, 0].values.round(1),
                        'Üst Sınır': forecast_ci.iloc[:, 1].values.round(1)
                    })
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        except ImportError:
            st.warning("⚠️ statsmodels kütüphanesi bulunamadı. `pip install statsmodels` ile yükleyin.")

# ─────────────────────────────────────────────
# SAYFA: SAYISALLAŞTIRMA
# ─────────────────────────────────────────────
elif page == "📸 Sayısallaştırma":
    st.markdown("## 📸 Sayısallaştırma (Digitization)")
    st.markdown("""
    <div class="prediction-box">
        <h4>📐 Analog Grafik → Dijital Veri</h4>
        <p style="color:#aaa">KRDAE Meteoroloji Laboratuvarı'nın 115 yıllık analog grafik kayıtları
        (termogram, barogram, aktinogram) Computer Vision teknikleriyle sayısallaştırılmaktadır.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🌡️ Termogram Pipeline", "📊 Arşiv İstatistikleri"])
    
    with tab1:
        st.markdown("### Termogram Sayısallaştırma Pipeline'ı")
        st.markdown("""
        **İşlem Adımları:**
        1. 🎨 HSV renk maskeleme (mavi/siyah mürekkep tespiti)
        2. 📏 Grid çizgileri tespiti (yatay morfolojik analiz)
        3. 📐 Y ekseni otomatik kalibrasyonu (major grid = 10°C aralıkları)
        4. 📈 Eğri çıkarımı (piksel → °C dönüşümü)
        5. 💾 Saatlik CSV çıktısı
        """)
        
        # Termogram arşiv istatistikleri
        termogram_dir = Path("Termogram/TERMOGRAM-1_1911-2005")
        if termogram_dir.exists():
            years = sorted([d.name for d in termogram_dir.iterdir() if d.is_dir() and d.name.isdigit()])
            st.info(f"📁 Termogram arşivi: **{len(years)} yıl** ({years[0]}–{years[-1]})")
            
            # Sayısallaştırılmış veri varsa göster
            output_dir = Path("alper/output_termogram")
            if output_dir.exists():
                csv_files = list(output_dir.glob("*_hourly.csv"))
                summary_file = output_dir / "termogram_ozet.csv"
                
                if csv_files:
                    st.success(f"✅ **{len(csv_files)} gün** başarıyla sayısallaştırıldı!")
                    
                    # Örnek gün göster
                    sample_csv = sorted(csv_files)[len(csv_files)//2]  # ortadaki
                    sample_df = pd.read_csv(sample_csv)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sample_df['hour'], y=sample_df['temperature_c'],
                        mode='lines+markers',
                        line=dict(color='#ff6b6b', width=2),
                        marker=dict(size=6),
                        name='Sayısallaştırılmış Sıcaklık'
                    ))
                    fig.update_layout(
                        title=f'Örnek Sayısallaştırma: {sample_df["date"].iloc[0]}',
                        template='plotly_dark', height=350,
                        xaxis_title='Saat', yaxis_title='°C',
                        xaxis=dict(dtick=2)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Özet tablo
                    if summary_file.exists():
                        ozet = pd.read_csv(summary_file)
                        ozet_ok = ozet[ozet['status'] == 'ok']
                        if not ozet_ok.empty:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Başarı Oranı", f"{len(ozet_ok)}/{len(ozet)} ({100*len(ozet_ok)/len(ozet):.0f}%)")
                            col2.metric("Ort. Max Sıcaklık", f"{ozet_ok['max_temp'].mean():.1f}°C")
                            col3.metric("Ort. Min Sıcaklık", f"{ozet_ok['min_temp'].mean():.1f}°C")
                else:
                    st.warning("Sayısallaştırma henüz çalıştırılmadı. `digitize_termogram.py` ile çalıştırın.")
        
        st.markdown("---")
        st.markdown("### 📂 Mevcut Grafik Arşivleri")
        
        archives = {
            "🌡️ Termogram": ("Termogram/TERMOGRAM-1_1911-2005", "1911-2005", "Sıcaklık günlük kaydı"),
            "📊 Barograph": ("Barograph/BAROGRAF GÜNLÜK 1975-2004", "1975-2004", "Basınç günlük kaydı"),
            "☀️ Aktinograph": ("aktinograph", "1975-2004", "Güneş radyasyonu kaydı"),
            "🌊 Mareogram": ("Mareogram/Mereogram-VANİKÖY HAFTALIK", "Çeşitli", "Deniz seviyesi kaydı"),
            "🌧️ Plüvyogram": ("Yağış/YAĞIŞ PLÜVYOGRAM_3_1965-2025", "1965-2025", "Yağış kaydı"),
            "📜 Osmanlıca MicroBarograph": ("MicroBarograf-Osmanlica/MİKROBAROGRAM_OSMANLI TÜRKÇESİ_1911-1913", "1911-1913", "Osmanlı Türkçesi basınç kaydı"),
        }
        
        cols = st.columns(3)
        for i, (name, (path, years, desc)) in enumerate(archives.items()):
            with cols[i % 3]:
                exists = Path(path).exists()
                status = "✅" if exists else "❌"
                st.markdown(f"""
                <div class="metric-card" style="border-color:#6c5ce7">
                    <div class="metric-label">{name}</div>
                    <div style="font-size:14px;color:#ddd;margin-top:4px">{years}</div>
                    <div style="font-size:12px;color:#888;margin-top:2px">{desc} {status}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Arşiv Kapsamı")
        
        # Veri mevcudiyeti ısı haritası
        st.markdown("**Yapılandırılmış Veri Mevcudiyeti (Excel Verileri)**")
        years_range = range(1911, 2025)
        params = ['max_temp', 'min_temp', 'ort_temp', 'yagis', 'nem']
        param_names = ['Max Sıcaklık', 'Min Sıcaklık', 'Ort Sıcaklık', 'Yağış', 'Nem']
        
        availability = []
        for yr in years_range:
            row = {'year': yr}
            yr_data = df[df['year'] == yr]
            for p, pn in zip(params, param_names):
                if p in yr_data.columns:
                    row[pn] = yr_data[p].notna().sum()
                else:
                    row[pn] = 0
            availability.append(row)
        
        avail_df = pd.DataFrame(availability).set_index('year')
        
        fig = px.imshow(avail_df.T, 
                        color_continuous_scale='Viridis',
                        title='Veri Noktası Sayısı (Yıl × Parametre)',
                        labels=dict(x='Yıl', y='Parametre', color='Gün'),
                        aspect='auto')
        fig.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align:center;color:#555;font-size:12px">
Boğaziçi Üniversitesi · KRDAE Meteoroloji Laboratuvarı · 
WMO Asırlık İstasyon Sertifikası · 1911–2024 · 
<b>115 Yıllık İklim Verisi Hackathon</b>
</div>
""", unsafe_allow_html=True)