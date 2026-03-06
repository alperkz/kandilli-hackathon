import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['doy']   = df['date'].dt.dayofyear
    df = df.sort_values('date').reset_index(drop=True)
    return df

with st.spinner("Veri yükleniyor..."):
    df = load_all_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/tr/5/5a/Bo%C4%9Fazi%C3%A7i_%C3%9Cniversitesi_logosu.png", width=80)
    st.markdown("### 🌡️ Kandilli Rasathanesi")
    st.markdown("**KRDAE Meteoroloji Lab.**  \n1911 – 2024")
    st.divider()

    page = st.radio("📊 Sayfa", [
        "🏠 Genel Bakış",
        "🌡️ Sıcaklık Analizi",
        "🌧️ Yağış Analizi",
        "💧 Nem Analizi",
        "🔥 İklim Değişikliği",
        "📅 Yıl Karşılaştırma"
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

    c1, c2, c3, c4, c5 = st.columns(5)
    son_yil = yillik.iloc[-1]
    ilk_yil = yillik.iloc[0]

    def metric_card(col, label, val, delta, unit, color):
        col.markdown(f"""
        <div class="metric-card" style="border-color:{color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{val:.1f}{unit}</div>
            <div style="font-size:12px;color:#aaa;margin-top:4px">{delta:+.1f}{unit} (ilk yıla göre)</div>
        </div>""", unsafe_allow_html=True)

    metric_card(c1, "Ort. Max Sıcaklık", son_yil.max_temp, son_yil.max_temp - ilk_yil.max_temp, "°C", "#ff6b6b")
    metric_card(c2, "Ort. Min Sıcaklık", son_yil.min_temp, son_yil.min_temp - ilk_yil.min_temp, "°C", "#74b9ff")
    metric_card(c3, "Ort. Sıcaklık",     son_yil.ort_temp, son_yil.ort_temp - ilk_yil.ort_temp, "°C", "#fdcb6e")
    metric_card(c4, "Yıllık Yağış",      son_yil.yagis,    son_yil.yagis    - ilk_yil.yagis,    "mm", "#00cec9")
    metric_card(c5, "Ort. Nem",          son_yil.nem,      son_yil.nem      - ilk_yil.nem,       "%",  "#a29bfe")

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
# SAYFA: YAĞIŞ ANALİZİ
# ─────────────────────────────────────────────
elif page == "🌧️ Yağış Analizi":
    st.markdown("## 🌧️ Yağış Analizi")

    yillik_yagis = dff.groupby('year')['yagis'].sum().reset_index()
    aylik_yagis  = dff.groupby('month')['yagis'].mean().reset_index()
    aylik_yagis['ay_isim'] = aylik_yagis['month'].map(AY_ISIM)

    col1, col2 = st.columns(2)
    col1.metric("Toplam Yıl Ortalaması", f"{yillik_yagis['yagis'].mean():.0f} mm")
    col2.metric("En Yağışlı Yıl",
                f"{int(yillik_yagis.loc[yillik_yagis['yagis'].idxmax(),'year'])} — {yillik_yagis['yagis'].max():.0f} mm")

    fig1 = go.Figure()
    fig1.add_bar(x=yillik_yagis['year'], y=yillik_yagis['yagis'],
                 marker_color='#00cec9', opacity=0.7)
    z = np.polyfit(yillik_yagis['year'], yillik_yagis['yagis'], 1)
    p = np.poly1d(z)
    fig1.add_trace(go.Scatter(x=yillik_yagis['year'], y=p(yillik_yagis['year']),
        name='Trend', line=dict(color='white', dash='dash', width=2)))
    fig1.update_layout(title='Yıllık Toplam Yağış', template='plotly_dark',
                       height=350, xaxis_title='Yıl', yaxis_title='mm')
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(aylik_yagis, x='ay_isim', y='yagis',
                      color='yagis', color_continuous_scale='Blues',
                      title='Aylık Ortalama Yağış',
                      labels={'ay_isim':'Ay','yagis':'mm'})
        fig2.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Yağış ısı haritası
        pivot_y = dff.groupby(['year','month'])['yagis'].sum().reset_index()
        pivot_tbl = pivot_y.pivot(index='year', columns='month', values='yagis')
        pivot_tbl.columns = [AY_ISIM[m] for m in pivot_tbl.columns]
        fig3 = px.imshow(pivot_tbl.T, color_continuous_scale='Blues',
                         title='Yağış Isı Haritası',
                         labels=dict(x='Yıl', y='Ay', color='mm'), aspect='auto')
        fig3.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig3, use_container_width=True)

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

    # Anomali: 1961-1990 referans dönemi
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

    # Istatistik
    col1, col2, col3 = st.columns(3)
    trend_z = np.polyfit(yillik['year'], yillik['ort_temp'], 1)
    col1.metric("Isınma Trendi", f"{trend_z[0]*10:.2f}°C / 10 yıl", "↑ İklim ısınıyor")
    hot_years = (yillik['anomali'] > 0).sum()
    col2.metric("Pozitif Anomali Yıl Sayısı", f"{hot_years} / {len(yillik)}")
    last20 = yillik[yillik['year']>=2000]['ort_temp'].mean()
    first20 = yillik[yillik['year']<1930]['ort_temp'].mean()
    col3.metric("2000'ler vs 1920'ler", f"+{last20-first20:.2f}°C")

    # Yaz sıcaklıkları
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

# ─────────────────────────────────────────────
# SAYFA: YIL KARŞILAŞTIRMA
# ─────────────────────────────────────────────
elif page == "📅 Yıl Karşılaştırma":
    st.markdown("## 📅 Yıl Karşılaştırma")

    available_years = sorted(df['year'].dropna().unique().astype(int).tolist())
    col1, col2 = st.columns(2)
    yil1 = col1.selectbox("1. Yıl", available_years, index=available_years.index(1950))
    yil2 = col2.selectbox("2. Yıl", available_years, index=available_years.index(2020))

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

# Footer
st.divider()
st.markdown("""
<div style="text-align:center;color:#555;font-size:12px">
Boğaziçi Üniversitesi · KRDAE Meteoroloji Laboratuvarı · 
WMO Asırlık İstasyon Sertifikası · 1911–2024
</div>
""", unsafe_allow_html=True)