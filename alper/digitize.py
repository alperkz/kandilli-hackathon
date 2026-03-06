"""
Aktinograf TIF Sayısallaştırma Pipeline
Kandilli Rasathanesi - KRDAE
Kullanım: python digitize.py <tif_dosyası_veya_klasör> [--output output_klasoru]
"""

import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import re
import argparse
from pathlib import Path

# ─── SABITLER ────────────────────────────────────────────
# Y ekseni: 0.0 → 2.0 cal/cm²/min (aktinograf skalası)
Y_VALUE_MIN = 0.0
Y_VALUE_MAX = 2.0

# X ekseni: saat 20 → saat 20 (ertesi gün) = 24 saat
# Saat 20'den başlayıp 20'ye dönüyor (gün ortasında gündüz)
HOUR_START = 20   # sol kenar saati
HOUR_TOTAL = 24   # toplam saat

# Renk maskeleri (HSV) — mor/mavi eğri
HSV_RANGES = [
    (np.array([120, 25, 25]), np.array([175, 255, 220])),  # mor
    (np.array([90,  25, 25]), np.array([125, 255, 220])),  # mavi
]

# ─── DOSYA ADI'NDAN TARİH ÇIKTAR ─────────────────────────
def parse_date_from_filename(fname):
    """1995_ARALIK-18.tif → 1995-12-18"""
    ay_map = {
        'OCAK':1,'SUBAT':2,'ŞUBAT':2,'MART':3,'NISAN':4,'NİSAN':4,
        'MAYIS':5,'HAZIRAN':6,'HAZİRAN':6,'HAZORAN':6,'HAZòRAN':6,
        'TEMMUZ':7,'AGUSTOS':8,'AĞUSTOS':8,'AGUSTOS':8,'A¶USTOS':8,
        'EYLUL':9,'EYLÜL':9,'EYLöL':9,'EKIM':10,'EKİM':10,'EKòM':10,
        'KASIM':11,'ARALIK':12
    }
    name = Path(fname).stem.upper()
    # Format: YYYY_AY-GUN veya YYYY_AY_GUN
    m = re.search(r'(\d{4})[_\-]([^_\-\d]+)[_\-](\d{1,2})', name)
    if m:
        year = int(m.group(1))
        month_str = m.group(2)
        day = int(m.group(3))
        month = ay_map.get(month_str, None)
        if month:
            return pd.Timestamp(year=year, month=month, day=day)
    return None

# ─── TEK TIF İŞLE ────────────────────────────────────────
def process_tif(tif_path, debug=False):
    pil_img = Image.open(tif_path).convert("RGB")
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # 1. Eğriyi maskele
    mask = np.zeros((h, w), dtype=np.uint8)
    for (lo, hi) in HSV_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    # 2. Gürültü temizle
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Her x sütununda eğrinin üst noktasını bul (yüksek değer = üstte)
    curve_xs, curve_ys = [], []
    for x in range(w):
        col_ys = np.where(mask[:, x] > 0)[0]
        if len(col_ys) > 0:
            curve_xs.append(x)
            curve_ys.append(int(col_ys.min()))  # en üst piksel

    if len(curve_xs) < 10:
        return None, None

    # numpy array'e çevir
    curve_xs = np.array(curve_xs)
    curve_ys = np.array(curve_ys)

    # Eksik gün tespiti
    x_span_raw = curve_xs.max() - curve_xs.min()
    if len(curve_xs) < 200 or x_span_raw < img.shape[1] * 0.3:
        return "EKSIK", None

    # 4. Grafik sınırlarını otomatik tespit
    img_w = img.shape[1]
    margin = int(img_w * 0.05)
    valid_mask = curve_xs > margin
    if valid_mask.sum() < 10:
        valid_mask = np.ones(len(curve_xs), dtype=bool)
    x_left  = int(curve_xs[valid_mask].min())
    x_right = int(curve_xs[valid_mask].max())
    # Sıfır hattı = en alttaki eğri pikseli (%98 percentile, y büyük = aşağı)
    y_zero  = int(np.percentile(curve_ys, 98))
    # Max hat = en üstteki eğri pikseli (%2 percentile)
    y_top   = int(np.percentile(curve_ys, 2))
    # Güvenlik: çok dar band ise genişlet
    if (y_zero - y_top) < 50:
        y_top  = max(0, y_top - 100)
        y_zero = min(img.shape[0] - 1, y_zero + 100)

    x_span = x_right - x_left
    y_span = y_zero  - y_top
    if x_span <= 0 or y_span <= 0:
        return None, None

    # Eğri X ekseninin yeterli kısmını kaplıyor mu? (eksik gün kontrolü)
    # Eğri pikselleri toplam genişliğin en az %30unu kaplamalı
    coverage = x_span / img_w
    if coverage < 0.30:
        print(f"    ⚠️  Eksik gün tespit edildi (kapsama: {coverage:.1%}) - atlanıyor")
        return None, None

    # 5. Piksel → Gerçek değer dönüşümü
    def px_to_value(y_px):
        # y_px: piksel (küçük y = yukarı = yüksek değer)
        ratio = (y_zero - y_px) / y_span
        ratio = np.clip(ratio, 0, 1)
        val = Y_VALUE_MIN + ratio * (Y_VALUE_MAX - Y_VALUE_MIN)
        return val if val > 0.05 else 0.0  # gece gürültüsünü sıfırla

    def px_to_hour(x_px):
        ratio = (x_px - x_left) / x_span
        hour_offset = ratio * HOUR_TOTAL
        hour = (HOUR_START + hour_offset) % 24
        return hour

    # 6. Saatlik örnekleme (her saat için medyan değer)
    hourly = {}
    for x, y in zip(curve_xs, curve_ys):
        hour = px_to_hour(x)
        val  = px_to_value(y)
        h_key = int(hour)
        if h_key not in hourly:
            hourly[h_key] = []
        hourly[h_key].append(val)

    hourly_avg = {h: np.median(v) for h, v in hourly.items()}

    # 7. Tüm 30 dakikalık nokta verisi
    records = []
    for x, y in zip(curve_xs, curve_ys):
        hour_f = px_to_hour(x)
        val    = px_to_value(y)
        records.append({'hour': round(hour_f, 3), 'radiation_cal_cm2_min': round(val, 4)})

    df_full = pd.DataFrame(records).sort_values('hour').reset_index(drop=True)

    # 8. Günlük toplam (integral tahmini - trapezoid)
    if len(df_full) > 2:
        daily_total = np.trapezoid(df_full['radiation_cal_cm2_min'],
                                   df_full['hour'] * 3600)  # cal/cm²
        daily_total = round(daily_total, 2)
    else:
        daily_total = None

    meta = {
        'x_left': x_left, 'x_right': x_right,
        'y_zero': y_zero,  'y_top': y_top,
        'daily_total_cal_cm2': daily_total,
        'points_detected': len(records)
    }

    # 9. Debug görsel
    if debug:
        debug_img = img_bgr.copy()
        for x, y in zip(curve_xs, curve_ys):
            cv2.circle(debug_img, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.line(debug_img, (x_left, y_zero),  (x_right, y_zero),  (0,0,255), 2)
        cv2.line(debug_img, (x_left, y_top),   (x_right, y_top),   (255,0,0), 2)
        debug_path = tif_path.replace('.tif', '_debug.png').replace('.TIF', '_debug.png')
        cv2.imwrite(debug_path, debug_img)

    return df_full, meta

# ─── TOPLU İŞLEME ────────────────────────────────────────
def process_folder(folder, output_dir):
    tif_files = sorted(Path(folder).rglob("*.tif")) + sorted(Path(folder).rglob("*.TIF"))
    print(f"{len(tif_files)} TIF dosyası bulundu.")

    all_daily = []
    os.makedirs(output_dir, exist_ok=True)

    for tif_path in tif_files:
        print(f"  İşleniyor: {tif_path.name} ...", end=" ")
        try:
            df, meta = process_tif(str(tif_path))
            if df is None:
                all_daily.append({
                    'date': parse_date_from_filename(tif_path.name),
                    'daily_total_cal_cm2': None,
                    'points_detected': 0,
                    'status': 'eksik_gun',
                    'file': tif_path.name
                })
                continue

            date = parse_date_from_filename(tif_path.name)
            date_str = date.strftime('%Y-%m-%d') if date else tif_path.stem

            # Saatlik CSV kaydet
            out_csv = os.path.join(output_dir, f"{date_str}_hourly.csv")
            df_hourly = df.groupby(df['hour'].astype(int))['radiation_cal_cm2_min'].median().reset_index()
            df_hourly.columns = ['hour', 'radiation_cal_cm2_min']
            df_hourly['date'] = date_str
            df_hourly[['date','hour','radiation_cal_cm2_min']].to_csv(out_csv, index=False)

            all_daily.append({
                'date': date_str,
                'daily_total_cal_cm2': meta['daily_total_cal_cm2'],
                'points_detected': meta['points_detected'],
                'status': 'ok',
                'file': tif_path.name
            })
            print(f"✅  {meta['points_detected']} nokta | günlük toplam: {meta['daily_total_cal_cm2']} cal/cm²")
        except Exception as e:
            print(f"❌ Hata: {e}")

    # Tüm günlük özet
    if all_daily:
        df_summary = pd.DataFrame(all_daily).sort_values('date')
        summary_path = os.path.join(output_dir, "ozet_gunluk.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n✅ Özet kaydedildi: {summary_path}")
        print(df_summary.to_string(index=False))

# ─── MAIN ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aktinograf TIF Sayısallaştırıcı')
    parser.add_argument('input',  help='TIF dosyası veya klasör')
    parser.add_argument('--output', default='output_csv', help='Çıktı klasörü')
    parser.add_argument('--debug', action='store_true', help='Debug görseli üret')
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        process_folder(str(inp), args.output)
    elif inp.suffix.lower() == '.tif':
        df, meta = process_tif(str(inp), debug=args.debug)
        if df is not None:
            os.makedirs(args.output, exist_ok=True)
            date = parse_date_from_filename(inp.name)
            date_str = date.strftime('%Y-%m-%d') if date else inp.stem
            out = os.path.join(args.output, f"{date_str}_raw.csv")
            df['date'] = date_str
            df.to_csv(out, index=False)
            print(f"✅ {meta['points_detected']} nokta çıkarıldı")
            print(f"   Günlük toplam: {meta['daily_total_cal_cm2']} cal/cm²")
            print(f"   CSV: {out}")
            print(meta)
        else:
            print("❌ Eğri bulunamadı")
    else:
        print("Hata: .tif dosyası veya klasör belirtin")