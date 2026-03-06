"""
Termogram TIF Sayısallaştırma Pipeline
Kandilli Rasathanesi - KRDAE
Sıcaklık grafik kayıtlarından (thermograph) eğri çıkarımı.

Y ekseni: 0 → 40 °C (bazı kağıtlarda -10 → 40)
X ekseni: saat 7 → saat 7 (ertesi gün) = 24 saat

Kullanım:
    python digitize_termogram.py <tif_dosyası_veya_klasör> [--output output_csv]
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
# Y ekseni: Termogram kağıdında sıcaklık aralığı
Y_VALUE_MIN = -10.0   # alt sınır (°C) - güvenlik marjı
Y_VALUE_MAX = 50.0    # üst sınır (°C) - güvenlik marjı

# X ekseni: saat başlangıcı
HOUR_START = 7    # sol kenar saati (termogramlar genelde 7'den başlar)
HOUR_TOTAL = 24   # toplam saat

# ─── DOSYA ADI'NDAN TARİH ÇIKAR ─────────────────────────
def parse_date_from_filename(fname):
    """1975_OCAK-01.tif → 1975-01-01"""
    ay_map = {
        'OCAK':1,'SUBAT':2,'ŞUBAT':2,'MART':3,'NISAN':4,'NİSAN':4,
        'MAYIS':5,'HAZIRAN':6,'HAZİRAN':6,'HAZORAN':6,'HAZòRAN':6,
        'TEMMUZ':7,'AGUSTOS':8,'AĞUSTOS':8,'A¶USTOS':8,
        'EYLUL':9,'EYLÜL':9,'EYLöL':9,'EKIM':10,'EKİM':10,'EKòM':10,
        'KASIM':11,'ARALIK':12
    }
    name = Path(fname).stem.upper()
    m = re.search(r'(\d{4})[_\-]([^_\-\d]+)[_\-](\d{1,2})', name)
    if m:
        year = int(m.group(1))
        month_str = m.group(2)
        day = int(m.group(3))
        month = ay_map.get(month_str, None)
        if month:
            try:
                return pd.Timestamp(year=year, month=month, day=day)
            except:
                pass
    return None

# ─── GRİD ÇIZGILERI ve Y EKSENİ ─────────────────────────
def detect_grid_and_calibrate(img_gray, h, w):
    """Yatay grid çizgilerini tespit et ve Y ekseni kalibrasyonu yap.
    
    Termogram kağıtlarında:
    - Kalın çizgiler: her 10°C (major)
    - İnce çizgiler: her 5°C veya 2°C (minor)
    - Tipik aralık: -10 ile 40°C arası
    
    Returns: (y_top_px, temp_top), (y_bot_px, temp_bot) or None pairs
    """
    # Uzun yatay yapıları bul
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    row_sums = horizontal_lines.sum(axis=1) / 255
    threshold = w * 0.25
    line_rows = np.where(row_sums > threshold)[0]
    
    if len(line_rows) < 2:
        return None, None
    
    # Yakın satırları grupla (±8 piksel = aynı çizgi)
    groups = []
    current_group = [line_rows[0]]
    for i in range(1, len(line_rows)):
        if line_rows[i] - line_rows[i-1] <= 8:
            current_group.append(line_rows[i])
        else:
            groups.append(int(np.mean(current_group)))
            current_group = [line_rows[i]]
    groups.append(int(np.mean(current_group)))
    
    if len(groups) < 3:
        return None, None
    
    # Aralıkları hesapla
    spacings = np.diff(groups)
    
    # Çok küçük aralıkları filtrele (gürültü, <20px)
    valid_indices = [0]
    for i in range(len(spacings)):
        if spacings[i] >= 20:
            valid_indices.append(i + 1)
    groups = [groups[i] for i in valid_indices]
    
    if len(groups) < 3:
        return None, None

    spacings = np.diff(groups)
    
    # BÜYÜK aralıkları bul (10°C major grid çizgileri)
    # Büyük aralık = en büyük spacing'e yakın olanlar
    median_spacing = np.median(spacings)
    large_spacings = spacings[spacings >= median_spacing * 1.3]
    small_spacings = spacings[spacings < median_spacing * 1.3]
    
    if len(large_spacings) >= 2:
        # Büyük aralıklar var → bunlar 10°C aralıkları
        px_per_10deg = np.median(large_spacings)
        px_per_deg = px_per_10deg / 10.0
        
        # Büyük aralığa sahip çizgileri (major grid) bul
        major_lines = [groups[0]]
        for i in range(len(spacings)):
            if spacings[i] >= median_spacing * 1.3:
                major_lines.append(groups[i + 1])
        
    elif len(small_spacings) >= 3 and len(large_spacings) == 0:
        # Sadece küçük aralıklar → bunlar 5°C aralıkları
        px_per_5deg = np.median(spacings)
        px_per_deg = px_per_5deg / 5.0
        major_lines = groups  # hepsi eşit aralıklı
    else:
        # Fallback: median spacing = 10°C
        px_per_deg = median_spacing / 10.0
        major_lines = groups
    
    # Major çizgileri kullanarak kalibrasyon
    # En üst major çizgi → en yüksek sıcaklık
    # En alt major çizgi → daha düşük sıcaklık
    # Her major çizgi arası 10°C
    
    y_top_grid = major_lines[0]
    y_bot_grid = major_lines[-1]
    n_major_intervals = len(major_lines) - 1
    
    # Toplam piksel → toplam derece
    total_degrees = (y_bot_grid - y_top_grid) / px_per_deg
    
    # Termogram kağıdında en alt MAJOR çizgi genelde 10°C
    # (grafikler genelde 0 altını göstermez ama güvenli alan var)
    # En alt major çizgi = n_major*10 aşağıdan
    # Tipik yapı: üstten alta 40, 30, 20, 10 (°C)
    
    # En alt major çizgiyi 10°C referans alalım
    # (Görüntülerde "10" etiketi en alt kalın çizgide görünüyor)
    temp_at_bottom_major = 10.0
    
    # Eğer major çizgi sayısı 3+ ise, üst major = 10 + n*10
    temp_at_top_major = temp_at_bottom_major + n_major_intervals * 10.0
    
    # Ama tüm grid alanını kapsayacak şekilde genişlet
    # (curve, major çizgilerin dışına da çıkabilir)
    y_top_grid = groups[0]
    y_bot_grid = groups[-1]
    
    temp_at_top = temp_at_top_major + (major_lines[0] - y_top_grid) * (1.0 / px_per_deg)
    temp_at_bottom = temp_at_bottom_major - (y_bot_grid - major_lines[-1]) * (1.0 / px_per_deg)
    
    # Sanity check
    total_range = temp_at_top - temp_at_bottom
    if total_range < 15 or total_range > 100:
        temp_at_bottom = -10.0
        temp_at_top = 40.0
    
    return (y_top_grid, temp_at_top), (y_bot_grid, temp_at_bottom)


# ─── EĞRİ TESPİTİ ───────────────────────────────────────
def detect_curve(img_bgr, hsv, h, w):
    """Mavi mürekkep VEYA siyah mürekkep eğrisini tespit et."""
    
    # Strateji 1: Mavi mürekkep (H=85-125, S>40)
    blue_mask = cv2.inRange(hsv, np.array([85, 35, 35]), np.array([130, 255, 255]))
    
    # Strateji 2: Koyu mürekkep (gri tonlarında ama çizgi şeklinde)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Grid çizgilerini kaldır (yatay yapılar)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 6))
    
    # Çok koyu pikseller
    _, dark_binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Yatay ve dikey grid çizgilerini çıkar
    grid_h = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, horizontal_kernel)
    grid_v = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # Grid'i çıkar, eğriyi bırak
    dark_curve_mask = dark_binary.copy()
    dark_curve_mask[grid_h > 0] = 0
    dark_curve_mask[grid_v > 0] = 0
    
    # Küçük gürültüleri temizle
    kernel_small = np.ones((2, 2), np.uint8)
    dark_curve_mask = cv2.morphologyEx(dark_curve_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Mavi pixel sayısını kontrol et
    blue_count = cv2.countNonZero(blue_mask)
    dark_count = cv2.countNonZero(dark_curve_mask)
    
    # Hangisi daha iyi çalışıyor?
    # Mavi maske: eğer yeterli mavi piksel varsa (genişliğin %5'i x yüksekliğin %1'i)
    min_curve_pixels = w * 0.15
    
    if blue_count > min_curve_pixels:
        # Mavi mürekkep tespit edildi
        mask = blue_mask
        method = "blue_ink"
    elif dark_count > min_curve_pixels:
        mask = dark_curve_mask
        method = "dark_ink"
    else:
        # Her ikisini birleştir
        mask = cv2.bitwise_or(blue_mask, dark_curve_mask)
        method = "combined"
    
    # Gürültü temizle
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask, method

# ─── Y SINIRLARINI OTOMATİK BUL ─────────────────────────
def find_chart_bounds(img_gray, h, w):
    """Grafik alanının sınırlarını bul (kağıdın grid alanı)."""
    # Dikey kenarları bul
    # Grid genelde belirli bir bölgeden başlar
    # Sol ve sağ marjinleri tespit et
    
    # Her sütundaki karanlık piksel sayısı
    col_darkness = []
    for x in range(w):
        col = img_gray[:, x]
        dark = np.sum(col < 180)
        col_darkness.append(dark)
    col_darkness = np.array(col_darkness)
    
    # Grid bölgesi: sürekli olarak karanlık piksellerin olduğu alan
    threshold = h * 0.1
    active_cols = np.where(col_darkness > threshold)[0]
    
    if len(active_cols) > 10:
        x_left = int(active_cols[0])
        x_right = int(active_cols[-1])
    else:
        x_left = int(w * 0.05)
        x_right = int(w * 0.95)
    
    return x_left, x_right

# ─── TEK TIF İŞLE ────────────────────────────────────────
def process_tif(tif_path, debug=False):
    """Bir Termogram TIF dosyasını işle, sıcaklık verisi çıkar."""
    pil_img = Image.open(tif_path).convert("RGB")
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # 1+2. Grid çizgileri + Y ekseni kalibrasyonu (birleşik)
    y_cal_top, y_cal_bottom = detect_grid_and_calibrate(gray, h, w)
    
    if y_cal_top is None:
        # Fallback: görüntünün %10-%90 arası
        y_top_px = int(h * 0.05)
        y_bot_px = int(h * 0.92)
        temp_top = 40.0
        temp_bot = 0.0
    else:
        y_top_px, temp_top = y_cal_top
        y_bot_px, temp_bot = y_cal_bottom

    # 3. X ekseni sınırları
    x_left, x_right = find_chart_bounds(gray, h, w)
    # Biraz daralt (marjinler için)
    x_left = max(x_left, int(w * 0.03))
    x_right = min(x_right, int(w * 0.97))
    
    x_span = x_right - x_left
    y_span = y_bot_px - y_top_px  # piksel cinsinden (pozitif)
    
    if x_span <= 0 or y_span <= 0:
        return None, None

    # 4. Eğriyi tespit et
    mask, method = detect_curve(img_bgr, hsv, h, w)

    # 5. Her x sütununda eğrinin medyan y'sini bul
    curve_xs, curve_ys = [], []
    for x in range(x_left, x_right):
        col_ys = np.where(mask[:, x] > 0)[0]
        if len(col_ys) > 0:
            # Medyan al (gürültülere karşı daha robust)
            curve_xs.append(x)
            curve_ys.append(int(np.median(col_ys)))

    if len(curve_xs) < 20:
        return None, None

    curve_xs = np.array(curve_xs)
    curve_ys = np.array(curve_ys)

    # Kapsama kontrolü
    coverage = (curve_xs.max() - curve_xs.min()) / x_span
    if coverage < 0.25:
        return "EKSIK", None

    # 6. Piksel → Gerçek değer dönüşümü
    def px_to_temp(y_px):
        # y_px küçük = yukarı = yüksek sıcaklık
        if y_span == 0:
            return 0
        ratio = (y_bot_px - y_px) / y_span
        ratio = np.clip(ratio, 0, 1)
        temp = temp_bot + ratio * (temp_top - temp_bot)
        return round(temp, 1)

    def px_to_hour(x_px):
        ratio = (x_px - x_left) / x_span
        hour_offset = ratio * HOUR_TOTAL
        hour = (HOUR_START + hour_offset) % 24
        return hour

    # 7. Veri çıkarımı
    records = []
    for x, y in zip(curve_xs, curve_ys):
        hour_f = px_to_hour(x)
        temp = px_to_temp(y)
        records.append({'hour': round(hour_f, 3), 'temperature_c': temp})

    df_full = pd.DataFrame(records).sort_values('hour').reset_index(drop=True)

    # Saatlik medyan
    df_full['hour_int'] = df_full['hour'].astype(int)
    hourly = df_full.groupby('hour_int')['temperature_c'].median().reset_index()
    hourly.columns = ['hour', 'temperature_c']

    # Günlük özet
    daily_max = df_full['temperature_c'].max()
    daily_min = df_full['temperature_c'].min()
    daily_mean = df_full['temperature_c'].mean()

    meta = {
        'x_left': x_left, 'x_right': x_right,
        'y_top_px': y_top_px, 'y_bot_px': y_bot_px,
        'temp_top': temp_top, 'temp_bot': temp_bot,
        'calibration': f'{temp_bot}°C-{temp_top}°C',
        'method': method,
        'daily_max_c': round(daily_max, 1),
        'daily_min_c': round(daily_min, 1),
        'daily_mean_c': round(daily_mean, 1),
        'points_detected': len(records),
        'coverage': round(coverage, 2),
    }

    # 8. Debug görsel
    if debug:
        debug_img = img_bgr.copy()
        for x, y in zip(curve_xs, curve_ys):
            cv2.circle(debug_img, (int(x), int(y)), 2, (0, 255, 0), -1)
        # Kalibrasyon çizgileri
        cv2.line(debug_img, (x_left, y_top_px), (x_right, y_top_px), (0, 0, 255), 2)
        cv2.line(debug_img, (x_left, y_bot_px), (x_right, y_bot_px), (0, 0, 255), 2)
        # Alan sınırları
        cv2.rectangle(debug_img, (x_left, y_top_px), (x_right, y_bot_px), (255, 0, 0), 2)
        
        debug_path = tif_path.replace('.tif', '_debug.png').replace('.TIF', '_debug.png')
        cv2.imwrite(debug_path, debug_img)

    return hourly, meta

# ─── TOPLU İŞLEME ────────────────────────────────────────
def process_folder(folder, output_dir):
    tif_files = sorted(Path(folder).rglob("*.tif")) + sorted(Path(folder).rglob("*.TIF"))
    # Tekrarlanan dosyaları (büyük/küçük harf farkı) kaldır
    seen = set()
    unique_tifs = []
    for f in tif_files:
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            unique_tifs.append(f)
    tif_files = unique_tifs
    
    print(f"📊 {len(tif_files)} TIF dosyası bulundu.")

    all_daily = []
    os.makedirs(output_dir, exist_ok=True)

    for i, tif_path in enumerate(tif_files):
        print(f"  [{i+1}/{len(tif_files)}] {tif_path.name} ...", end=" ")
        try:
            df, meta = process_tif(str(tif_path))
            
            if df is None:
                print("⏭️  Eğri bulunamadı")
                continue
            if isinstance(df, str) and df == "EKSIK":
                print("⚠️  Eksik gün")
                all_daily.append({
                    'date': parse_date_from_filename(tif_path.name),
                    'max_temp': None, 'min_temp': None, 'mean_temp': None,
                    'status': 'eksik', 'file': tif_path.name
                })
                continue

            date = parse_date_from_filename(tif_path.name)
            date_str = date.strftime('%Y-%m-%d') if date else tif_path.stem

            # Saatlik CSV kaydet
            out_csv = os.path.join(output_dir, f"{date_str}_hourly.csv")
            df['date'] = date_str
            df[['date', 'hour', 'temperature_c']].to_csv(out_csv, index=False)

            all_daily.append({
                'date': date_str,
                'max_temp': meta['daily_max_c'],
                'min_temp': meta['daily_min_c'],
                'mean_temp': meta['daily_mean_c'],
                'method': meta['method'],
                'points': meta['points_detected'],
                'coverage': meta['coverage'],
                'status': 'ok',
                'file': tif_path.name
            })
            print(f"✅ Max:{meta['daily_max_c']}°C Min:{meta['daily_min_c']}°C ({meta['method']})")
        except Exception as e:
            print(f"❌ Hata: {e}")
            all_daily.append({
                'date': parse_date_from_filename(tif_path.name),
                'max_temp': None, 'min_temp': None, 'mean_temp': None,
                'status': f'hata: {str(e)[:50]}', 'file': tif_path.name
            })

    # Tüm günlük özet
    if all_daily:
        df_summary = pd.DataFrame(all_daily)
        if 'date' in df_summary.columns:
            df_summary['date'] = df_summary['date'].astype(str)
            df_summary = df_summary.sort_values('date')
        summary_path = os.path.join(output_dir, "termogram_ozet.csv")
        df_summary.to_csv(summary_path, index=False)
        ok_count = sum(1 for d in all_daily if d.get('status') == 'ok')
        print(f"\n✅ {ok_count}/{len(all_daily)} gün başarıyla sayısallaştırıldı")
        print(f"📄 Özet: {summary_path}")

# ─── MAIN ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Termogram TIF Sayısallaştırıcı')
    parser.add_argument('input', help='TIF dosyası veya klasör')
    parser.add_argument('--output', default='output_termogram', help='Çıktı klasörü')
    parser.add_argument('--debug', action='store_true', help='Debug görseli üret')
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        process_folder(str(inp), args.output)
    elif inp.suffix.lower() == '.tif':
        df, meta = process_tif(str(inp), debug=args.debug)
        if df is not None and not isinstance(df, str):
            os.makedirs(args.output, exist_ok=True)
            date = parse_date_from_filename(inp.name)
            date_str = date.strftime('%Y-%m-%d') if date else inp.stem
            out = os.path.join(args.output, f"{date_str}_hourly.csv")
            df['date'] = date_str
            df.to_csv(out, index=False)
            print(f"✅ {meta['points_detected']} nokta çıkarıldı")
            print(f"   Max: {meta['daily_max_c']}°C | Min: {meta['daily_min_c']}°C | Ort: {meta['daily_mean_c']}°C")
            print(f"   Yöntem: {meta['method']} | Kapsama: {meta['coverage']:.0%}")
            print(f"   CSV: {out}")
        else:
            print("❌ Eğri bulunamadı veya eksik gün")
    else:
        print("Hata: .tif dosyası veya klasör belirtin")
