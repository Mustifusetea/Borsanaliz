"""
╔══════════════════════════════════════════════════════════════╗
║   KANTİTATİF TEKNİK & TEMEL ANALİZ ARACI - v24.0 WEB       ║
║   Güçlü Retry · Tam Tarayıcı Profili    ║
╚══════════════════════════════════════════════════════════════╝

Çalıştırma:
    pip install streamlit plotly yfinance pandas numpy
    streamlit run hisse_analiz_web.py
"""

import os, warnings, random, logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════
#  SAYFA AYARLARI
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Kantitatif Analiz Aracı v24",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.4rem; }
  .card {
      background: #161b22; border: 1px solid #21262d;
      border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
  }
  .card-label { font-size:0.75rem; color:#8b949e; margin-bottom:3px; }
  .card-value { font-size:1.45rem; font-weight:700; color:#e6edf3; }
  .card-sub   { font-size:0.78rem; color:#8b949e; margin-top:4px; }
  .news-item  { border-left:3px solid #21262d; padding:8px 12px;
                margin:5px 0; border-radius:0 6px 6px 0; background:#1c2128; }
  .sec-hdr    { font-size:1.02rem; font-weight:700; color:#58a6ff;
                border-bottom:1px solid #21262d; padding-bottom:5px;
                margin:16px 0 10px 0; }
  .badge-pos  { background:#1a3a2a; color:#3fb950; border-radius:5px;
                padding:2px 10px; font-weight:700; font-size:0.85rem; display:inline-block; }
  .badge-neg  { background:#3a1a1a; color:#f85149; border-radius:5px;
                padding:2px 10px; font-weight:700; font-size:0.85rem; display:inline-block; }
  .badge-ntr  { background:#2a2a1a; color:#e3b341; border-radius:5px;
                padding:2px 10px; font-weight:700; font-size:0.85rem; display:inline-block; }
  .forecast-band { background:#161b22; border-left:4px solid #58a6ff;
                   padding:8px 14px; border-radius:0 8px 8px 0; margin:4px 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  BULUT UYUMLU SESSION & LOG  (Gelişmiş — v24)
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")
_LOG = logging.getLogger("hisse_analiz")

# Gerçek tarayıcı profillerini tam taklit eden header setleri
_BROWSER_PROFILES = [
    {   # Chrome 124 / Windows
        "User-Agent"     : ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"),
        "Accept"         : ("text/html,application/xhtml+xml,application/xml;"
                            "q=0.9,image/avif,image/webp,*/*;q=0.8"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer"        : "https://finance.yahoo.com/",
        "Origin"         : "https://finance.yahoo.com",
        "Connection"     : "keep-alive",
        "Sec-Fetch-Dest" : "document",
        "Sec-Fetch-Mode" : "navigate",
        "Sec-Fetch-Site" : "same-origin",
        "DNT"            : "1",
    },
    {   # Safari 17 / macOS
        "User-Agent"     : ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                            "Version/17.4 Safari/605.1.15"),
        "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer"        : "https://finance.yahoo.com/",
        "Connection"     : "keep-alive",
    },
    {   # Firefox 125 / Linux
        "User-Agent"     : ("Mozilla/5.0 (X11; Linux x86_64; rv:125.0) "
                            "Gecko/20100101 Firefox/125.0"),
        "Accept"         : ("text/html,application/xhtml+xml,application/xml;"
                            "q=0.9,image/avif,image/webp,*/*;q=0.8"),
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer"        : "https://finance.yahoo.com/",
        "Connection"     : "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    },
]


def _yeni_session(profil_idx: int = -1) -> requests.Session:
    """Tam tarayıcı profilini taklit eden session üretir (-1 = rastgele)."""
    profil = (_BROWSER_PROFILES[profil_idx % len(_BROWSER_PROFILES)]
              if profil_idx >= 0
              else random.choice(_BROWSER_PROFILES))
    s = requests.Session()
    s.headers.update(profil)
    retry = Retry(
        total=4, backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s


def _df_temizle(df) -> "pd.DataFrame | None":
    """MultiIndex düzelt, datetime'a çevir, NaN satırları temizle."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[df["Close"].notna()]
    return df if not df.empty else None

# ══════════════════════════════════════════════════════════════
#  RENK PALETİ
# ══════════════════════════════════════════════════════════════
C = {
    "bg":"#0d1117","panel":"#161b22","card":"#1c2128","grid":"#21262d",
    "white":"#e6edf3","muted":"#8b949e",
    "price":"#58a6ff","sma50":"#f0883e","sma200":"#bc8cff",
    "macd":"#58a6ff","signal":"#ff7b72",
    "pos":"#3fb950","neg":"#f85149","vol":"#388bfd","rsi":"#d2a8ff","gold":"#e3b341",
}

# ══════════════════════════════════════════════════════════════
#  NaN-GÜVENLİ YARDIMCILAR
# ══════════════════════════════════════════════════════════════
def _f(d, default=None):
    try:
        v = float(d)
        return default if (v != v or abs(v) == float("inf")) else v
    except: return default

def _fp(d, pb="$"):
    v = _f(d)
    return f"{pb}{v:.2f}" if v is not None else "Veri Yok"

def _pct(d, isaret=True):
    v = _f(d)
    if v is None: return "-"
    return f"%{v:+.1f}" if isaret else f"%{v:.1f}"

def badge(karar):
    css = {"POZITIF":"badge-pos","NEGATIF":"badge-neg"}.get(karar,"badge-ntr")
    return f'<span class="{css}">{karar}</span>'

def karar_rengi(k): return {"POZITIF":C["pos"],"NEGATIF":C["neg"]}.get(k,C["gold"])
def karar_emoji(k): return {"POZITIF":"🟢","NEGATIF":"🔴"}.get(k,"🟡")

# ══════════════════════════════════════════════════════════════
#  TOOLTIP METİNLERİ
# ══════════════════════════════════════════════════════════════
TIP = {
    "RSI"  : ("RSI (Göreceli Güç Endeksi): Fiyatın hızını ve değişimini ölçer. "
              "0-100 arası değer alır. 70 üzeri → aşırı alım (satış baskısı riski), "
              "30 altı → aşırı satım (alım fırsatı sinyali). İdeal nötr: 40-60."),
    "MACD" : ("MACD (Hareketli Ortalama Yakınsama/Iraksama): 12 ve 26 günlük EMA farkı. "
              "MACD > Sinyal Hattı → yükseliş momentumu. "
              "MACD < Sinyal Hattı → düşüş momentumu. "
              "Histogram sıfırı kestiğinde trend dönüşü sinyali verir."),
    "ATR"  : ("ATR (Ortalama Gerçek Aralık): Hissenin günlük ortalama fiyat dalgalanmasını "
              "gösterir. Yüksek ATR → volatil piyasa, Stop-Loss geniş tutulmalı. "
              "Düşük ATR → sakin piyasa. Stop-Loss hesabı için 1.5-2x ATR önerilir."),
    "SMA50": ("SMA 50 (50 Günlük Basit Hareketli Ortalama): Kısa-orta vade trend göstergesi. "
              "Fiyat > SMA50 → yükseliş trendi. Fiyat < SMA50 → düşüş baskısı. "
              "Destek/direnç seviyesi olarak da kullanılır."),
    "SMA200":("SMA 200 (200 Günlük Basit Hareketli Ortalama): Uzun vadeli trend göstergesi. "
              "SMA50 > SMA200 → Altın Kesişim (boğa sinyali). "
              "SMA50 < SMA200 → Ölüm Kesişimi (ayı sinyali). "
              "Kurumsal yatırımcıların temel referansı."),
    "FK"   : ("F/K Oranı (Fiyat/Kazanç): Hisse fiyatının hisse başı kazanca oranı. "
              "< 15 → ucuz/değer hissesi, 15-25 → adil fiyat, > 30 → primli/pahalı. "
              "Sektöre göre değişir; teknoloji hisseleri genelde yüksek F/K taşır."),
    "KM"   : ("Kâr Marjı: Net kârın toplam gelire oranı. "
              "> %15 → yüksek kârlılık (güçlü rekabet avantajı), "
              "%5-15 → orta, < %5 → düşük marj, < 0 → zarar. "
              "Yüksek marj = daha dayanıklı iş modeli."),
    "BD"   : ("Borç/Özkaynak Oranı: Şirketin toplam borcunun özkaynaklarına oranı. "
              "< 50 → düşük borç yükü (finansal güç), 50-150 → orta, "
              "> 150 → yüksek borç (faiz riski). Sektöre göre yorumlanmalı."),
    "TEMETTU": ("Temettü (Kâr Payı): Şirketin hissedarlarına dağıttığı kâr payı. "
                "Düzenli temettü veren şirketler genelde finansal olgunluk göstergesidir. "
                "Temettü verimi = Yıllık temettü / Hisse fiyatı."),
    "HEDEF": ("Analist Hedef Fiyatı: Wall Street analistlerinin 12 aylık fiyat tahminlerinin "
              "ortalaması. Mevcut fiyatın üzerindeyse potansiyel yükseliş, altındaysa "
              "aşırı değerlenme riski işareti olarak yorumlanır."),
    "MA5"  : ("MA5 (5 Günlük Hareketli Ortalama): En kısa vadeli momentum göstergesi. ""Günlük trader'ların takip ettiği hızlı trend çizgisi. ""Fiyat > MA5 → güçlü kısa vadeli ivme. Fiyat < MA5 → zayıflama sinyali."),
    "MA10" : ("MA10 (10 Günlük Hareketli Ortalama): İki haftalık momentum ölçer. ""MA5 ile birlikte kullanılır. MA5 > MA10 → kısa vade yükseliş, MA5 < MA10 → düşüş. ""Destek/direnç testi için orta kısa vadede güvenilir referans."),
    "STOCH": ("Stokastik Osilatör %K(9)/%D(6): Kapanış fiyatının son 9 günün ""fiyat aralığı içindeki konumunu ölçer. ""0-100 arası. > 80 → aşırı alım (dikkat), < 20 → aşırı satım (alım fırsatı). ""%K çizgisi %D'yi yukarı keserse → alım sinyali; aşağı keserse → satış sinyali."),
    "PIVOT": ("Pivot Noktaları (Klasik): Önceki günün High/Low/Close değerleri kullanılarak ""Ana Pivot (P = (H+L+C)/3), Direnç (R1=2P−L, R2=P+(H−L)), ""Destek (S1=2P−H, S2=P−(H−L)) seviyeleri hesaplanır. ""Kurumsal ve algoritmik trader'ların kısa vadeli hedef belirleme aracı."),
    "STOP" : ("Dinamik Stop-Loss: ATR tabanlı risk yönetim seviyesi. "
              "1.5x ATR = makul stop seviyesi. Bu seviyenin altına kapanış → "
              "pozisyondan çıkış değerlendirilmeli. Volatilite arttıkça stop uzaklaşır."),
}

# ══════════════════════════════════════════════════════════════
#  1. VERİ ÇEKİMİ
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=900, show_spinner=False)
def veri_cek(sembol: str) -> "pd.DataFrame | None":
    """
    3 kademeli, bulut sunucu blok-aşan veri çekimi:
      1. Tam tarayıcı profilli session + Ticker.history("1y")
      2. Farklı profil + yf.download() ile 1y tarih aralığı
      3. Daha kısa aralık (6ay) + session'sız yf.download()
    """
    bitis     = datetime.today()
    bas_1y    = (bitis - timedelta(days=365)).strftime("%Y-%m-%d")
    bas_6m    = (bitis - timedelta(days=185)).strftime("%Y-%m-%d")
    bitis_str = bitis.strftime("%Y-%m-%d")

    # ── Kademe 1: Ticker.history + tam tarayıcı session ─────────
    try:
        ses1    = _yeni_session(0)          # Chrome profili
        ticker1 = yf.Ticker(sembol, session=ses1)
        df1     = ticker1.history(period="1y", auto_adjust=True)
        result  = _df_temizle(df1)
        if result is not None:
            _LOG.info("veri_cek K1 basarili [%s] %d gun", sembol, len(result))
            return result
        _LOG.warning("veri_cek K1 bos dondü [%s]", sembol)
    except Exception as e1:
        _LOG.warning("veri_cek K1 hata [%s]: %s", sembol, e1)

    # ── Kademe 2: yf.download + farklı tarayıcı profili ─────────
    try:
        import time; time.sleep(1.5)        # kısa bekleme — rate limit aşımı
        ses2 = _yeni_session(1)             # Safari profili
        df2  = yf.download(sembol, start=bas_1y, end=bitis_str,
                           progress=False, auto_adjust=True, session=ses2)
        result = _df_temizle(df2)
        if result is not None:
            _LOG.info("veri_cek K2 basarili [%s] %d gun", sembol, len(result))
            return result
        _LOG.warning("veri_cek K2 bos dondü [%s]", sembol)
    except Exception as e2:
        _LOG.warning("veri_cek K2 hata [%s]: %s", sembol, e2)

    # ── Kademe 3: Kısa aralık + session'sız ─────────────────────
    try:
        import time; time.sleep(2.0)
        df3 = yf.download(sembol, start=bas_6m, end=bitis_str,
                          progress=False, auto_adjust=True)
        result = _df_temizle(df3)
        if result is not None:
            _LOG.warning("veri_cek K3 (6ay kisaltilmis) basarili [%s]", sembol)
            return result
    except Exception as e3:
        _LOG.error("veri_cek K3 hata [%s]: %s", sembol, e3)

    _LOG.error("veri_cek tum kademeler basarisiz [%s]", sembol)
    return None


@st.cache_data(ttl=900, show_spinner=False)
def temel_veri_cek(sembol: str) -> dict:
    """
    Bulut uyumlu temel veri çekimi:
    1. User-Agent session ile ticker.info dene
    2. info boş/eksikse fast_info ile kritik alanları doldur
    3. Yedek: session'siz Ticker (son çare)
    """
    ses = _yeni_session()

    # ── Aşama 1: session ile ticker.info ─────────────────────
    info = {}
    try:
        ticker = yf.Ticker(sembol, session=ses)
        raw    = ticker.info
        # info bazen {'trailingPegRatio': None} gibi sadece 1-2 alan döner
        if raw and len(raw) > 5:
            info = raw
            _LOG.info("temel_veri_cek: info OK [%s] (%d alan)", sembol, len(info))
        else:
            _LOG.warning("temel_veri_cek: info zayif [%s] (%d alan) — fast_info deneniyor",
                         sembol, len(raw) if raw else 0)
    except Exception as e:
        _LOG.warning("temel_veri_cek: ticker.info hatasi [%s]: %s", sembol, e)
        # Yedek: session'siz dene
        try:
            ticker = yf.Ticker(sembol)
            raw    = ticker.info
            if raw and len(raw) > 5:
                info = raw
        except Exception as e2:
            _LOG.error("temel_veri_cek: tamamen basarisiz [%s]: %s", sembol, e2)
            ticker = yf.Ticker(sembol)  # en azından fast_info için

    # ── Aşama 2: fast_info fallback ──────────────────────────
    # Kritik sayısal alanlar boşsa fast_info / basic_info ile doldur
    def _fast_fallback(field_info, fast_attr, ticker_obj):
        """info'da yoksa fast_info/basic_info'dan almayı dene."""
        if _f(field_info) is not None:
            return field_info
        try:
            fi = getattr(ticker_obj, 'fast_info', None)
            if fi and hasattr(fi, fast_attr):
                return getattr(fi, fast_attr)
        except Exception:
            pass
        return field_info

    # fast_info alanları: three_month_average_volume, last_price, year_high, year_low vb.
    try:
        fi = getattr(ticker, 'fast_info', None)
        if fi:
            if not info.get('fiftyTwoWeekLow'):  info['fiftyTwoWeekLow']  = getattr(fi,'year_low',  None)
            if not info.get('fiftyTwoWeekHigh'): info['fiftyTwoWeekHigh'] = getattr(fi,'year_high', None)
            if not info.get('marketCap'):        info['marketCap']        = getattr(fi,'market_cap',None)
            if not info.get('currentPrice'):     info['currentPrice']     = getattr(fi,'last_price', None)
            _LOG.info("temel_veri_cek: fast_info tamamlama yapildi [%s]", sembol)
    except Exception as fe:
        _LOG.warning("temel_veri_cek: fast_info hatasi [%s]: %s", sembol, fe)

    # ── Veri sözlüğü ─────────────────────────────────────────
    veri = {
        "sirket_adi"    : info.get("longName") or info.get("shortName") or sembol,
        "sektor"        : info.get("sector",   "---"),
        "endustri"      : info.get("industry", "---"),
        "ozet"          : info.get("longBusinessSummary", ""),
        "ulke"          : info.get("country", "---"),
        "web"           : info.get("website", ""),
        "calisan_sayisi": info.get("fullTimeEmployees"),
        "piyasa_degeri" : info.get("marketCap"),
        "hedef_fiyat"   : _f(info.get("targetMeanPrice")),
        "fk_orani"      : _f(info.get("trailingPE")),
        "kar_marji"     : _f(info.get("profitMargins")),
        "borc_ozkaynak" : _f(info.get("debtToEquity")),
        "haftalik_dip"  : _f(info.get("fiftyTwoWeekLow")),
        "haftalik_zirve": _f(info.get("fiftyTwoWeekHigh")),
    }

    _LOG.info("temel_veri_cek: FK=%s KM=%s BD=%s [%s]",
             veri['fk_orani'], veri['kar_marji'], veri['borc_ozkaynak'], sembol)

    # ── Bilanço tarihi ────────────────────────────────────────
    bilanco = "Bulunamadı"
    try:
        cal = ticker.calendar
        if cal is not None:
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed: bilanco = str(ed[0] if isinstance(ed,list) else ed)[:10]
            elif hasattr(cal,"columns") and "Earnings Date" in cal.columns:
                bilanco = str(cal["Earnings Date"].iloc[0])[:10]
    except Exception as be:
        _LOG.debug("bilanco tarihi alinamadi [%s]: %s", sembol, be)
    veri["bilanco_tarihi"] = bilanco

    # ── Temettü geçmişi ───────────────────────────────────────
    veri["son_temettu_tarih"] = "Veri Yok"
    veri["_temettu_miktar"]   = None
    veri["_temettu_verimi"]   = None
    try:
        divs = ticker.dividends
        if divs is not None and not divs.empty:
            son_miktar = _f(float(divs.iloc[-1]))
            if son_miktar is not None:
                veri["_temettu_miktar"]   = son_miktar
                veri["son_temettu_tarih"] = str(divs.index[-1])[:10]
                yillik = _f(info.get("trailingAnnualDividendRate"))
                fiyat  = _f(info.get("currentPrice") or info.get("regularMarketPrice"))
                if yillik and fiyat and fiyat > 0:
                    veri["_temettu_verimi"] = (yillik / fiyat) * 100
    except Exception as de:
        _LOG.debug("temettu alinamadi [%s]: %s", sembol, de)

    return veri


@st.cache_data(ttl=900, show_spinner=False)
def makro_veri_cek(baslangic: str, bitis: str, bist_modu: bool) -> dict:
    semboller = {"SPY": "XU100.IS" if bist_modu else "SPY",
                 "GLD": "GLD", "USO": "USO"}
    sonuc = {}
    for anahtar, gercek in semboller.items():
        try:
            df = yf.download(gercek, start=baslangic, end=bitis,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if not df.empty: sonuc[anahtar] = df
        except: pass
    return sonuc


@st.cache_data(ttl=900, show_spinner=False)
def haber_cek(sembol: str) -> list:
    _POZ = {"up","upgrade","upgraded","buy","outperform","overweight","win","wins",
            "contract","profit","beat","beats","record","growth","surge","rally",
            "positive","strong","raise","boost","acquire","deal","dividend","bullish"}
    _NEG = {"down","downgrade","downgraded","sell","underperform","underweight","loss",
            "lawsuit","miss","misses","cut","decline","drop","weak","warning",
            "layoff","recall","investigation","fine","penalty","bearish","negative",
            "debt","bankrupt"}
    def ton(baslik):
        kw = set(baslik.lower().replace(","," ").replace("."," ").split())
        p,n = kw&_POZ, kw&_NEG
        if p and not n: return "POZITIF"
        if n and not p: return "NEGATIF"
        if p and n:     return "POZITIF" if len(p)>=len(n) else "NEGATIF"
        return "NOTR"
    try:
        haberler = yf.Ticker(sembol).news or []
        sonuc = []
        for h in haberler[:4]:
            ic = h.get("content", h)
            baslik  = ic.get("title") or h.get("title") or "Başlık yok"
            yayinci = (ic.get("provider",{}).get("displayName")
                       or h.get("publisher") or "---")
            ts = ic.get("pubDate") or h.get("providerPublishTime")
            if isinstance(ts,(int,float)): zaman = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
            elif isinstance(ts,str):       zaman = ts[:16]
            else:                          zaman = "---"
            sonuc.append({"baslik":baslik,"yayinci":yayinci,"zaman":zaman,"ton":ton(baslik)})
        return sonuc
    except: return []

# ══════════════════════════════════════════════════════════════
#  2. TEKNİK GÖSTERGELER
# ══════════════════════════════════════════════════════════════
def hesapla_gostergeler(df: pd.DataFrame) -> pd.DataFrame:
    k = df["Close"].squeeze(); h = df["Volume"].squeeze()
    d = k.diff(); kaz = d.clip(lower=0); kayip = -d.clip(upper=0)
    df["RSI"]         = 100 - 100/(1 + kaz.ewm(com=13,min_periods=14).mean()
                                      / kayip.ewm(com=13,min_periods=14).mean())
    e12 = k.ewm(span=12,adjust=False).mean()
    e26 = k.ewm(span=26,adjust=False).mean()
    df["MACD"]        = e12 - e26
    df["MACD_Signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df["MA5"]         = k.rolling(5).mean()
    df["MA10"]        = k.rolling(10).mean()
    df["SMA50"]       = k.rolling(50).mean()
    df["SMA200"]      = k.rolling(200).mean()
    # Stokastik %K(9) ve %D(6)
    low9  = df["Low"].squeeze().rolling(9).min()
    high9 = df["High"].squeeze().rolling(9).max()
    df["STOCH_K"]     = (df["Close"].squeeze() - low9) / (high9 - low9 + 1e-9) * 100
    df["STOCH_D"]     = df["STOCH_K"].rolling(6).mean()
    prev = k.shift(1)
    tr = pd.concat([df["High"].squeeze()-df["Low"].squeeze(),
                    (df["High"].squeeze()-prev).abs(),
                    (df["Low"].squeeze()-prev).abs()], axis=1).max(axis=1)
    df["ATR"]         = tr.ewm(com=13,min_periods=14).mean()
    ort = h.rolling(20).mean()
    df["Vol_Ort20"]   = ort
    df["Vol_Yuksek"]  = h > ort*1.5
    return df


def son_degerler(df: pd.DataFrame) -> dict:
    k = df["Close"].squeeze().dropna()
    son = df.loc[k.index[-1]]
    sk    = _f(k.iloc[-1], 0.0)
    rsi   = _f(son.get("RSI"),    50.0)
    macd  = _f(son.get("MACD"),   0.0)
    sig   = _f(son.get("MACD_Signal"), 0.0)
    hist  = _f(son.get("MACD_Hist"),   0.0)
    sma50  = _f(son.get("SMA50"))
    sma200 = _f(son.get("SMA200"))
    ma5    = _f(son.get("MA5"))
    ma10   = _f(son.get("MA10"))
    stoch_k= _f(son.get("STOCH_K"))
    stoch_d= _f(son.get("STOCH_D"))
    atr    = _f(son.get("ATR"))
    hacim  = _f(son.get("Volume"), 0.0)
    ort_h  = _f(son.get("Vol_Ort20"), hacim)
    # Pivot Noktaları (önceki gün H/L/C ile)
    idx = k.index
    if len(idx) >= 2:
        prev_row = df.loc[idx[-2]]
        ph = _f(prev_row.get("High"))
        pl = _f(prev_row.get("Low"))
        pc = _f(prev_row.get("Close"))
        if ph and pl and pc:
            pvt = (ph+pl+pc)/3
            r1  = 2*pvt - pl
            r2  = pvt + (ph-pl)
            s1  = 2*pvt - ph
            s2  = pvt - (ph-pl)
        else:
            pvt=r1=r2=s1=s2=None
    else:
        pvt=r1=r2=s1=s2=None
    if macd>sig and hist>0:    macd_yon = "YUKSELIS"
    elif macd<sig and hist<0:  macd_yon = "DUSUS"
    elif macd>sig:             macd_yon = "ZAYIFLAYAN YUKSELIS"
    else:                      macd_yon = "ZAYIFLAYAN DUSUS"
    if sma50 and sma200:
        if sk>sma50>sma200:    sma_durum = 1
        elif sk<sma50<sma200:  sma_durum = -1
        else:                  sma_durum = 0
    else: sma_durum = 0
    puan = 0
    if 40<rsi<70:  puan+=1
    if rsi<=30:    puan+=2
    puan += {"YUKSELIS":2,"ZAYIFLAYAN DUSUS":1,"ZAYIFLAYAN YUKSELIS":-1,"DUSUS":-2}.get(macd_yon,0)
    if sma50  and sk>sma50:  puan+=1
    if sma200 and sk>sma200: puan+=1
    return {"sk":sk,"rsi":rsi,"macd":macd,"sig":sig,"hist":hist,
            "sma50":sma50,"sma200":sma200,"ma5":ma5,"ma10":ma10,
            "stoch_k":stoch_k,"stoch_d":stoch_d,
            "pvt":pvt,"r1":r1,"r2":r2,"s1":s1,"s2":s2,
            "atr":atr,"hacim":hacim,"ort_hacim":ort_h,
            "macd_yon":macd_yon,"sma_durum":sma_durum,"puan":puan}

# ══════════════════════════════════════════════════════════════
#  3. ÜÇLÜ VADE TAHMİN (YÜZDESELLİ)
# ══════════════════════════════════════════════════════════════
def _karar(puan, max_p):
    oran = puan/max_p if max_p else 0
    p100 = max(0, min(100, round((puan+max_p)/(2*max_p)*100)))
    if oran>=0.35:  return "POZITIF", p100
    if oran<=-0.20: return "NEGATIF", p100
    return "NOTR", p100


def getiri_hesapla(df, gun):
    try:
        k = df["Close"].squeeze().dropna(); g = min(gun,len(k)-1)
        if g<1: return None
        return ((float(k.iloc[-1])-float(k.iloc[-g]))/float(k.iloc[-g]))*100
    except: return None


def tahmin_hesapla(sd: dict, temel: dict, makro_df: dict,
                   hisse_df: pd.DataFrame, haber_tonu: str,
                   bist_modu: bool) -> dict:
    sk  = sd["sk"]; rsi = sd["rsi"]; macd_yon = sd["macd_yon"]
    sma_durum = sd["sma_durum"]; atr = sd["atr"]; puan = sd["puan"]

    # ── Kısa Vade (1-4 Hafta) ──────────────────────────────────
    kv, kv_max = 0, 8
    if rsi<=30:   kv+=2
    elif rsi<=50: kv+=1
    elif rsi>=70: kv-=2
    kv += {"YUKSELIS":2,"ZAYIFLAYAN DUSUS":1,"ZAYIFLAYAN YUKSELIS":-1,"DUSUS":-2}.get(macd_yon,0)
    if atr and sk:
        atr_pct = atr/sk*100
        if atr_pct<1.5: kv+=1
        elif atr_pct>3: kv-=1
    kv += {"POZITIF":1,"NEGATIF":-1}.get(haber_tonu, 0)
    kv_karar, kv_100 = _karar(kv, kv_max)

    # Yüzdesel bant: ATR tabanlı (sqrt(20 işlem günü) * ATR)
    if atr and sk:
        alt_kv = round(atr*2.0/sk*100, 1)
        ust_kv = round(atr*4.5/sk*100, 1)
        if kv_karar == "POZITIF":
            kv_bant = f"%+{alt_kv} ile %+{ust_kv} arasında yükseliş marjı öngörülüyor."
        elif kv_karar == "NEGATIF":
            kv_bant = f"%-{ust_kv} ile %-{alt_kv} arasında düşüş riski öngörülüyor."
        else:
            kv_bant = f"±%{alt_kv} ile ±%{ust_kv} arasında yatay/değişken seyir bekleniyor."
    else:
        kv_bant = "ATR verisi eksik — yüzdesel marj hesaplanamadı."

    # ── Orta Vade (3-6 Ay) ─────────────────────────────────────
    ov, ov_max = 0, 7
    ov += {1:3, -1:-3, 0:0}.get(sma_durum, 0)
    spy_df = makro_df.get("SPY")
    h6a = getiri_hesapla(hisse_df, 126)
    s6a = getiri_hesapla(spy_df, 126) if spy_df is not None else None
    makro_durum = "Veri Yok"; fark_makro = None
    if h6a is not None and s6a is not None:
        fark_makro = h6a - s6a
        if fark_makro>=2:    ov+=2; makro_durum = f"ENDEKS ÜSTÜNDE (%{fark_makro:+.1f})"
        elif fark_makro<=-2: ov-=2; makro_durum = f"ENDEKS ALTINDA (%{fark_makro:+.1f})"
        else:                        makro_durum = f"ENDEKS PARALEL (%{fark_makro:+.1f})"
    hedef  = _f(temel.get("hedef_fiyat"))
    potans = None
    if hedef and sk:
        potans = _f(((hedef-sk)/sk)*100)
        if potans and potans>0:  ov+=2
        elif potans:             ov-=1
    ov_karar, ov_100 = _karar(ov, ov_max)

    # Yüzdesel bant: SMA200'den sapmayı ve analist hedefini harmanlıyoruz
    sma200 = sd.get("sma200")
    ov_bant_parts = []
    if sma200 and sk and sma_durum == 1:
        sma_hedef_pct = round(((sma200*1.05 - sk)/sk)*100, 1)  # SMA200'ün %5 üzeri
        ov_bant_parts.append(f"SMA200 bazlı orta hedef: %{sma_hedef_pct:+.1f}")
    if potans is not None:
        ov_bant_parts.append(f"analist hedefi: %{potans:+.1f}")
    if ov_bant_parts:
        birlesik = " | ".join(ov_bant_parts)
        if ov_karar == "POZITIF":
            ov_bant = f"Orta vadede yükseliş potansiyeli — {birlesik}."
        elif ov_karar == "NEGATIF":
            ov_bant = f"Orta vadede baskı bekleniyor — {birlesik}."
        else:
            ov_bant = f"Orta vade belirsiz — {birlesik}."
    else:
        ov_bant = ("SMA trendi güçlü, orta vadeli destek var." if sma_durum==1
                   else "SMA baskısı sürebilir, dikkatli olun." if sma_durum==-1
                   else "SMA trendi karışık, yatay seyir olası.")

    # ── Uzun Vade (12+ Ay) ─────────────────────────────────────
    uv, uv_max = 0, 8
    fk  = _f(temel.get("fk_orani"))
    km  = _f(temel.get("kar_marji"))
    bd  = _f(temel.get("borc_ozkaynak"))
    div = _f(temel.get("_temettu_miktar"))
    tv  = _f(temel.get("_temettu_verimi"))
    if fk is not None:
        if fk<0:    uv-=2
        elif fk<15: uv+=2
        elif fk>35: uv-=2
        elif fk>25: uv-=1
    if km is not None:
        kmp = km*100
        if kmp<0:   uv-=2
        elif kmp<5: uv-=1
        elif kmp>=15: uv+=2
    if bd is not None:
        if bd<50:   uv+=2
        elif bd>150:uv-=2
    if div and div>0: uv+=2
    uv_karar, uv_100 = _karar(uv, uv_max)

    # Yüzdesel bant: analist hedefi + temettü verimi
    uv_bant_parts = []
    h1y = getiri_hesapla(hisse_df, 252)
    if potans is not None:
        uv_bant_parts.append(f"analist hedefi: %{potans:+.1f}")
    if tv is not None:
        uv_bant_parts.append(f"temettü verimi: %{tv:.2f}")
    if h1y is not None:
        uv_bant_parts.append(f"geçen yıl getiri: %{h1y:+.1f}")
    if uv_bant_parts:
        birlesik = " | ".join(uv_bant_parts)
        if uv_karar == "POZITIF":
            uv_bant = f"Uzun vadeli birikim için olumlu zemin — {birlesik}."
        elif uv_karar == "NEGATIF":
            uv_bant = f"Uzun vadeli riskler ağır basıyor — {birlesik}."
        else:
            uv_bant = f"Uzun vade karışık tablo — {birlesik}."
    else:
        uv_bant = ("Temel göstergeler güçlü." if uv_karar=="POZITIF"
                   else "Temel göstergeler zayıf." if uv_karar=="NEGATIF"
                   else "Temel göstergeler nötr.")

    # ── Genel ──────────────────────────────────────────────────
    g_puan = kv+ov+uv; g_max = kv_max+ov_max+uv_max
    genel_karar, genel_100 = _karar(g_puan, g_max)

    return {
        "genel_karar":genel_karar,"genel_100":genel_100,
        "kv_karar":kv_karar,"kv_100":kv_100,"kv_bant":kv_bant,
        "ov_karar":ov_karar,"ov_100":ov_100,"ov_bant":ov_bant,
        "uv_karar":uv_karar,"uv_100":uv_100,"uv_bant":uv_bant,
        "makro_durum":makro_durum,"potans":potans,
    }



# ══════════════════════════════════════════════════════════════
#  KORKU & AÇGÖZLÜLÜK MODÜLÜ
# ══════════════════════════════════════════════════════════════
def korku_skoru_hesapla(sd: dict, df: pd.DataFrame) -> dict:
    """
    Hisseye özel Korku & Açgözlülük skoru (0-100).
    Bileşenler ve ağırlıklar:
      RSI tabanlı         : %40
      SMA sapması         : %30
      Hacim-fiyat dinamiği: %30
    """
    sk     = sd["sk"]
    rsi    = sd["rsi"]
    sma50  = sd["sma50"]
    sma200 = sd["sma200"]

    # ── Bileşen 1: RSI (%40) ─────────────────────────────────
    # RSI 0→Aşırı Korku, 100→Aşırı Açgözlülük; doğrusal normalize
    rsi_skor = float(rsi)          # RSI zaten 0-100

    # ── Bileşen 2: SMA Sapması (%30) ─────────────────────────
    # Fiyatın SMA50 ve SMA200'den % sapması (ortalaması)
    sma_skoru = 50.0               # varsayılan: nötr
    sapma_listesi = []
    if sma50 and sma50 > 0:
        sapma_listesi.append((sk - sma50) / sma50 * 100)
    if sma200 and sma200 > 0:
        sapma_listesi.append((sk - sma200) / sma200 * 100)
    if sapma_listesi:
        ort_sapma = sum(sapma_listesi) / len(sapma_listesi)
        # +20% sapma → 90 puan, -20% sapma → 10 puan; ±10% → 65/35
        sma_skoru = max(0.0, min(100.0, 50.0 + ort_sapma * 2.0))

    # ── Bileşen 3: Hacim-Fiyat Dinamiği (%30) ────────────────
    # Son 5 günün fiyat yönü × hacim değişimi
    hfp_skor = 50.0
    try:
        kapanis = df["Close"].squeeze().dropna()
        hacim   = df["Volume"].squeeze().dropna()
        if len(kapanis) >= 6 and len(hacim) >= 6:
            son5_fiyat  = kapanis.iloc[-5:]
            son5_hacim  = hacim.iloc[-5:]
            onceki_hac  = hacim.iloc[-10:-5].mean() if len(hacim) >= 10 else hacim.iloc[-5:].mean()

            fiyat_yon   = float(son5_fiyat.iloc[-1] - son5_fiyat.iloc[0])
            hacim_degis = float(son5_hacim.mean() - onceki_hac) / (float(onceki_hac) + 1e-9)

            # Fiyat ↑ + Hacim ↑ → güçlü açgözlülük
            # Fiyat ↓ + Hacim ↑ → güçlü korku
            sinyal = fiyat_yon * hacim_degis   # + → aç, - → korku
            # normalize: sinyal ∈ (-∞,+∞) → [0,100]
            hfp_skor = max(0.0, min(100.0, 50.0 + sinyal * 5.0))
    except Exception:
        pass

    # ── Ağırlıklı Birleşim ───────────────────────────────────
    ham_skor = (rsi_skor * 0.40) + (sma_skoru * 0.30) + (hfp_skor * 0.30)
    skor     = int(round(max(0.0, min(100.0, ham_skor))))

    # ── Etiket & Renk ────────────────────────────────────────
    if skor <= 25:
        etiket = "Aşırı Korku";    renk = "#f85149"
    elif skor <= 45:
        etiket = "Korku";          renk = "#e3813b"
    elif skor <= 55:
        etiket = "Nötr";           renk = "#8b949e"
    elif skor <= 75:
        etiket = "Açgözlülük";     renk = "#56d364"
    else:
        etiket = "Aşırı Açgözlülük"; renk = "#26a641"

    return {
        "skor"        : skor,
        "etiket"      : etiket,
        "renk"        : renk,
        "rsi_skor"    : round(rsi_skor, 1),
        "sma_skor"    : round(sma_skoru, 1),
        "hfp_skor"    : round(hfp_skor, 1),
    }


def korku_gauge_ciz(psi: dict, sembol: str) -> go.Figure:
    """Plotly Gauge Chart: Korku & Açgözlülük Hız Göstergesi."""
    skor   = psi["skor"]
    etiket = psi["etiket"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=skor,
        delta={"reference": 50, "increasing": {"color": "#26a641"},
               "decreasing": {"color": "#f85149"}},
        number={"font": {"size": 40, "color": "#e6edf3"}, "suffix": ""},
        title={"text": f"<b>Psikoloji Endeksi</b><br>"
                       f"<span style='font-size:0.85em;color:#8b949e'>{sembol} · {etiket}</span>",
               "font": {"color": "#e6edf3", "size": 14}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 25, 45, 55, 75, 100],
                "ticktext": ["0", "25", "45", "55", "75", "100"],
                "tickcolor": "#8b949e",
                "tickfont": {"size": 10, "color": "#8b949e"},
            },
            "bar": {"color": psi["renk"], "thickness": 0.28},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "#3a1214"},   # Aşırı Korku
                {"range": [25, 45], "color": "#3a2010"},   # Korku
                {"range": [45, 55], "color": "#1c2128"},   # Nötr
                {"range": [55, 75], "color": "#112010"},   # Açgözlülük
                {"range": [75,100], "color": "#0e2010"},   # Aşırı Açgözlülük
            ],
            "threshold": {
                "line": {"color": "#e6edf3", "width": 2},
                "thickness": 0.8,
                "value": skor,
            },
        },
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
    )
    return fig

# ══════════════════════════════════════════════════════════════
#  STRATEJİK YOL HARİTASI
# ══════════════════════════════════════════════════════════════
def strateji_goster(tahmin: dict, sd: dict, temel: dict) -> None:
    """
    Vade puanlarını + teknik/temel verileri harmanlayarak
    kullanıcıya net, aksiyona dönüşebilir strateji önerileri sunar.
    """
    kv_100  = tahmin.get("kv_100", 50)
    ov_100  = tahmin.get("ov_100", 50)
    uv_100  = tahmin.get("uv_100", 50)
    kv_k    = tahmin.get("kv_karar","NOTR")
    ov_k    = tahmin.get("ov_karar","NOTR")
    uv_k    = tahmin.get("uv_karar","NOTR")

    rsi      = sd.get("rsi", 50)
    macd_yon = sd.get("macd_yon","NOTR")
    sma_dur  = sd.get("sma_durum", 0)
    fk       = _f(temel.get("fk_orani"))
    km       = _f(temel.get("kar_marji"))
    div      = _f(temel.get("_temettu_miktar"))

    oneriler = []

    # ── Haftalık Trade / Kısa Vade ────────────────────────────
    kv_teknik_ok = (
        kv_100 >= 60
        and rsi < 70              # aşırı alımda değil
        and macd_yon in ("YUKSELIS", "ZAYIFLAYAN DUSUS")
    )
    if kv_teknik_ok:
        oneriler.append({
            "tip"  : "success",
            "ikon" : "🟢",
            "baslik": "HAFTALIK TRADE / KISA VADELİ ALIM",
            "metin": (
                f"Kısa vade puanı ({kv_100}/100), RSI ({rsi:.1f}) ve MACD "
                f"göstergeleri uyumlu bir tablo çiziyor. "
                f"Bu hisse **haftalık bazda kısa süreli trade ve alım fırsatları** "
                f"için uygun görünüyor. Stop-Loss'u ATR tabanlı olarak belirleyin."
            ),
        })

    # ── Aylık Yatırım / Orta Vade ─────────────────────────────
    ov_teknik_ok = (
        ov_100 >= 58
        and sma_dur >= 0          # SMA trendi negatif değil
    )
    if ov_teknik_ok:
        sma_acik = ("SMA50 > SMA200 altın kesişim desteği ile"
                    if sma_dur == 1 else "SMA trendi karışık olmakla birlikte")
        oneriler.append({
            "tip"  : "info",
            "ikon" : "🔵",
            "baslik": "AYLIK YATIRIM / KADEMELİ ALIM",
            "metin": (
                f"Orta vade puanı ({ov_100}/100) ve {sma_acik} makro tablo, "
                f"**aylık vadede pozisyon korumak veya kademeli alım yapmak** "
                f"için olumlu bir zemin sunuyor. "
                f"Endeks performansına göre portföy ağırlığını ayarlayın."
            ),
        })

    # ── Uzun Vadeli Portföy / Birikim ─────────────────────────
    uv_temel_ok = (
        uv_100 >= 55
        and (fk is None or fk < 35)     # makul F/K
        and (km is None or km >= 0)     # zarar etmiyor
    )
    if uv_temel_ok:
        div_notu = (f"Temettü ödemesi ({div:.4f}/hisse) pasif gelir de sağlıyor. "
                    if div and div > 0 else "")
        oneriler.append({
            "tip"  : "success",
            "ikon" : "🟣",
            "baslik": "UZUN VADELİ PORTFÖY / BİRİKTİRME",
            "metin": (
                f"Uzun vade puanı ({uv_100}/100) ve temel rasyolar "
                f"**uzun vadeli (1 yıl+) portföylerde biriktirilmeye değer** "
                f"bir tablo ortaya koyuyor. {div_notu}"
                f"Düzenli kademeli alım (DCA) stratejisi değerlendirilebilir."
            ),
        })

    # ── Bekle ve İzle (hiçbir öneri yoksa veya genel negatifse) ─
    genel_100 = tahmin.get("genel_100", 50)
    if not oneriler or genel_100 < 40:
        oneriler.append({
            "tip"  : "warning",
            "ikon" : "🟡",
            "baslik": "BEKLE VE İZLE",
            "metin": (
                f"Mevcut tabloda riskler yüksek (Genel puan: {genel_100}/100). "
                f"**Yeni pozisyon açmak yerine kenarda bekleyip izlemek** "
                f"daha güvenli olabilir. Teknik ve temel göstergeler netleşene "
                f"kadar sabırlı olmak en sağlıklı yaklaşımdır."
            ),
        })

    # ── Ekrana yaz ───────────────────────────────────────────
    st.header("💡 Stratejik Yol Haritası", anchor="strateji-onerisi", divider="orange")
    st.caption(
        "⚠️ Aşağıdaki öneriler tamamen algoritmik gösterge analizine dayalıdır; "
        "kesinlikle yatırım tavsiyesi değildir."
    )

    for o in oneriler:
        icerik = f"**{o['ikon']} {o['baslik']}**\n\n{o['metin']}"
        if o["tip"] == "success":
            st.success(icerik)
        elif o["tip"] == "info":
            st.info(icerik)
        elif o["tip"] == "warning":
            st.warning(icerik)
        else:
            st.error(icerik)

# ══════════════════════════════════════════════════════════════
#  4. PLOTLY GRAFİĞİ  (Zaman filtresi butonları dahil)
# ══════════════════════════════════════════════════════════════
def grafik_ciz(df: pd.DataFrame, sembol: str, pb: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        vertical_spacing=0.025,
        subplot_titles=(f"{sembol} — Fiyat & Hareketli Ortalamalar",
                        "Hacim", "MACD", "RSI (14)")
    )
    # datetime garantisi — rangeselector step ayarları buna bağlı
    t = pd.to_datetime(df.index)
    k = df["Close"].squeeze()

    # ── Mum (Candlestick) + SMA overlay ──────────────────────
    fig.add_trace(go.Candlestick(
        x=t,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=k,
        name="OHLC",
        increasing=dict(line=dict(color=C["pos"],width=1), fillcolor=C["pos"]),
        decreasing=dict(line=dict(color=C["neg"],width=1), fillcolor=C["neg"]),
        hovertext=[f"A:{o:.2f}  Y:{h:.2f}  D:{l:.2f}  K:{c:.2f}"
                   for o,h,l,c in zip(df["Open"].squeeze(),
                                       df["High"].squeeze(),
                                       df["Low"].squeeze(), k)],
        hoverinfo="x+text",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["MA5"].squeeze(), name="MA 5",
        line=dict(color="#f0c040",width=1.0,dash="solid"),
        hovertemplate="MA5: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["MA10"].squeeze(), name="MA 10",
        line=dict(color="#80d4ff",width=1.0,dash="solid"),
        hovertemplate="MA10: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["SMA50"].squeeze(), name="SMA 50",
        line=dict(color=C["sma50"],width=1.3,dash="dash"),
        hovertemplate="SMA50: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["SMA200"].squeeze(), name="SMA 200",
        line=dict(color=C["sma200"],width=1.3,dash="dot"),
        hovertemplate="SMA200: %{y:.2f}<extra></extra>"), row=1, col=1)

    # Hacim
    rk = [C["neg"] if v else C["vol"] for v in df["Vol_Yuksek"]]
    fig.add_trace(go.Bar(x=t, y=df["Volume"].squeeze(), name="Hacim",
        marker_color=rk, opacity=0.8,
        hovertemplate="Hacim: %{y:,.0f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["Vol_Ort20"].squeeze(), name="Ort-20",
        line=dict(color=C["gold"],width=1,dash="dash")), row=2, col=1)

    # MACD
    hist = df["MACD_Hist"].squeeze()
    hc = [C["pos"] if v>=0 else C["neg"] for v in hist]
    fig.add_trace(go.Bar(x=t, y=hist, name="Histogram", marker_color=hc, opacity=0.75,
        hovertemplate="Histogram: %{y:.4f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["MACD"].squeeze(), name="MACD",
        line=dict(color=C["macd"],width=1.3),
        hovertemplate="MACD: %{y:.4f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["MACD_Signal"].squeeze(), name="Sinyal",
        line=dict(color=C["signal"],width=1,dash="dash"),
        hovertemplate="Sinyal: %{y:.4f}<extra></extra>"), row=3, col=1)

    # RSI
    rsi = df["RSI"].squeeze()
    fig.add_trace(go.Scatter(x=t, y=rsi, name="RSI",
        line=dict(color=C["rsi"],width=1.4),
        hovertemplate="RSI: %{y:.1f}<extra></extra>"), row=4, col=1)
    fig.add_hrect(y0=70,y1=100,fillcolor=C["neg"],opacity=0.07,row=4,col=1,line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=C["pos"],opacity=0.07,row=4,col=1,line_width=0)
    for y, clr in [(70,C["neg"]),(30,C["pos"]),(50,C["muted"])]:
        fig.add_hline(y=y, line_color=clr, line_dash="dash", line_width=0.8, row=4, col=1)

    # ── Zaman Filtresi Butonları ─────────────────────────────
    # Sadece row=1'e rangeselector; shared_xaxes=True sayesinde
    # tüm alt paneller otomatik senkronize olur.
    # rangebreaks ayrı bir update_xaxes çağrısıyla TÜM satırlara uygulanır.
    fig.update_xaxes(
        rangeselector=dict(
            bgcolor=C["card"],
            activecolor=C["price"],
            bordercolor=C["grid"],
            borderwidth=1,
            font=dict(color=C["white"], size=11),
            buttons=[
                dict(count=7,  label="1H",  step="day",   stepmode="backward"),
                dict(count=14, label="2H",  step="day",   stepmode="backward"),
                dict(count=1,  label="1A",  step="month", stepmode="backward"),
                dict(count=3,  label="3A",  step="month", stepmode="backward"),
                dict(count=6,  label="6A",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(step="all", label="MAX"),
            ],
        ),
        rangeslider=dict(visible=False),
        row=1, col=1,
    )
    # rangebreaks: hafta sonu boşluklarını tüm satırlarda kapat
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )

    fig.update_layout(
        height=800, paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(color=C["white"],size=11),
        legend=dict(bgcolor=C["card"],bordercolor=C["grid"],borderwidth=1,font=dict(size=10)),
        margin=dict(l=10,r=10,t=42,b=10),
        hovermode="x unified",
    )
    for i in range(1,5):
        fig.update_xaxes(gridcolor=C["grid"],showgrid=True,row=i,col=1)
        fig.update_yaxes(gridcolor=C["grid"],showgrid=True,row=i,col=1)
    fig.update_yaxes(title_text=f"Fiyat ({pb})", row=1, col=1)
    fig.update_yaxes(title_text="Hacim",  row=2, col=1)
    fig.update_yaxes(title_text="MACD",   row=3, col=1)
    fig.update_yaxes(title_text="RSI", range=[0,100], row=4, col=1)
    return fig

# ══════════════════════════════════════════════════════════════
#  5. MAKRO TABLO
# ══════════════════════════════════════════════════════════════
def makro_tablo_olustur(sembol, hisse_df, makro_df, bist_modu) -> pd.DataFrame:
    spy_etk = "BIST 100 (XU100)" if bist_modu else "S&P 500 (SPY)"
    h6 = getiri_hesapla(hisse_df,126); h1 = getiri_hesapla(hisse_df,252)
    rows = [{"Enstrüman":sembol,
             "6 Ay":f"%{h6:+.1f}" if h6 is not None else "---",
             "1 Yıl":f"%{h1:+.1f}" if h1 is not None else "---"}]
    for sem, etk in [("SPY",spy_etk),("GLD","Altın (GLD)"),("USO","H.Petrol (USO)")]:
        d = makro_df.get(sem)
        g6 = getiri_hesapla(d,126) if d is not None else None
        g1 = getiri_hesapla(d,252) if d is not None else None
        rows.append({"Enstrüman":etk,
                     "6 Ay":f"%{g6:+.1f}" if g6 is not None else "---",
                     "1 Yıl":f"%{g1:+.1f}" if g1 is not None else "---"})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  HISSE RADAR — Sektörel Havuz (Dev Kütüphane v21)
# ══════════════════════════════════════════════════════════════
RADAR_SEKTORLER = {
    # ─────────────────────────────────────────────────────────
    #  ABD BORSASI  (S&P 500 & Nasdaq 100 ağırlıklı)
    # ─────────────────────────────────────────────────────────
    "ABD": {
        "🌐 Tümü (Karışık)": [
            # Teknoloji / Yazılım / Çip
            "AAPL","MSFT","NVDA","GOOGL","META","AMZN","AMD","TSM",
            "INTC","CRM","ADBE","AVGO","QCOM","TXN","IBM","NOW","SNOW","PLTR",
            # Finans / Ödeme
            "V","MA","JPM","BAC","WFC","GS","MS","AXP","C","PYPL","SQ",
            # Savunma / Havacılık / Uzay
            "LMT","RTX","GD","NOC","BA","HWM","TDG","LHX",
            # Enerji / Endüstri
            "XOM","CVX","COP","SLB","GE","CAT","MMM","UNP","DE","HON",
            # Sağlık / Biyoteknoloji
            "JNJ","UNH","PFE","MRK","LLY","ABBV","AMGN","GILD","BNTX","MRNA",
            # Tüketim / Medya / Otomotiv
            "TSLA","WMT","PG","KO","PEP","MCD","NKE","NFLX",
            "DIS","SBUX","HD","COST","ABNB","UBER",
            # Yeni — Dijital / SaaS / Kripto / Yenilenebilir
            "SHOP","SPOT","ROKU","ZM","DOCU","RBLX",
            "CRWD","PANW","FTNT","DDOG","NET",
            "COIN","HOOD","MSTR","PLUG","ENPH",
        ],
        "💻 Teknoloji & Yazılım & Çip & Siber": [
            "AAPL","MSFT","NVDA","GOOGL","META","AMZN","AMD","TSM",
            "INTC","CRM","ADBE","AVGO","QCOM","TXN","IBM","NOW","SNOW","PLTR",
            "MU","AMAT","LRCX","KLAC","ASML","MRVL","PANW","CRWD","ZS","DDOG",
            # Yeni — Yazılım/SaaS
            "SHOP","SPOT","ROKU","ZM","DOCU","RBLX",
            # Yeni — Siber Güvenlik
            "FTNT","DDOG","NET",
        ],
        "🏦 Finans & Ödeme & Sigorta": [
            "JPM","BAC","WFC","GS","MS","C","AXP","BLK",
            "V","MA","PYPL","SQ","COF","USB","TFC","PNC","SCHW",
            "CB","MMC","AON","MET","PRU",
            # Yeni — Kripto / Fintech
            "COIN","HOOD","MSTR",
        ],
        "🛡️ Savunma & Havacılık & Uzay": [
            "LMT","RTX","GD","NOC","BA","HWM","TDG","LHX",
            "HII","LDOS","SAIC","AXON","KTOS","RKLB","SPCE",
        ],
        "⚡ Enerji & Endüstri & Altyapı": [
            "XOM","CVX","COP","SLB","EOG","MPC","VLO","PSX","OXY",
            "GE","CAT","MMM","UNP","DE","HON","EMR","ETN","ITW","PH",
            # Yeni — Temiz Enerji
            "PLUG","ENPH",
        ],
        "💊 Sağlık & Biyoteknoloji & İlaç": [
            "JNJ","UNH","PFE","MRK","LLY","ABBV","AMGN","GILD",
            "BNTX","MRNA","REGN","VRTX","ISRG","BMY","ABT","TMO","DHR","CI",
        ],
        "🛒 Tüketim & Medya & Otomotiv": [
            "TSLA","WMT","PG","KO","PEP","MCD","NKE","NFLX",
            "DIS","SBUX","HD","COST","ABNB","UBER","TGT","LOW",
            "F","GM","CMCSA","PARA",
        ],
    },
    # ─────────────────────────────────────────────────────────
    #  BORSA İSTANBUL  (BIST — .IS uzantılı)
    # ─────────────────────────────────────────────────────────
    "BIST": {
        "🌐 Tümü (Karışık)": [
            # Bankacılık / Finans
            "AKBNK.IS","GARAN.IS","ISCTR.IS","YKBNK.IS","HALKB.IS",
            "VAKBN.IS","ISMEN.IS","TSKB.IS","ALBRK.IS","SKBNK.IS",
            "TURSG.IS","ANSGR.IS","AKGRT.IS",
            # Holding / Yatırım
            "KCHOL.IS","SAHOL.IS","ENKAI.IS","ALARK.IS",
            "DOHOL.IS","TEZOL.IS","TKFEN.IS",
            # Sanayi / Üretim / Otomotiv
            "TUPRS.IS","EREGL.IS","KRDMD.IS","FROTO.IS","TOASO.IS",
            "SISE.IS","PETKM.IS","SASA.IS","HEKTS.IS","VESBE.IS",
            "ARCLK.IS","BRISA.IS","KORDS.IS","CEMTS.IS",
            # Havacılık / Ulaştırma
            "THYAO.IS","PGSUS.IS","DOAS.IS","TAVHL.IS","CLEBI.IS",
            # Teknoloji / Savunma / Telekom
            "ASELS.IS","TCELL.IS","TTKOM.IS","MIATK.IS","KONTK.IS",
            "YEOTK.IS","ALFAS.IS","SDTTR.IS","ARDYZ.IS","VBTYZ.IS",
            # Enerji / Elektrik
            "ASTOR.IS","GESAN.IS","SMRTG.IS","ENJSA.IS","GWIND.IS",
            "ODAS.IS","CANTE.IS","ZOREN.IS","AKSEN.IS",
            # Perakende / Gıda / İçecek
            "BIMAS.IS","MGROS.IS","SOKM.IS","CCOLA.IS","AEFES.IS",
            "TUKAS.IS","TATGD.IS","ULKER.IS",
            # Çimento / Gayrimenkul
            "EKGYO.IS","TRGYO.IS","OYAKC.IS","CIMSA.IS","AKCNS.IS","ENERY.IS",
            # Yeni eklemeler
            "MAVI.IS","YYLGD.IS","TTRAK.IS","OTKAR.IS",
            "SELEC.IS","LOGO.IS","TKNSA.IS","BRSAN.IS","JANTS.IS",
            "AYDEM.IS","KCAER.IS","CWENE.IS","EUPWR.IS","KMPUR.IS","KONTR.IS",
        ],
        "🏦 Bankacılık & Finans & Sigorta": [
            "AKBNK.IS","GARAN.IS","ISCTR.IS","YKBNK.IS","HALKB.IS",
            "VAKBN.IS","ISMEN.IS","TSKB.IS","ALBRK.IS","SKBNK.IS",
            "TURSG.IS","ANSGR.IS","AKGRT.IS","QNBFB.IS",
        ],
        "🏛️ Holding & Yatırım": [
            "KCHOL.IS","SAHOL.IS","ENKAI.IS","ALARK.IS",
            "DOHOL.IS","TEZOL.IS","TKFEN.IS","ISGSY.IS",
        ],
        "🏭 Sanayi & Üretim & Otomotiv": [
            "TUPRS.IS","EREGL.IS","KRDMD.IS","FROTO.IS","TOASO.IS",
            "SISE.IS","PETKM.IS","SASA.IS","HEKTS.IS","VESBE.IS",
            "ARCLK.IS","BRISA.IS","KORDS.IS","CEMTS.IS",
            "VESTL.IS","TTRAK.IS","OTKAR.IS","DURDO.IS",
            # Yeni — Sanayi/Metal
            "BRSAN.IS","JANTS.IS",
        ],
        "✈️ Havacılık & Ulaştırma": [
            "THYAO.IS","PGSUS.IS","DOAS.IS","TAVHL.IS","CLEBI.IS",
        ],
        "🛡️ Teknoloji & Savunma & Telekom": [
            "ASELS.IS","TCELL.IS","TTKOM.IS","MIATK.IS","KONTK.IS",
            "YEOTK.IS","ALFAS.IS","SDTTR.IS","ARDYZ.IS","VBTYZ.IS",
            "NETAS.IS","LOGO.IS","INDES.IS","KAREL.IS",
            # Yeni — Teknoloji/Yazılım/Savunma
            "SELEC.IS","TKNSA.IS","KONTR.IS",
        ],
        "⚡ Enerji & Elektrik & Yenilenebilir": [
            "ASTOR.IS","GESAN.IS","SMRTG.IS","ENJSA.IS","GWIND.IS",
            "ODAS.IS","CANTE.IS","ZOREN.IS","AKSEN.IS",
            "TUPRS.IS","PETKM.IS","SASA.IS","AYGAZ.IS","KOZAL.IS",
            # Yeni — Enerji/Elektrik
            "AYDEM.IS","KCAER.IS","CWENE.IS","EUPWR.IS","KMPUR.IS",
        ],
        "🛒 Perakende & Gıda & İçecek": [
            "BIMAS.IS","MGROS.IS","SOKM.IS","CCOLA.IS","AEFES.IS",
            "TUKAS.IS","TATGD.IS","ULKER.IS","MAVI.IS",
            # Yeni — Gıda/Perakende
            "YYLGD.IS",
        ],
        "🏗️ Çimento & Gayrimenkul & İnşaat": [
            "EKGYO.IS","TRGYO.IS","OYAKC.IS","CIMSA.IS","AKCNS.IS","ENERY.IS",
            "TKFEN.IS","ENKAI.IS","MPARK.IS","VKGYO.IS","ISGYO.IS","OZGYO.IS",
        ],
    },
}

# Geriye dönük uyumluluk — düz listeler (makro veri çekiminde kullanılır)
RADAR_ABD  = RADAR_SEKTORLER["ABD"]["🌐 Tümü (Karışık)"]
RADAR_BIST = RADAR_SEKTORLER["BIST"]["🌐 Tümü (Karışık)"]


@st.cache_data(ttl=300, show_spinner=False)
def hizli_puan_hesapla(sembol: str) -> int:
    """Hızlı 0-100 puan: yalnızca RSI+MACD+SMA (temel veri çekmeden)."""
    try:
        df = veri_cek(sembol)
        if df is None: return -1
        df = hesapla_gostergeler(df.copy())
        sd = son_degerler(df)
        p = 0; mx = 7
        rsi = sd["rsi"]
        if rsi<=30:   p+=2
        elif rsi<=50: p+=1
        elif rsi>=70: p-=2
        p += {"YUKSELIS":2,"ZAYIFLAYAN DUSUS":1,"ZAYIFLAYAN YUKSELIS":-1,"DUSUS":-2}.get(sd["macd_yon"],0)
        if sd["sma50"]  and sd["sk"] > sd["sma50"]:  p+=1
        if sd["sma200"] and sd["sk"] > sd["sma200"]: p+=1
        return max(0, min(100, round((p + mx) / (2*mx) * 100)))
    except: return -1


def radar_tara(liste: list, hedef_min: int, hedef_max: int) -> tuple:
    """
    Tüm listeyi karıştırılmış sırayla tarar.
    Kritere uyan İLK hisseyi bulunca döngüyü kırar ve döndürür.
    """
    rastgele = list(liste)
    random.shuffle(rastgele)          # her çağrıda farklı sıra
    for sem in rastgele:              # tüm liste — erken çıkışlı
        p = hizli_puan_hesapla(sem)
        if p >= 0 and hedef_min <= p <= hedef_max:
            return sem, p             # bulundu → hemen dön
    return None, -1                   # liste bitti, uygun yok

# ══════════════════════════════════════════════════════════════
#  SESSION STATE BAŞLATICI
# ══════════════════════════════════════════════════════════════
for _k, _v in [("radar_piyasa","ABD"),("radar_sembol",None),
               ("analiz_sembol","AAPL"),("analiz_baslat",False),
               ("radar_sektor","🌐 Tümü (Karışık)")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Analiz Aracı v24.0")
    st.markdown("---")

    # ── Radar Bölümü ──────────────────────────────────────────
    st.markdown("### 🔍 Yapay Zeka Hisse Radarı")
    st.caption("Piyasa ve sektör seçin, sistem ilgili havuzdan rastgele bir hisse analiz eder.")

    # 1. Piyasa seçimi (2 buton yan yana)
    rb1, rb2 = st.columns(2)
    if rb1.button("🇺🇸 ABD Borsası", use_container_width=True,
                  type="primary" if st.session_state["radar_piyasa"]=="ABD" else "secondary"):
        st.session_state["radar_piyasa"] = "ABD"
        st.session_state["radar_sektor"] = "🌐 Tümü (Karışık)"  # piyasa değişince sektörü sıfırla
        st.rerun()
    if rb2.button("🇹🇷 BIST", use_container_width=True,
                  type="primary" if st.session_state["radar_piyasa"]=="BIST" else "secondary"):
        st.session_state["radar_piyasa"] = "BIST"
        st.session_state["radar_sektor"] = "🌐 Tümü (Karışık)"
        st.rerun()

    # 2. Sektör selectbox — seçili piyasanın sektörlerini göster
    _piyasa_sek = st.session_state["radar_piyasa"]
    _sektor_listesi = list(RADAR_SEKTORLER[_piyasa_sek].keys())

    # Kaydedilmiş sektör geçerli mi? (piyasa değişmiş olabilir)
    _kayitli = st.session_state["radar_sektor"]
    _default_idx = (_sektor_listesi.index(_kayitli)
                    if _kayitli in _sektor_listesi else 0)

    secili_sektor = st.selectbox(
        "🎯 Sektör / Tema Seçin",
        options=_sektor_listesi,
        index=_default_idx,
        key="sektor_secim_widget",
        help="Tümü seçilirse tüm sektörlerden karma seçim yapılır.",
    )
    # Selectbox değişikliğini session'a kaydet (rerun gerekmez)
    st.session_state["radar_sektor"] = secili_sektor

    # Seçilen havuzu göster
    _secili_liste = RADAR_SEKTORLER[_piyasa_sek][secili_sektor]
    st.caption(f"Havuzda **{len(_secili_liste)}** hisse var.")

    # 3. "Bana Hisse Bul" butonu
    if st.button("🎲 Bana Hisse Bul", use_container_width=True, type="primary"):
        secilen = random.choice(_secili_liste)
        st.session_state["analiz_sembol"] = secilen
        st.session_state["analiz_baslat"] = True
        st.session_state["radar_sembol"]  = secilen
        st.rerun()

    st.markdown("---")
    # ── Manuel Giriş ──────────────────────────────────────────
    st.markdown("### ✏️ Manuel Sembol Girişi")
    sembol_input = st.text_input(
        "Hisse Sembolü",
        value=st.session_state["analiz_sembol"],
        placeholder="Örn: AAPL, THYAO.IS, LMT",
        help="BIST hisseleri için .IS ekleyin. Örnek: THYAO.IS, GARAN.IS",
        key="sembol_input_widget",
    ).strip().upper()
    analiz_btn = st.button("🔍 Analizi Başlat", use_container_width=True, type="primary")

    # Radar veya buton tetiklediyse session güncelle
    if analiz_btn:
        st.session_state["analiz_sembol"]  = sembol_input
        st.session_state["analiz_baslat"]  = True

    st.markdown("---")
    # ── Hızlı Menü (Anchor Navigasyon) ──────────────────────
    st.markdown("### 🗂️ Sekmeler")
    st.markdown(
        "- 📈 **Tab 1** — Grafik & Trend\n"
        "- 🧠 **Tab 2** — YZ Kararı & Psikoloji\n"
        "- 📊 **Tab 3** — Derin Analiz"
    )
    st.markdown("---")
    st.markdown("""
**Özellikler (v24.0):**
- 🔍 YZ Hisse Radarı (ABD/BIST)
- 📐 RSI · MACD · ATR · SMA50/200
- 📊 MA5/10 · Stokastik · Pivot
- 🏢 Temel Analiz + Temettü
- 🌍 Makro Kıyaslama (SPY/BIST100)
- 📰 Haber & Duyarlılık Analizi
- 🎯 Kısa · Orta · Uzun Vade Tahmin
""")
    st.markdown("---")
    st.caption("⚠️ Yalnızca bilgilendirme amaçlıdır. Yatırım tavsiyesi değildir.")

# ══════════════════════════════════════════════════════════════
#  ANA UYGULAMA
# ══════════════════════════════════════════════════════════════
st.title("📊 Kantitatif Teknik & Temel Analiz Aracı")
st.caption("v17.0  |  Güçlü Retry · Tam Tarayıcı Profili · Tek Buton Mantığı")

# Session state'den aktif sembol ve tetik bilgisini al
_analiz_baslat = st.session_state.get("analiz_baslat", False) or analiz_btn
_aktif_sembol  = st.session_state.get("analiz_sembol", sembol_input) or sembol_input

if not _analiz_baslat:
    st.info("👈 Soldaki panelden hisse sembolünü girin ve **Analizi Başlat** butonuna tıklayın.")
    st.stop()

# Tetik tüketildi — bir sonraki yenilemede tekrar çalışmasın
st.session_state["analiz_baslat"] = False
sembol_input = _aktif_sembol

# ── Veri Çekimi ──────────────────────────────────────────────
bist_modu     = sembol_input.endswith(".IS")
pb            = "₺" if bist_modu else "$"
zaman_str     = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
bitis_str     = datetime.today().strftime("%Y-%m-%d")
baslangic_str = (datetime.today()-timedelta(days=365)).strftime("%Y-%m-%d")

with st.spinner(f"📡 {sembol_input} verileri çekiliyor..."):
    df_ham   = veri_cek(sembol_input)
    temel    = temel_veri_cek(sembol_input)
    makro_df = makro_veri_cek(baslangic_str, bitis_str, bist_modu)
    haberler = haber_cek(sembol_input)

if df_ham is None:
    st.error(
    f"❌ **{sembol_input}** için fiyat verisi alınamadı. "
    "Yahoo Finance sunucuları şu an yanıt vermiyor olabilir. "
    "Lütfen 10–15 saniye bekleyip tekrar deneyin. "
    "Sembol yanlışsa düzeltin (BIST için .IS ekleyin: THYAO.IS)."
)
    st.stop()

df = hesapla_gostergeler(df_ham.copy())
sd = son_degerler(df)

haber_tonu = "NOTR"
if haberler:
    p = sum(1 for h in haberler if h["ton"]=="POZITIF")
    n = sum(1 for h in haberler if h["ton"]=="NEGATIF")
    haber_tonu = "POZITIF" if p>n else ("NEGATIF" if n>p else "NOTR")

tahmin = tahmin_hesapla(sd, temel, makro_df, df, haber_tonu, bist_modu)

# ══════════════════════════════════════════════════════════════
#  HEADER: Fiyat + 52 Haftalık Aralık (her zaman görünür)
# ══════════════════════════════════════════════════════════════
st.markdown("---")

genel_k   = tahmin["genel_karar"]
genel_100 = tahmin["genel_100"]
renk      = karar_rengi(genel_k)
bar_str   = "█"*int(genel_100/10) + "░"*(10-int(genel_100/10))

hdr_l, hdr_r = st.columns([3, 1])
hdr_l.markdown(
    f"## {karar_emoji(genel_k)} {temel.get('sirket_adi', sembol_input)}  "
    f"<span style='color:#8b949e;font-size:1rem;'>({sembol_input})</span>",
    unsafe_allow_html=True
)
hdr_r.markdown(
    f'<div style="text-align:right;">'
    f'<div style="font-size:1.8rem;font-weight:800;color:{renk};">{_fp(sd["sk"],pb)}</div>'
    f'<div style="font-size:0.78rem;color:{renk};">{genel_k} · {genel_100}/100 {bar_str}</div>'
    f'</div>',
    unsafe_allow_html=True
)

# 52H progress bar
_dip_h  = _f(temel.get("haftalik_dip"))
_zirve_h= _f(temel.get("haftalik_zirve"))
_sk_h   = sd["sk"]
if _dip_h is not None and _zirve_h is not None and _zirve_h > _dip_h:
    _aralik    = _zirve_h - _dip_h
    _konum     = max(0.0, min(1.0, (_sk_h - _dip_h) / _aralik))
    _konum_pct = round(_konum * 100, 1)
    _dip_r  = C["neg"] if _konum < 0.25 else C["muted"]
    _zir_r  = C["pos"] if _konum > 0.75 else C["muted"]
    _hbc1, _hbc2, _hbc3 = st.columns([1, 5, 1])
    _hbc1.markdown(
        f'<div style="font-size:0.7rem;color:#8b949e;">52H Dip</div>'
        f'<div style="font-size:0.95rem;font-weight:700;color:{_dip_r};">{_fp(_dip_h,pb)}</div>',
        unsafe_allow_html=True)
    with _hbc2:
        st.progress(_konum)
        _lok = ("📍 Dip yakını" if _konum < 0.25
                else "📍 Zirve yakını" if _konum > 0.75 else "📍 Orta bölge")
        st.caption(f"Fiyat 52H aralığının **%{_konum_pct}**'inde  {_lok}")
    _hbc3.markdown(
        f'<div style="text-align:right;font-size:0.7rem;color:#8b949e;">52H Zirve</div>'
        f'<div style="text-align:right;font-size:0.95rem;font-weight:700;color:{_zir_r};">{_fp(_zirve_h,pb)}</div>',
        unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  3 SEKMELİ PANEL
# ══════════════════════════════════════════════════════════════
psi     = korku_skoru_hesapla(sd, df)
atr_pct = (sd["atr"]/sd["sk"]*100) if sd["atr"] and sd["sk"] else None

tab1, tab2, tab3 = st.tabs([
    "📈 Grafik & Trend",
    "🧠 YZ Kararı & Psikoloji",
    "📊 Derin Analiz",
])

# ─────────────────────────────────────────────────────────────
#  TAB 1 — GRAFİK & TREND
# ─────────────────────────────────────────────────────────────
with tab1:
    # Stop-Loss / ATR
    if sd["atr"] and sd["sk"] and atr_pct is not None:
        _sl1  = sd["sk"] - 1.5*sd["atr"]
        _slp1 = (_sl1 - sd["sk"]) / sd["sk"] * 100
        _vs1  = "🟢 Düşük" if atr_pct<1.5 else ("🔴 Yüksek" if atr_pct>3 else "🟡 Orta")
        st.info(f"🛡️ **Stop-Loss:** {_fp(_sl1,pb)} (kapanışın **%{_slp1:.2f}** altı)  |  "
                f"ATR: {_fp(sd['atr'],pb)}  |  Volatilite: {_vs1}")
        with st.expander("ℹ️ Stop-Loss & ATR"):
            st.caption(TIP["STOP"]); st.caption(TIP["ATR"])

    # Mum grafiği
    st.plotly_chart(grafik_ciz(df, sembol_input, pb), use_container_width=True)

    # Makro kıyaslama
    st.subheader("🌍 Makro Kıyaslama")
    st.dataframe(makro_tablo_olustur(sembol_input, df, makro_df, bist_modu),
                 use_container_width=True, hide_index=True)

    # Haberler
    st.subheader("📰 Güncel Haberler")
    _ton_rk = {"POZITIF":C["pos"],"NEGATIF":C["neg"],"NOTR":C["gold"]}
    if haberler:
        for _h in haberler:
            _hr = _ton_rk.get(_h["ton"], C["gold"])
            st.markdown(
                f'<div class="news-item" style="border-left-color:{_hr};">'
                f'<span style="color:{_hr};font-weight:700;font-size:0.78rem;">'
                f'{karar_emoji(_h["ton"])} {_h["ton"]}</span>'
                f'<span style="color:#8b949e;font-size:0.78rem;margin-left:10px;">'
                f'{_h["zaman"]} · {_h["yayinci"]}</span><br>'
                f'<span style="color:#e6edf3;font-size:0.9rem;">{_h["baslik"]}</span></div>',
                unsafe_allow_html=True
            )
        st.caption(f"Genel Haber Tonu: **{haber_tonu}**")
    else:
        st.caption("Bu sembol için güncel haber akışı bulunamadı.")

# ─────────────────────────────────────────────────────────────
#  TAB 2 — YZ KARARI & PSİKOLOJİ
# ─────────────────────────────────────────────────────────────
with tab2:
    # Genel karar banner
    st.markdown(f"""
<div style="background:{renk}18;border:2px solid {renk};border-radius:12px;
            padding:18px 24px;margin-bottom:16px;text-align:center;">
  <div style="font-size:1.8rem;font-weight:800;color:{renk};">{karar_emoji(genel_k)} {genel_k}</div>
  <div style="font-size:0.9rem;color:#e6edf3;margin-top:4px;">
    Algoritma Puanı: <b>{genel_100}/100</b> &nbsp;|&nbsp; {bar_str}
  </div>
  <div style="font-size:0.78rem;color:#8b949e;margin-top:4px;">{zaman_str}</div>
</div>
""", unsafe_allow_html=True)

    # 3 vade kartı
    _vc1, _vc2, _vc3 = st.columns(3)
    for _vcol, _vikon, _vetk, _vk, _v100, _vbant in [
        (_vc1,"⚡","Kısa Vade (1-4H)", tahmin["kv_karar"], tahmin["kv_100"], tahmin["kv_bant"]),
        (_vc2,"📅","Orta Vade (3-6A)", tahmin["ov_karar"], tahmin["ov_100"], tahmin["ov_bant"]),
        (_vc3,"🏦","Uzun Vade (12A+)", tahmin["uv_karar"], tahmin["uv_100"], tahmin["uv_bant"]),
    ]:
        _vr = karar_rengi(_vk)
        _vcol.markdown(f"""
<div style="background:{_vr}14;border:1px solid {_vr};border-radius:10px;padding:14px;">
  <div style="font-size:0.77rem;color:#8b949e;">{_vikon} {_vetk}</div>
  <div style="font-size:1.3rem;font-weight:700;color:{_vr};">{karar_emoji(_vk)} {_vk}</div>
  <div style="font-size:0.9rem;color:#e6edf3;font-weight:600;">{_v100}/100</div>
  <div class="forecast-band" style="margin-top:8px;border-left-color:{_vr};font-size:0.78rem;color:#c9d1d9;">
    {_vbant}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Gauge + psikoloji badge
    _gc, _bc = st.columns([1, 1.6])
    with _gc:
        st.plotly_chart(korku_gauge_ciz(psi, sembol_input),
                        use_container_width=True, config={"displayModeBar": False})
    with _bc:
        _pr = psi["renk"]
        st.markdown(
            f'<div style="background:{_pr}15;border:1px solid {_pr};border-radius:10px;'
            f'padding:16px 20px;">'
            f'<div style="font-size:1.7rem;font-weight:800;color:{_pr};">'
            f'{psi["skor"]}/100 — {psi["etiket"]}</div>'
            f'<div style="color:#8b949e;font-size:0.82rem;margin-top:8px;">'
            f'Bu hisse için mevcut piyasa psikolojisi</div></div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        with st.expander("ℹ️ Psikoloji Skoru nasıl hesaplanır?"):
            st.markdown(
                "Bu skor üç bileşenden oluşur:\n\n"
                f"- **RSI Bileşeni (%40):** {psi['rsi_skor']:.1f}\n"
                f"- **SMA Sapması (%30):** {psi['sma_skor']:.1f}\n"
                f"- **Hacim-Fiyat Dinamiği (%30):** {psi['hfp_skor']:.1f}\n\n"
                "**0-25:** 🔴 Aşırı Korku  |  **25-45:** 🟠 Korku  |  "
                "**45-55:** ⚪ Nötr  |  **55-75:** 🟢 Açgözlülük  |  "
                "**75-100:** 💚 Aşırı Açgözlülük"
            )

    st.markdown("---")
    strateji_goster(tahmin, sd, temel)

# ─────────────────────────────────────────────────────────────
#  TAB 3 — DERİN ANALİZ
# ─────────────────────────────────────────────────────────────
with tab3:
    # ATR / Stop-Loss özet metrik
    _da1, _da2 = st.columns(2)
    _da1.metric("Son Kapanış", _fp(sd["sk"],pb))
    if atr_pct is not None:
        _sl3  = sd["sk"] - 1.5*sd["atr"]
        _slp3 = (_sl3 - sd["sk"]) / sd["sk"] * 100
        _da2.metric("Önerilen Stop-Loss", _fp(_sl3,pb),
                    f"%{_slp3:.2f}  (ATR: {_fp(sd['atr'],pb)})",
                    delta_color="off", help=TIP["STOP"])

    # Teknik iç sekmeler
    st.subheader("📐 Teknik İndikatörler")
    _tm, _tt, _tp = st.tabs(["📊 Momentum", "📈 Trend Ortalamaları", "🎯 Pivot / Destek-Direnç"])

    with _tm:
        _mm1, _mm2, _mm3 = st.columns(3)
        _rv = sd["rsi"]
        if   _rv >= 70: _rb, _rc = "🔴 Aşırı Alım — Satış riski", C["neg"]
        elif _rv <= 30: _rb, _rc = "🟢 Aşırı Satım — Alım fırsatı", C["pos"]
        elif _rv >= 60: _rb, _rc = "🟢 Güçlü Momentum", C["pos"]
        elif _rv <= 40: _rb, _rc = "🔴 Zayıf Momentum", C["neg"]
        else:           _rb, _rc = "🟡 Nötr Bölge", C["gold"]
        _mm1.metric("RSI (14)", f"{_rv:.1f}", delta_color="off", help=TIP["RSI"])
        _mm1.markdown(f'<span style="color:{_rc};font-size:0.82rem;">{_rb}</span>',
                      unsafe_allow_html=True)
        _myn = sd["macd_yon"]
        if   _myn=="YUKSELIS":          _mb, _mc = "🟢 Yükseliş Momentumu", C["pos"]
        elif _myn=="DUSUS":             _mb, _mc = "🔴 Düşüş Momentumu", C["neg"]
        elif _myn=="ZAYIFLAYAN YUKSELIS":_mb,_mc = "🟡 Yükseliş Zayıflıyor", C["gold"]
        else:                           _mb, _mc = "🟡 Düşüş Yavaşlıyor", C["gold"]
        _mm2.metric("MACD Yönü", _myn, delta_color="off", help=TIP["MACD"])
        _mm2.markdown(f'<span style="color:{_mc};font-size:0.82rem;">{_mb}</span>',
                      unsafe_allow_html=True)
        _skv = sd.get("stoch_k"); _sdv = sd.get("stoch_d")
        if _skv is not None:
            if   _skv>=80: _sb,_sc = "🔴 Aşırı Alım — Dikkat", C["neg"]
            elif _skv<=20: _sb,_sc = "🟢 Aşırı Satım — Alım Fırsatı", C["pos"]
            else:          _sb,_sc = "🟡 Nötr Bölge (20-80)", C["gold"]
            _mm3.metric("Stokastik %K(9)", f"{_skv:.1f}", delta_color="off", help=TIP["STOCH"])
            _mm3.markdown(f'<span style="color:{_sc};font-size:0.82rem;">{_sb}</span>',
                          unsafe_allow_html=True)
            if _sdv is not None:
                _kdb = "🟢 K>D (Alım)" if _skv>_sdv else "🔴 K<D (Satış)"
                _kdc = C["pos"] if _skv>_sdv else C["neg"]
                _mm3.markdown(f'<span style="color:{_kdc};font-size:0.78rem;">%D={_sdv:.1f} — {_kdb}</span>',
                              unsafe_allow_html=True)
        else:
            _mm3.metric("Stokastik %K(9)", "Veri Yok", delta_color="off", help=TIP["STOCH"])

    with _tt:
        _t1, _t2, _t3, _t4 = st.columns(4)
        _m5=sd.get("ma5"); _m10=sd.get("ma10")
        _s50=sd["sma50"]; _s200=sd["sma200"]; _sk_t=sd["sk"]
        def _mab(col_,lbl_,val_,tip_):
            col_.metric(lbl_, _fp(val_,pb) if val_ else "—", delta_color="off", help=TIP[tip_])
            if val_:
                _u = _sk_t > val_
                col_.markdown(
                    f'<span style="color:{C["pos"] if _u else C["neg"]};font-size:0.82rem;">'
                    f'{"🟢 Fiyat Üstünde" if _u else "🔴 Fiyat Altında"}</span>',
                    unsafe_allow_html=True)
        _mab(_t1,"MA 5",  _m5,  "MA5")
        _mab(_t2,"MA 10", _m10, "MA10")
        _mab(_t3,"SMA 50",_s50, "SMA50")
        _mab(_t4,"SMA 200",_s200,"SMA200")
        if _m5 and _m10:
            if _m5>_m10: st.success("📈 **MA5 > MA10** — Kısa vade yükseliş yönünde.")
            else:        st.error("📉 **MA5 < MA10** — Kısa vade baskı devam ediyor.")
        if _s50 and _s200:
            if _s50>_s200: st.success("✨ **SMA50 > SMA200** — Altın Kesişim. Uzun vade yükseliş.")
            else:          st.error("💀 **SMA50 < SMA200** — Ölüm Kesişimi. Uzun vade düşüş.")

    with _tp:
        _pvt=sd.get("pvt"); _r1=sd.get("r1"); _r2=sd.get("r2")
        _s1=sd.get("s1");   _s2=sd.get("s2"); _skp=sd["sk"]
        if _pvt is not None:
            def _pb2(col_,lbl_,sev_,tip_):
                col_.metric(lbl_, _fp(sev_,pb), delta_color="off", help=TIP["PIVOT"])
                if sev_ is not None:
                    if tip_=="destek":
                        _u=_skp>=sev_; _c=C["pos"] if _u else C["neg"]
                        _l="🟢 Üstünde — Tutuyor" if _u else "🔴 Altında — Kırıldı!"
                    else:
                        _u=_skp>=sev_; _c=C["pos"] if _u else "#e3b341"
                        _l="🟢 Kırdı! — Yükseliş" if _u else "🟡 Geçilemedi"
                    col_.markdown(f'<span style="color:{_c};font-size:0.80rem;">{_l}</span>',
                                  unsafe_allow_html=True)
            _pp1,_pp2,_pp3,_pp4,_pp5 = st.columns(5)
            _pb2(_pp1,"S2",_s2,"destek"); _pb2(_pp2,"S1",_s1,"destek")
            _pb2(_pp3,"P (Pivot)",_pvt,"destek")
            _pb2(_pp4,"R1",_r1,"direnc"); _pb2(_pp5,"R2",_r2,"direnc")
            st.markdown("---")
            if _r2 and _skp>_r2:    st.success(f"💥 R2 kırdı ({_fp(_r2,pb)}) — Çok güçlü yükseliş!")
            elif _r1 and _skp>_r1:  st.success(f"📈 R1 kırdı ({_fp(_r1,pb)}) — Yükseliş momentumu.")
            elif _skp>_pvt:         st.info(f"🟡 Pivot ({_fp(_pvt,pb)}) üstünde — Pozitif bölge.")
            elif _s1 and _skp>_s1:  st.warning(f"⚠️ S1 ({_fp(_s1,pb)}) üstünde — Zayıf bölge.")
            elif _s2 and _skp>_s2:  st.error(f"📉 S1 kırıldı ({_fp(_s1,pb) if _s1 else '?'}) — Düşüş baskısı.")
            else:                    st.error("💀 S2 de kırıldı — Kritik düşüş!")
            with st.expander("ℹ️ Pivot Formülü"): st.caption(TIP["PIVOT"])
        else:
            st.info("Pivot için yeterli veri yok.")

    # Temel Analiz
    st.markdown("---")
    st.subheader("🏢 Temel Analiz & Şirket Sağlığı")

    def _fmt_m3(d_, pb_):
        if d_ is None: return "Veri Yok"
        if d_>=1e12: return f"{pb_}{d_/1e12:.2f}T"
        if d_>=1e9:  return f"{pb_}{d_/1e9:.2f}Mrd"
        if d_>=1e6:  return f"{pb_}{d_/1e6:.2f}Mn"
        return f"{pb_}{d_:,.0f}"

    _dmik = temel.get("_temettu_miktar"); _dtar = temel.get("son_temettu_tarih","Veri Yok")
    _dver = temel.get("_temettu_verimi")
    _dstr = (f"{_dtar}  ({pb}{_dmik:.4f}/hisse)" +
             (f"  · Verim: %{_dver:.2f}" if _dver else "")
             if _dmik else "Açıklanmadı / Veri Yok")

    _dt1, _dt2 = st.columns(2)
    with _dt1:
        st.markdown(f"""
<div class="card">
  <div class="card-label">Şirket Kimliği</div>
  <div class="card-value" style="font-size:1.05rem;">{temel.get('sirket_adi','---')}</div>
  <div class="card-sub">
    {temel.get('sektor','---')} · {temel.get('endustri','---')}<br>
    <b>Piyasa Değeri:</b> {_fmt_m3(temel.get('piyasa_degeri'),pb)}<br>
    <b>Yaklaşan Bilanço:</b> {temel.get('bilanco_tarihi','---')}<br>
    <b>Son Temettü:</b> {_dstr}
  </div>
</div>
""", unsafe_allow_html=True)
        _hdf = temel.get("hedef_fiyat"); _pts = tahmin.get("potans")
        _rh  = C["pos"] if _pts and _pts>0 else C["neg"] if _pts else C["muted"]
        st.markdown(f"""
<div class="card">
  <div class="card-label">Analist Hedef Fiyatı</div>
  <div class="card-value">{_fp(_hdf,pb)}</div>
  <div class="card-sub">Mevcut: {_fp(sd['sk'],pb)} &nbsp;|&nbsp;
    Potansiyel: <b style="color:{_rh};">{_pct(_pts)}</b></div>
</div>
""", unsafe_allow_html=True)
        with st.expander("ℹ️ Analist Hedef"): st.caption(TIP["HEDEF"])

    with _dt2:
        _fk=temel.get("fk_orani"); _km=temel.get("kar_marji"); _bd=temel.get("borc_ozkaynak")
        _fkr = C["pos"] if _fk and _fk<15 else (C["neg"] if _fk and _fk>30 else C["gold"])
        _kmr = C["pos"] if _km and _km*100>15 else (C["neg"] if _km and _km*100<0 else C["gold"])
        _bdr = C["pos"] if _bd and _bd<50 else (C["neg"] if _bd and _bd>150 else C["gold"])
        st.markdown(f"""
<div class="card">
  <div class="card-label">Temel Rasyolar</div>
  <table style="width:100%;color:#e6edf3;font-size:0.88rem;border-collapse:collapse;">
    <tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 0;">F/K Oranı</td>
        <td style="color:{_fkr};font-weight:700;text-align:right;">{f"{_fk:.1f}x" if _fk is not None else "Veri Yok"}</td></tr>
    <tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 0;">Kâr Marjı</td>
        <td style="color:{_kmr};font-weight:700;text-align:right;">{f"%{_km*100:.1f}" if _km is not None else "Veri Yok"}</td></tr>
    <tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 0;">Borç/Özkaynak</td>
        <td style="color:{_bdr};font-weight:700;text-align:right;">{f"{_bd:.1f}" if _bd is not None else "Veri Yok"}</td></tr>
    <tr><td style="padding:5px 0;">Temettü Verimi</td>
        <td style="color:#3fb950;font-weight:700;text-align:right;">{f"%{_dver:.2f}" if _dver else "---"}</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)
        with st.expander("ℹ️ Rasyolar Hakkında"):
            st.caption(TIP["FK"]); st.caption(TIP["KM"])
            st.caption(TIP["BD"]); st.caption(TIP["TEMETTU"])
        _artı=[]; _eksi=[]
        if _fk:
            if _fk<15: _artı.append(f"F/K {_fk:.1f}x — ucuz")
            elif _fk>30: _eksi.append(f"F/K {_fk:.1f}x — primli")
        if _km:
            if _km*100>15: _artı.append(f"Kâr Marjı %{_km*100:.1f} — yüksek")
            elif _km*100<0: _eksi.append("Kâr Marjı negatif — zarar")
        if _bd:
            if _bd<50: _artı.append(f"B/Ö {_bd:.0f} — düşük borç")
            elif _bd>150: _eksi.append(f"B/Ö {_bd:.0f} — yüksek borç")
        if _pts and _pts>0: _artı.append(f"Hedef potansiyeli: %{_pts:.1f}")
        elif _pts and _pts<0: _eksi.append(f"Fiyat hedefi %{abs(_pts):.1f} aştı")
        if _dmik: _artı.append(f"Temettü: {pb}{_dmik:.4f}/hisse")
        st.markdown(
            '<div class="card"><div class="card-label">Artıları & Eksileri</div>' +
            ''.join(f'<div style="color:#3fb950;font-size:0.83rem;">✅ {a}</div>' for a in _artı) +
            ('' if _artı else '<div style="color:#8b949e;font-size:0.83rem;">—</div>') +
            ''.join(f'<div style="color:#f85149;font-size:0.83rem;">❌ {e}</div>' for e in _eksi) +
            '</div>',
            unsafe_allow_html=True
        )

    # Şirket Profili
    st.markdown("---")
    _ozet = temel.get("ozet","")
    if _ozet:
        with st.expander("🏛️ Şirket Profili — Hakkında", expanded=False):
            _sp1,_sp2,_sp3 = st.columns(3)
            _sp1.markdown(f"**Sektör:** {temel.get('sektor','---')}")
            _sp1.markdown(f"**Endüstri:** {temel.get('endustri','---')}")
            _sp2.markdown(f"**Ülke:** {temel.get('ulke','---')}")
            _cal = temel.get("calisan_sayisi")
            _sp2.markdown(f"**Çalışan:** {f'{_cal:,}' if _cal else '---'}")
            _web = temel.get("web","")
            _sp3.markdown(f"**Web:** [{_web}]({_web})" if _web else "**Web:** ---")
            st.markdown("---")
            st.caption(_ozet[:400]+("..." if len(_ozet)>400 else ""))
    else:
        with st.expander("🏛️ Şirket Profili — Hakkında", expanded=False):
            st.caption(f"**Sektör:** {temel.get('sektor','---')}  |  "
                       f"**Endüstri:** {temel.get('endustri','---')}  |  "
                       f"**Ülke:** {temel.get('ulke','---')}")
            st.caption("Detaylı özet mevcut değil.")

st.caption("⚠️ Bu araç yalnızca bilgilendirme amaçlıdır. Yatırım kararlarınızı lisanslı bir finansal danışmanla alınız.")
