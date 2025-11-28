import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr

st.set_page_config(page_title="Dashboard LD Guru & CT Siswa", layout="wide")
st.title("Literasi Digital Guru & Computational Thinking Siswa")

# =====================================================
# 1. LOAD DATA
# =====================================================
guru_path = r"files/data_LD_guru.csv"
siswa_path = r"files/data_CT_siswa.csv"

df_guru = pd.read_csv(guru_path, sep=";")
df_siswa = pd.read_csv(siswa_path, sep=";")

st.success(f"Data guru: {len(df_guru)} baris, Data siswa: {len(df_siswa)} baris")

# =====================================================
# 2. PROSES DATA GURU
# =====================================================
likert_cols = [col for col in df_guru.columns if pd.api.types.is_numeric_dtype(df_guru[col]) 
               and df_guru[col].min() >=1 and df_guru[col].max() <=5]

dim_skill = likert_cols[0:18]
dim_etika = likert_cols[18:25]
dim_keamanan = likert_cols[25:33]
dim_budaya = likert_cols[33:40]

for dim, cols in zip(["Skill_Digital","Etika_Digital","Keamanan_Digital","Budaya_Digital"],
                     [dim_skill, dim_etika, dim_keamanan, dim_budaya]):
    df_guru[dim] = df_guru[cols].mean(axis=1)

df_guru["Total_LD"] = df_guru[["Skill_Digital","Etika_Digital","Keamanan_Digital","Budaya_Digital"]].sum(axis=1)
df_guru["Mean_LD"] = df_guru[["Skill_Digital","Etika_Digital","Keamanan_Digital","Budaya_Digital"]].mean(axis=1)

def kategori_ld(skor):
    if skor < 2.34:
        return "Rendah"
    elif skor < 3.67:
        return "Sedang"
    else:
        return "Tinggi"

df_guru["Level_LD"] = df_guru["Mean_LD"].apply(kategori_ld)

# =====================================================
# 3. PROSES DATA SISWA
# =====================================================
ct_cols = [col for col in df_siswa.columns if col.startswith("S ")]
for col in ct_cols:
    df_siswa[col] = pd.to_numeric(df_siswa[col].astype(str).str.replace(",", "."), errors="coerce").fillna(0)

df_siswa["Total_CT"] = df_siswa[ct_cols].sum(axis=1)
df_siswa["Mean_CT"] = df_siswa[ct_cols].mean(axis=1)

soal_cols_per_jenjang = {
    "Siaga": [col for col in ct_cols if df_siswa.loc[df_siswa["Kategori"]=="Siaga", col].notna().any()],
    "Penggalang": [col for col in ct_cols if df_siswa.loc[df_siswa["Kategori"]=="Penggalang", col].notna().any()],
    "Penegak": [col for col in ct_cols if df_siswa.loc[df_siswa["Kategori"]=="Penegak", col].notna().any()]
}

def skor_max(row):
    jenjang = row["Kategori"]
    if jenjang == "Siaga":
        return len(soal_cols_per_jenjang["Siaga"]) * 8.33
    else:
        return len(soal_cols_per_jenjang[jenjang]) * 6.67

df_siswa["Persentase_CT"] = df_siswa.apply(lambda row: row["Total_CT"]/skor_max(row), axis=1)

def kategori_ct(df, skor_col="Persentase_CT", jenjang_col="Kategori"):
    df["Level_CT"] = ""
    for jenjang in df[jenjang_col].unique():
        mask = df[jenjang_col] == jenjang
        q1 = df.loc[mask, skor_col].quantile(0.33)
        q2 = df.loc[mask, skor_col].quantile(0.66)
        df.loc[mask, "Level_CT"] = df.loc[mask, skor_col].apply(lambda x: "Rendah" if x<=q1 else ("Sedang" if x<=q2 else "Tinggi"))
    return df

df_siswa = kategori_ct(df_siswa)

# =====================================================
# 4. VISUALISASI DENGAN TABS
# =====================================================
tab_guru, tab_siswa, tab_perbandingan = st.tabs(["Guru", "Siswa", "Perbandingan"])

with tab_guru:
    st.subheader("Distribusi Level LD Guru")
    col1, col2 = st.columns(2)
    with col1:
        st.write(df_guru["Level_LD"].value_counts())
    with col2:
        fig1 = px.pie(df_guru, names="Level_LD", title="Proporsi Level LD Guru")
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Level LD Guru per Sekolah")
    level_ld_options = ["Rendah","Sedang","Tinggi"]
    for level in level_ld_options:
        st.markdown(f"**Level {level}**")
        sekolah_level = df_guru[df_guru["Level_LD"]==level][["NAMA", "Asal Instansi","Mean_LD"]]
        st.dataframe(sekolah_level.reset_index(drop=True))

with tab_siswa:
    st.subheader("Distribusi Level CT Siswa")
    col1, col2 = st.columns(2)
    with col1:
        st.write(df_siswa["Level_CT"].value_counts())
    with col2:
        fig2 = px.pie(df_siswa, names="Level_CT", title="Proporsi Level CT Siswa")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Level CT Siswa per Sekolah")
    level_ct_options = ["Rendah","Sedang","Tinggi"]
    for level in level_ct_options:
        st.markdown(f"**Level {level}**")
        sekolah_level = df_siswa[df_siswa["Level_CT"]==level][["Nama" ,"Kategori", "SekolahNama","Mean_CT"]]
        st.dataframe(sekolah_level.reset_index(drop=True))

with tab_perbandingan:
    df_guru['sekolah_key'] = df_guru['Asal Instansi'].apply(lambda x: " ".join(str(x).lower().split()[:3]))
    df_siswa['sekolah_key'] = df_siswa['SekolahNama'].apply(lambda x: " ".join(str(x).lower().split()[:3]))

    guru_per_sekolah = df_guru.groupby('sekolah_key').agg({
        'Asal Instansi':'first',
        'Mean_LD':'mean',
        'Level_LD': lambda x: x.mode()[0]
    }).reset_index()

    siswa_per_sekolah = df_siswa.groupby(['sekolah_key','SekolahNama']).agg({
        'Mean_CT':'mean',
        'Level_CT': lambda x: x.mode()[0]
    }).reset_index()

    df_perbandingan = guru_per_sekolah.merge(siswa_per_sekolah, on="sekolah_key", how="inner")[[
        'Asal Instansi','SekolahNama','Level_LD','Level_CT','Mean_LD','Mean_CT'
    ]]

    st.subheader("Perbandingan Guru & Siswa per Sekolah")
    st.dataframe(df_perbandingan)
    r, p_value = pearsonr(df_perbandingan['Mean_LD'], df_perbandingan['Mean_CT'])
    st.success(f"Korelasi Pearson r = {r:.2f}, p-value = {p_value:.2f}")

    # Interpretasi
    if abs(r) < 0.2:
        strength = "sangat lemah"
    elif abs(r) < 0.4:
        strength = "lemah"
    elif abs(r) < 0.6:
        strength = "sedang"
    elif abs(r) < 0.8:
        strength = "kuat"
    else:
        strength = "sangat kuat"

    significance = "signifikan" if p_value < 0.05 else "tidak signifikan"

    st.warning(f"Hubungan antara rata-rata Literasi Digital Guru dan rata-rata Computational Thinking Siswa tergolong **{strength}** dan **{significance}** secara statistik.")

    st.subheader("Visualisasi Scatter Guru vs Siswa")
    fig3 = px.scatter(df_perbandingan, x="Mean_LD", y="Mean_CT",
                      color="Level_LD", hover_data=["Asal Instansi","SekolahNama"],
                      title="Scatter Mean LD Guru vs Mean CT Siswa")
    st.plotly_chart(fig3, use_container_width=True)
