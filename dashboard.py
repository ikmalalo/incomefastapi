import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("🚀 Income Data Dashboard")

# Upload Data
uploaded_file = st.file_uploader("Upload file adultincome_bersih2.csv", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File berhasil dimuat!")

    # Mapping kolom income dan gender
    if 'income' in df.columns:
        df['income'] = df['income'].map({1: '>50K', 0: '<=50K'})

    if 'gender_Male' in df.columns:
        df['gender'] = df['gender_Male'].map({True: 'Male', False: 'Female'})

    # Cek Unik Value Income dan Gender
    st.subheader("🔎 Cek Nilai Unik Income dan Gender")
    st.write("Unique income values:", df['income'].unique())
    st.dataframe(df[['income', 'gender_Male']].head(20))
    st.divider()

    # Preview Data
    st.subheader("📄 Preview Data")
    st.dataframe(df.head())
    st.write("List kolom:", df.columns.tolist())
    st.divider()

    # 📊 Distribusi Umur
    st.subheader("📊 Distribusi Umur Pekerja")
    if "age" in df.columns:
        fig_age = px.histogram(df, x="age", nbins=20, title="Distribusi Umur")
        st.plotly_chart(fig_age)
        st.divider()

    # 🛠️ Rata-rata Jam Kerja per Workclass
    st.subheader("🛠️ Rata-rata Jam Kerja per Workclass")
    if "workclass" in df.columns and "hours_per_week" in df.columns:
        avg_hours = df.groupby('workclass')['hours_per_week'].mean().reset_index()
        fig_hours = px.bar(avg_hours, x="workclass", y="hours_per_week", title="Rata-rata Jam Kerja per Workclass")
        st.plotly_chart(fig_hours)
        st.divider()

    # 💵 Perbandingan Income
    st.subheader("💵 Perbandingan Income (<=50K vs >50K)")
    if "income" in df.columns:
        fig_income = px.pie(df, names="income", title="Distribusi Income")
        st.plotly_chart(fig_income)
        st.divider()

    # 💼 Income Berdasarkan Pekerjaan (>50K)
    st.subheader("💼 Income Berdasarkan Pekerjaan (>50K)")
    if "occupation" in df.columns and "income" in df.columns:
        occ_income = df[df['income'] == ">50K"]
        if not occ_income.empty:
            occ_income = occ_income.groupby('occupation').size().reset_index(name="count")
            fig_occ_income = px.bar(occ_income, x="occupation", y="count", title="Pekerjaan yang Banyak Menghasilkan >50K")
            st.plotly_chart(fig_occ_income)
            st.divider()

    # 👨‍👩‍👧‍👦 Income Berdasarkan Gender
    st.subheader("👨‍👩‍👧‍👦 Income Berdasarkan Gender")
    if "gender" in df.columns and "income" in df.columns:
        sex_income = df.groupby(['gender', 'income']).size().reset_index(name="count")
        if not sex_income.empty:
            fig_sex_income = px.bar(sex_income, x="gender", y="count", color="income", barmode="group", title="Income Berdasarkan Gender")
            st.plotly_chart(fig_sex_income)
            st.divider()

    # 📈 Income Berdasarkan Kelompok Umur
    st.subheader("📈 Income Berdasarkan Kelompok Umur")
    if "age_group" in df.columns and "income" in df.columns:
        agegroup_income = df.groupby(['age_group', 'income']).size().reset_index(name="count")
        fig_age_income = px.bar(agegroup_income, x="age_group", y="count", color="income", barmode="group", title="Income Berdasarkan Kelompok Umur")
        st.plotly_chart(fig_age_income)
        st.divider()

    # 🌎 Sebaran Negara Asal
    st.subheader("🌎 Sebaran Negara Asal")
    if "native_country" in df.columns:
        country_count = df['native_country'].value_counts().reset_index()
        country_count.columns = ['native_country', 'count']
        fig_country = px.bar(country_count, x="native_country", y="count",
                             labels={'native_country': 'Country', 'count': 'Count'},
                             title="Sebaran Negara Asal")
        st.plotly_chart(fig_country)
        st.divider()

    # 🔗 Korelasi Umur vs Jam Kerja (Rata-rata per Kelompok Umur)
    st.subheader("🔗 Rata-rata Jam Kerja per Kelompok Umur")
    if "age" in df.columns and "hours_per_week" in df.columns:
        df['age_bin'] = pd.cut(df['age'], bins=[0,20,30,40,50,60,70,80,100], labels=['0-20','21-30','31-40','41-50','51-60','61-70','71-80','81-100'])
        age_hours = df.groupby('age_bin')['hours_per_week'].mean().reset_index()
        fig_line = px.line(age_hours, x='age_bin', y='hours_per_week', title="Rata-rata Jam Kerja per Kelompok Umur")
        st.plotly_chart(fig_line)
        st.divider()


    # 📊 Distribusi Income berdasarkan Relationship
    st.subheader("📊 Distribusi Income berdasarkan Relationship")
    if "relationship" in df.columns and "income" in df.columns:
        rel_income = df.groupby(['relationship', 'income']).size().reset_index(name='count')
        fig_rel_income = px.bar(rel_income, x='relationship', y='count', color='income', barmode='group',
                                title="Income Berdasarkan Relationship")
        st.plotly_chart(fig_rel_income)
        st.divider()

    # 📊 Distribusi Relationship berdasarkan Gender
    st.subheader("📊 Distribusi Relationship berdasarkan Gender")
    if "relationship" in df.columns and "gender" in df.columns:
        rel_gender = df.groupby(['relationship', 'gender']).size().reset_index(name='count')
        fig_rel_gender = px.bar(rel_gender, x='relationship', y='count', color='gender', barmode='group',
                                title="Relationship Berdasarkan Gender")
        st.plotly_chart(fig_rel_gender)
        st.divider()
